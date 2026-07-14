from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock
from urllib.parse import unquote

from ui.file_staging import cleanup_stale_staging, stage_files_for_gradio


class FileStagingTests(unittest.TestCase):
    def test_stages_allowed_posix_and_windows_style_sources_neutrally(self) -> None:
        with tempfile.TemporaryDirectory() as source_tmp, tempfile.TemporaryDirectory() as stage_tmp:
            source_root = Path(source_tmp)
            staging_root = Path(stage_tmp) / "ui-stage"
            posix_source = source_root / "SECRET_WORKSPACE" / "SECRET_USER_original.wav"
            windows_style_source = source_root / "nested" / "C:\\SECRET_USER\\private_full.wav"
            posix_source.parent.mkdir(parents=True)
            windows_style_source.parent.mkdir(parents=True)
            posix_source.write_bytes(b"original-bytes")
            windows_style_source.write_bytes(b"full-bytes")

            staged = stage_files_for_gradio(
                {
                    "original_audio": posix_source,
                    "full_audio": windows_style_source,
                },
                allowed_roots=(source_root,),
                base_dir=source_root,
                staging_root=staging_root,
            )

            original = Path(staged["original_audio"] or "")
            full = Path(staged["full_audio"] or "")
            self.assertEqual(original.read_bytes(), b"original-bytes")
            self.assertEqual(full.read_bytes(), b"full-bytes")
            self.assertEqual(original.name, "original.wav")
            self.assertEqual(full.name, "full.wav")

            decoded_paths = unquote(" ".join(str(value) for value in staged.values()))
            self.assertNotIn("SECRET_WORKSPACE", decoded_paths)
            self.assertNotIn("SECRET_USER", decoded_paths)
            self.assertNotIn("private_full.wav", decoded_paths)
            self.assertNotIn("SECRET_USER_original.wav", decoded_paths)
            self.assertTrue(original.is_relative_to(staging_root))
            self.assertTrue(full.is_relative_to(staging_root))

    def test_rejects_existing_files_outside_allowed_roots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as stage_tmp:
            root = Path(tmp)
            allowed_root = root / "allowed"
            outside = root / "outside" / "SECRET_USER.wav"
            allowed_root.mkdir()
            outside.parent.mkdir()
            outside.write_bytes(b"secret")

            staged = stage_files_for_gradio(
                {"original_audio": outside},
                allowed_roots=(allowed_root,),
                base_dir=root,
                staging_root=Path(stage_tmp) / "ui-stage",
            )

            self.assertIsNone(staged["original_audio"])

    @unittest.skipUnless(hasattr(os, "symlink"), "symlink is not available on this platform")
    def test_rejects_allowed_root_symlink_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as stage_tmp:
            root = Path(tmp)
            allowed_root = root / "allowed"
            outside = root / "outside" / "outside.wav"
            allowed_root.mkdir()
            outside.parent.mkdir()
            outside.write_bytes(b"outside")
            link = allowed_root / "link.wav"
            os.symlink(outside, link)

            staged = stage_files_for_gradio(
                {"original_audio": link},
                allowed_roots=(allowed_root,),
                base_dir=root,
                staging_root=Path(stage_tmp) / "ui-stage",
            )

            self.assertIsNone(staged["original_audio"])

    def test_rejects_relative_traversal_after_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as stage_tmp:
            root = Path(tmp)
            allowed_root = root / "allowed"
            outside = root / "outside" / "SECRET_USER.wav"
            allowed_root.mkdir()
            outside.parent.mkdir()
            outside.write_bytes(b"outside")

            staged = stage_files_for_gradio(
                {"original_audio": "allowed/../outside/SECRET_USER.wav"},
                allowed_roots=(allowed_root,),
                base_dir=root,
                staging_root=Path(stage_tmp) / "ui-stage",
            )

            self.assertIsNone(staged["original_audio"])

    def test_cleanup_only_removes_stale_staging_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as stage_tmp:
            staging_root = Path(stage_tmp) / "ui-stage"
            stale = staging_root / ("0" * 32)
            fresh = staging_root / ("1" * 32)
            unrelated = staging_root / "not-a-session"
            stale.mkdir(parents=True)
            fresh.mkdir()
            unrelated.mkdir()
            old_time = time.time() - 7200
            os.utime(stale, (old_time, old_time))

            cleanup_stale_staging(staging_root=staging_root, ttl_seconds=3600)

            self.assertFalse(stale.exists())
            self.assertTrue(fresh.exists())
            self.assertTrue(unrelated.exists())

    @unittest.skipUnless(hasattr(os, "symlink"), "symlink is not available on this platform")
    def test_cleanup_does_not_follow_malicious_session_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as stage_tmp:
            root = Path(stage_tmp)
            staging_root = root / "ui-stage"
            outside = root / "outside"
            outside.mkdir()
            protected = outside / "keep.txt"
            protected.write_text("do-not-delete", encoding="utf-8")
            staging_root.mkdir()
            link = staging_root / ("a" * 32)
            os.symlink(outside, link)

            cleanup_stale_staging(staging_root=staging_root, ttl_seconds=60, now=time.time() + 7200)

            self.assertTrue(link.exists())
            self.assertTrue(protected.exists())
            self.assertEqual(protected.read_text(encoding="utf-8"), "do-not-delete")

    @unittest.skipUnless(hasattr(os, "symlink"), "symlink is not available on this platform")
    def test_cleanup_removes_nested_symlink_without_deleting_target(self) -> None:
        with tempfile.TemporaryDirectory() as stage_tmp:
            root = Path(stage_tmp)
            staging_root = root / "ui-stage"
            outside = root / "outside"
            session = staging_root / ("b" * 32)
            outside.mkdir()
            session.mkdir(parents=True)
            protected = outside / "keep.txt"
            protected.write_text("do-not-delete", encoding="utf-8")
            os.symlink(outside, session / "nested-link")
            old_time = time.time() - 7200
            os.utime(session, (old_time, old_time))

            cleanup_stale_staging(staging_root=staging_root, ttl_seconds=3600)

            self.assertFalse(session.exists())
            self.assertTrue(protected.exists())
            self.assertEqual(protected.read_text(encoding="utf-8"), "do-not-delete")

    def test_isolates_consecutive_sessions_for_same_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as stage_tmp:
            root = Path(tmp)
            staging_root = Path(stage_tmp) / "ui-stage"
            source = root / "allowed" / "SECRET_USER_same.wav"
            source.parent.mkdir()
            source.write_bytes(b"first")

            first = stage_files_for_gradio(
                {"original_audio": source},
                allowed_roots=(source.parent,),
                base_dir=root,
                staging_root=staging_root,
            )
            source.write_bytes(b"second")
            second = stage_files_for_gradio(
                {"original_audio": source},
                allowed_roots=(source.parent,),
                base_dir=root,
                staging_root=staging_root,
            )

            first_path = Path(first["original_audio"] or "")
            second_path = Path(second["original_audio"] or "")
            self.assertNotEqual(first_path.parent, second_path.parent)
            self.assertEqual(first_path.name, "original.wav")
            self.assertEqual(second_path.name, "original.wav")
            self.assertEqual(first_path.read_bytes(), b"first")
            self.assertEqual(second_path.read_bytes(), b"second")

    def test_missing_directory_or_disappearing_sources_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as stage_tmp:
            root = Path(tmp)
            allowed_root = root / "allowed"
            allowed_root.mkdir()
            missing = allowed_root / "missing.wav"
            directory = allowed_root / "directory.wav"
            directory.mkdir()

            staged = stage_files_for_gradio(
                {"original_audio": missing, "mixed_audio": directory},
                allowed_roots=(allowed_root,),
                base_dir=root,
                staging_root=Path(stage_tmp) / "ui-stage",
            )

            self.assertIsNone(staged["original_audio"])
            self.assertIsNone(staged["mixed_audio"])

    def test_copy_failure_fails_closed_without_source_path_in_exception(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as stage_tmp:
            root = Path(tmp)
            source = root / "allowed" / "SECRET_USER_unreadable.wav"
            source.parent.mkdir()
            source.write_bytes(b"secret")

            with mock.patch("ui.file_staging.shutil.copyfile", side_effect=OSError("copy denied")):
                staged = stage_files_for_gradio(
                    {"original_audio": source},
                    allowed_roots=(source.parent,),
                    base_dir=root,
                    staging_root=Path(stage_tmp) / "ui-stage",
                )

            self.assertIsNone(staged["original_audio"])

    def test_prepare_or_chmod_failure_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as stage_tmp:
            root = Path(tmp)
            source = root / "allowed" / "SECRET_USER_original.wav"
            source.parent.mkdir()
            source.write_bytes(b"secret")
            staging_root = Path(stage_tmp) / "ui-stage"

            with mock.patch("ui.file_staging.os.chmod", side_effect=OSError("SECRET_USER")):
                staged = stage_files_for_gradio(
                    {"original_audio": source},
                    allowed_roots=(source.parent,),
                    base_dir=root,
                    staging_root=staging_root,
                )
            self.assertIsNone(staged["original_audio"])

            with mock.patch(
                "ui.file_staging._prepare_session_dir",
                side_effect=RuntimeError("SECRET_USER"),
            ):
                staged = stage_files_for_gradio(
                    {"mixed_audio": source},
                    allowed_roots=(source.parent,),
                    base_dir=root,
                    staging_root=staging_root,
                )
            self.assertIsNone(staged["mixed_audio"])

    def test_many_sensitive_source_names_do_not_appear_in_staged_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as stage_tmp:
            root = Path(tmp)
            staging_root = Path(stage_tmp) / "ui-stage"
            exposed = []
            for index in range(50):
                sentinel = f"SECRET_USER_{index:02d}"
                source = root / "outputs" / "job" / f"SECRET_WORKSPACE_{index:02d}" / f"{sentinel}.wav"
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_bytes(f"content-{index}".encode("ascii"))
                staged = stage_files_for_gradio(
                    {"original_audio": source},
                    allowed_roots=(root / "outputs" / "job",),
                    base_dir=root,
                    staging_root=staging_root,
                )
                exposed.append(str(staged["original_audio"]))

            decoded = unquote(" ".join(exposed))
            self.assertNotIn("SECRET_USER", decoded)
            self.assertNotIn("SECRET_WORKSPACE", decoded)
            self.assertNotIn("outputs/job", decoded)
            self.assertEqual(decoded.count("original.wav"), 50)


if __name__ == "__main__":
    unittest.main()
