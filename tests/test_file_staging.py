from __future__ import annotations

import os
import tempfile
import time
import unittest
import wave
from pathlib import Path
from unittest import mock
from urllib.parse import unquote

from ui.file_staging import cleanup_stale_staging, is_valid_pcm_wav, stage_files_for_gradio


def _write_pcm_wav(path: Path, frames: bytes = b"\x00\x00\x01\x00") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(48000)
        wav_file.writeframes(frames)


def _write_zero_frame_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(48000)


def _write_truncated_wav(path: Path) -> None:
    _write_pcm_wav(path, b"\x00\x00\x01\x00")
    path.write_bytes(path.read_bytes()[:-1])


def _write_float_wav_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = b"\x00" * 4
    fmt = (
        (3).to_bytes(2, "little")
        + (1).to_bytes(2, "little")
        + (48000).to_bytes(4, "little")
        + (48000 * 4).to_bytes(4, "little")
        + (4).to_bytes(2, "little")
        + (32).to_bytes(2, "little")
    )
    path.write_bytes(
        b"RIFF"
        + (4 + 8 + len(fmt) + 8 + len(data)).to_bytes(4, "little")
        + b"WAVEfmt "
        + len(fmt).to_bytes(4, "little")
        + fmt
        + b"data"
        + len(data).to_bytes(4, "little")
        + data
    )


class FileStagingTests(unittest.TestCase):
    def test_stages_allowed_posix_and_windows_style_sources_neutrally(self) -> None:
        with tempfile.TemporaryDirectory() as source_tmp, tempfile.TemporaryDirectory() as stage_tmp:
            source_root = Path(source_tmp)
            staging_root = Path(stage_tmp) / "ui-stage"
            posix_source = source_root / "SECRET_WORKSPACE" / "SECRET_USER_original.wav"
            windows_style_source = source_root / "nested" / "C:\\SECRET_USER\\private_full.wav"
            posix_source.parent.mkdir(parents=True)
            windows_style_source.parent.mkdir(parents=True)
            _write_pcm_wav(posix_source)
            _write_pcm_wav(windows_style_source, b"\x02\x00\x03\x00")
            expected_original = posix_source.read_bytes()
            expected_full = windows_style_source.read_bytes()

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
            self.assertEqual(original.read_bytes(), expected_original)
            self.assertEqual(full.read_bytes(), expected_full)
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
            _write_pcm_wav(source, b"\x01\x00\x02\x00")
            first_bytes = source.read_bytes()

            first = stage_files_for_gradio(
                {"original_audio": source},
                allowed_roots=(source.parent,),
                base_dir=root,
                staging_root=staging_root,
            )
            _write_pcm_wav(source, b"\x03\x00\x04\x00")
            second_bytes = source.read_bytes()
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
            self.assertEqual(first_path.read_bytes(), first_bytes)
            self.assertEqual(second_path.read_bytes(), second_bytes)

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
            _write_pcm_wav(source)

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
            _write_pcm_wav(source)
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
                _write_pcm_wav(source, index.to_bytes(2, "little") * 2)
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

    def test_rejects_invalid_wav_roles_before_staging(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as stage_tmp:
            root = Path(tmp)
            invalid_root = root / "allowed"
            invalid_root.mkdir()
            cases = {
                "empty": invalid_root / "empty.wav",
                "random": invalid_root / "random.wav",
                "zero_frame": invalid_root / "zero_frame.wav",
                "truncated": invalid_root / "truncated.wav",
                "non_pcm": invalid_root / "non_pcm.wav",
                "malformed": invalid_root / "malformed.wav",
            }
            cases["empty"].write_bytes(b"")
            cases["random"].write_bytes(b"not a wav")
            _write_zero_frame_wav(cases["zero_frame"])
            _write_truncated_wav(cases["truncated"])
            _write_float_wav_header(cases["non_pcm"])
            cases["malformed"].write_bytes(b"RIFF\x04\x00\x00\x00WAVE")

            for name, source in cases.items():
                with self.subTest(name=name):
                    self.assertFalse(is_valid_pcm_wav(source))
                    staged = stage_files_for_gradio(
                        {
                            "original_audio": source,
                            "mixed_audio": source,
                            "full_audio": source,
                        },
                        allowed_roots=(invalid_root,),
                        base_dir=root,
                        staging_root=Path(stage_tmp) / f"ui-stage-{name}",
                    )
                    self.assertIsNone(staged["original_audio"])
                    self.assertIsNone(staged["mixed_audio"])
                    self.assertIsNone(staged["full_audio"])


if __name__ == "__main__":
    unittest.main()
