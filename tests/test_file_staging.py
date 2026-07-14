from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path
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


if __name__ == "__main__":
    unittest.main()
