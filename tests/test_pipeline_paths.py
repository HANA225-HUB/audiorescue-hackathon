import tempfile
import unittest
from pathlib import Path

from core.pipeline import create_job_paths


class PipelinePathContractTest(unittest.TestCase):
    def test_pipeline_creates_expected_job_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = create_job_paths(temp_dir, "job_001")
            self.assertTrue(paths.root.is_dir())
            self.assertEqual(paths.original.name, "original.wav")
            self.assertEqual(paths.enhanced_full.name, "enhanced_full.wav")
            self.assertEqual(paths.enhanced_mix.name, "enhanced_mix.wav")
            self.assertTrue(paths.original.is_absolute())

    def test_pipeline_rejects_path_traversal_job_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(ValueError):
                create_job_paths(Path(temp_dir), "../escape")


if __name__ == "__main__":
    unittest.main()
