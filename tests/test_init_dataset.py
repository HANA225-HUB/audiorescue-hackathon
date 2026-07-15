import csv
import tempfile
import unittest
from pathlib import Path

from scripts.dataset_spec import DatasetSpecError
from scripts.init_dataset import PROJECT_ROOT, initialize_dataset

try:
    from tests.test_dataset_spec import write_synthetic_spec
except ModuleNotFoundError:
    from test_dataset_spec import write_synthetic_spec


class InitDatasetTest(unittest.TestCase):
    def test_requires_explicit_spec_unless_example_mode_is_requested(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "data_local"

            with self.assertRaisesRegex(DatasetSpecError, "explicit dataset spec"):
                initialize_dataset(root)

            report = initialize_dataset(root, example_mode=True)
            self.assertEqual(report.root, root.resolve())

    def test_creates_neutral_workspace_templates_from_spec(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            spec_path = write_synthetic_spec(temp / "dataset_spec.json")
            root = temp / "data_local"

            report = initialize_dataset(root, spec_path=spec_path)

            self.assertEqual(report.root, root.resolve())
            expected_dirs = {
                "raw/clean/speaker_1",
                "raw/clean/speaker_2",
                "raw/noise",
                "raw/real",
                "controlled/dev",
                "controlled/holdout",
                "source_original/clean",
                "source_original/noise",
                "source_original/real",
                "outputs",
            }
            for relative in expected_dirs:
                self.assertTrue((root / relative).is_dir(), relative)

            with (root / "transcripts.tsv").open(
                encoding="utf-8", newline=""
            ) as handle:
                transcripts = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(len(transcripts), 2)
            self.assertEqual(
                {row["speaker_id"] for row in transcripts},
                {"speaker_1", "speaker_2"},
            )
            self.assertEqual(
                {row["reference_text"] for row in transcripts},
                {"synthetic reference one", "synthetic reference two"},
            )

            with (root / "recording_metadata.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                metadata = list(csv.DictReader(handle))
            self.assertEqual(len(metadata), 4)
            self.assertEqual(
                {row["source_type"] for row in metadata},
                {"clean", "noise", "real"},
            )
            self.assertTrue(
                all(row["consent_status"] == "pending_local_authorization" for row in metadata)
            )
            self.assertEqual(list(root.rglob("*.wav")), [])

            rendered = "\n".join(path.read_text(encoding="utf-8") for path in root.glob("*.*"))
            self.assertIn("speaker_", rendered)
            self.assertIn("sentence_", rendered)
            self.assertIn("noise_class_", rendered)
            self.assertNotRegex(rendered, r"speaker_id,.*\n[A-Z],")

    def test_rerun_preserves_human_edits_and_audio(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            spec_path = write_synthetic_spec(temp / "dataset_spec.json")
            root = temp / "data_local"
            initialize_dataset(root, spec_path=spec_path)
            licenses = root / "LICENSES.md"
            licenses.write_text("human decision\n", encoding="utf-8")
            audio = root / "raw/clean/speaker_1/sample_clean_1.wav"
            audio.write_bytes(b"recording")

            report = initialize_dataset(root, spec_path=spec_path)

            self.assertEqual(licenses.read_text(encoding="utf-8"), "human decision\n")
            self.assertEqual(audio.read_bytes(), b"recording")
            self.assertIn(licenses.resolve(), report.preserved_files)
            self.assertEqual(report.created_files, ())

    def test_rejects_file_as_dataset_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            spec_path = write_synthetic_spec(Path(temp_dir) / "dataset_spec.json")
            root = Path(temp_dir) / "not-a-directory"
            root.write_text("x", encoding="utf-8")
            with self.assertRaises(NotADirectoryError):
                initialize_dataset(root, spec_path=spec_path)

    def test_rejects_unignored_private_root_inside_repository(self) -> None:
        unsafe = PROJECT_ROOT / "unsafe_recordings_unit_test"
        self.assertFalse(unsafe.exists())
        with self.assertRaisesRegex(ValueError, "ignored by Git"):
            initialize_dataset(unsafe, example_mode=True)
        self.assertFalse(unsafe.exists())

    def test_rejects_symlink_escape_before_creating_templates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            spec_path = write_synthetic_spec(temp / "dataset_spec.json")
            root = temp / "data_local"
            outside = temp / "outside"
            root.mkdir()
            outside.mkdir()
            link = root / "raw"
            try:
                link.symlink_to(outside, target_is_directory=True)
            except (OSError, NotImplementedError) as exc:
                self.skipTest(f"directory symlink unavailable: {exc}")

            with self.assertRaisesRegex(ValueError, "unsafe path"):
                initialize_dataset(root, spec_path=spec_path)
            self.assertEqual(list(outside.rglob("*")), [])


if __name__ == "__main__":
    unittest.main()
