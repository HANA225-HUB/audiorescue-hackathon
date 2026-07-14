import csv
import tempfile
import unittest
from pathlib import Path

from scripts.init_dataset import (
    DIRECTORIES,
    LATIN_SQUARE,
    PROJECT_ROOT,
    SENTENCES,
    initialize_dataset,
)


class InitDatasetTest(unittest.TestCase):
    def test_creates_exact_private_workspace_templates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "data_local"
            report = initialize_dataset(root)

            self.assertEqual(report.root, root.resolve())
            for relative in DIRECTORIES:
                self.assertTrue((root / relative).is_dir())

            with (root / "transcripts.tsv").open(
                encoding="utf-8", newline=""
            ) as handle:
                transcripts = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(len(transcripts), 9)
            self.assertEqual(
                {(row["speaker_id"], row["sentence_id"]) for row in transcripts},
                {
                    (speaker, sentence.upper())
                    for speaker in ("A", "B", "C")
                    for sentence in SENTENCES
                },
            )
            for row in transcripts:
                speaker = row["speaker_id"]
                sentence = row["sentence_id"].lower()
                self.assertEqual(row["noise_type"], LATIN_SQUARE[speaker][sentence])
                self.assertEqual(
                    row["split"],
                    "dev" if sentence in {"s01", "s02"} else "locked_test",
                )
                self.assertEqual(row["reference_text"], SENTENCES[sentence])

            with (root / "recording_metadata.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                metadata = list(csv.DictReader(handle))
            self.assertEqual(len(metadata), 15)
            self.assertEqual(
                {row["source_type"] for row in metadata},
                {"clean", "noise", "real"},
            )
            self.assertTrue(all(row["consent_status"] == "pending" for row in metadata))
            self.assertTrue(all(row["source_original_path"] == "" for row in metadata))
            self.assertTrue(
                all(row["source_original_sha256"] == "" for row in metadata)
            )
            self.assertTrue(
                all(row["standardized_sha256"] == "" for row in metadata)
            )
            self.assertEqual(list(root.rglob("*.wav")), [])

    def test_rerun_preserves_human_edits_and_audio(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "data_local"
            initialize_dataset(root)
            licenses = root / "LICENSES.md"
            licenses.write_text("human decision\n", encoding="utf-8")
            audio = root / "raw/clean/spkA/clean_spkA_s01.wav"
            audio.write_bytes(b"recording")

            report = initialize_dataset(root)

            self.assertEqual(licenses.read_text(encoding="utf-8"), "human decision\n")
            self.assertEqual(audio.read_bytes(), b"recording")
            self.assertIn(licenses.resolve(), report.preserved_files)
            self.assertEqual(report.created_files, ())

    def test_rejects_file_as_dataset_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "not-a-directory"
            root.write_text("x", encoding="utf-8")
            with self.assertRaises(NotADirectoryError):
                initialize_dataset(root)

    def test_rejects_unignored_private_root_inside_repository(self) -> None:
        unsafe = PROJECT_ROOT / "unsafe_recordings_unit_test"
        self.assertFalse(unsafe.exists())
        with self.assertRaisesRegex(ValueError, "ignored by Git"):
            initialize_dataset(unsafe)
        self.assertFalse(unsafe.exists())


if __name__ == "__main__":
    unittest.main()
