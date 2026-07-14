import csv
import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_blind_ab import (
    BlindPair,
    _ensure_private_destination,
    prepare_blind_test,
    read_pairs_csv,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PrepareBlindABTest(unittest.TestCase):
    def test_prepares_balanced_anonymous_byte_identical_ballots(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            original = root / "original.wav"
            enhanced = root / "enhanced.wav"
            original.write_bytes(b"RIFF-original-audio")
            enhanced.write_bytes(b"RIFF-enhanced-audio")
            destination = root / "session"

            report = prepare_blind_test(
                [BlindPair("demo", original, enhanced)],
                destination,
                seed=7,
            )

            self.assertEqual(report.pair_count, 1)
            with report.answer_key.open(encoding="utf-8", newline="") as handle:
                answers = list(csv.DictReader(handle))
            self.assertEqual(len(answers), 3)
            original_as_a = sum(row["A_role"] == "original" for row in answers)
            self.assertIn(original_as_a, {1, 2})

            source_digests = {digest(original), digest(enhanced)}
            for rater in ("A", "B", "C"):
                ballot_path = destination / f"rater_{rater}" / "ballot.csv"
                with ballot_path.open(encoding="utf-8", newline="") as handle:
                    ballot = list(csv.DictReader(handle))
                self.assertEqual(len(ballot), 1)
                self.assertNotIn("role", " ".join(ballot[0]).lower())
                copied = {
                    digest(destination / f"rater_{rater}" / ballot[0]["clip_A"]),
                    digest(destination / f"rater_{rater}" / ballot[0]["clip_B"]),
                }
                self.assertEqual(copied, source_digests)

    def test_multi_pair_schedule_is_balanced_for_every_rater(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pairs = []
            for index in range(6):
                original = root / f"original_{index}.wav"
                enhanced = root / f"enhanced_{index}.wav"
                original.write_bytes(f"original-{index}".encode())
                enhanced.write_bytes(f"enhanced-{index}".encode())
                pairs.append(BlindPair(f"sample_{index}", original, enhanced))

            report = prepare_blind_test(pairs, root / "balanced", seed=11)
            with report.answer_key.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            for rater in ("A", "B", "C"):
                rater_rows = [row for row in rows if row["rater"] == rater]
                original_as_a = sum(
                    row["A_role"] == "original" for row in rater_rows
                )
                self.assertLessEqual(abs(original_as_a - (len(rater_rows) - original_as_a)), 1)

    def test_repository_output_must_be_git_ignored(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        with self.assertRaisesRegex(ValueError, "ignored by Git"):
            _ensure_private_destination(project_root / "unsafe_blind_answers")

    def test_reads_relative_paths_from_pairs_csv_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "before.wav").write_bytes(b"before")
            (root / "after.wav").write_bytes(b"after")
            pairs_csv = root / "pairs.csv"
            pairs_csv.write_text(
                "sample_id,original_path,enhanced_path\n"
                "demo,before.wav,after.wav\n",
                encoding="utf-8",
            )
            pairs = read_pairs_csv(pairs_csv)
            self.assertEqual(pairs[0].original_path, (root / "before.wav").resolve())
            self.assertEqual(pairs[0].enhanced_path, (root / "after.wav").resolve())

    def test_refuses_existing_output_duplicate_ids_and_same_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            before = root / "before.wav"
            after = root / "after.wav"
            before.write_bytes(b"before")
            after.write_bytes(b"after")
            destination = root / "session"
            destination.mkdir()
            pair = BlindPair("demo", before, after)
            with self.assertRaises(FileExistsError):
                prepare_blind_test([pair], destination)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                prepare_blind_test([pair, pair], root / "duplicate")
            with self.assertRaisesRegex(ValueError, "same file"):
                prepare_blind_test(
                    [BlindPair("same", before, before)], root / "same"
                )


if __name__ == "__main__":
    unittest.main()
