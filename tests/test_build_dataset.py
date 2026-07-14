import csv
import hashlib
import json
import math
import random
import tempfile
import unittest
import wave
from array import array
from pathlib import Path

from scripts.build_dataset import (
    APPROVED_CONSENT_TOKEN,
    LATIN_SQUARE,
    MANIFEST_FIELDS,
    PEAK_LIMIT,
    PENDING_CONSENT_TOKEN,
    SAMPLE_RATE,
    SENTENCES,
    DatasetBuildError,
    build_dataset,
    mix_pcm16,
    read_pcm16_mono,
)


def write_test_wav(
    path: Path,
    samples: list[int],
    *,
    sample_rate: int = SAMPLE_RATE,
    channels: int = 1,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = array("h", samples)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(encoded.tobytes())


def make_sources(root: Path, *, include_real: bool = True) -> None:
    for speaker_index, speaker_id in enumerate(("A", "B", "C"), start=1):
        for sentence_index, sentence_id in enumerate(SENTENCES, start=1):
            frequency_step = 3 + speaker_index + sentence_index
            samples = [
                round(9_000 * math.sin(2 * math.pi * frequency_step * index / 401))
                for index in range(800 + sentence_index * 20)
            ]
            write_test_wav(
                root
                / "raw"
                / "clean"
                / f"spk{speaker_id}"
                / f"clean_spk{speaker_id}_{sentence_id.lower()}.wav",
                samples,
            )

    for noise_index, noise_type in enumerate(("fan", "keyboard", "traffic"), start=1):
        rng = random.Random(100 + noise_index)
        samples = [rng.randint(-7_000, 7_000) for _ in range(2_400)]
        write_test_wav(
            root / "raw" / "noise" / f"noise_{noise_type}_take01.wav",
            samples,
        )

    if include_real:
        real_specs = (("A", "fan"), ("B", "keyboard"), ("C", "traffic"))
        for real_index, (speaker_id, noise_type) in enumerate(real_specs, start=1):
            samples = [
                round(8_000 * math.sin(2 * math.pi * (5 + real_index) * index / 431))
                for index in range(900 + real_index * 10)
            ]
            write_test_wav(
                root
                / "raw"
                / "real"
                / f"real_spk{speaker_id}_{noise_type}_r01.wav",
                samples,
            )


def approve_sources(root: Path) -> None:
    rows: list[dict[str, str]] = []
    for standardized in sorted((root / "raw").rglob("*.wav")):
        relative = standardized.relative_to(root)
        original = root / "source_original" / relative.relative_to("raw")
        original.parent.mkdir(parents=True, exist_ok=True)
        original.write_bytes(standardized.read_bytes())
        digest = hashlib.sha256(standardized.read_bytes()).hexdigest()
        rows.append(
            {
                "asset_id": standardized.stem,
                "source_original_path": original.relative_to(root).as_posix(),
                "relative_path": relative.as_posix(),
                "consent_status": "yes",
                "source_original_sha256": digest,
                "standardized_sha256": digest,
            }
        )
    with (root / "recording_metadata.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

class MixPcm16Test(unittest.TestCase):
    def test_exact_snr_and_peak_protection(self) -> None:
        clean = [30_000, -30_000, 25_000, -25_000] * 100
        noise = [25_000, 20_000, -25_000, -20_000] * 130

        gains = []
        for target_snr in (5.0, 0.0, -5.0):
            with self.subTest(target_snr=target_snr):
                result = mix_pcm16(
                    clean,
                    noise,
                    target_snr,
                    noise_offset_samples=20,
                )
                achieved_snr = 10 * math.log10(
                    result.clean_power
                    / (result.alpha * result.alpha * result.noise_power)
                )
                self.assertAlmostEqual(achieved_snr, target_snr, places=12)
                self.assertLessEqual(
                    max(abs(value) for value in result.samples) / 32768,
                    PEAK_LIMIT,
                )
                gains.append(result.final_gain)
        self.assertTrue(any(gain < 1.0 for gain in gains))

    def test_insufficient_noise_is_rejected_instead_of_looped(self) -> None:
        with self.assertRaisesRegex(DatasetBuildError, "不能循环填充"):
            mix_pcm16([1, -1, 2, -2], [1, -1], 0.0)


class BuildDatasetTest(unittest.TestCase):
    def test_builds_names_latin_square_splits_and_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "data_local"
            make_sources(root)
            approve_sources(root)

            rows = build_dataset(
                root, consent_or_license=APPROVED_CONSENT_TOKEN
            )

            self.assertEqual(len(rows), 33)
            self.assertEqual(sum(row["split"] == "dev" for row in rows), 18)
            self.assertEqual(
                sum(row["split"] == "locked_test" for row in rows), 9
            )
            self.assertEqual(
                sum(row["source_type"] == "clean_control" for row in rows), 3
            )
            self.assertEqual(sum(row["source_type"] == "real" for row in rows), 3)
            self.assertEqual(tuple(rows[0]), MANIFEST_FIELDS)

            by_id = {row["sample_id"]: row for row in rows}
            for row in rows:
                if row["source_type"] != "controlled_mix":
                    continue
                self.assertEqual(
                    row["noise_type"],
                    LATIN_SQUARE[(row["speaker_id"], row["sentence_id"])],
                )
            main = by_id["mix_spkB_s03_fan_snr000"]
            self.assertEqual(LATIN_SQUARE[("B", "S03")], "fan")
            self.assertEqual(main["split"], "locked_test")
            self.assertTrue(main["is_locked"])
            self.assertTrue(main["is_demo_candidate"])
            self.assertEqual(main["snr_db"], 0)
            self.assertEqual(
                main["consent_or_license"], APPROVED_CONSENT_TOKEN
            )
            self.assertEqual(
                main["mixed_path"],
                "controlled/locked_test/mix_spkB_s03_fan_snr000.wav",
            )

            same_cell = [
                row
                for row in rows
                if row["speaker_id"] == "B"
                and row["sentence_id"] == "S03"
                and row["source_type"] == "controlled_mix"
            ]
            self.assertEqual(len({row["mix_seed"] for row in same_cell}), 1)
            self.assertEqual(
                len({row["noise_offset_seconds"] for row in same_cell}), 1
            )

            manifest_json = json.loads(
                (root / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest_json, rows)
            with (root / "manifest.csv").open(
                "r", encoding="utf-8", newline=""
            ) as file_handle:
                csv_rows = list(csv.DictReader(file_handle))
            self.assertEqual(len(csv_rows), 33)
            self.assertEqual(tuple(csv_rows[0]), MANIFEST_FIELDS)
            self.assertEqual(csv_rows[0]["sample_id"], rows[0]["sample_id"])

            mixed = read_pcm16_mono(root / main["mixed_path"])
            self.assertEqual(mixed.sample_rate, SAMPLE_RATE)
            self.assertEqual(len(main["sha256"]), 64)
            self.assertEqual(
                main["sha256"],
                hashlib.sha256((root / main["mixed_path"]).read_bytes()).hexdigest(),
            )

    def test_rebuild_with_same_seed_is_byte_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "data_local"
            make_sources(root)

            first_rows = build_dataset(root, base_seed=12345)
            first_csv = (root / "manifest.csv").read_bytes()
            first_json = (root / "manifest.json").read_bytes()
            first_hashes = [row["sha256"] for row in first_rows]

            second_rows = build_dataset(root, base_seed=12345, overwrite=True)

            self.assertEqual(second_rows, first_rows)
            self.assertEqual((root / "manifest.csv").read_bytes(), first_csv)
            self.assertEqual((root / "manifest.json").read_bytes(), first_json)
            self.assertEqual(
                [row["sha256"] for row in second_rows],
                first_hashes,
            )

    def test_existing_outputs_require_explicit_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "data_local"
            make_sources(root)
            build_dataset(root)

            with self.assertRaisesRegex(DatasetBuildError, "--overwrite"):
                build_dataset(root)

    def test_consent_uses_exact_workflow_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "data_local"
            make_sources(root)

            for invalid_value in ("team-approved", "yes", "TODO", ""):
                with self.subTest(consent_or_license=invalid_value):
                    with self.assertRaisesRegex(
                        DatasetBuildError, "exact workflow token"
                    ):
                        build_dataset(
                            root, consent_or_license=invalid_value
                        )

            self.assertFalse((root / "manifest.csv").exists())
            rows = build_dataset(root)
            self.assertTrue(rows)
            self.assertEqual(
                {row["consent_or_license"] for row in rows},
                {PENDING_CONSENT_TOKEN},
            )

    def test_formal_consent_requires_all_15_hashed_master_records(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "data_local"
            make_sources(root)
            with self.assertRaisesRegex(DatasetBuildError, "授权台账"):
                build_dataset(root, consent_or_license=APPROVED_CONSENT_TOKEN)

            approve_sources(root)
            with (root / "recording_metadata.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["consent_status"] = "no"
            with (root / "recording_metadata.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(DatasetBuildError, "必须为 yes"):
                build_dataset(root, consent_or_license=APPROVED_CONSENT_TOKEN)

    def test_wrong_source_format_requires_explicit_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "data_local"
            make_sources(root)
            bad_path = root / "raw" / "clean" / "spkA" / "clean_spkA_s01.wav"
            write_test_wav(bad_path, [1, -1] * 400, sample_rate=44_100)

            with self.assertRaisesRegex(DatasetBuildError, "不会静默重采样"):
                build_dataset(root)

            self.assertFalse((root / "manifest.csv").exists())

    def test_missing_real_requires_an_explicit_intermediate_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "data_local"
            make_sources(root, include_real=False)

            with self.assertRaisesRegex(DatasetBuildError, "real_spkA_fan_r01"):
                build_dataset(root)

            rows = build_dataset(root, allow_missing_real=True)
            self.assertEqual(len(rows), 30)
            self.assertEqual(sum(row["source_type"] == "real" for row in rows), 0)


if __name__ == "__main__":
    unittest.main()
