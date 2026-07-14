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
    MANIFEST_FIELDS,
    PEAK_LIMIT,
    PENDING_CONSENT_TOKEN,
    SAMPLE_RATE,
    DatasetBuildError,
    build_dataset,
    mix_pcm16,
    read_pcm16_mono,
)
from scripts.dataset_spec import load_dataset_spec

try:
    from tests.test_dataset_spec import write_synthetic_spec
except ModuleNotFoundError:
    from test_dataset_spec import write_synthetic_spec


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


def make_sources(root: Path, spec_path: Path, *, include_real: bool = True) -> None:
    spec = load_dataset_spec(spec_path)
    for index, item in enumerate(spec.clean_recordings, start=1):
        samples = [
            round(9_000 * math.sin(2 * math.pi * (index + 3) * frame / 401))
            for frame in range(900 + index * 40)
        ]
        write_test_wav(root / item.path, samples)
    for index, item in enumerate(spec.noise_recordings, start=1):
        rng = random.Random(100 + index)
        samples = [rng.randint(-7_000, 7_000) for _ in range(2_400)]
        write_test_wav(root / item.path, samples)
    if include_real:
        for index, item in enumerate(spec.real_recordings, start=1):
            samples = [
                round(8_000 * math.sin(2 * math.pi * (index + 5) * frame / 431))
                for frame in range(900 + index * 30)
            ]
            write_test_wav(root / item.path, samples)


def approve_sources(root: Path, spec_path: Path) -> None:
    spec = load_dataset_spec(spec_path)
    rows: list[dict[str, str]] = []
    for asset_id, standardized in spec.master_audio_paths().items():
        standardized_path = root / standardized
        original = root / "source_original" / standardized
        original.parent.mkdir(parents=True, exist_ok=True)
        original.write_bytes(standardized_path.read_bytes())
        digest = hashlib.sha256(standardized_path.read_bytes()).hexdigest()
        rows.append(
            {
                "asset_id": asset_id,
                "source_original_path": original.relative_to(root).as_posix(),
                "relative_path": standardized.as_posix(),
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
        with self.assertRaisesRegex(DatasetBuildError, "loop"):
            mix_pcm16([1, -1, 2, -2], [1, -1], 0.0)


class BuildDatasetTest(unittest.TestCase):
    def test_builds_from_explicit_synthetic_spec(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "data_local"
            spec_path = write_synthetic_spec(Path(temp_dir) / "dataset_spec.json")
            make_sources(root, spec_path)
            approve_sources(root, spec_path)

            rows = build_dataset(
                root,
                spec_path=spec_path,
                consent_or_license=APPROVED_CONSENT_TOKEN,
            )

            self.assertEqual(len(rows), 4)
            self.assertEqual(sum(row["split"] == "dev" for row in rows), 1)
            self.assertEqual(sum(row["split"] == "holdout" for row in rows), 1)
            self.assertEqual(sum(row["source_type"] == "clean_control" for row in rows), 1)
            self.assertEqual(sum(row["source_type"] == "real" for row in rows), 1)
            self.assertEqual(tuple(rows[0]), MANIFEST_FIELDS)

            by_id = {row["sample_id"]: row for row in rows}
            main = by_id["sample_mix_1"]
            self.assertEqual(main["noise_type"], "noise_class_1")
            self.assertEqual(main["split"], "dev")
            self.assertFalse(main["is_locked"])
            self.assertTrue(main["is_demo_candidate"])
            self.assertEqual(main["snr_db"], 0)
            self.assertEqual(main["consent_or_license"], APPROVED_CONSENT_TOKEN)
            self.assertEqual(main["mixed_path"], "controlled/dev/sample_mix_1.wav")

            manifest_json = json.loads(
                (root / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest_json, rows)
            with (root / "manifest.csv").open(
                "r", encoding="utf-8", newline=""
            ) as file_handle:
                csv_rows = list(csv.DictReader(file_handle))
            self.assertEqual(len(csv_rows), 4)
            self.assertEqual(tuple(csv_rows[0]), MANIFEST_FIELDS)

            mixed = read_pcm16_mono(root / main["mixed_path"])
            self.assertEqual(mixed.sample_rate, SAMPLE_RATE)
            self.assertEqual(
                main["sha256"],
                hashlib.sha256((root / main["mixed_path"]).read_bytes()).hexdigest(),
            )

    def test_rebuild_with_same_seed_is_byte_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "data_local"
            spec_path = write_synthetic_spec(Path(temp_dir) / "dataset_spec.json")
            make_sources(root, spec_path)

            first_rows = build_dataset(root, spec_path=spec_path, base_seed=12345)
            first_csv = (root / "manifest.csv").read_bytes()
            first_json = (root / "manifest.json").read_bytes()
            first_hashes = [row["sha256"] for row in first_rows]

            second_rows = build_dataset(
                root,
                spec_path=spec_path,
                base_seed=12345,
                overwrite=True,
            )

            self.assertEqual(second_rows, first_rows)
            self.assertEqual((root / "manifest.csv").read_bytes(), first_csv)
            self.assertEqual((root / "manifest.json").read_bytes(), first_json)
            self.assertEqual([row["sha256"] for row in second_rows], first_hashes)

    def test_existing_outputs_require_explicit_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "data_local"
            spec_path = write_synthetic_spec(Path(temp_dir) / "dataset_spec.json")
            make_sources(root, spec_path)
            build_dataset(root, spec_path=spec_path)

            with self.assertRaisesRegex(DatasetBuildError, "--overwrite"):
                build_dataset(root, spec_path=spec_path)

    def test_consent_uses_exact_workflow_tokens_from_spec_module(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "data_local"
            spec_path = write_synthetic_spec(Path(temp_dir) / "dataset_spec.json")
            make_sources(root, spec_path)

            for invalid_value in ("team-approved", "yes", "TODO", ""):
                with self.subTest(consent_or_license=invalid_value):
                    with self.assertRaisesRegex(DatasetBuildError, "workflow token"):
                        build_dataset(
                            root,
                            spec_path=spec_path,
                            consent_or_license=invalid_value,
                        )

            rows = build_dataset(root, spec_path=spec_path)
            self.assertEqual(
                {row["consent_or_license"] for row in rows},
                {PENDING_CONSENT_TOKEN},
            )

    def test_formal_consent_requires_spec_masters_not_fixed_count(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "data_local"
            spec_path = write_synthetic_spec(Path(temp_dir) / "dataset_spec.json")
            make_sources(root, spec_path)
            with self.assertRaisesRegex(DatasetBuildError, "recording metadata"):
                build_dataset(
                    root,
                    spec_path=spec_path,
                    consent_or_license=APPROVED_CONSENT_TOKEN,
                )

            approve_sources(root, spec_path)
            rows = build_dataset(
                root,
                spec_path=spec_path,
                consent_or_license=APPROVED_CONSENT_TOKEN,
            )
            self.assertEqual(len(rows), 4)

    def test_wrong_source_format_requires_explicit_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "data_local"
            spec_path = write_synthetic_spec(Path(temp_dir) / "dataset_spec.json")
            make_sources(root, spec_path)
            bad_path = root / "raw/clean/speaker_1/sample_clean_1.wav"
            write_test_wav(bad_path, [1, -1] * 400, sample_rate=44_100)

            with self.assertRaisesRegex(DatasetBuildError, "resampling"):
                build_dataset(root, spec_path=spec_path)
            self.assertFalse((root / "manifest.csv").exists())

    def test_missing_real_requires_explicit_intermediate_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "data_local"
            spec_path = write_synthetic_spec(Path(temp_dir) / "dataset_spec.json")
            make_sources(root, spec_path, include_real=False)

            with self.assertRaisesRegex(DatasetBuildError, "real recording"):
                build_dataset(root, spec_path=spec_path)

            rows = build_dataset(root, spec_path=spec_path, allow_missing_real=True)
            self.assertEqual(len(rows), 3)
            self.assertEqual(sum(row["source_type"] == "real" for row in rows), 0)


if __name__ == "__main__":
    unittest.main()
