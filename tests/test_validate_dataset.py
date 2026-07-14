import csv
import hashlib
import io
import tempfile
import unittest
import wave
from array import array
from contextlib import redirect_stdout
from pathlib import Path

import yaml

from scripts.build_dataset import build_dataset
from scripts.validate_dataset import (
    APPROVED_CONSENT_TOKEN,
    DATASET_VERSION,
    DEFAULT_POLICY,
    DEMO_SAMPLE_IDS,
    MANIFEST_FIELDS,
    REFERENCE_TEXTS,
    SNR_CODES,
    ValidationPolicy,
    expected_dataset_artifacts,
    inspect_wav,
    main,
    normalize_reference,
    required_manifest_primary_paths,
    sha256_file,
    validate_dataset,
    validate_manifest,
)
from tests.test_build_dataset import approve_sources


def write_pcm_wav(
    path: Path,
    *,
    duration: float = 0.02,
    sample_rate: int = 48_000,
    channels: int = 1,
    sample_width: int = 2,
    sample_value: int = 1_200,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = max(1, round(duration * sample_rate))
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        if sample_width == 2:
            samples = array("h", [sample_value] * frames * channels)
            wav_file.writeframes(samples.tobytes())
        else:
            wav_file.writeframes(bytes([128] * frames * channels * sample_width))


def manifest_row(root: Path, relative: Path) -> dict[str, str]:
    spec = expected_dataset_artifacts()[relative]
    reference_text = REFERENCE_TEXTS[spec.sentence or ""]
    row = {field: "" for field in MANIFEST_FIELDS}
    row.update(
        {
            "sample_id": relative.stem,
            "dataset_version": DATASET_VERSION,
            "split": spec.split or ("clean_control" if spec.kind == "clean" else spec.kind),
            "speaker_id": spec.speaker or "",
            "sentence_id": (spec.sentence or "").upper(),
            "reference_text": reference_text,
            "reference_normalized": normalize_reference(reference_text),
            "source_type": (
                "controlled_mix"
                if spec.kind == "mix"
                else "clean_control" if spec.kind == "clean" else "real"
            ),
            "noise_type": spec.noise or "",
            "noise_source": "team_recorded",
            "consent_or_license": APPROVED_CONSENT_TOKEN,
            "sample_rate": "48000",
            "channels": "1",
            "duration_seconds": "0.02",
            "recording_device": "test",
            "sha256": sha256_file(root / relative),
            "is_demo_candidate": str(relative.stem in DEMO_SAMPLE_IDS).lower(),
            "is_locked": str(
                spec.kind == "clean"
                or (spec.kind == "mix" and spec.split == "locked_test")
            ).lower(),
        }
    )
    if spec.kind == "mix":
        row["mixed_path"] = relative.as_posix()
        row["clean_path"] = (
            Path("raw") / "clean" / f"spk{spec.speaker}" / f"clean_spk{spec.speaker}_{spec.sentence}.wav"
        ).as_posix()
        row["noise_path"] = (Path("raw") / "noise" / f"noise_{spec.noise}_take01.wav").as_posix()
        row["snr_db"] = str(SNR_CODES[spec.snr_code or ""])
        row["mix_seed"] = "20260714"
        row["noise_offset_seconds"] = "0.0"
        row["mix_alpha"] = "0.5"
        row["final_gain"] = "1.0"
    elif spec.kind == "real":
        row["mixed_path"] = relative.as_posix()
    else:
        row["clean_path"] = relative.as_posix()
    return row


def complete_manifest_rows(root: Path) -> list[dict[str, str]]:
    return [
        manifest_row(root, relative)
        for relative in sorted(
            required_manifest_primary_paths(), key=lambda path: path.as_posix()
        )
    ]


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_varied_pcm_wav(path: Path, frames: int, phase: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = array(
        "h",
        (
            (1 if (index + phase) % 2 else -1)
            * (1_000 + ((index * (phase + 3)) % 700))
            for index in range(frames)
        ),
    )
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(48_000)
        wav_file.writeframes(samples.tobytes())


class DatasetMatrixTest(unittest.TestCase):
    def test_frozen_matrix_counts_and_key_examples(self) -> None:
        artifacts = expected_dataset_artifacts()
        counts: dict[str, int] = {}
        for spec in artifacts.values():
            counts[spec.kind] = counts.get(spec.kind, 0) + 1

        self.assertEqual(counts, {"clean": 9, "noise": 3, "mix": 27, "real": 3})
        self.assertIn(
            Path("controlled/locked_test/mix_spkB_s03_fan_snr000.wav"),
            artifacts,
        )
        self.assertIn(Path("raw/real/real_spkA_traffic_r01.wav"), artifacts)
        self.assertEqual(len(required_manifest_primary_paths()), 33)


class WavInspectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = ValidationPolicy(
            audio_min_seconds=0.01,
            noise_min_seconds=0.01,
            noise_max_seconds=1.0,
        )

    def test_valid_pcm16_wav(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "clean_spkA_s01.wav"
            write_pcm_wav(path)
            spec = expected_dataset_artifacts()[Path("raw/clean/spkA/clean_spkA_s01.wav")]
            info, issues = inspect_wav(path, spec, self.policy)

            self.assertIsNotNone(info)
            self.assertFalse([issue for issue in issues if issue.severity == "error"])
            self.assertEqual(info.sample_rate, 48_000)  # type: ignore[union-attr]
            self.assertEqual(info.channels, 1)  # type: ignore[union-attr]

    def test_silence_and_clipping_are_rejected(self) -> None:
        spec = expected_dataset_artifacts()[Path("raw/clean/spkA/clean_spkA_s01.wav")]
        with tempfile.TemporaryDirectory() as temp_dir:
            silent_path = Path(temp_dir) / "silent.wav"
            clipped_path = Path(temp_dir) / "clipped.wav"
            write_pcm_wav(silent_path, sample_value=0)
            write_pcm_wav(clipped_path, sample_value=32_767)

            _, silent_issues = inspect_wav(silent_path, spec, self.policy)
            _, clipped_issues = inspect_wav(clipped_path, spec, self.policy)
            self.assertIn("WAV_SILENT", {issue.code for issue in silent_issues})
            self.assertIn("WAV_CLIPPED", {issue.code for issue in clipped_issues})

    def test_header_format_and_duration_are_checked(self) -> None:
        spec = expected_dataset_artifacts()[Path("raw/clean/spkA/clean_spkA_s01.wav")]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bad.wav"
            write_pcm_wav(
                path,
                duration=0.005,
                sample_rate=44_100,
                channels=2,
                sample_width=1,
            )
            _, issues = inspect_wav(path, spec, self.policy)
            codes = {issue.code for issue in issues}
            self.assertTrue(
                {"WAV_SAMPLE_RATE", "WAV_CHANNELS", "WAV_BIT_DEPTH", "WAV_DURATION"}.issubset(codes)
            )

    def test_unreadable_wav_is_rejected(self) -> None:
        spec = expected_dataset_artifacts()[Path("raw/clean/spkA/clean_spkA_s01.wav")]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "broken.wav"
            path.write_bytes(b"not a wav file")
            info, issues = inspect_wav(path, spec, self.policy)
            self.assertIsNone(info)
            self.assertIn("WAV_UNREADABLE", {issue.code for issue in issues})

    def test_default_audio_minimum_is_one_second(self) -> None:
        app_config = yaml.safe_load(
            (Path(__file__).resolve().parents[1] / "configs" / "app.yaml").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            DEFAULT_POLICY.audio_min_seconds,
            app_config["audio"]["min_duration_seconds"],
        )
        self.assertEqual(DEFAULT_POLICY.audio_min_seconds, 1.0)

        spec = expected_dataset_artifacts()[
            Path("raw/clean/spkA/clean_spkA_s01.wav")
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "too_short.wav"
            write_pcm_wav(path, duration=0.5)

            _, issues = inspect_wav(path, spec)

            self.assertIn("WAV_DURATION", {issue.code for issue in issues})


class ManifestValidationTest(unittest.TestCase):
    def test_manifest_checks_unique_ids_and_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            relative = Path("controlled/dev/mix_spkA_s01_fan_snr000.wav")
            write_pcm_wav(root / relative)
            spec = expected_dataset_artifacts()[relative]
            info, issues = inspect_wav(
                root / relative,
                spec,
                ValidationPolicy(audio_min_seconds=0.01),
            )
            self.assertFalse([issue for issue in issues if issue.severity == "error"])
            row = manifest_row(root, relative)
            manifest = root / "manifest.csv"
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
                writer.writeheader()
                writer.writerow(row)
                duplicate = dict(row)
                duplicate["sha256"] = "0" * 64
                writer.writerow(duplicate)

            manifest_issues, rows, hashes = validate_manifest(
                root,
                manifest,
                {relative: info},  # type: ignore[dict-item]
                required_primary_paths={relative},
            )
            codes = {issue.code for issue in manifest_issues}
            self.assertEqual(rows, 2)
            self.assertEqual(hashes, 1)
            self.assertIn("MANIFEST_DUPLICATE_ID", codes)
            self.assertIn("MANIFEST_DUPLICATE_PATH", codes)
            self.assertIn("MANIFEST_HASH", codes)


class ManifestSemanticContractTest(unittest.TestCase):
    policy = ValidationPolicy(
        audio_min_seconds=0.01,
        noise_min_seconds=0.01,
        noise_max_seconds=1.0,
    )

    def make_complete_fixture(self, root: Path) -> list[dict[str, str]]:
        for relative in expected_dataset_artifacts():
            write_pcm_wav(root / relative)
        return complete_manifest_rows(root)

    def validate_rows(
        self, root: Path, rows: list[dict[str, str]]
    ) -> set[str]:
        write_manifest(root / "manifest.csv", rows)
        report = validate_dataset(root, policy=self.policy)
        return {issue.code for issue in report.errors}

    def test_rejects_row_semantic_and_numeric_tampering(self) -> None:
        cases = (
            (
                "sample id",
                "mix_spkA_s01_fan_snrp05",
                "sample_id",
                "wrong_id",
                "MANIFEST_SAMPLE_ID",
            ),
            (
                "source type",
                "mix_spkA_s01_fan_snrp05",
                "source_type",
                "mix",
                "MANIFEST_SEMANTICS",
            ),
            (
                "speaker case",
                "mix_spkA_s01_fan_snrp05",
                "speaker_id",
                "a",
                "MANIFEST_SEMANTICS",
            ),
            (
                "sentence case",
                "mix_spkA_s01_fan_snrp05",
                "sentence_id",
                "s01",
                "MANIFEST_SEMANTICS",
            ),
            (
                "noise assignment",
                "mix_spkA_s01_fan_snrp05",
                "noise_type",
                "traffic",
                "MANIFEST_SEMANTICS",
            ),
            (
                "split",
                "mix_spkA_s01_fan_snrp05",
                "split",
                "locked_test",
                "MANIFEST_SEMANTICS",
            ),
            (
                "dev lock",
                "mix_spkA_s01_fan_snrp05",
                "is_locked",
                "true",
                "MANIFEST_LOCKED",
            ),
            (
                "mix clean path",
                "mix_spkA_s01_fan_snrp05",
                "clean_path",
                "raw/clean/spkB/clean_spkB_s01.wav",
                "MANIFEST_PATH_SEMANTICS",
            ),
            (
                "clean extra mixed path",
                "clean_spkA_s03",
                "mixed_path",
                "raw/clean/spkA/clean_spkA_s03.wav",
                "MANIFEST_PATH_SEMANTICS",
            ),
            (
                "real extra clean path",
                "real_spkA_traffic_r01",
                "clean_path",
                "raw/clean/spkA/clean_spkA_s03.wav",
                "MANIFEST_PATH_SEMANTICS",
            ),
            (
                "missing demo flag",
                "mix_spkB_s03_fan_snr000",
                "is_demo_candidate",
                "false",
                "MANIFEST_DEMO_CANDIDATE",
            ),
            (
                "spurious demo flag",
                "mix_spkA_s01_fan_snrp05",
                "is_demo_candidate",
                "true",
                "MANIFEST_DEMO_CANDIDATE",
            ),
            (
                "reference text",
                "mix_spkA_s01_fan_snrp05",
                "reference_text",
                "被篡改的文本",
                "MANIFEST_REFERENCE",
            ),
            (
                "normalized reference",
                "mix_spkA_s01_fan_snrp05",
                "reference_normalized",
                "wrong",
                "MANIFEST_REFERENCE",
            ),
            (
                "nan snr",
                "mix_spkA_s01_fan_snrp05",
                "snr_db",
                "nan",
                "MANIFEST_SNR",
            ),
            (
                "fractional seed",
                "mix_spkA_s01_fan_snrp05",
                "mix_seed",
                "1.5",
                "MANIFEST_MIX_META",
            ),
            (
                "negative offset",
                "mix_spkA_s01_fan_snrp05",
                "noise_offset_seconds",
                "-0.1",
                "MANIFEST_MIX_META",
            ),
            (
                "zero alpha",
                "mix_spkA_s01_fan_snrp05",
                "mix_alpha",
                "0",
                "MANIFEST_MIX_META",
            ),
            (
                "infinite alpha",
                "mix_spkA_s01_fan_snrp05",
                "mix_alpha",
                "inf",
                "MANIFEST_MIX_META",
            ),
            (
                "infinite gain",
                "mix_spkA_s01_fan_snrp05",
                "final_gain",
                "inf",
                "MANIFEST_MIX_META",
            ),
            (
                "zero gain",
                "mix_spkA_s01_fan_snrp05",
                "final_gain",
                "0",
                "MANIFEST_MIX_META",
            ),
            (
                "infinite distance",
                "mix_spkA_s01_fan_snrp05",
                "recording_distance_cm",
                "inf",
                "MANIFEST_AUDIO_META",
            ),
            (
                "development-only consent",
                "mix_spkA_s01_fan_snrp05",
                "consent_or_license",
                "pending_team_confirmation",
                "MANIFEST_CONSENT",
            ),
            (
                "invented consent",
                "mix_spkA_s01_fan_snrp05",
                "consent_or_license",
                "yes",
                "MANIFEST_CONSENT",
            ),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base_rows = self.make_complete_fixture(root)
            for label, sample_id, field_name, replacement, expected_code in cases:
                with self.subTest(case=label):
                    rows = [dict(row) for row in base_rows]
                    row = next(item for item in rows if item["sample_id"] == sample_id)
                    row[field_name] = replacement
                    codes = self.validate_rows(root, rows)
                    self.assertIn(expected_code, codes)

    def test_requires_exact_row_count_and_primary_path_set(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base_rows = self.make_complete_fixture(root)

            extra_rows = [dict(row) for row in base_rows]
            extra_rows.append(dict(extra_rows[0]))
            self.assertIn(
                "MANIFEST_ROW_COUNT", self.validate_rows(root, extra_rows)
            )

            rows = [dict(row) for row in base_rows]
            replaced = next(
                row
                for row in rows
                if row["sample_id"] == "mix_spkA_s01_fan_snrp05"
            )
            unexpected = Path("controlled/dev/unexpected.wav")
            write_pcm_wav(root / unexpected)
            replaced["sample_id"] = unexpected.stem
            replaced["mixed_path"] = unexpected.as_posix()
            replaced["sha256"] = sha256_file(root / unexpected)
            self.assertIn("MANIFEST_COVERAGE", self.validate_rows(root, rows))

    def test_three_snr_variants_share_seed_and_offset_per_cell(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            rows = self.make_complete_fixture(root)
            changed = next(
                row
                for row in rows
                if row["sample_id"] == "mix_spkA_s01_fan_snr000"
            )
            changed["mix_seed"] = "20260715"
            changed["noise_offset_seconds"] = "0.25"
            self.assertIn("MANIFEST_MIX_CELL", self.validate_rows(root, rows))


class FullDatasetValidationTest(unittest.TestCase):
    def test_complete_short_fixture_passes_with_injected_duration_policy(self) -> None:
        policy = ValidationPolicy(
            audio_min_seconds=0.01,
            noise_min_seconds=0.01,
            noise_max_seconds=1.0,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for relative in expected_dataset_artifacts():
                write_pcm_wav(root / relative)

            with (root / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
                writer.writeheader()
                for relative in sorted(required_manifest_primary_paths(), key=lambda path: path.as_posix()):
                    writer.writerow(manifest_row(root, relative))

            report = validate_dataset(root, policy=policy)
            self.assertTrue(report.ok, report.render())
            self.assertEqual(report.checked_wavs, 42)
            self.assertEqual(report.manifest_rows, 33)
            self.assertEqual(report.hashes_verified, 33)

    def test_current_builder_output_passes_the_strict_manifest_contract(self) -> None:
        policy = ValidationPolicy(
            audio_min_seconds=0.005,
            audio_max_seconds=1.0,
            noise_min_seconds=0.02,
            noise_max_seconds=1.0,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            phase = 0
            for relative, spec in expected_dataset_artifacts().items():
                if spec.kind == "mix":
                    continue
                phase += 1
                frames = 1_440 if spec.kind == "noise" else 600
                write_varied_pcm_wav(root / relative, frames, phase)

            approve_sources(root)

            rows = build_dataset(
                root, consent_or_license=APPROVED_CONSENT_TOKEN
            )
            self.assertEqual(len(rows), 33)
            report = validate_dataset(root, policy=policy)
            self.assertTrue(report.ok, report.render())
            self.assertEqual(report.manifest_rows, 33)
            self.assertEqual(report.hashes_verified, 33)

    def test_cli_returns_failure_and_human_readable_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = io.StringIO()
            with redirect_stdout(output):
                exit_code = main([temp_dir])
            self.assertEqual(exit_code, 1)
            self.assertIn("AudioRescue dataset validation: FAIL", output.getvalue())
            self.assertIn("Errors:", output.getvalue())


if __name__ == "__main__":
    unittest.main()
