import csv
import hashlib
import io
import tempfile
import unittest
import wave
from array import array
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from scripts.build_dataset import APPROVED_CONSENT_TOKEN, build_dataset
from scripts.dataset_spec import DatasetSpecError, load_dataset_spec, load_example_dataset_spec
from scripts.validate_dataset import (
    DEFAULT_POLICY,
    MANIFEST_FIELDS,
    ValidationPolicy,
    expected_dataset_artifacts,
    inspect_wav,
    main,
    required_manifest_primary_paths,
    sha256_file,
    validate_dataset,
    validate_manifest,
)

try:
    from tests.test_build_dataset import approve_sources, make_sources
    from tests.test_dataset_spec import write_synthetic_spec
except ModuleNotFoundError:
    from test_build_dataset import approve_sources, make_sources
    from test_dataset_spec import write_synthetic_spec


def read_audio_min_duration(config_path: Path) -> float:
    in_audio_block = False
    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if line == "audio:":
            in_audio_block = True
            continue
        if line and not line.startswith((" ", "\t")):
            in_audio_block = False
        if in_audio_block and line.strip().startswith("min_duration_seconds:"):
            return float(line.split(":", 1)[1].strip())
    raise AssertionError("audio.min_duration_seconds not found in app config")


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


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


class WavInspectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = ValidationPolicy(
            audio_min_seconds=0.01,
            noise_min_seconds=0.01,
            noise_max_seconds=1.0,
        )
        self.spec = load_example_dataset_spec()
        self.artifact = expected_dataset_artifacts(self.spec)[
            Path("raw/clean/speaker_1/sample_clean_1.wav")
        ]

    def test_valid_pcm16_wav(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.wav"
            write_pcm_wav(path)
            info, issues = inspect_wav(path, self.artifact, self.policy)

            self.assertIsNotNone(info)
            self.assertFalse([issue for issue in issues if issue.severity == "error"])
            self.assertEqual(info.sample_rate, 48_000)  # type: ignore[union-attr]
            self.assertEqual(info.channels, 1)  # type: ignore[union-attr]

    def test_silence_clipping_header_and_duration_are_checked(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            cases = [
                ("silent", {"sample_value": 0}, "WAV_SILENT"),
                ("clipped", {"sample_value": 32_767}, "WAV_CLIPPED"),
                ("rate", {"sample_rate": 44_100}, "WAV_SAMPLE_RATE"),
                ("channels", {"channels": 2}, "WAV_CHANNELS"),
                ("bit_depth", {"sample_width": 1}, "WAV_BIT_DEPTH"),
                ("duration", {"duration": 0.005}, "WAV_DURATION"),
            ]
            for label, kwargs, code in cases:
                with self.subTest(label=label):
                    path = temp / f"{label}.wav"
                    write_pcm_wav(path, **kwargs)
                    _info, issues = inspect_wav(path, self.artifact, self.policy)
                    self.assertIn(code, {issue.code for issue in issues})

    def test_unreadable_wav_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "broken.wav"
            path.write_bytes(b"not a wav file")
            info, issues = inspect_wav(path, self.artifact, self.policy)
            self.assertIsNone(info)
            self.assertIn("WAV_UNREADABLE", {issue.code for issue in issues})

    def test_default_audio_minimum_matches_config(self) -> None:
        audio_min_duration = read_audio_min_duration(
            Path(__file__).resolve().parents[1] / "configs" / "app.yaml"
        )
        self.assertEqual(
            DEFAULT_POLICY.audio_min_seconds,
            audio_min_duration,
        )
        self.assertEqual(DEFAULT_POLICY.audio_min_seconds, 1.0)


class ManifestValidationTest(unittest.TestCase):
    policy = ValidationPolicy(
        audio_min_seconds=0.005,
        audio_max_seconds=1.0,
        noise_min_seconds=0.02,
        noise_max_seconds=1.0,
    )

    def build_fixture(self, root: Path, spec_path: Path) -> list[dict[str, str]]:
        make_sources(root, spec_path)
        approve_sources(root, spec_path)
        return build_dataset(
            root,
            spec_path=spec_path,
            consent_or_license=APPROVED_CONSENT_TOKEN,
        )

    def validate_rows(
        self,
        root: Path,
        spec_path: Path,
        rows: list[dict[str, str]],
    ) -> set[str]:
        write_manifest(root / "manifest.csv", rows)
        report = validate_dataset(root, spec_path=spec_path, policy=self.policy)
        return {issue.code for issue in report.errors}

    def test_manifest_checks_unique_ids_and_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            spec_path = write_synthetic_spec(temp / "dataset_spec.json")
            root = temp / "data_local"
            rows = self.build_fixture(root, spec_path)
            duplicate = dict(rows[0])
            duplicate["sha256"] = "0" * 64
            write_manifest(root / "manifest.csv", [rows[0], duplicate])
            relative = Path(rows[0]["mixed_path"])
            artifact = expected_dataset_artifacts(load_dataset_spec(spec_path))[relative]
            info, _issues = inspect_wav(root / relative, artifact, self.policy)

            manifest_issues, count, hashes = validate_manifest(
                root,
                root / "manifest.csv",
                {relative: info},  # type: ignore[dict-item]
                spec=load_dataset_spec(spec_path),
                required_primary_paths={relative},
            )
            codes = {issue.code for issue in manifest_issues}
            self.assertEqual(count, 2)
            self.assertEqual(hashes, 1)
            self.assertIn("MANIFEST_DUPLICATE_ID", codes)
            self.assertIn("MANIFEST_DUPLICATE_PATH", codes)
            self.assertIn("MANIFEST_HASH", codes)

    def test_requires_explicit_spec_for_dataset_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(DatasetSpecError, "explicit dataset spec"):
                validate_dataset(Path(temp_dir))

    def test_manifest_argument_must_stay_inside_dataset_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            spec_path = write_synthetic_spec(temp / "dataset_spec.json")
            root = temp / "data_local"
            rows = self.build_fixture(root, spec_path)
            external_manifest = temp / "external_manifest.csv"
            write_manifest(external_manifest, rows)

            report = validate_dataset(
                root,
                manifest_path=external_manifest,
                spec_path=spec_path,
                policy=self.policy,
            )

            self.assertIn("MANIFEST_PATH", {issue.code for issue in report.errors})

    def test_rejects_traversal_symlink_hash_split_and_consent_errors(self) -> None:
        cases = [
            ("traversal", "mixed_path", "../outside.wav", "MANIFEST_PATH"),
            ("hash", "sha256", "0" * 64, "MANIFEST_HASH"),
            ("split", "split", "unexpected_split", "MANIFEST_SPLIT"),
            (
                "consent",
                "consent_or_license",
                "pending_local_authorization",
                "MANIFEST_CONSENT",
            ),
            ("license", "consent_or_license", "yes", "MANIFEST_CONSENT"),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            spec_path = write_synthetic_spec(temp / "dataset_spec.json")
            root = temp / "data_local"
            base_rows = self.build_fixture(root, spec_path)
            for label, field_name, replacement, expected_code in cases:
                with self.subTest(label=label):
                    rows = [dict(row) for row in base_rows]
                    rows[0][field_name] = replacement
                    self.assertIn(expected_code, self.validate_rows(root, spec_path, rows))

            symlink_path = root / "controlled" / "dev" / "sample_symlink.wav"
            symlink_path.write_bytes((root / base_rows[0]["mixed_path"]).read_bytes())
            rows = [dict(row) for row in base_rows]
            rows[0]["mixed_path"] = "controlled/dev/sample_symlink.wav"
            rows[0]["sample_id"] = "sample_symlink"
            rows[0]["sha256"] = hashlib.sha256(symlink_path.read_bytes()).hexdigest()
            original_is_symlink = Path.is_symlink

            def fake_is_symlink(path: Path) -> bool:
                if path == symlink_path:
                    return True
                return original_is_symlink(path)

            with mock.patch.object(Path, "is_symlink", fake_is_symlink):
                self.assertIn("MANIFEST_PATH", self.validate_rows(root, spec_path, rows))

    def test_rejects_controlled_source_path_deletion_replacement_and_type_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            spec_path = write_synthetic_spec(temp / "dataset_spec.json")
            root = temp / "data_local"
            base_rows = self.build_fixture(root, spec_path)
            alternate = root / "raw" / "clean" / "speaker_2" / "sample_clean_2.wav"
            self.assertTrue(alternate.is_file())

            cases = [
                ("missing_clean", "clean_path", "", "MANIFEST_SOURCE_PATH"),
                (
                    "replaced_clean",
                    "clean_path",
                    "raw/clean/speaker_2/sample_clean_2.wav",
                    "MANIFEST_SOURCE_PATH",
                ),
                ("wrong_source_type", "source_type", "real", "MANIFEST_SOURCE_TYPE"),
            ]
            for label, field_name, value, expected_code in cases:
                with self.subTest(label=label):
                    rows = [dict(row) for row in base_rows]
                    rows[0][field_name] = value
                    self.assertIn(expected_code, self.validate_rows(root, spec_path, rows))

    def test_complete_synthetic_fixture_passes_with_spec(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            spec_path = write_synthetic_spec(temp / "dataset_spec.json")
            root = temp / "data_local"
            rows = self.build_fixture(root, spec_path)

            report = validate_dataset(root, spec_path=spec_path, policy=self.policy)

            self.assertTrue(report.ok, report.render())
            self.assertEqual(report.checked_wavs, 6)
            self.assertEqual(report.manifest_rows, len(rows))
            self.assertEqual(report.hashes_verified, len(rows))

    def test_current_builder_output_passes_the_spec_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            spec_path = write_synthetic_spec(temp / "dataset_spec.json")
            root = temp / "data_local"
            rows = self.build_fixture(root, spec_path)

            self.assertEqual(
                required_manifest_primary_paths(load_dataset_spec(spec_path)),
                {Path(row["mixed_path"] or row["clean_path"]) for row in rows},
            )
            report = validate_dataset(root, spec_path=spec_path, policy=self.policy)
            self.assertTrue(report.ok, report.render())

    def test_cli_returns_failure_and_human_readable_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = io.StringIO()
            with redirect_stdout(output):
                exit_code = main([temp_dir])
            self.assertEqual(exit_code, 2)
            rendered = output.getvalue()
            self.assertIn("Dataset validation refused", rendered)
            self.assertNotIn(temp_dir, rendered)

    def test_report_rendering_uses_root_relative_paths_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            spec_path = write_synthetic_spec(temp / "dataset_spec.json")
            root = temp / "data_local"
            rows = self.build_fixture(root, spec_path)
            rows[0]["mixed_path"] = "../outside.wav"
            write_manifest(root / "manifest.csv", rows)

            report = validate_dataset(root, spec_path=spec_path, policy=self.policy)
            rendered = report.render()

            self.assertIn("AudioRescue dataset validation: FAIL", rendered)
            self.assertNotIn(temp_dir, rendered)
            self.assertNotIn(Path(temp_dir).as_posix().replace("/", "%2F"), rendered)


if __name__ == "__main__":
    unittest.main()
