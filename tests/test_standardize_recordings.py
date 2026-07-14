import csv
import hashlib
import os
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

from scripts.standardize_recordings import (
    ConversionError,
    MetadataError,
    MissingSourceError,
    OutputExistsError,
    SourceIntegrityError,
    WavValidationError,
    select_converter,
    sha256_file,
    standardize_recordings,
    validate_standard_wav,
)


FIELDS = [
    "asset_id",
    "source_type",
    "source_original_path",
    "relative_path",
    "source_original_sha256",
    "standardized_sha256",
    "notes",
]


def metadata_row(
    asset_id: str,
    *,
    source_path: str = "",
    output_path: str | None = None,
    source_sha256: str = "",
    standardized_sha256: str = "",
) -> dict[str, str]:
    return {
        "asset_id": asset_id,
        "source_type": "clean",
        "source_original_path": source_path,
        "relative_path": output_path or f"raw/clean/{asset_id}.wav",
        "source_original_sha256": source_sha256,
        "standardized_sha256": standardized_sha256,
        "notes": "preserve me",
    }


def create_data_root(parent: Path) -> Path:
    root = parent / "data_local"
    (root / "source_original/clean").mkdir(parents=True)
    (root / "raw/clean").mkdir(parents=True)
    return root


def write_metadata(root: Path, rows: list[dict[str, str]]) -> Path:
    path = root / "recording_metadata.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def read_metadata(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_wav(
    path: Path,
    *,
    sample_rate: int = 48_000,
    channels: int = 1,
    sample_width: int = 2,
    frames: int = 480,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00" * frames * channels * sample_width)


class StandardizeRecordingsTest(unittest.TestCase):
    def test_success_preserves_master_and_atomically_records_both_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = create_data_root(Path(temp_dir))
            source = root / "source_original/clean/take01.m4a"
            source_bytes = b"immutable master bytes\x00\x01"
            source.write_bytes(source_bytes)
            ledger = write_metadata(
                root,
                [
                    metadata_row(
                        "clean_spkA_s01",
                        source_path="source_original/clean/take01.m4a",
                    )
                ],
            )
            calls: list[tuple[Path, Path]] = []

            def fake_converter(input_path: Path, output_path: Path) -> None:
                calls.append((input_path, output_path))
                write_wav(output_path)

            report = standardize_recordings(root, converter=fake_converter)

            output = root / "raw/clean/clean_spkA_s01.wav"
            self.assertEqual(report.processed_assets, ("clean_spkA_s01",))
            self.assertEqual(report.processed_outputs, (output.resolve(),))
            self.assertEqual(report.skipped_count, 0)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][0], source.resolve())
            self.assertNotEqual(calls[0][1], output.resolve())
            self.assertEqual(source.read_bytes(), source_bytes)
            validate_standard_wav(output)

            row = read_metadata(ledger)[0]
            self.assertEqual(row["source_original_sha256"], sha256_file(source))
            self.assertEqual(row["standardized_sha256"], sha256_file(output))
            self.assertEqual(row["notes"], "preserve me")

    def test_final_mode_rejects_all_15_blank_paths_before_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = create_data_root(Path(temp_dir))
            write_metadata(
                root,
                [metadata_row(f"asset_{index:02d}") for index in range(15)],
            )
            converter = mock.Mock()

            with self.assertRaisesRegex(MissingSourceError, r"15 .*source_original_path"):
                standardize_recordings(root, converter=converter)

            converter.assert_not_called()

    def test_available_only_processes_filled_rows_and_skips_the_rest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = create_data_root(Path(temp_dir))
            source = root / "source_original/clean/ready.aac"
            source.write_bytes(b"ready master")
            rows = [
                metadata_row(
                    "ready",
                    source_path="source_original/clean/ready.aac",
                )
            ]
            rows.extend(metadata_row(f"pending_{index:02d}") for index in range(14))
            ledger = write_metadata(root, rows)

            def fake_converter(_source: Path, output: Path) -> None:
                write_wav(output)

            report = standardize_recordings(
                root,
                available_only=True,
                converter=fake_converter,
            )

            self.assertEqual(report.processed_assets, ("ready",))
            self.assertEqual(report.skipped_count, 14)
            self.assertTrue((root / "raw/clean/ready.wav").is_file())
            updated_rows = read_metadata(ledger)
            self.assertTrue(updated_rows[0]["source_original_sha256"])
            self.assertTrue(updated_rows[0]["standardized_sha256"])
            self.assertTrue(
                all(not row["standardized_sha256"] for row in updated_rows[1:])
            )

    def test_available_only_with_no_ready_rows_does_not_select_a_converter(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = create_data_root(Path(temp_dir))
            write_metadata(root, [metadata_row(f"asset_{index}") for index in range(15)])

            with mock.patch(
                "scripts.standardize_recordings.select_converter",
                side_effect=AssertionError("converter selection must be skipped"),
            ):
                report = standardize_recordings(root, available_only=True)

            self.assertEqual(report.processed_count, 0)
            self.assertEqual(report.skipped_count, 15)

    def test_refuses_overwrite_by_default_and_allows_explicit_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = create_data_root(Path(temp_dir))
            source = root / "source_original/clean/take.wav"
            source.write_bytes(b"master")
            output = root / "raw/clean/asset.wav"
            old_output = b"existing output must remain"
            output.write_bytes(old_output)
            ledger = write_metadata(
                root,
                [
                    metadata_row(
                        "asset",
                        source_path="source_original/clean/take.wav",
                    )
                ],
            )
            converter = mock.Mock(side_effect=lambda _source, target: write_wav(target))

            with self.assertRaises(OutputExistsError):
                standardize_recordings(root, converter=converter)
            converter.assert_not_called()
            self.assertEqual(output.read_bytes(), old_output)
            self.assertFalse(read_metadata(ledger)[0]["standardized_sha256"])

            report = standardize_recordings(
                root,
                overwrite=True,
                converter=converter,
            )
            self.assertEqual(report.processed_count, 1)
            validate_standard_wav(output)
            self.assertNotEqual(output.read_bytes(), old_output)

    def test_invalid_conversion_never_commits_output_or_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = create_data_root(Path(temp_dir))
            source = root / "source_original/clean/take.m4a"
            source.write_bytes(b"master")
            ledger = write_metadata(
                root,
                [
                    metadata_row(
                        "asset",
                        source_path="source_original/clean/take.m4a",
                    )
                ],
            )
            original_ledger = ledger.read_bytes()

            def wrong_format(_source: Path, output: Path) -> None:
                write_wav(output, sample_rate=44_100, channels=2)

            with self.assertRaises(WavValidationError):
                standardize_recordings(root, converter=wrong_format)

            self.assertFalse((root / "raw/clean/asset.wav").exists())
            self.assertEqual(ledger.read_bytes(), original_ledger)
            self.assertEqual(source.read_bytes(), b"master")

    def test_converter_failure_never_commits_output_or_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = create_data_root(Path(temp_dir))
            source = root / "source_original/clean/take.m4a"
            source.write_bytes(b"master")
            ledger = write_metadata(
                root,
                [
                    metadata_row(
                        "asset",
                        source_path="source_original/clean/take.m4a",
                    )
                ],
            )
            original_ledger = ledger.read_bytes()

            def failed_converter(_source: Path, _output: Path) -> None:
                raise RuntimeError("synthetic conversion failure")

            with self.assertRaisesRegex(ConversionError, "synthetic conversion failure"):
                standardize_recordings(root, converter=failed_converter)

            self.assertFalse((root / "raw/clean/asset.wav").exists())
            self.assertEqual(ledger.read_bytes(), original_ledger)

    def test_detects_master_change_during_conversion_before_commit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = create_data_root(Path(temp_dir))
            source = root / "source_original/clean/take.m4a"
            source.write_bytes(b"master before")
            ledger = write_metadata(
                root,
                [
                    metadata_row(
                        "asset",
                        source_path="source_original/clean/take.m4a",
                    )
                ],
            )
            original_ledger = ledger.read_bytes()

            def malicious_converter(input_path: Path, output: Path) -> None:
                write_wav(output)
                input_path.write_bytes(b"master changed")

            with self.assertRaises(SourceIntegrityError):
                standardize_recordings(root, converter=malicious_converter)

            self.assertFalse((root / "raw/clean/asset.wav").exists())
            self.assertEqual(ledger.read_bytes(), original_ledger)

    def test_rejects_wrong_stored_master_hash_before_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = create_data_root(Path(temp_dir))
            source = root / "source_original/clean/take.wav"
            source.write_bytes(b"master")
            write_metadata(
                root,
                [
                    metadata_row(
                        "asset",
                        source_path="source_original/clean/take.wav",
                        source_sha256="0" * 64,
                    )
                ],
            )
            converter = mock.Mock()

            with self.assertRaises(SourceIntegrityError):
                standardize_recordings(root, converter=converter)

            converter.assert_not_called()

    def test_rejects_source_and_output_paths_outside_frozen_roots(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = create_data_root(Path(temp_dir))
            outside_source = root / "outside.m4a"
            outside_source.write_bytes(b"outside")
            write_metadata(
                root,
                [
                    metadata_row(
                        "outside_source",
                        source_path="outside.m4a",
                    )
                ],
            )
            converter = mock.Mock()

            with self.assertRaisesRegex(MetadataError, "source_original"):
                standardize_recordings(root, converter=converter)
            converter.assert_not_called()

            valid_source = root / "source_original/clean/take.m4a"
            valid_source.write_bytes(b"inside")
            write_metadata(
                root,
                [
                    metadata_row(
                        "outside_output",
                        source_path="source_original/clean/take.m4a",
                        output_path="controlled/escape.wav",
                    )
                ],
            )
            with self.assertRaisesRegex(MetadataError, "raw"):
                standardize_recordings(root, converter=converter)
            converter.assert_not_called()

    def test_csv_failure_rolls_back_overwritten_output_and_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = create_data_root(Path(temp_dir))
            source = root / "source_original/clean/take.m4a"
            source.write_bytes(b"master")
            output = root / "raw/clean/asset.wav"
            write_wav(output, frames=17)
            old_output = output.read_bytes()
            ledger = write_metadata(
                root,
                [
                    metadata_row(
                        "asset",
                        source_path="source_original/clean/take.m4a",
                        source_sha256=sha256_file(source),
                        standardized_sha256=sha256_file(output),
                    )
                ],
            )
            old_ledger = ledger.read_bytes()

            def fake_converter(_source: Path, target: Path) -> None:
                write_wav(target, frames=41)

            with mock.patch(
                "scripts.standardize_recordings._atomic_write_metadata",
                side_effect=OSError("synthetic disk failure"),
            ):
                with self.assertRaisesRegex(OSError, "synthetic disk failure"):
                    standardize_recordings(
                        root,
                        overwrite=True,
                        converter=fake_converter,
                    )

            self.assertEqual(output.read_bytes(), old_output)
            self.assertEqual(ledger.read_bytes(), old_ledger)

    def test_converter_selection_prefers_ffmpeg_then_macos_afconvert(self) -> None:
        runner = mock.Mock()
        preferred = select_converter(
            which=lambda _name: "/opt/bin/ffmpeg",
            platform="darwin",
            afconvert_path="/does/not/matter",
            runner=runner,
        )
        self.assertEqual(preferred.backend, "ffmpeg")
        self.assertEqual(preferred.executable, "/opt/bin/ffmpeg")

        with tempfile.TemporaryDirectory() as temp_dir:
            fallback_path = Path(temp_dir) / "afconvert"
            fallback_path.write_text("#!/bin/sh\n", encoding="utf-8")
            fallback_path.chmod(0o755)
            fallback = select_converter(
                which=lambda _name: None,
                platform="darwin",
                afconvert_path=fallback_path,
                runner=runner,
            )
            self.assertEqual(fallback.backend, "afconvert")
            self.assertEqual(fallback.executable, str(fallback_path))

        with self.assertRaises(ConversionError):
            select_converter(
                which=lambda _name: None,
                platform="linux",
                runner=runner,
            )

    def test_command_converter_builds_fixed_format_commands(self) -> None:
        calls: list[list[str]] = []

        def runner(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            return subprocess.CompletedProcess(command, 0, "", "")

        ffmpeg = select_converter(
            which=lambda _name: "/fake/ffmpeg",
            platform="linux",
            runner=runner,
        )
        ffmpeg(Path("input.m4a"), Path("output.wav"))

        command = calls[0]
        self.assertEqual(command[0], "/fake/ffmpeg")
        self.assertEqual(command[command.index("-ac") + 1], "1")
        self.assertEqual(command[command.index("-ar") + 1], "48000")
        self.assertEqual(command[command.index("-c:a") + 1], "pcm_s16le")


if __name__ == "__main__":
    unittest.main()
