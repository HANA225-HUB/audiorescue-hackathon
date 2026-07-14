import csv
import hashlib
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import yaml

from scripts.freeze_experiment import (
    DATASET_VALIDATOR_ID,
    DATASET_VALIDATOR_VERSION,
    EXPECTED_SPLITS,
    FREEZE_SCHEMA_VERSION,
    FreezeError,
    GitSnapshot,
    freeze_experiment,
    main,
)


@dataclass(frozen=True)
class StubIssue:
    code: str
    path: str
    message: str


@dataclass(frozen=True)
class StubValidationReport:
    checked_wavs: int = 42
    expected_wavs: int = 42
    manifest_rows: int = 33
    hashes_verified: int = 33
    errors: tuple[StubIssue, ...] = ()
    warnings: tuple[StubIssue, ...] = (
        StubIssue("LEVEL_WARNING", "raw/clean/example.wav", "review level"),
    )


def passing_dataset_validator(
    _dataset_root: str | Path,
    _manifest_path: str | Path | None,
) -> StubValidationReport:
    return StubValidationReport()


def write_config(path: Path, *, initial_prompt: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "contract_version": "v0.1-contract",
                "audio": {"target_sample_rate": 48_000, "channels": 1},
                "enhancement": {
                    "model": "DeepFilterNet3",
                    "default_strength": 0.75,
                },
                "asr": {
                    "model": "base",
                    "language": "zh",
                    "initial_prompt": initial_prompt,
                },
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )


def write_manifest(
    path: Path,
    *,
    pending: bool = False,
    locked_flags_by_split: dict[str, str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "dataset_version",
        "split",
        "reference_text",
        "consent_or_license",
        "sha256",
        "is_locked",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        index = 0
        for split, count in EXPECTED_SPLITS.items():
            for _ in range(count):
                index += 1
                writer.writerow(
                    {
                        "sample_id": f"sample_{index:02d}",
                        "dataset_version": "AudioRescue-CN-Mini-v1",
                        "split": split,
                        "reference_text": "测试参考文本",
                        "consent_or_license": (
                            "pending_team_confirmation"
                            if pending
                            else "team-approved-for-competition-evaluation"
                        ),
                        "sha256": hashlib.sha256(str(index).encode()).hexdigest(),
                        "is_locked": (locked_flags_by_split or {}).get(
                            split,
                            str(split in {"locked_test", "clean_control"}).lower(),
                        ),
                    }
                )


class FreezeExperimentTest(unittest.TestCase):
    def test_writes_strict_immutable_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = root / "configs/app.yaml"
            manifest = root / "data_local/manifest.csv"
            output = root / "data_local/config_frozen.json"
            (root / "core").mkdir()
            (root / "core/pipeline.py").write_text("VERSION = 1\n", encoding="utf-8")
            write_config(config)
            write_manifest(manifest)

            record = freeze_experiment(
                config_path=config,
                manifest_path=manifest,
                output_path=output,
                project_root=root,
                git_snapshot=GitSnapshot("a" * 40, ()),
                now=lambda: datetime(2026, 7, 15, 2, 0, tzinfo=timezone.utc),
                dataset_validator=passing_dataset_validator,
            )

            persisted = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(persisted, record)
            self.assertEqual(
                record["freeze_schema_version"],
                "audiorescue-experiment-freeze-v2",
            )
            self.assertEqual(record["freeze_schema_version"], FREEZE_SCHEMA_VERSION)
            self.assertEqual(record["manifest"]["row_count"], 33)
            self.assertEqual(record["manifest"]["split_counts"], EXPECTED_SPLITS)
            self.assertEqual(len(record["manifest"]["locked_sample_ids"]), 9)
            self.assertFalse(record["git"]["dirty"])
            self.assertEqual(
                record["dataset_validation"],
                {
                    "validator_id": DATASET_VALIDATOR_ID,
                    "validator_version": DATASET_VALIDATOR_VERSION,
                    "dataset_root": str(manifest.parent.resolve()),
                    "checked_wavs": 42,
                    "expected_wavs": 42,
                    "manifest_rows": 33,
                    "hashes_verified": 33,
                    "warning_count": 1,
                },
            )
            self.assertEqual(
                persisted["dataset_validation"], record["dataset_validation"]
            )
            self.assertEqual(
                record["code_sha256"]["core/pipeline.py"],
                hashlib.sha256(b"VERSION = 1\n").hexdigest(),
            )
            with self.assertRaisesRegex(FreezeError, "already exists"):
                freeze_experiment(
                    config_path=config,
                    manifest_path=manifest,
                    output_path=output,
                    project_root=root,
                    git_snapshot=GitSnapshot("a" * 40, ()),
                    dataset_validator=passing_dataset_validator,
                )

    def test_rejects_incomplete_validator_coverage(self) -> None:
        cases = {
            "checked_wavs": 41,
            "expected_wavs": 41,
            "manifest_rows": 32,
            "hashes_verified": 32,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = root / "app.yaml"
            manifest = root / "manifest.csv"
            write_config(config)
            write_manifest(manifest)

            for field_name, bad_value in cases.items():
                with self.subTest(field_name=field_name):
                    values = {
                        "checked_wavs": 42,
                        "expected_wavs": 42,
                        "manifest_rows": 33,
                        "hashes_verified": 33,
                    }
                    values[field_name] = bad_value
                    report = StubValidationReport(**values)
                    output = root / f"bad-{field_name}.json"
                    with self.assertRaisesRegex(
                        FreezeError,
                        rf"coverage is incomplete.*{field_name} must be",
                    ):
                        freeze_experiment(
                            config_path=config,
                            manifest_path=manifest,
                            output_path=output,
                            project_root=root,
                            git_snapshot=GitSnapshot("a" * 40, ()),
                            dataset_validator=lambda *_args, report=report: report,
                        )
                    self.assertFalse(output.exists())

    def test_rejects_non_hex_git_sha(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = root / "app.yaml"
            manifest = root / "manifest.csv"
            output = root / "invalid-git.json"
            write_config(config)
            write_manifest(manifest)

            with self.assertRaisesRegex(FreezeError, "hexadecimal"):
                freeze_experiment(
                    config_path=config,
                    manifest_path=manifest,
                    output_path=output,
                    project_root=root,
                    git_snapshot=GitSnapshot("g" * 40, ()),
                    dataset_validator=passing_dataset_validator,
                )
            self.assertFalse(output.exists())

    def test_all_split_locked_flags_are_semantically_fixed(self) -> None:
        cases = {
            "dev": "true",
            "real": "yes",
            "locked_test": "false",
            "clean_control": "0",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = root / "app.yaml"
            manifest = root / "manifest.csv"
            write_config(config)

            for split, bad_flag in cases.items():
                with self.subTest(split=split):
                    write_manifest(
                        manifest,
                        locked_flags_by_split={split: bad_flag},
                    )
                    output = root / f"bad-lock-{split}.json"
                    expected = str(split in {"locked_test", "clean_control"}).lower()
                    with self.assertRaisesRegex(
                        FreezeError,
                        rf"{split} row is_locked must be {expected}",
                    ):
                        freeze_experiment(
                            config_path=config,
                            manifest_path=manifest,
                            output_path=output,
                            project_root=root,
                            git_snapshot=GitSnapshot("a" * 40, ()),
                            dataset_validator=passing_dataset_validator,
                        )
                    self.assertFalse(output.exists())

    def test_config_or_manifest_change_during_validation_is_rejected(self) -> None:
        for changed_file in ("config", "manifest"):
            with self.subTest(changed_file=changed_file), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                config = root / "app.yaml"
                manifest = root / "manifest.csv"
                output = root / "frozen.json"
                write_config(config)
                write_manifest(manifest)

                def mutating_validator(*_args: object) -> StubValidationReport:
                    target = config if changed_file == "config" else manifest
                    with target.open("ab") as handle:
                        handle.write(b"\n")
                    return StubValidationReport()

                expected_label = "app config" if changed_file == "config" else "manifest"
                with self.assertRaisesRegex(
                    FreezeError,
                    rf"{expected_label} changed during freeze",
                ):
                    freeze_experiment(
                        config_path=config,
                        manifest_path=manifest,
                        output_path=output,
                        project_root=root,
                        git_snapshot=GitSnapshot("a" * 40, ()),
                        dataset_validator=mutating_validator,
                    )
                self.assertFalse(output.exists())

    def test_exclusive_create_preserves_racing_writer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = root / "app.yaml"
            manifest = root / "manifest.csv"
            output = root / "frozen.json"
            write_config(config)
            write_manifest(manifest)
            racing_payload = b"racing writer won\n"
            real_os_open = os.open

            def racing_open(path: object, flags: int, mode: int = 0o777) -> int:
                output.write_bytes(racing_payload)
                return real_os_open(path, flags, mode)

            with mock.patch(
                "scripts.freeze_experiment.os.open",
                side_effect=racing_open,
            ), self.assertRaisesRegex(FreezeError, "already exists"):
                freeze_experiment(
                    config_path=config,
                    manifest_path=manifest,
                    output_path=output,
                    project_root=root,
                    git_snapshot=GitSnapshot("a" * 40, ()),
                    dataset_validator=passing_dataset_validator,
                )

            self.assertEqual(output.read_bytes(), racing_payload)

    def test_refuses_dirty_worktree_pending_consent_and_asr_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = root / "app.yaml"
            manifest = root / "manifest.csv"
            write_config(config)
            write_manifest(manifest)

            with self.assertRaisesRegex(FreezeError, "dirty"):
                freeze_experiment(
                    config_path=config,
                    manifest_path=manifest,
                    output_path=root / "dirty.json",
                    project_root=root,
                    git_snapshot=GitSnapshot("b" * 40, (" M core/pipeline.py",)),
                    dataset_validator=passing_dataset_validator,
                )

            write_manifest(manifest, pending=True)
            with self.assertRaisesRegex(FreezeError, "exact formal token"):
                freeze_experiment(
                    config_path=config,
                    manifest_path=manifest,
                    output_path=root / "pending.json",
                    project_root=root,
                    git_snapshot=GitSnapshot("b" * 40, ()),
                    dataset_validator=passing_dataset_validator,
                )

            write_manifest(manifest)
            write_config(config, initial_prompt="参考文本")
            with self.assertRaisesRegex(FreezeError, "initial_prompt"):
                freeze_experiment(
                    config_path=config,
                    manifest_path=manifest,
                    output_path=root / "prompt.json",
                    project_root=root,
                    git_snapshot=GitSnapshot("b" * 40, ()),
                    dataset_validator=passing_dataset_validator,
                )

    def test_allow_dirty_records_development_exception(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = root / "app.yaml"
            manifest = root / "manifest.csv"
            write_config(config)
            write_manifest(manifest)
            record = freeze_experiment(
                config_path=config,
                manifest_path=manifest,
                output_path=root / "dev-freeze.json",
                project_root=root,
                allow_dirty=True,
                git_snapshot=GitSnapshot("c" * 40, ("?? scratch.txt",)),
                dataset_validator=passing_dataset_validator,
            )
            self.assertTrue(record["git"]["dirty"])
            self.assertEqual(record["git"]["dirty_entries"], ["?? scratch.txt"])

    def test_validator_error_refuses_freeze_without_creating_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = root / "configs/app.yaml"
            manifest = root / "data_local/manifest.csv"
            dataset_root = root / "dataset"
            output = root / "freeze/config_frozen.json"
            write_config(config)
            write_manifest(manifest)
            calls: list[tuple[Path, Path]] = []

            def failing_validator(
                selected_root: str | Path,
                selected_manifest: str | Path | None,
            ) -> StubValidationReport:
                calls.append((Path(selected_root), Path(selected_manifest or "")))
                return StubValidationReport(
                    checked_wavs=41,
                    errors=(
                        StubIssue(
                            "WAV_SAMPLE_RATE",
                            "raw/clean/spkA/clean_spkA_s01.wav",
                            "expected 48000 Hz, got 44100 Hz",
                        ),
                    ),
                )

            with self.assertRaisesRegex(
                FreezeError,
                r"dataset validation reported 1 error.*WAV_SAMPLE_RATE",
            ):
                freeze_experiment(
                    config_path=config,
                    manifest_path=manifest,
                    output_path=output,
                    project_root=root,
                    dataset_root=dataset_root,
                    git_snapshot=GitSnapshot("d" * 40, ()),
                    dataset_validator=failing_validator,
                )

            self.assertEqual(calls, [(dataset_root.resolve(), manifest.resolve())])
            self.assertFalse(output.exists())
            self.assertFalse(output.parent.exists())

    def test_cli_cannot_bypass_default_validator(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = root / "app.yaml"
            manifest = root / "metadata/manifest.csv"
            dataset_root = root / "dataset"
            output = root / "frozen.json"
            write_config(config)
            write_manifest(manifest)
            validator = mock.Mock(
                return_value=StubValidationReport(
                    errors=(
                        StubIssue("MISSING_WAV", "raw/noise.wav", "missing"),
                    )
                )
            )

            with mock.patch(
                "scripts.freeze_experiment.inspect_git",
                return_value=GitSnapshot("e" * 40, ()),
            ), mock.patch(
                "scripts.freeze_experiment.validate_dataset",
                validator,
            ), redirect_stdout(io.StringIO()):
                exit_code = main(
                    [
                        "--config",
                        str(config),
                        "--manifest",
                        str(manifest),
                        "--dataset-root",
                        str(dataset_root),
                        "--output",
                        str(output),
                        "--project-root",
                        str(root),
                    ]
                )

            self.assertEqual(exit_code, 1)
            self.assertFalse(output.exists())
            validator.assert_called_once_with(dataset_root.resolve(), manifest.resolve())


if __name__ == "__main__":
    unittest.main()
