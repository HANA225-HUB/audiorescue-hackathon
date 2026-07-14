import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.run_evaluation import (
    EXPECTED_SPLIT_COUNTS,
    EvaluationError,
    REQUIRED_MANIFEST_FIELDS,
    RuntimeIdentity,
    _canonical_primary_paths,
    _inspect_runtime_identity,
    locked_receipt_path,
    run_evaluation,
)


MANIFEST_FIELDS = tuple(sorted(REQUIRED_MANIFEST_FIELDS))
FREEZE_SCHEMA_VERSION = "audiorescue-experiment-freeze-v2"
FROZEN_COMMIT = "a" * 40
FROZEN_CONFIG_SHA256 = "b" * 64


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def make_row(
    root: Path,
    sample_id: str,
    *,
    split: str = "dev",
    source_type: str | None = None,
    noise_type: str = "fan",
    snr_db: str = "0",
    relative_path: str | None = None,
) -> dict[str, str]:
    if relative_path is not None:
        relative = Path(relative_path)
    elif split == "clean_control":
        relative = Path("raw/clean/spkA") / f"{sample_id}.wav"
    elif split == "real":
        relative = Path("raw/real") / f"{sample_id}.wav"
    else:
        relative = Path("controlled") / split / f"{sample_id}.wav"
    absolute = root / relative
    absolute.parent.mkdir(parents=True, exist_ok=True)
    absolute.write_bytes(f"test audio {sample_id}".encode("utf-8"))
    if source_type is None:
        source_type = {
            "clean_control": "clean_control",
            "real": "real",
        }.get(split, "controlled_mix")
    is_locked = split in {"locked_test", "clean_control"}
    return {
        "sample_id": sample_id,
        "dataset_version": "AudioRescue-CN-Mini-v1",
        "split": split,
        "source_type": source_type,
        "reference_text": f"参考文本-{sample_id}",
        "clean_path": relative.as_posix() if split == "clean_control" else "",
        "mixed_path": "" if split == "clean_control" else relative.as_posix(),
        "noise_type": noise_type,
        "snr_db": snr_db,
        "sha256": hashlib.sha256(absolute.read_bytes()).hexdigest(),
        "is_locked": str(is_locked),
    }


def make_split_rows(root: Path, split: str) -> list[dict[str, str]]:
    noises = ("fan", "keyboard", "traffic")
    snrs = ("5", "0", "-5")
    return [
        make_row(
            root,
            sample_id,
            split=split,
            noise_type="" if split == "clean_control" else noises[index % 3],
            snr_db=snrs[index % 3] if split in {"dev", "locked_test"} else "",
            relative_path=relative_path,
        )
        for index, (sample_id, relative_path) in enumerate(
            sorted(_canonical_primary_paths(split).items())
        )
    ]


def make_full_manifest_rows(root: Path) -> list[dict[str, str]]:
    return [
        row
        for split in EXPECTED_SPLIT_COUNTS
        for row in make_split_rows(root, split)
    ]


def result_payload(
    status: str = "success",
    before: float | None = 0.5,
    after: float | None = 0.25,
) -> dict[str, object]:
    return {
        "job_id": "job",
        "status": status,
        "cer_before": None if before is None else {"cer": before},
        "cer_after": None if after is None else {"cer": after},
    }


def write_freeze_record(
    path: Path,
    manifest: Path,
    *,
    strength: float = 0.75,
    dirty: bool = False,
    manifest_sha256: str | None = None,
    schema_version: str = FREEZE_SCHEMA_VERSION,
    git_commit: str = FROZEN_COMMIT,
    config_sha256: str = FROZEN_CONFIG_SHA256,
    dataset_version: str = "AudioRescue-CN-Mini-v1",
) -> None:
    payload = {
        "freeze_schema_version": schema_version,
        "dataset_version": dataset_version,
        "contract_version": "v0.1-contract",
        "manifest": {
            "path": str(manifest.resolve()),
            "sha256": manifest_sha256
            or hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "row_count": 33,
            "split_counts": EXPECTED_SPLIT_COUNTS,
            "locked_sample_ids": sorted(_canonical_primary_paths("locked_test")),
        },
        "config": {
            "sha256": config_sha256,
            "processing": {
                "enhancement": {"default_strength": strength},
                "asr": {"initial_prompt": None},
            },
        },
        "git": {"commit": git_commit, "dirty": dirty},
        "dataset_validation": {
            "validator_id": "scripts.validate_dataset.validate_dataset",
            "validator_version": "audiorescue-dataset-validator-v1",
            "dataset_root": str(manifest.parent.resolve()),
            "checked_wavs": 42,
            "expected_wavs": 42,
            "manifest_rows": 33,
            "hashes_verified": 33,
            "warning_count": 0,
        },
        "rules": {
            "reference_text_not_used_as_asr_prompt": True,
            "locked_test_is_one_shot": True,
            "config_changes_require_new_freeze_record": True,
        },
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def runtime_identity(**overrides: object) -> RuntimeIdentity:
    values: dict[str, object] = {
        "git_commit": FROZEN_COMMIT,
        "git_dirty": False,
        "config_path": "/tmp/audiorescue-app.yaml",
        "config_sha256": FROZEN_CONFIG_SHA256,
    }
    values.update(overrides)
    return RuntimeIdentity(**values)  # type: ignore[arg-type]


class RuntimeIdentityTest(unittest.TestCase):
    def test_git_status_includes_untracked_files(self) -> None:
        completed = [
            mock.Mock(stdout=FROZEN_COMMIT + "\n"),
            mock.Mock(stdout="?? core/new_module.py\n"),
        ]
        with mock.patch(
            "scripts.run_evaluation.subprocess.run", side_effect=completed
        ) as run:
            identity = _inspect_runtime_identity()

        self.assertTrue(identity.git_dirty)
        self.assertEqual(
            run.call_args_list[1].args[0],
            ["git", "status", "--porcelain", "--untracked-files=all"],
        )


class ManifestPreflightTest(unittest.TestCase):
    def test_all_splits_require_the_frozen_row_count(self) -> None:
        for split, expected_count in EXPECTED_SPLIT_COUNTS.items():
            with self.subTest(split=split), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir) / "data_local"
                root.mkdir()
                rows = make_split_rows(root, split)[:-1]
                manifest = root / "manifest.csv"
                write_manifest(manifest, rows)
                arguments: dict[str, object] = {
                    "manifest_path": manifest,
                    "dataset_root": root,
                    "split": split,
                    "output_dir": Path(temp_dir) / "out",
                    "process_callable": lambda **_: result_payload(),
                }
                if split == "locked_test":
                    arguments.update(
                        confirm_locked=True,
                        frozen_config=Path(temp_dir) / "placeholder.json",
                        force_recompute=True,
                    )
                    arguments.pop("process_callable", None)
                elif split == "clean_control":
                    arguments["frozen_config"] = Path(temp_dir) / "placeholder.json"
                with self.assertRaisesRegex(
                    EvaluationError,
                    f"exactly {expected_count} rows",
                ):
                    run_evaluation(**arguments)  # type: ignore[arg-type]

    def test_flags_are_split_specific(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "data_local"
            root.mkdir()
            rows = make_split_rows(root, "dev")
            rows[0]["is_locked"] = "true"
            manifest = root / "manifest.csv"
            write_manifest(manifest, rows)
            with self.assertRaisesRegex(EvaluationError, "is_locked must be false"):
                run_evaluation(
                    manifest_path=manifest,
                    dataset_root=root,
                    split="dev",
                    output_dir=Path(temp_dir) / "out",
                    process_callable=lambda **_: result_payload(),
                )

    def test_dev_cannot_relabel_or_open_a_locked_test_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "data_local"
            root.mkdir()
            rows = make_split_rows(root, "dev")
            locked_path = next(iter(_canonical_primary_paths("locked_test").values()))
            locked_file = root / locked_path
            locked_file.parent.mkdir(parents=True, exist_ok=True)
            locked_file.write_bytes(b"locked bytes must not enter dev")
            rows[0]["mixed_path"] = locked_path
            rows[0]["sha256"] = hashlib.sha256(locked_file.read_bytes()).hexdigest()
            manifest = root / "manifest.csv"
            write_manifest(manifest, rows)
            with self.assertRaisesRegex(EvaluationError, "primary path must be"):
                run_evaluation(
                    manifest_path=manifest,
                    dataset_root=root,
                    split="dev",
                    output_dir=Path(temp_dir) / "out",
                    process_callable=lambda **_: result_payload(),
                )

    def test_all_hashes_are_checked_before_any_process_call(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "data_local"
            root.mkdir()
            rows = make_split_rows(root, "dev")
            rows[-1]["sha256"] = "0" * 64
            manifest = root / "manifest.csv"
            write_manifest(manifest, rows)
            calls = 0

            def fake_process(**_: object) -> dict[str, object]:
                nonlocal calls
                calls += 1
                return result_payload()

            with self.assertRaisesRegex(EvaluationError, "sha256 does not match"):
                run_evaluation(
                    manifest_path=manifest,
                    dataset_root=root,
                    split="dev",
                    output_dir=Path(temp_dir) / "out",
                    process_callable=fake_process,
                )
            self.assertEqual(calls, 0)
            self.assertFalse((Path(temp_dir) / "out").exists())


class EvaluationSelectionTest(unittest.TestCase):
    def test_dev_passes_reference_only_to_pipeline_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "data_local"
            root.mkdir()
            rows = make_split_rows(root, "dev")
            manifest = root / "manifest.csv"
            write_manifest(manifest, rows)
            calls: list[dict[str, object]] = []

            def fake_process(**kwargs: object) -> dict[str, object]:
                calls.append(kwargs)
                return result_payload()

            index, summary = run_evaluation(
                manifest_path=manifest,
                dataset_root=root,
                split="dev",
                output_dir=Path(temp_dir) / "evaluation",
                process_callable=fake_process,
            )
            self.assertEqual(len(calls), 18)
            self.assertEqual(
                set(calls[0]),
                {
                    "input_path",
                    "strength",
                    "enable_events",
                    "reference_text",
                    "force_recompute",
                },
            )
            self.assertEqual(calls[0]["reference_text"], rows[0]["reference_text"])
            self.assertNotIn("initial_prompt", calls[0])
            self.assertEqual(len(index["samples"]), 18)
            self.assertEqual(summary["sample_count"], 18)

    def test_clean_control_requires_completed_locked_run_then_is_repeatable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "data_local"
            root.mkdir()
            rows = make_full_manifest_rows(root)
            manifest = root / "manifest.csv"
            write_manifest(manifest, rows)
            freeze = Path(temp_dir) / "frozen.json"
            config = Path(temp_dir) / "app.yaml"
            config.write_text("audiorescue-clean-gate-config\n", encoding="utf-8")
            selected_runtime = runtime_identity(
                config_path=str(config),
                config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
            )
            write_freeze_record(
                freeze,
                manifest,
                config_sha256=selected_runtime.config_sha256,
            )
            received: list[str] = []
            receipt_root = Path(temp_dir) / "receipt-state"

            def fake_process(**kwargs: object) -> dict[str, object]:
                received.append(str(kwargs["input_path"]))
                return result_payload()

            with (
                mock.patch(
                    "scripts.run_evaluation.LOCKED_RECEIPT_ROOT",
                    receipt_root,
                ),
                mock.patch("scripts.run_evaluation._validate_full_dataset"),
                mock.patch(
                    "scripts.run_evaluation._inspect_runtime_identity",
                    return_value=selected_runtime,
                ),
                mock.patch(
                    "core.pipeline.process_audio",
                    side_effect=lambda **_: result_payload(),
                ),
            ):
                with self.assertRaisesRegex(
                    EvaluationError,
                    "requires the same dataset's completed locked_test receipt",
                ):
                    run_evaluation(
                        manifest_path=manifest,
                        dataset_root=root,
                        split="clean_control",
                        output_dir=Path(temp_dir) / "clean_before_locked",
                        frozen_config=freeze,
                        runtime_identity=selected_runtime,
                        process_callable=fake_process,
                    )

                run_evaluation(
                    manifest_path=manifest,
                    dataset_root=root,
                    split="locked_test",
                    output_dir=Path(temp_dir) / "locked_out",
                    frozen_config=freeze,
                    confirm_locked=True,
                    force_recompute=True,
                )

                for run_number in (1, 2):
                    run_evaluation(
                        manifest_path=manifest,
                        dataset_root=root,
                        split="clean_control",
                        output_dir=Path(temp_dir) / f"clean_out_{run_number}",
                        frozen_config=freeze,
                        runtime_identity=selected_runtime,
                        process_callable=fake_process,
                    )
                receipt_path = locked_receipt_path(freeze)
            self.assertEqual(len(received), 6)
            self.assertTrue(all("raw/clean" in path for path in received))
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            self.assertEqual(receipt["status"], "complete")

    def test_dev_accepts_an_ordinary_json_record(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "data_local"
            root.mkdir()
            rows = make_split_rows(root, "dev")
            manifest = root / "manifest.csv"
            write_manifest(manifest, rows)
            ordinary_json = Path(temp_dir) / "notes.json"
            ordinary_json.write_text('{"note": "development run"}\n', encoding="utf-8")
            index, _ = run_evaluation(
                manifest_path=manifest,
                dataset_root=root,
                split="dev",
                output_dir=Path(temp_dir) / "out",
                frozen_config=ordinary_json,
                process_callable=lambda **_: result_payload(),
            )
            self.assertEqual(
                index["frozen_config"]["sha256"],
                hashlib.sha256(ordinary_json.read_bytes()).hexdigest(),
            )


class LockedEvaluationGuardTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root = self.base / "data_local"
        self.root.mkdir()
        receipt_patch = mock.patch(
            "scripts.run_evaluation.LOCKED_RECEIPT_ROOT",
            self.base / "receipt-state",
        )
        receipt_patch.start()
        self.addCleanup(receipt_patch.stop)
        self.all_rows = make_full_manifest_rows(self.root)
        self.rows = [row for row in self.all_rows if row["split"] == "locked_test"]
        self.manifest = self.root / "manifest.csv"
        write_manifest(self.manifest, self.all_rows)
        self.freeze = self.base / "frozen.json"
        self.config = self.base / "app.yaml"
        self.config.write_text("audiorescue-test-config\n", encoding="utf-8")
        self.runtime = runtime_identity(
            config_path=str(self.config),
            config_sha256=hashlib.sha256(self.config.read_bytes()).hexdigest(),
        )

    def write_freeze(self, **overrides: object) -> None:
        overrides.setdefault("config_sha256", self.runtime.config_sha256)
        write_freeze_record(
            self.freeze,
            self.manifest,
            **overrides,  # type: ignore[arg-type]
        )

    def call(
        self,
        *,
        fake_process: object | None = None,
        inspected_runtime: RuntimeIdentity | None = None,
        dataset_gate_side_effect: object | None = None,
        **overrides: object,
    ) -> tuple[dict[str, object], dict[str, object]]:
        process = fake_process or (lambda **_: result_payload())
        arguments: dict[str, object] = {
            "manifest_path": self.manifest,
            "dataset_root": self.root,
            "split": "locked_test",
            "output_dir": self.base / "evaluation",
            "frozen_config": self.freeze,
            "confirm_locked": True,
            "force_recompute": True,
        }
        arguments.update(overrides)
        with (
            mock.patch(
                "scripts.run_evaluation._validate_full_dataset",
                side_effect=dataset_gate_side_effect,
            ),
            mock.patch(
                "scripts.run_evaluation._inspect_runtime_identity",
                return_value=inspected_runtime or self.runtime,
            ),
            mock.patch("core.pipeline.process_audio", side_effect=process),
        ):
            return run_evaluation(**arguments)  # type: ignore[arg-type,return-value]

    def test_requires_confirmation_and_frozen_config(self) -> None:
        with self.assertRaisesRegex(EvaluationError, "confirm-locked"):
            self.call(confirm_locked=False)
        with self.assertRaisesRegex(EvaluationError, "frozen-config"):
            self.call(frozen_config=None)

    def test_rejects_invalid_or_wrong_schema_freeze(self) -> None:
        self.freeze.write_text("not json", encoding="utf-8")
        with self.assertRaisesRegex(EvaluationError, "valid JSON"):
            self.call()
        self.write_freeze(schema_version="wrong")
        with self.assertRaisesRegex(EvaluationError, "freeze_schema_version"):
            self.call()

    def test_rejects_handwritten_minimal_freeze_without_validation_evidence(self) -> None:
        payload = {
            "freeze_schema_version": FREEZE_SCHEMA_VERSION,
            "dataset_version": "AudioRescue-CN-Mini-v1",
            "manifest": {
                "path": str(self.manifest),
                "sha256": hashlib.sha256(self.manifest.read_bytes()).hexdigest(),
                "row_count": 33,
                "split_counts": EXPECTED_SPLIT_COUNTS,
                "locked_sample_ids": sorted(_canonical_primary_paths("locked_test")),
            },
            "config": {
                "sha256": self.runtime.config_sha256,
                "processing": {
                    "enhancement": {"default_strength": 0.75},
                    "asr": {"initial_prompt": None},
                },
            },
            "git": {"commit": FROZEN_COMMIT, "dirty": False},
        }
        self.freeze.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(EvaluationError, "dataset_validation evidence"):
            self.call()

    def test_rejects_frozen_manifest_strength_or_dirty_drift(self) -> None:
        self.write_freeze(manifest_sha256="0" * 64)
        with self.assertRaisesRegex(EvaluationError, "manifest sha256"):
            self.call()
        self.write_freeze(strength=0.5)
        with self.assertRaisesRegex(EvaluationError, "requested strength"):
            self.call()
        self.write_freeze(dirty=True)
        with self.assertRaisesRegex(EvaluationError, "dirty freeze"):
            self.call()

    def test_rejects_runtime_commit_dirty_or_config_drift(self) -> None:
        self.write_freeze()
        with self.assertRaisesRegex(EvaluationError, "Git HEAD"):
            self.call(inspected_runtime=runtime_identity(git_commit="c" * 40))
        with self.assertRaisesRegex(
            EvaluationError, "tracked or untracked changes"
        ):
            self.call(inspected_runtime=runtime_identity(git_dirty=True))
        with self.assertRaisesRegex(EvaluationError, "pipeline config sha256"):
            self.call(inspected_runtime=runtime_identity(config_sha256="c" * 64))

    def test_locked_forbids_overwrite(self) -> None:
        self.write_freeze()
        with self.assertRaisesRegex(EvaluationError, "forbids --overwrite"):
            self.call(overwrite=True)

    def test_success_consumes_freeze_and_second_output_cannot_bypass(self) -> None:
        self.write_freeze()
        index, summary = self.call()
        receipt_path = locked_receipt_path(self.freeze)
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        self.assertEqual(len(index["samples"]), 9)
        self.assertEqual(summary["status_counts"]["success"], 9)
        self.assertEqual(receipt["status"], "complete")
        self.assertEqual(receipt["status_counts"]["success"], 9)
        identity = receipt["locked_dataset"]["identity_payload"]
        audit = receipt["locked_dataset"]["audit"]
        self.assertEqual(audit["dataset_version"], "AudioRescue-CN-Mini-v1")
        self.assertEqual(
            audit["manifest_sha256"],
            hashlib.sha256(self.manifest.read_bytes()).hexdigest(),
        )
        self.assertEqual(len(identity["locked_samples"]), 9)
        self.assertEqual(len(audit["reference_text_sha256"]), 9)
        self.assertEqual(
            receipt["locked_dataset"]["dataset_id"],
            receipt_path.stem,
        )
        self.assertEqual(len(receipt["evaluation_index"]["sha256"]), 64)
        self.assertEqual(len(receipt["summary"]["sha256"]), 64)
        with self.assertRaisesRegex(EvaluationError, "already been consumed"):
            self.call(output_dir=self.base / "different-output")

    def test_copy_or_reformatted_freeze_uses_the_same_one_shot_receipt(self) -> None:
        self.write_freeze()
        original_receipt = locked_receipt_path(self.freeze)
        copied = self.base / "copied-and-reformatted.json"
        copied.write_text(
            json.dumps(
                json.loads(self.freeze.read_text(encoding="utf-8")),
                ensure_ascii=False,
                indent=4,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        self.assertEqual(locked_receipt_path(copied), original_receipt)
        self.call()
        with self.assertRaisesRegex(EvaluationError, "already been consumed"):
            self.call(
                frozen_config=copied,
                output_dir=self.base / "copied-freeze-output",
            )

    def test_new_code_config_and_freeze_still_hit_the_same_dataset_receipt(self) -> None:
        self.write_freeze()
        original_receipt = locked_receipt_path(self.freeze)
        self.call()

        new_config = self.base / "new-app.yaml"
        new_config.write_text("new frozen pipeline config\n", encoding="utf-8")
        new_runtime = runtime_identity(
            git_commit="c" * 40,
            config_path=str(new_config),
            config_sha256=hashlib.sha256(new_config.read_bytes()).hexdigest(),
        )
        new_freeze = self.base / "new-code-config-freeze.json"
        write_freeze_record(
            new_freeze,
            self.manifest,
            git_commit=new_runtime.git_commit,
            config_sha256=new_runtime.config_sha256,
        )

        self.assertEqual(locked_receipt_path(new_freeze), original_receipt)
        with self.assertRaisesRegex(EvaluationError, "already been consumed"):
            self.call(
                frozen_config=new_freeze,
                inspected_runtime=new_runtime,
                output_dir=self.base / "new-code-config-output",
            )

    def test_receipt_key_only_changes_when_a_locked_audio_hash_changes(self) -> None:
        self.write_freeze()
        baseline = locked_receipt_path(self.freeze)

        variants: list[tuple[str, list[dict[str, str]], str, bool]] = []
        variants.append(
            (
                "row-order",
                [dict(row) for row in reversed(self.all_rows)],
                "AudioRescue-CN-Mini-v1",
                False,
            )
        )
        dev_changed = [dict(row) for row in self.all_rows]
        next(row for row in dev_changed if row["split"] == "dev")[
            "reference_text"
        ] = "只修改开发集参考文本"
        variants.append(
            ("dev-row", dev_changed, "AudioRescue-CN-Mini-v1", False)
        )
        locked_reference_changed = [dict(row) for row in self.all_rows]
        next(
            row
            for row in locked_reference_changed
            if row["split"] == "locked_test"
        )["reference_text"] = "只修改锁定集参考文本，不改变音频"
        variants.append(
            (
                "locked-reference",
                locked_reference_changed,
                "AudioRescue-CN-Mini-v1",
                False,
            )
        )
        version_changed = [dict(row) for row in self.all_rows]
        for row in version_changed:
            row["dataset_version"] = "AudioRescue-CN-Mini-v2"
        variants.append(
            ("dataset-version", version_changed, "AudioRescue-CN-Mini-v2", False)
        )
        variants.append(
            (
                "csv-format",
                [dict(row) for row in self.all_rows],
                "AudioRescue-CN-Mini-v1",
                True,
            )
        )

        for name, rows, dataset_version, append_blank_line in variants:
            with self.subTest(unchanged_locked_audio=name):
                manifest = self.root / f"manifest-{name}.csv"
                write_manifest(manifest, rows)
                if append_blank_line:
                    with manifest.open("a", encoding="utf-8") as handle:
                        handle.write("\n")
                freeze = self.base / f"freeze-{name}.json"
                write_freeze_record(
                    freeze,
                    manifest,
                    dataset_version=dataset_version,
                )
                self.assertEqual(locked_receipt_path(freeze), baseline)

        locked_changed = [dict(row) for row in self.all_rows]
        changed_row = next(
            row for row in locked_changed if row["split"] == "locked_test"
        )
        changed_audio = self.root / changed_row["mixed_path"]
        changed_audio.write_bytes(b"changed locked audio bytes")
        changed_row["sha256"] = hashlib.sha256(changed_audio.read_bytes()).hexdigest()
        changed_manifest = self.root / "manifest-locked-audio-changed.csv"
        write_manifest(changed_manifest, locked_changed)
        changed_freeze = self.base / "freeze-locked-audio-changed.json"
        write_freeze_record(changed_freeze, changed_manifest)
        self.assertNotEqual(locked_receipt_path(changed_freeze), baseline)

    def test_started_receipt_blocks_locked_retry_and_clean_control(self) -> None:
        self.write_freeze()

        def interrupting_process(**_: object) -> dict[str, object]:
            raise KeyboardInterrupt("simulated process interruption")

        with self.assertRaises(KeyboardInterrupt):
            self.call(fake_process=interrupting_process)

        receipt_path = locked_receipt_path(self.freeze)
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        self.assertEqual(receipt["status"], "started")
        with self.assertRaisesRegex(EvaluationError, "already been consumed"):
            self.call(output_dir=self.base / "locked-retry-after-started")

        with mock.patch("scripts.run_evaluation._validate_full_dataset"):
            with self.assertRaisesRegex(
                EvaluationError,
                "requires a completed locked_test receipt.*started",
            ):
                run_evaluation(
                    manifest_path=self.manifest,
                    dataset_root=self.root,
                    split="clean_control",
                    output_dir=self.base / "clean-after-started",
                    frozen_config=self.freeze,
                    runtime_identity=self.runtime,
                    process_callable=lambda **_: result_payload(),
                )

    def test_manifest_change_during_validation_is_refused_before_receipt(self) -> None:
        self.write_freeze()
        receipt_path = locked_receipt_path(self.freeze)

        def mutate_manifest(*_: object) -> None:
            with self.manifest.open("ab") as handle:
                handle.write(b"\n")

        with self.assertRaisesRegex(EvaluationError, "manifest changed"):
            self.call(dataset_gate_side_effect=mutate_manifest)
        self.assertFalse(receipt_path.exists())

    def test_formal_locked_api_rejects_injected_test_hooks(self) -> None:
        self.write_freeze()
        with self.assertRaisesRegex(EvaluationError, "forbids injected"):
            run_evaluation(
                manifest_path=self.manifest,
                dataset_root=self.root,
                split="locked_test",
                output_dir=self.base / "injected",
                frozen_config=self.freeze,
                confirm_locked=True,
                force_recompute=True,
                runtime_identity=self.runtime,
                process_callable=lambda **_: result_payload(),
            )

    def test_hash_preflight_failure_does_not_consume(self) -> None:
        self.rows[-1]["sha256"] = "0" * 64
        write_manifest(self.manifest, self.all_rows)
        self.write_freeze()
        with self.assertRaisesRegex(EvaluationError, "sha256 does not match"):
            self.call()
        self.assertFalse(locked_receipt_path(self.freeze).exists())
        self.assertFalse((self.base / "evaluation").exists())

    def test_sample_exception_still_consumes_and_completes_receipt(self) -> None:
        self.write_freeze()

        def failing_process(**_: object) -> dict[str, object]:
            raise RuntimeError("model failed")

        _, summary = self.call(fake_process=failing_process)
        self.assertEqual(summary["status_counts"]["failed"], 9)
        receipt = json.loads(
            locked_receipt_path(self.freeze).read_text(encoding="utf-8")
        )
        self.assertEqual(receipt["status"], "complete")
        self.assertEqual(receipt["status_counts"]["failed"], 9)
        with self.assertRaisesRegex(EvaluationError, "already been consumed"):
            self.call(output_dir=self.base / "retry-output")


class EvaluationAggregationTest(unittest.TestCase):
    def test_exceptions_continue_and_summary_has_all_required_groups(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "data_local"
            root.mkdir()
            rows = make_split_rows(root, "dev")
            rows[0].update(noise_type="fan", snr_db="5")
            rows[1].update(noise_type="fan", snr_db="0")
            rows[2].update(noise_type="keyboard", snr_db="-5")
            rows[3].update(noise_type="traffic", snr_db="0")
            manifest = root / "manifest.csv"
            write_manifest(manifest, rows)
            outcomes = {
                Path(rows[0]["mixed_path"]).stem: result_payload("success", 0.5, 0.25),
                Path(rows[1]["mixed_path"]).stem: result_payload("success", 0.2, 0.2),
                Path(rows[2]["mixed_path"]).stem: result_payload("partial", 0.1, 0.3),
            }

            def fake_process(**kwargs: object) -> dict[str, object]:
                file_stem = Path(str(kwargs["input_path"])).stem
                if file_stem == Path(rows[3]["mixed_path"]).stem:
                    raise RuntimeError("isolated failure")
                return outcomes.get(file_stem, result_payload("success", None, None))

            output_dir = Path(temp_dir) / "evaluation"
            index, summary = run_evaluation(
                manifest_path=manifest,
                dataset_root=root,
                split="dev",
                output_dir=output_dir,
                process_callable=fake_process,
            )
            self.assertEqual(
                summary["status_counts"],
                {"success": 16, "partial": 1, "failed": 1},
            )
            self.assertEqual(summary["cer"]["improved"], 1)
            self.assertEqual(summary["cer"]["tied"], 1)
            self.assertEqual(summary["cer"]["worsened"], 1)
            self.assertEqual(summary["cer"]["unavailable"], 15)
            self.assertEqual(summary["cer"]["before_median"], 0.2)
            self.assertEqual(summary["cer"]["after_median"], 0.25)
            self.assertEqual(summary["by_noise_type"]["fan"]["sample_count"], 6)
            self.assertEqual(summary["by_snr_db"]["0"]["sample_count"], 7)
            self.assertEqual(index["samples"][3]["error"]["type"], "RuntimeError")
            self.assertTrue((output_dir / "evaluation_index.json").is_file())
            self.assertTrue((output_dir / "summary.json").is_file())

    def test_existing_output_directory_requires_explicit_overwrite_for_dev(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "data_local"
            root.mkdir()
            rows = make_split_rows(root, "dev")
            manifest = root / "manifest.csv"
            write_manifest(manifest, rows)
            output = Path(temp_dir) / "out"
            output.mkdir()
            with self.assertRaisesRegex(EvaluationError, "already exists"):
                run_evaluation(
                    manifest_path=manifest,
                    dataset_root=root,
                    split="dev",
                    output_dir=output,
                    process_callable=lambda **_: result_payload(),
                )
            _, summary = run_evaluation(
                manifest_path=manifest,
                dataset_root=root,
                split="dev",
                output_dir=output,
                overwrite=True,
                process_callable=lambda **_: result_payload(),
            )
            self.assertEqual(summary["status_counts"]["success"], 18)


if __name__ == "__main__":
    unittest.main()
