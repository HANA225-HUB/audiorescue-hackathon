#!/usr/bin/env python3
"""Run a manifest split through the frozen AudioRescue pipeline.

Only the pipeline receives ``reference_text``, where it is used after ASR for
CER. This runner has no Whisper prompt option and never forwards the reference
to a model-facing parameter.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import shutil
import statistics
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SPLITS = ("dev", "locked_test", "real", "clean_control")
STATUSES = ("success", "partial", "failed")
REQUIRED_MANIFEST_FIELDS = {
    "sample_id",
    "split",
    "source_type",
    "reference_text",
    "clean_path",
    "mixed_path",
    "noise_type",
    "snr_db",
    "sha256",
    "is_locked",
}
EXPECTED_SPLIT_COUNTS = {
    "dev": 18,
    "locked_test": 9,
    "real": 3,
    "clean_control": 3,
}
INTEGRITY_GATED_SPLITS = {"locked_test", "clean_control"}
CER_TIE_TOLERANCE = 1e-12
SCHEMA_VERSION = "audiorescue-evaluation-v1"
FREEZE_SCHEMA_VERSION = "audiorescue-experiment-freeze-v2"
RECEIPT_SCHEMA_VERSION = "audiorescue-locked-test-receipt-v1"
DATASET_VALIDATOR_ID = "scripts.validate_dataset.validate_dataset"
DATASET_VALIDATOR_VERSION = "audiorescue-dataset-validator-v1"
EXPECTED_DATASET_WAVS = 42
EXPECTED_MANIFEST_ROWS = 33


class EvaluationError(RuntimeError):
    """Raised for a run-level precondition error before sample processing."""


@dataclass(frozen=True, slots=True)
class RuntimeIdentity:
    """Current code/config identity checked immediately before locked_test."""

    git_commit: str
    git_dirty: bool
    config_path: str
    config_sha256: str


@dataclass(frozen=True, slots=True)
class PreparedSample:
    row: dict[str, str]
    input_path: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class ManifestSnapshot:
    path: Path
    raw_bytes: bytes
    sha256: str
    rows: tuple[dict[str, str], ...]


@dataclass(frozen=True, slots=True)
class FrozenIdentity:
    path: Path
    file_sha256: str
    freeze_id: str
    receipt_path: Path
    payload: Mapping[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _is_hex_digest(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def _inspect_runtime_identity() -> RuntimeIdentity:
    """Read the exact Git/config identity used by the formal CLI run."""

    project_root = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise EvaluationError(f"cannot inspect current Git identity: {exc}") from exc

    configured_path = os.environ.get("AUDIORESCUE_CONFIG")
    config_path = (
        Path(configured_path).expanduser().resolve()
        if configured_path
        else (project_root / "configs" / "app.yaml").resolve()
    )
    if not config_path.is_file():
        raise EvaluationError(f"current pipeline config does not exist: {config_path}")
    try:
        config_sha256 = sha256_file(config_path)
    except OSError as exc:
        raise EvaluationError(f"cannot hash current pipeline config {config_path}: {exc}") from exc
    return RuntimeIdentity(
        git_commit=commit,
        git_dirty=bool(status.strip()),
        config_path=str(config_path),
        config_sha256=config_sha256,
    )


def _coerce_runtime_identity(value: RuntimeIdentity | Mapping[str, Any]) -> RuntimeIdentity:
    if isinstance(value, RuntimeIdentity):
        return value
    if not isinstance(value, Mapping):
        raise EvaluationError("runtime_identity must be RuntimeIdentity or a mapping")
    try:
        return RuntimeIdentity(
            git_commit=value["git_commit"],
            git_dirty=value["git_dirty"],
            config_path=value["config_path"],
            config_sha256=value["config_sha256"],
        )
    except KeyError as exc:
        raise EvaluationError(f"runtime_identity is missing {exc.args[0]}") from exc


def _validate_frozen_config(
    split: str,
    frozen_config: str | Path | None,
    confirm_locked: bool,
    *,
    manifest_sha256: str,
    manifest_path: Path,
    dataset_root: Path,
    requested_strength: float,
    runtime_identity: RuntimeIdentity | None,
) -> FrozenIdentity | dict[str, str] | None:
    if split == "locked_test" and not confirm_locked:
        raise EvaluationError(
            "locked_test requires explicit --confirm-locked after parameters are frozen"
        )
    if split in INTEGRITY_GATED_SPLITS and frozen_config is None:
        raise EvaluationError(f"{split} requires --frozen-config")
    if frozen_config is None:
        return None

    path = Path(frozen_config).expanduser().resolve()
    if not path.is_file():
        raise EvaluationError(f"frozen config does not exist: {path}")
    try:
        raw_bytes = path.read_bytes()
        parsed_config = json.loads(raw_bytes.decode("utf-8"))
        # Python accepts NaN/Infinity by default even though JSON does not.
        json.dumps(parsed_config, allow_nan=False)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise EvaluationError(f"frozen config is not valid JSON: {path}: {exc}") from exc

    if split in INTEGRITY_GATED_SPLITS:
        if not isinstance(parsed_config, Mapping):
            raise EvaluationError("integrity-gated freeze record must be a JSON object")
        if parsed_config.get("freeze_schema_version") != FREEZE_SCHEMA_VERSION:
            raise EvaluationError(
                f"locked freeze_schema_version must be {FREEZE_SCHEMA_VERSION!r}"
            )

        frozen_manifest = parsed_config.get("manifest")
        if not isinstance(frozen_manifest, Mapping):
            raise EvaluationError("locked freeze manifest section is missing")
        frozen_manifest_hash = (
            frozen_manifest.get("sha256")
        )
        if (
            not isinstance(frozen_manifest_hash, str)
            or frozen_manifest_hash.lower() != manifest_sha256
        ):
            raise EvaluationError(
                "locked manifest sha256 does not match the manifest being evaluated"
            )
        frozen_manifest_path = frozen_manifest.get("path")
        if not isinstance(frozen_manifest_path, str):
            raise EvaluationError("locked freeze manifest.path is missing")
        try:
            recorded_manifest_path = Path(frozen_manifest_path).expanduser().resolve()
        except (OSError, RuntimeError, ValueError) as exc:
            raise EvaluationError("locked freeze manifest.path is invalid") from exc
        if recorded_manifest_path != manifest_path:
            raise EvaluationError(
                "locked freeze manifest.path does not match the manifest being evaluated"
            )
        if frozen_manifest.get("row_count") != EXPECTED_MANIFEST_ROWS:
            raise EvaluationError("locked freeze manifest.row_count must be 33")
        split_counts = frozen_manifest.get("split_counts")
        if not isinstance(split_counts, Mapping) or dict(split_counts) != EXPECTED_SPLIT_COUNTS:
            raise EvaluationError(
                f"locked freeze split_counts must be {EXPECTED_SPLIT_COUNTS}"
            )
        expected_locked_ids = sorted(_canonical_primary_paths("locked_test"))
        locked_ids = frozen_manifest.get("locked_sample_ids")
        if (
            not isinstance(locked_ids, list)
            or len(locked_ids) != len(expected_locked_ids)
            or sorted(locked_ids) != expected_locked_ids
        ):
            raise EvaluationError(
                "locked freeze manifest.locked_sample_ids does not match the canonical split"
            )

        validation = parsed_config.get("dataset_validation")
        if not isinstance(validation, Mapping):
            raise EvaluationError("locked freeze dataset_validation evidence is missing")
        expected_validation = {
            "validator_id": DATASET_VALIDATOR_ID,
            "validator_version": DATASET_VALIDATOR_VERSION,
            "checked_wavs": EXPECTED_DATASET_WAVS,
            "expected_wavs": EXPECTED_DATASET_WAVS,
            "manifest_rows": EXPECTED_MANIFEST_ROWS,
            "hashes_verified": EXPECTED_MANIFEST_ROWS,
        }
        for key, expected in expected_validation.items():
            if validation.get(key) != expected:
                raise EvaluationError(
                    f"locked freeze dataset_validation.{key} must be {expected!r}"
                )
        recorded_root = validation.get("dataset_root")
        if not isinstance(recorded_root, str):
            raise EvaluationError("locked freeze dataset_validation.dataset_root is missing")
        try:
            resolved_recorded_root = Path(recorded_root).expanduser().resolve()
        except (OSError, RuntimeError, ValueError) as exc:
            raise EvaluationError(
                "locked freeze dataset_validation.dataset_root is invalid"
            ) from exc
        if resolved_recorded_root != dataset_root:
            raise EvaluationError(
                "locked freeze dataset root does not match the evaluated dataset root"
            )

        rules = parsed_config.get("rules")
        required_rules = (
            "reference_text_not_used_as_asr_prompt",
            "locked_test_is_one_shot",
            "config_changes_require_new_freeze_record",
        )
        if not isinstance(rules, Mapping) or any(
            rules.get(rule) is not True for rule in required_rules
        ):
            raise EvaluationError("locked freeze fairness rules are missing or false")

        frozen_git = parsed_config.get("git")
        if not isinstance(frozen_git, Mapping) or frozen_git.get("dirty") is not False:
            raise EvaluationError("locked evaluation refuses a dirty freeze record")
        frozen_commit = frozen_git.get("commit")
        if not _is_hex_digest(frozen_commit, 40):
            raise EvaluationError("locked freeze git.commit must be a 40-character SHA")
        if runtime_identity is None:
            raise EvaluationError("locked evaluation requires a current runtime identity")
        if not _is_hex_digest(runtime_identity.git_commit, 40):
            raise EvaluationError("current Git HEAD must be a 40-character SHA")
        if str(frozen_commit).lower() != runtime_identity.git_commit.lower():
            raise EvaluationError("current Git HEAD does not match frozen git.commit")
        if runtime_identity.git_dirty is not False:
            raise EvaluationError(
                "current Git worktree has tracked or untracked changes"
            )

        frozen_config_section = parsed_config.get("config")
        processing = (
            frozen_config_section.get("processing")
            if isinstance(frozen_config_section, Mapping)
            else None
        )
        enhancement = (
            processing.get("enhancement") if isinstance(processing, Mapping) else None
        )
        frozen_strength = (
            enhancement.get("default_strength")
            if isinstance(enhancement, Mapping)
            else None
        )
        if (
            isinstance(frozen_strength, bool)
            or not isinstance(frozen_strength, (int, float))
            or not math.isfinite(float(frozen_strength))
            or not 0.0 <= float(frozen_strength) <= 1.0
        ):
            raise EvaluationError(
                "locked freeze config.processing.enhancement.default_strength "
                "must be a finite number from 0 to 1"
            )
        if float(frozen_strength) != requested_strength:
            raise EvaluationError(
                "requested strength does not match frozen "
                "config.processing.enhancement.default_strength"
            )

        frozen_config_hash = (
            frozen_config_section.get("sha256")
            if isinstance(frozen_config_section, Mapping)
            else None
        )
        if not _is_hex_digest(frozen_config_hash, 64):
            raise EvaluationError("locked freeze config.sha256 must be a SHA-256 digest")
        if not _is_hex_digest(runtime_identity.config_sha256, 64):
            raise EvaluationError("current pipeline config SHA-256 is invalid")
        if str(frozen_config_hash).lower() != runtime_identity.config_sha256.lower():
            raise EvaluationError(
                "current pipeline config sha256 does not match frozen config.sha256"
            )
        asr = processing.get("asr") if isinstance(processing, Mapping) else None
        if not isinstance(asr, Mapping) or asr.get("initial_prompt") is not None:
            raise EvaluationError(
                "locked freeze config.processing.asr.initial_prompt must be null"
            )

        identity_payload = {
            "freeze_schema_version": FREEZE_SCHEMA_VERSION,
            "dataset_version": parsed_config.get("dataset_version"),
            "contract_version": parsed_config.get("contract_version"),
            "git_commit": str(frozen_commit).lower(),
            "config_sha256": str(frozen_config_hash).lower(),
            "manifest_sha256": str(frozen_manifest_hash).lower(),
            "locked_sample_ids": expected_locked_ids,
            "dataset_validation": expected_validation,
        }
        freeze_id = hashlib.sha256(
            json.dumps(
                identity_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        receipt_path = dataset_root / ".locked_receipts" / f"{freeze_id}.json"
        return FrozenIdentity(
            path=path,
            file_sha256=hashlib.sha256(raw_bytes).hexdigest(),
            freeze_id=freeze_id,
            receipt_path=receipt_path,
            payload=parsed_config,
        )
    return {"path": str(path), "sha256": hashlib.sha256(raw_bytes).hexdigest()}


def _resolve_in_root(dataset_root: Path, raw_path: str) -> Path:
    value = raw_path.strip()
    if not value:
        raise ValueError("manifest audio path is empty")
    supplied = Path(value)
    if supplied.is_absolute():
        resolved = supplied.resolve()
    else:
        if supplied.parts and supplied.parts[0] == dataset_root.name:
            supplied = Path(*supplied.parts[1:])
        resolved = (dataset_root / supplied).resolve()
    try:
        resolved.relative_to(dataset_root)
    except ValueError as exc:
        raise ValueError(f"manifest audio path escapes dataset root: {raw_path}") from exc
    return resolved


def _read_manifest_snapshot(manifest_path: Path) -> ManifestSnapshot:
    """Parse and hash one immutable byte snapshot of the manifest."""

    if not manifest_path.is_file():
        raise EvaluationError(f"manifest does not exist: {manifest_path}")
    try:
        raw_bytes = manifest_path.read_bytes()
        text = raw_bytes.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(text, newline=""))
        headers = set(reader.fieldnames or [])
        missing = sorted(REQUIRED_MANIFEST_FIELDS - headers)
        if missing:
            raise EvaluationError(
                "manifest is missing required columns: " + ", ".join(missing)
            )
        rows = tuple(
            {
                str(key): (value or "").strip()
                for key, value in row.items()
                if key is not None
            }
            for row in reader
        )
    except EvaluationError:
        raise
    except (csv.Error, OSError, UnicodeError) as exc:
        raise EvaluationError(f"cannot read manifest: {manifest_path}: {exc}") from exc
    if not rows:
        raise EvaluationError("manifest is empty")
    return ManifestSnapshot(
        path=manifest_path,
        raw_bytes=raw_bytes,
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
        rows=rows,
    )


def _canonical_primary_paths(split: str) -> dict[str, str]:
    """Return the fixed sample-id to primary-path map for one dataset split."""

    try:
        from scripts.validate_dataset import (
            expected_dataset_artifacts,
            required_manifest_primary_paths,
        )
    except ModuleNotFoundError:  # pragma: no cover - direct script fallback
        from validate_dataset import (  # type: ignore[no-redef]
            expected_dataset_artifacts,
            required_manifest_primary_paths,
        )

    artifacts = expected_dataset_artifacts()
    expected: dict[str, str] = {}
    for relative_path in required_manifest_primary_paths():
        spec = artifacts[relative_path]
        canonical_split = (
            "clean_control"
            if spec.kind == "clean"
            else "real"
            if spec.kind == "real"
            else spec.split
        )
        if canonical_split == split:
            expected[relative_path.stem] = relative_path.as_posix()
    return expected


def _load_split_rows(snapshot: ManifestSnapshot, split: str) -> list[dict[str, str]]:
    rows = [row for row in snapshot.rows if row.get("split") == split]
    if not rows:
        raise EvaluationError(f"manifest contains no rows for split {split!r}")
    expected_count = EXPECTED_SPLIT_COUNTS[split]
    if len(rows) != expected_count:
        raise EvaluationError(
            f"split {split!r} must contain exactly {expected_count} rows, got {len(rows)}"
        )

    expected_paths = _canonical_primary_paths(split)
    if len(expected_paths) != expected_count:
        raise EvaluationError(
            f"internal canonical split matrix for {split!r} is incomplete"
        )
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    for row in rows:
        sample_id = row.get("sample_id", "")
        if not sample_id:
            raise EvaluationError(f"split {split!r} contains an empty sample_id")
        if sample_id in seen_ids:
            raise EvaluationError(f"split {split!r} contains duplicate sample_id {sample_id!r}")
        seen_ids.add(sample_id)
        expected_path = expected_paths.get(sample_id)
        if expected_path is None:
            raise EvaluationError(
                f"sample {sample_id!r} is not part of the canonical {split!r} split"
            )
        actual_path = _select_audio_path(row).replace("\\", "/")
        while actual_path.startswith("./"):
            actual_path = actual_path[2:]
        if actual_path.startswith(snapshot.path.parent.name + "/"):
            actual_path = actual_path.split("/", 1)[1]
        if actual_path != expected_path:
            raise EvaluationError(
                f"sample {sample_id!r} primary path must be {expected_path!r}, "
                f"got {actual_path!r}"
            )
        if actual_path in seen_paths:
            raise EvaluationError(
                f"split {split!r} reuses primary audio path {actual_path!r}"
            )
        seen_paths.add(actual_path)
    missing_ids = sorted(set(expected_paths) - seen_ids)
    if missing_ids:
        raise EvaluationError(
            f"split {split!r} is missing canonical sample IDs: {', '.join(missing_ids)}"
        )
    return rows


def _select_audio_path(row: Mapping[str, str]) -> str:
    if row.get("split") == "clean_control" or row.get("source_type") == "clean_control":
        return row.get("clean_path", "")
    return row.get("mixed_path", "")


def _parse_locked_flag(value: str, *, sample_id: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise EvaluationError(
        f"sample {sample_id!r} has invalid is_locked value {value!r}"
    )


def _prepare_samples(
    rows: Sequence[dict[str, str]],
    dataset_root: Path,
    split: str,
) -> list[PreparedSample]:
    """Resolve and hash every selected input before any pipeline call."""

    expected_locked = split in {"locked_test", "clean_control"}
    prepared: list[PreparedSample] = []
    for row in rows:
        sample_id = row["sample_id"]
        actual_locked = _parse_locked_flag(
            row.get("is_locked", ""),
            sample_id=sample_id,
        )
        if actual_locked is not expected_locked:
            raise EvaluationError(
                f"sample {sample_id!r} is_locked must be "
                f"{str(expected_locked).lower()} for split {split!r}"
            )
        reference_text = row.get("reference_text", "")
        if not reference_text:
            raise EvaluationError(f"sample {sample_id!r} has empty reference_text")
        try:
            input_path = _resolve_in_root(dataset_root, _select_audio_path(row))
        except ValueError as exc:
            raise EvaluationError(f"sample {sample_id!r}: {exc}") from exc
        if not input_path.is_file():
            raise EvaluationError(
                f"sample {sample_id!r} input audio does not exist: {input_path}"
            )
        declared_hash = row.get("sha256", "")
        if not _is_hex_digest(declared_hash, 64):
            raise EvaluationError(
                f"sample {sample_id!r} sha256 must be a 64-character digest"
            )
        try:
            actual_hash = sha256_file(input_path)
        except OSError as exc:
            raise EvaluationError(
                f"cannot hash sample {sample_id!r} at {input_path}: {exc}"
            ) from exc
        if declared_hash.lower() != actual_hash:
            raise EvaluationError(
                f"sample {sample_id!r} sha256 does not match input audio"
            )
        prepared.append(
            PreparedSample(
                row=dict(row),
                input_path=input_path,
                sha256=actual_hash,
            )
        )
    return prepared


def _validate_full_dataset(dataset_root: Path, manifest_path: Path) -> None:
    """Re-run the authoritative 42-WAV/33-row validator before gated runs."""

    try:
        from scripts.validate_dataset import validate_dataset
    except ModuleNotFoundError:  # pragma: no cover - direct script fallback
        from validate_dataset import validate_dataset  # type: ignore[no-redef]

    try:
        report = validate_dataset(dataset_root, manifest_path)
    except Exception as exc:
        raise EvaluationError(f"dataset validator could not complete: {exc}") from exc
    if report.errors:
        preview = "; ".join(
            f"{issue.code} {issue.path}: {issue.message}" for issue in report.errors[:5]
        )
        raise EvaluationError(
            f"dataset validation failed with {len(report.errors)} error(s): {preview}"
        )
    expected_counts = {
        "checked_wavs": EXPECTED_DATASET_WAVS,
        "expected_wavs": EXPECTED_DATASET_WAVS,
        "manifest_rows": EXPECTED_MANIFEST_ROWS,
        "hashes_verified": EXPECTED_MANIFEST_ROWS,
    }
    for name, expected in expected_counts.items():
        if getattr(report, name, None) != expected:
            raise EvaluationError(
                f"dataset validator coverage {name} must be {expected}, "
                f"got {getattr(report, name, None)!r}"
            )


def _verify_file_bytes(path: Path, expected: bytes, label: str) -> None:
    try:
        current = path.read_bytes()
    except OSError as exc:
        raise EvaluationError(f"cannot re-read {label} before execution: {exc}") from exc
    if current != expected:
        raise EvaluationError(f"{label} changed after validation; run refused")


def _stage_locked_inputs(
    samples: Sequence[PreparedSample], destination: Path
) -> list[PreparedSample]:
    """Copy verified locked bytes into this one-shot run before the receipt."""

    staging = destination / "_locked_inputs"
    try:
        staging.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        raise EvaluationError(f"cannot create locked input staging directory: {exc}") from exc
    staged: list[PreparedSample] = []
    try:
        for index, sample in enumerate(samples, start=1):
            staged_path = staging / f"{index:02d}_{sample.row['sample_id']}.wav"
            shutil.copyfile(sample.input_path, staged_path)
            actual_hash = sha256_file(staged_path)
            if actual_hash != sample.sha256:
                raise EvaluationError(
                    f"sample {sample.row['sample_id']!r} changed while staging"
                )
            staged_path.chmod(0o400)
            staged.append(
                PreparedSample(
                    row=sample.row,
                    input_path=staged_path,
                    sha256=actual_hash,
                )
            )
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return staged


def _ensure_private_output(dataset_root: Path, destination: Path) -> None:
    """Prevent reference text and locked outputs from landing in a tracked path."""

    project_root = Path(__file__).resolve().parents[1]
    try:
        destination.relative_to(project_root)
    except ValueError:
        return
    try:
        destination.relative_to(dataset_root)
    except ValueError as exc:
        raise EvaluationError(
            "evaluation output inside the repository must stay under the private "
            f"dataset root: {dataset_root}"
        ) from exc

    try:
        dataset_root.relative_to(project_root)
    except ValueError:
        return
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", str(dataset_root)],
        cwd=project_root,
        check=False,
        capture_output=True,
    )
    if ignored.returncode != 0:
        raise EvaluationError(
            f"dataset root is inside the repository but is not ignored by Git: {dataset_root}"
        )


def _validate_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not output_dir.is_dir():
            raise EvaluationError(f"output path exists and is not a directory: {output_dir}")
        if not overwrite:
            raise EvaluationError(
                f"output directory already exists: {output_dir}; use --overwrite explicitly"
            )


def locked_receipt_path(frozen_config: str | Path) -> Path:
    """Return the canonical receipt path independent of freeze filename/formatting."""

    freeze_path = Path(frozen_config).expanduser().resolve()
    try:
        payload = json.loads(freeze_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise TypeError("freeze root is not an object")
        manifest = payload["manifest"]
        config = payload["config"]
        git = payload["git"]
        validation = payload["dataset_validation"]
        if not all(
            isinstance(section, Mapping)
            for section in (manifest, config, git, validation)
        ):
            raise TypeError("freeze sections are not objects")
        identity_payload = {
            "freeze_schema_version": payload["freeze_schema_version"],
            "dataset_version": payload["dataset_version"],
            "contract_version": payload["contract_version"],
            "git_commit": str(git["commit"]).lower(),
            "config_sha256": str(config["sha256"]).lower(),
            "manifest_sha256": str(manifest["sha256"]).lower(),
            "locked_sample_ids": sorted(manifest["locked_sample_ids"]),
            "dataset_validation": {
                key: validation[key]
                for key in (
                    "validator_id",
                    "validator_version",
                    "checked_wavs",
                    "expected_wavs",
                    "manifest_rows",
                    "hashes_verified",
                )
            },
        }
        freeze_id = hashlib.sha256(
            json.dumps(
                identity_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        manifest_path = Path(str(manifest["path"])).expanduser().resolve()
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise EvaluationError(
            f"cannot derive canonical locked receipt from {freeze_path}: {exc}"
        ) from exc
    return manifest_path.parent / ".locked_receipts" / f"{freeze_id}.json"


def _write_started_receipt(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise EvaluationError(
            f"locked_test freeze record has already been consumed: {path}"
        ) from exc
    except OSError as exc:
        raise EvaluationError(f"cannot create locked_test receipt {path}: {exc}") from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        # Never unlink: once O_EXCL succeeds, this one-shot attempt is consumed.
        raise EvaluationError(f"cannot persist started receipt {path}: {exc}") from exc


def _json_clone(value: Any) -> Any:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise TypeError(f"pipeline result is not strict JSON data: {exc}") from exc
    return json.loads(encoded)


def _result_payload(result: Any) -> dict[str, Any]:
    if hasattr(result, "to_dict") and callable(result.to_dict):
        payload = result.to_dict()
    elif isinstance(result, Mapping):
        payload = dict(result)
    else:
        raise TypeError("process_audio must return ProcessResult or a mapping")
    if not isinstance(payload, Mapping):
        raise TypeError("process_audio.to_dict() must return a mapping")
    cloned = _json_clone(dict(payload))
    status = cloned.get("status")
    if status not in STATUSES:
        raise ValueError(f"pipeline result has invalid status: {status!r}")
    return cloned


def _finite_cer(payload: Mapping[str, Any], field_name: str) -> float | None:
    value = payload.get(field_name)
    if not isinstance(value, Mapping):
        return None
    cer = value.get("cer")
    if isinstance(cer, bool) or not isinstance(cer, (int, float)):
        return None
    numeric = float(cer)
    if not math.isfinite(numeric) or numeric < 0:
        return None
    return numeric


def _cer_direction(before: float | None, after: float | None) -> tuple[str, float | None]:
    if before is None or after is None:
        return "unavailable", None
    delta = before - after
    if delta > CER_TIE_TOLERANCE:
        return "improved", delta
    if delta < -CER_TIE_TOLERANCE:
        return "worsened", delta
    return "tied", delta


def _snr_group(value: str) -> str:
    raw = value.strip()
    if not raw:
        return "not_applicable"
    try:
        numeric = float(raw)
    except ValueError:
        return raw
    if not math.isfinite(numeric):
        return raw
    return f"{numeric:g}"


def _aggregate(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    statuses = {status: 0 for status in STATUSES}
    directions = {"improved": 0, "tied": 0, "worsened": 0, "unavailable": 0}
    before_values: list[float] = []
    after_values: list[float] = []

    for record in records:
        status = record.get("status")
        if status in statuses:
            statuses[str(status)] += 1
        direction = str(record.get("cer_direction", "unavailable"))
        directions[direction if direction in directions else "unavailable"] += 1
        before = record.get("cer_before")
        after = record.get("cer_after")
        if isinstance(before, (int, float)) and isinstance(after, (int, float)):
            before_values.append(float(before))
            after_values.append(float(after))

    return {
        "sample_count": len(records),
        "status_counts": statuses,
        "cer": {
            "paired_count": len(before_values),
            "improved": directions["improved"],
            "tied": directions["tied"],
            "worsened": directions["worsened"],
            "unavailable": directions["unavailable"],
            "before_median": statistics.median(before_values) if before_values else None,
            "after_median": statistics.median(after_values) if after_values else None,
        },
    }


def build_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    split: str,
    generated_at: str,
    manifest_identity: Mapping[str, str],
    frozen_config_identity: Mapping[str, str] | None,
    evaluation_index_sha256: str,
) -> dict[str, Any]:
    by_noise: dict[str, list[Mapping[str, Any]]] = {}
    by_snr: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        noise_key = str(record.get("noise_type") or "not_applicable")
        snr_key = _snr_group(str(record.get("snr_db") or ""))
        by_noise.setdefault(noise_key, []).append(record)
        by_snr.setdefault(snr_key, []).append(record)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "split": split,
        "manifest": dict(manifest_identity),
        "frozen_config": dict(frozen_config_identity) if frozen_config_identity else None,
        "evaluation_index_sha256": evaluation_index_sha256,
        **_aggregate(records),
        "by_noise_type": {
            key: _aggregate(by_noise[key]) for key in sorted(by_noise)
        },
        "by_snr_db": {key: _aggregate(by_snr[key]) for key in sorted(by_snr)},
    }
    return _json_clone(summary)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def run_evaluation(
    *,
    manifest_path: str | Path,
    dataset_root: str | Path,
    split: str,
    output_dir: str | Path,
    frozen_config: str | Path | None = None,
    confirm_locked: bool = False,
    overwrite: bool = False,
    strength: float = 0.75,
    enable_events: bool = False,
    force_recompute: bool = False,
    process_callable: Callable[..., Any] | None = None,
    runtime_identity: RuntimeIdentity | Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate one manifest split and persist index plus aggregate summary."""

    if split not in SPLITS:
        raise EvaluationError(f"split must be one of: {', '.join(SPLITS)}")
    if isinstance(strength, bool) or not isinstance(strength, (int, float)):
        raise EvaluationError("strength must be a finite number from 0 to 1")
    strength = float(strength)
    if not math.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise EvaluationError("strength must be a finite number from 0 to 1")
    if split == "locked_test" and not confirm_locked:
        raise EvaluationError(
            "locked_test requires explicit --confirm-locked after parameters are frozen"
        )
    if split == "locked_test" and frozen_config is None:
        raise EvaluationError("locked_test requires --frozen-config")
    if split == "clean_control" and frozen_config is None:
        raise EvaluationError("clean_control requires --frozen-config")
    if split == "locked_test" and overwrite:
        raise EvaluationError("locked_test is one-shot and forbids --overwrite")
    if split == "locked_test" and not force_recompute:
        raise EvaluationError("locked_test requires --force-recompute")
    if split == "locked_test" and enable_events:
        raise EvaluationError("locked_test freezes enable_events=false")
    if split == "locked_test" and (
        process_callable is not None or runtime_identity is not None
    ):
        raise EvaluationError(
            "locked_test formal API forbids injected process/runtime test hooks"
        )

    manifest = Path(manifest_path).expanduser().resolve()
    root = Path(dataset_root).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if not root.is_dir():
        raise EvaluationError(f"dataset root is not a directory: {root}")
    _ensure_private_output(root, destination)

    manifest_snapshot = _read_manifest_snapshot(manifest)
    rows = _load_split_rows(manifest_snapshot, split)
    manifest_sha256 = manifest_snapshot.sha256
    prepared_samples = _prepare_samples(rows, root, split)
    current_runtime: RuntimeIdentity | None = None
    if split in INTEGRITY_GATED_SPLITS:
        _validate_full_dataset(root, manifest)
        current_runtime = (
            _inspect_runtime_identity()
            if runtime_identity is None
            else _coerce_runtime_identity(runtime_identity)
        )
    frozen_identity = _validate_frozen_config(
        split,
        frozen_config,
        confirm_locked,
        manifest_sha256=manifest_sha256,
        manifest_path=manifest,
        dataset_root=root,
        requested_strength=strength,
        runtime_identity=current_runtime,
    )
    _validate_output_dir(destination, overwrite)

    receipt_path: Path | None = None
    if split == "locked_test":
        if not isinstance(frozen_identity, FrozenIdentity):  # pragma: no cover
            raise EvaluationError("locked_test freeze identity is incomplete")
        receipt_path = frozen_identity.receipt_path
        if receipt_path.exists():
            raise EvaluationError(
                f"locked_test freeze record has already been consumed: {receipt_path}"
            )

    # Delayed import keeps manifest/unit-test tooling usable without model deps.
    if process_callable is None:
        try:
            from core.pipeline import process_audio as process_callable
        except Exception as exc:
            raise EvaluationError(f"cannot import core.pipeline.process_audio: {exc}") from exc

    try:
        destination.mkdir(parents=True, exist_ok=overwrite)
    except OSError as exc:
        raise EvaluationError(f"cannot create output directory {destination}: {exc}") from exc

    config_snapshot_path: Path | None = None
    if split == "locked_test":
        try:
            prepared_samples = _stage_locked_inputs(prepared_samples, destination)
            _verify_file_bytes(manifest, manifest_snapshot.raw_bytes, "manifest")
            if not isinstance(frozen_identity, FrozenIdentity):  # pragma: no cover
                raise EvaluationError("locked_test freeze identity is incomplete")
            if sha256_file(frozen_identity.path) != frozen_identity.file_sha256:
                raise EvaluationError("freeze record changed after validation; run refused")
            refreshed_runtime = _inspect_runtime_identity()
            if refreshed_runtime != current_runtime:
                raise EvaluationError(
                    "Git/config runtime identity changed after validation; run refused"
                )
            config_snapshot_path = destination / "_frozen_app.yaml"
            shutil.copyfile(refreshed_runtime.config_path, config_snapshot_path)
            if sha256_file(config_snapshot_path) != refreshed_runtime.config_sha256:
                raise EvaluationError("pipeline config changed while staging; run refused")
            config_snapshot_path.chmod(0o400)
        except Exception:
            shutil.rmtree(destination, ignore_errors=True)
            raise

    if isinstance(frozen_identity, FrozenIdentity):
        frozen_public_identity: dict[str, str] | None = {
            "path": str(frozen_identity.path),
            "sha256": frozen_identity.file_sha256,
            "freeze_id": frozen_identity.freeze_id,
            "receipt_path": str(frozen_identity.receipt_path),
        }
    else:
        frozen_public_identity = frozen_identity

    generated_at = _utc_now()
    manifest_identity = {"path": str(manifest), "sha256": manifest_sha256}
    started_receipt: dict[str, Any] | None = None
    if receipt_path is not None:
        if current_runtime is None or frozen_public_identity is None:  # pragma: no cover
            raise EvaluationError("locked_test receipt identity is incomplete")
        started_receipt = {
            "receipt_schema_version": RECEIPT_SCHEMA_VERSION,
            "status": "started",
            "started_at": generated_at,
            "split": "locked_test",
            "freeze": {
                "path": frozen_public_identity["path"],
                "sha256": frozen_public_identity["sha256"],
                "freeze_id": frozen_public_identity["freeze_id"],
            },
            "manifest": manifest_identity,
            "runtime": {
                "git_commit": current_runtime.git_commit,
                "config_path": current_runtime.config_path,
                "config_sha256": current_runtime.config_sha256,
            },
            "output_dir": str(destination),
            "sample_ids": [sample.row["sample_id"] for sample in prepared_samples],
        }
        _write_started_receipt(receipt_path, started_receipt)

    records: list[dict[str, Any]] = []
    previous_config = os.environ.get("AUDIORESCUE_CONFIG")
    if config_snapshot_path is not None:
        os.environ["AUDIORESCUE_CONFIG"] = str(config_snapshot_path)
    try:
        for prepared in prepared_samples:
            row = prepared.row
            input_path = prepared.input_path
            sample_id = row["sample_id"]
            record: dict[str, Any] = {
                "sample_id": sample_id,
                "split": split,
                "source_type": row.get("source_type", ""),
                "noise_type": row.get("noise_type", ""),
                "snr_db": row.get("snr_db", ""),
                "reference_text": row.get("reference_text", ""),
                "input_path": str(input_path),
                "status": "failed",
                "cer_before": None,
                "cer_after": None,
                "cer_delta": None,
                "cer_direction": "unavailable",
                "result": None,
                "error": None,
            }
            try:
                if split == "locked_test" and sha256_file(input_path) != prepared.sha256:
                    raise EvaluationError(
                        f"staged locked sample {sample_id!r} changed before processing"
                    )
                reference_text = row["reference_text"]

                # This is the only reference-text handoff. Pipeline computes CER
                # after both ASR calls; no model prompt argument exists here.
                result = process_callable(
                    input_path=str(input_path),
                    strength=strength,
                    enable_events=enable_events,
                    reference_text=reference_text,
                    force_recompute=force_recompute,
                )
                if split == "locked_test" and sha256_file(input_path) != prepared.sha256:
                    raise EvaluationError(
                        f"staged locked sample {sample_id!r} changed during processing"
                    )
                payload = _result_payload(result)
                before = _finite_cer(payload, "cer_before")
                after = _finite_cer(payload, "cer_after")
                direction, delta = _cer_direction(before, after)
                record.update(
                    {
                        "status": payload["status"],
                        "cer_before": before,
                        "cer_after": after,
                        "cer_delta": delta,
                        "cer_direction": direction,
                        "result": payload,
                    }
                )
            except Exception as exc:
                record["status"] = "failed"
                record["error"] = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
            records.append(record)
    finally:
        if config_snapshot_path is not None:
            if previous_config is None:
                os.environ.pop("AUDIORESCUE_CONFIG", None)
            else:
                os.environ["AUDIORESCUE_CONFIG"] = previous_config

    evaluation_index = _json_clone(
        {
            "schema_version": SCHEMA_VERSION,
            "generated_at": generated_at,
            "split": split,
            "dataset_root": str(root),
            "manifest": manifest_identity,
            "frozen_config": frozen_public_identity,
            "options": {
                "strength": strength,
                "enable_events": enable_events,
                "force_recompute": force_recompute,
            },
            "samples": records,
        }
    )
    index_path = destination / "evaluation_index.json"
    _write_json_atomic(index_path, evaluation_index)
    summary = build_summary(
        records,
        split=split,
        generated_at=generated_at,
        manifest_identity=manifest_identity,
        frozen_config_identity=frozen_public_identity,
        evaluation_index_sha256=sha256_file(index_path),
    )
    summary_path = destination / "summary.json"
    _write_json_atomic(summary_path, summary)

    if receipt_path is not None and started_receipt is not None:
        completed_receipt = {
            **started_receipt,
            "status": "complete",
            "completed_at": _utc_now(),
            "evaluation_index": {
                "path": str(index_path),
                "sha256": sha256_file(index_path),
            },
            "summary": {
                "path": str(summary_path),
                "sha256": sha256_file(summary_path),
            },
            "status_counts": dict(summary["status_counts"]),
        }
        _write_json_atomic(receipt_path, completed_receipt)
    return evaluation_index, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one AudioRescue manifest split and write frozen evaluation JSON",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frozen-config", type=Path)
    parser.add_argument("--confirm-locked", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strength", type=float, default=0.75)
    parser.add_argument("--enable-events", action="store_true")
    parser.add_argument("--force-recompute", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset_root = args.dataset_root or args.manifest.expanduser().resolve().parent
    try:
        _, summary = run_evaluation(
            manifest_path=args.manifest,
            dataset_root=dataset_root,
            split=args.split,
            output_dir=args.output_dir,
            frozen_config=args.frozen_config,
            confirm_locked=args.confirm_locked,
            overwrite=args.overwrite,
            strength=args.strength,
            enable_events=args.enable_events,
            force_recompute=args.force_recompute,
        )
    except EvaluationError as exc:
        print(f"Evaluation refused: {exc}")
        return 2

    print(
        f"Evaluation complete: split={summary['split']} samples={summary['sample_count']} "
        f"success={summary['status_counts']['success']} "
        f"partial={summary['status_counts']['partial']} "
        f"failed={summary['status_counts']['failed']}"
    )
    return 1 if summary["status_counts"]["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
