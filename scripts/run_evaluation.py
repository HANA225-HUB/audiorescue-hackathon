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

from scripts.dataset_spec import (
    DatasetSpec,
    DatasetSpecError,
    load_dataset_spec,
    root_relative_label,
    safe_join,
    spec_content_hash,
)

STATUSES = ("success", "partial", "failed")
REQUIRED_MANIFEST_FIELDS = {
    "sample_id",
    "dataset_version",
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
INTEGRITY_GATED_SPLITS = {"locked_test", "clean_control"}
CER_TIE_TOLERANCE = 1e-12
SCHEMA_VERSION = "audiorescue-evaluation-v1"
FREEZE_SCHEMA_VERSION = "audiorescue-experiment-freeze-v2"
RECEIPT_SCHEMA_VERSION = "audiorescue-locked-test-receipt-v2"
LOCKED_DATASET_IDENTITY_VERSION = "audiorescue-locked-dataset-identity-v1"
DATASET_VALIDATOR_ID = "scripts.validate_dataset.validate_dataset"
DATASET_VALIDATOR_VERSION = "audiorescue-dataset-validator-v1"
LOCKED_RECEIPT_ROOT = (
    Path(__file__).resolve().parents[1] / "data_local" / ".locked_receipts"
)


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
    dataset_id: str
    dataset_identity: Mapping[str, Any]
    receipt_path: Path
    payload: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class LoadedDatasetSpec:
    spec: DatasetSpec
    sha256: str | None


def _load_evaluation_spec(value: str | Path | DatasetSpec | None) -> LoadedDatasetSpec:
    if isinstance(value, DatasetSpec):
        return LoadedDatasetSpec(value, None)
    if value is None:
        raise EvaluationError("explicit dataset spec is required")
    try:
        return LoadedDatasetSpec(load_dataset_spec(value), spec_content_hash(value))
    except (DatasetSpecError, OSError) as exc:
        raise EvaluationError("dataset spec is invalid") from exc


def _safe_manifest_path(root: Path, manifest_path: str | Path) -> Path:
    supplied = Path(manifest_path).expanduser()
    if supplied.is_absolute():
        resolved = supplied.resolve()
        try:
            relative = resolved.relative_to(root)
        except ValueError as exc:
            raise EvaluationError("manifest must stay inside dataset root") from exc
        try:
            return safe_join(root, relative, must_exist=True)
        except DatasetSpecError as exc:
            raise EvaluationError("manifest path is unsafe") from exc
    try:
        return safe_join(root, supplied, must_exist=True)
    except DatasetSpecError as exc:
        raise EvaluationError("manifest path is unsafe") from exc


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


def _manifest_dataset_version(snapshot: ManifestSnapshot) -> str:
    versions = {row.get("dataset_version", "").strip() for row in snapshot.rows}
    if len(versions) != 1 or "" in versions:
        raise EvaluationError(
            "manifest must contain exactly one non-empty dataset_version"
        )
    return next(iter(versions))


def _build_locked_dataset_identity(
    *,
    locked_samples: Sequence[tuple[str, str]],
    expected_ids: Sequence[str],
) -> tuple[str, dict[str, Any]]:
    """Build the one-shot identity from data, never code/config/freeze bytes."""

    normalized: dict[str, str] = {}
    for sample_id, audio_sha256 in locked_samples:
        if not isinstance(sample_id, str) or not sample_id:
            raise EvaluationError("locked dataset identity contains an empty sample_id")
        if sample_id in normalized:
            raise EvaluationError(
                f"locked dataset identity contains duplicate sample_id {sample_id!r}"
            )
        if not _is_hex_digest(audio_sha256, 64):
            raise EvaluationError(
                f"locked sample {sample_id!r} audio sha256 is invalid"
            )
        normalized[sample_id] = audio_sha256.lower()

    expected_id_set = set(expected_ids)
    if set(normalized) != expected_id_set:
        missing = sorted(expected_id_set - set(normalized))
        unexpected = sorted(set(normalized) - expected_id_set)
        raise EvaluationError(
            "locked dataset identity does not match the selected dataset spec; "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )

    canonical_samples = [
        {
            "sample_id": sample_id,
            "audio_sha256": normalized[sample_id],
        }
        for sample_id in sorted(normalized)
    ]
    identity = {
        "identity_schema_version": LOCKED_DATASET_IDENTITY_VERSION,
        "locked_samples": canonical_samples,
    }
    # The one-shot key intentionally excludes manifest formatting/hash,
    # dataset_version, references, Git, config, and freeze representation.
    dataset_id = hashlib.sha256(
        json.dumps(
            canonical_samples,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return dataset_id, identity


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
        raise EvaluationError("current pipeline config does not exist")
    try:
        config_sha256 = sha256_file(config_path)
    except OSError as exc:
        raise EvaluationError("cannot hash current pipeline config") from exc
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
    spec: DatasetSpec,
    manifest_sha256: str,
    manifest_path: Path,
    manifest_dataset_version: str,
    locked_samples: Sequence[tuple[str, str]],
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
        raise EvaluationError("frozen config does not exist")
    try:
        raw_bytes = path.read_bytes()
        parsed_config = json.loads(raw_bytes.decode("utf-8"))
        # Python accepts NaN/Infinity by default even though JSON does not.
        json.dumps(parsed_config, allow_nan=False)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise EvaluationError("frozen config is not valid JSON") from exc

    if split in INTEGRITY_GATED_SPLITS:
        if not isinstance(parsed_config, Mapping):
            raise EvaluationError("integrity-gated freeze record must be a JSON object")
        if parsed_config.get("freeze_schema_version") != FREEZE_SCHEMA_VERSION:
            raise EvaluationError(
                f"locked freeze_schema_version must be {FREEZE_SCHEMA_VERSION!r}"
            )
        frozen_dataset_version = parsed_config.get("dataset_version")
        if (
            not isinstance(frozen_dataset_version, str)
            or frozen_dataset_version != manifest_dataset_version
        ):
            raise EvaluationError(
                "locked freeze dataset_version does not match the manifest"
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
        expected_manifest_rows = sum(spec.split_counts().values())
        if frozen_manifest.get("row_count") != expected_manifest_rows:
            raise EvaluationError("locked freeze manifest.row_count does not match spec")
        split_counts = frozen_manifest.get("split_counts")
        if not isinstance(split_counts, Mapping) or dict(split_counts) != spec.split_counts():
            raise EvaluationError(
                "locked freeze split_counts does not match spec"
            )
        expected_locked_ids = sorted(spec.locked_sample_ids())
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
        try:
            from scripts.validate_dataset import expected_dataset_artifacts
        except ModuleNotFoundError:  # pragma: no cover - direct script fallback
            from validate_dataset import expected_dataset_artifacts  # type: ignore[no-redef]
        expected_wavs = len(expected_dataset_artifacts(spec))
        expected_validation = {
            "validator_id": DATASET_VALIDATOR_ID,
            "validator_version": DATASET_VALIDATOR_VERSION,
            "checked_wavs": expected_wavs,
            "expected_wavs": expected_wavs,
            "manifest_rows": expected_manifest_rows,
            "hashes_verified": expected_manifest_rows,
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

        dataset_id, dataset_identity = _build_locked_dataset_identity(
            locked_samples=locked_samples,
            expected_ids=expected_locked_ids,
        )
        receipt_path = LOCKED_RECEIPT_ROOT / f"{dataset_id}.json"
        return FrozenIdentity(
            path=path,
            file_sha256=hashlib.sha256(raw_bytes).hexdigest(),
            dataset_id=dataset_id,
            dataset_identity=dataset_identity,
            receipt_path=receipt_path,
            payload=parsed_config,
        )
    return {"role": "frozen_config", "sha256": hashlib.sha256(raw_bytes).hexdigest()}


def _resolve_in_root(dataset_root: Path, raw_path: str) -> Path:
    try:
        return safe_join(dataset_root, raw_path, must_exist=True)
    except DatasetSpecError as exc:
        raise ValueError("manifest audio path is unsafe") from exc


def _read_manifest_snapshot(manifest_path: Path) -> ManifestSnapshot:
    """Parse and hash one immutable byte snapshot of the manifest."""

    if not manifest_path.is_file():
        raise EvaluationError("manifest does not exist")
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
        raise EvaluationError("cannot read manifest") from exc
    if not rows:
        raise EvaluationError("manifest is empty")
    return ManifestSnapshot(
        path=manifest_path,
        raw_bytes=raw_bytes,
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
        rows=rows,
    )


def _canonical_primary_paths(split: str, spec: DatasetSpec) -> dict[str, str]:
    """Return the selected spec's sample-id to primary-path map for one split."""

    clean_by_id = spec.clean_by_id
    expected: dict[str, str] = {}
    for item in spec.mixes:
        if item.split == split:
            expected[item.sample_id] = item.path.as_posix()
    for item in spec.clean_controls:
        if item.split == split:
            expected[item.sample_id] = clean_by_id[item.clean_id].path.as_posix()
    for item in spec.real_recordings:
        if item.split == split:
            expected[item.sample_id] = item.path.as_posix()
    return expected


def _expected_locked_flags(spec: DatasetSpec) -> dict[str, bool]:
    flags: dict[str, bool] = {}
    for item in (*spec.mixes, *spec.clean_controls, *spec.real_recordings):
        flags[item.sample_id] = item.is_locked
    return flags


def _load_split_rows(
    snapshot: ManifestSnapshot,
    split: str,
    spec: DatasetSpec,
) -> list[dict[str, str]]:
    rows = [row for row in snapshot.rows if row.get("split") == split]
    if not rows:
        raise EvaluationError(f"manifest contains no rows for split {split!r}")
    expected_paths = _canonical_primary_paths(split, spec)
    if not expected_paths:
        raise EvaluationError("requested split is not declared by dataset spec")
    expected_count = len(expected_paths)
    if len(rows) != expected_count:
        raise EvaluationError(
            "manifest split row count does not match dataset spec"
        )

    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    for row in rows:
        sample_id = row.get("sample_id", "")
        if not sample_id:
            raise EvaluationError("manifest split contains an empty sample_id")
        if sample_id in seen_ids:
            raise EvaluationError("manifest split contains duplicate sample_id")
        seen_ids.add(sample_id)
        expected_path = expected_paths.get(sample_id)
        if expected_path is None:
            raise EvaluationError("manifest sample_id is not declared by dataset spec")
        actual_path = _select_audio_path(row).replace("\\", "/")
        while actual_path.startswith("./"):
            actual_path = actual_path[2:]
        if actual_path.startswith(snapshot.path.parent.name + "/"):
            actual_path = actual_path.split("/", 1)[1]
        if actual_path != expected_path:
            raise EvaluationError("manifest primary path does not match dataset spec")
        if actual_path in seen_paths:
            raise EvaluationError("manifest split reuses a primary audio path")
        seen_paths.add(actual_path)
    missing_ids = sorted(set(expected_paths) - seen_ids)
    if missing_ids:
        raise EvaluationError("manifest split is missing sample IDs declared by spec")
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
    raise EvaluationError("manifest is_locked value is invalid")


def _prepare_samples(
    rows: Sequence[dict[str, str]],
    dataset_root: Path,
    split: str,
    spec: DatasetSpec,
) -> list[PreparedSample]:
    """Resolve and hash every selected input before any pipeline call."""

    expected_locked_by_id = _expected_locked_flags(spec)
    prepared: list[PreparedSample] = []
    for row in rows:
        sample_id = row["sample_id"]
        actual_locked = _parse_locked_flag(
            row.get("is_locked", ""),
            sample_id=sample_id,
        )
        expected_locked = expected_locked_by_id.get(sample_id)
        if expected_locked is None or actual_locked is not expected_locked:
            raise EvaluationError(
                "manifest is_locked does not match dataset spec"
            )
        reference_text = row.get("reference_text", "")
        if not reference_text:
            raise EvaluationError("manifest reference_text is empty")
        try:
            input_path = _resolve_in_root(dataset_root, _select_audio_path(row))
        except ValueError as exc:
            raise EvaluationError("manifest input path is unsafe") from exc
        if not input_path.is_file():
            raise EvaluationError("manifest input audio does not exist")
        declared_hash = row.get("sha256", "")
        if not _is_hex_digest(declared_hash, 64):
            raise EvaluationError(
                "manifest sha256 must be a 64-character digest"
            )
        try:
            actual_hash = sha256_file(input_path)
        except OSError as exc:
            raise EvaluationError(
                "cannot hash manifest input audio"
            ) from exc
        if declared_hash.lower() != actual_hash:
            raise EvaluationError(
                "manifest sha256 does not match input audio"
            )
        prepared.append(
            PreparedSample(
                row=dict(row),
                input_path=input_path,
                sha256=actual_hash,
            )
        )
    return prepared


def _validate_full_dataset(
    dataset_root: Path,
    manifest_path: Path,
    spec: DatasetSpec,
) -> None:
    """Re-run the authoritative validator before evaluation."""

    try:
        from scripts.validate_dataset import validate_dataset
    except ModuleNotFoundError:  # pragma: no cover - direct script fallback
        from validate_dataset import validate_dataset  # type: ignore[no-redef]

    try:
        report = validate_dataset(dataset_root, manifest_path, spec=spec)
    except Exception as exc:
        raise EvaluationError("dataset validator could not complete") from exc
    if report.errors:
        preview = "; ".join(
            f"{issue.code} {issue.path}: {issue.message}" for issue in report.errors[:5]
        )
        raise EvaluationError(
            f"dataset validation failed with {len(report.errors)} error(s): {preview}"
        )
    expected_manifest_rows = sum(spec.split_counts().values())
    expected_counts = {
        "checked_wavs": report.expected_wavs,
        "expected_wavs": report.expected_wavs,
        "manifest_rows": expected_manifest_rows,
        "hashes_verified": expected_manifest_rows,
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
        raise EvaluationError("cannot create locked input staging directory") from exc
    staged: list[PreparedSample] = []
    try:
        for index, sample in enumerate(samples, start=1):
            staged_path = staging / f"{index:02d}.wav"
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
            "evaluation output inside the repository must stay under the private dataset root"
        ) from exc

    try:
        dataset_root.relative_to(project_root)
    except ValueError:
        return
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", "--", dataset_root.relative_to(project_root).as_posix()],
        cwd=project_root,
        check=False,
        capture_output=True,
    )
    if ignored.returncode != 0:
        raise EvaluationError(
            "dataset root inside the repository must be ignored by Git"
        )


def _validate_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not output_dir.is_dir():
            raise EvaluationError("output path exists and is not a directory")
        if not overwrite:
            raise EvaluationError(
                "output directory already exists; use --overwrite explicitly"
            )


def locked_receipt_path(
    frozen_config: str | Path,
    *,
    dataset_spec: str | Path | DatasetSpec | None = None,
) -> Path:
    """Derive the data-only receipt path from a freeze and its frozen manifest."""

    loaded_spec = _load_evaluation_spec(dataset_spec).spec
    freeze_path = Path(frozen_config).expanduser().resolve()
    try:
        payload = json.loads(freeze_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise TypeError("freeze root is not an object")
        manifest = payload["manifest"]
        if not isinstance(manifest, Mapping):
            raise TypeError("freeze sections are not objects")
        manifest_path = Path(str(manifest["path"])).expanduser().resolve()
        snapshot = _read_manifest_snapshot(manifest_path)
        manifest_sha256 = str(manifest["sha256"]).lower()
        if snapshot.sha256 != manifest_sha256:
            raise EvaluationError("freeze manifest sha256 does not match manifest bytes")
        manifest_dataset_version = _manifest_dataset_version(snapshot)
        if payload["dataset_version"] != manifest_dataset_version:
            raise EvaluationError("freeze dataset_version does not match manifest")
        locked_rows = _load_split_rows(snapshot, "locked_test", loaded_spec)
        dataset_id, _ = _build_locked_dataset_identity(
            locked_samples=[
                (row["sample_id"], row.get("sha256", "")) for row in locked_rows
            ],
            expected_ids=[
                sample_id
                for sample_id, path in _canonical_primary_paths("locked_test", loaded_spec).items()
            ],
        )
    except (
        EvaluationError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        raise EvaluationError(
            "cannot derive canonical locked receipt"
        ) from exc
    return LOCKED_RECEIPT_ROOT / f"{dataset_id}.json"


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
            "locked_test freeze record has already been consumed"
        ) from exc
    except OSError as exc:
        raise EvaluationError("cannot create locked_test receipt") from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        # Never unlink: once O_EXCL succeeds, this one-shot attempt is consumed.
        raise EvaluationError("cannot persist started receipt") from exc


def _require_completed_locked_receipt(identity: FrozenIdentity) -> None:
    path = identity.receipt_path
    if not path.is_file():
        raise EvaluationError(
            "clean_control requires the same dataset's completed locked_test receipt"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        json.dumps(payload, allow_nan=False)
    except (OSError, UnicodeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise EvaluationError("locked_test receipt is invalid") from exc
    if not isinstance(payload, Mapping):
        raise EvaluationError("locked_test receipt is not a JSON object")
    if payload.get("receipt_schema_version") != RECEIPT_SCHEMA_VERSION:
        raise EvaluationError(
            f"locked_test receipt schema must be {RECEIPT_SCHEMA_VERSION!r}"
        )
    locked_dataset = payload.get("locked_dataset")
    if not isinstance(locked_dataset, Mapping):
        raise EvaluationError("locked_test receipt has no locked_dataset identity")
    if locked_dataset.get("dataset_id") != identity.dataset_id:
        raise EvaluationError("locked_test receipt dataset identity does not match")
    if locked_dataset.get("identity_payload") != dict(identity.dataset_identity):
        raise EvaluationError("locked_test receipt identity payload does not match")
    if payload.get("split") != "locked_test":
        raise EvaluationError("locked_test receipt split is invalid")
    if payload.get("status") != "complete":
        raise EvaluationError(
            "clean_control requires a completed locked_test receipt; "
            f"found status {payload.get('status')!r}"
        )


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
    dataset_spec: str | Path | DatasetSpec | None = None,
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

    loaded_dataset_spec = _load_evaluation_spec(dataset_spec)
    spec = loaded_dataset_spec.spec
    if split not in spec.split_counts():
        raise EvaluationError("split must be declared by dataset spec")
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

    root = Path(dataset_root).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if not root.is_dir():
        raise EvaluationError("dataset root is not a directory")
    manifest = _safe_manifest_path(root, manifest_path)
    _ensure_private_output(root, destination)

    manifest_snapshot = _read_manifest_snapshot(manifest)
    manifest_dataset_version = _manifest_dataset_version(manifest_snapshot)
    _validate_full_dataset(root, manifest, spec)
    rows = _load_split_rows(manifest_snapshot, split, spec)
    manifest_sha256 = manifest_snapshot.sha256
    prepared_samples = _prepare_samples(rows, root, split, spec)
    current_runtime: RuntimeIdentity | None = None
    locked_identity_samples: list[tuple[str, str]] = []
    if split in INTEGRITY_GATED_SPLITS:
        locked_rows = _load_split_rows(manifest_snapshot, "locked_test", spec)
        locked_prepared = (
            prepared_samples
            if split == "locked_test"
            else _prepare_samples(locked_rows, root, "locked_test", spec)
        )
        locked_identity_samples = [
            (sample.row["sample_id"], sample.sha256) for sample in locked_prepared
        ]
        current_runtime = (
            _inspect_runtime_identity()
            if runtime_identity is None
            else _coerce_runtime_identity(runtime_identity)
        )
    frozen_identity = _validate_frozen_config(
        split,
        frozen_config,
        confirm_locked,
        spec=spec,
        manifest_sha256=manifest_sha256,
        manifest_path=manifest,
        manifest_dataset_version=manifest_dataset_version,
        locked_samples=locked_identity_samples,
        dataset_root=root,
        requested_strength=strength,
        runtime_identity=current_runtime,
    )
    _validate_output_dir(destination, overwrite)

    receipt_path: Path | None = None
    if split in INTEGRITY_GATED_SPLITS:
        if not isinstance(frozen_identity, FrozenIdentity):  # pragma: no cover
            raise EvaluationError("integrity-gated freeze identity is incomplete")
        if split == "locked_test":
            receipt_path = frozen_identity.receipt_path
        if split == "locked_test" and frozen_identity.receipt_path.exists():
            raise EvaluationError(
                "locked_test dataset has already been consumed"
            )
        if split == "clean_control":
            _require_completed_locked_receipt(frozen_identity)

    # Delayed import keeps manifest/unit-test tooling usable without model deps.
    if process_callable is None:
        try:
            from core.pipeline import process_audio as process_callable
        except Exception as exc:
            raise EvaluationError("cannot import core.pipeline.process_audio") from exc

    try:
        destination.mkdir(parents=True, exist_ok=overwrite)
    except OSError as exc:
        raise EvaluationError("cannot create output directory") from exc

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
            "role": "frozen_config",
            "sha256": frozen_identity.file_sha256,
            "locked_dataset_id": frozen_identity.dataset_id,
            "receipt_role": "locked_receipt",
        }
    else:
        frozen_public_identity = frozen_identity

    generated_at = _utc_now()
    manifest_identity = {
        "path": root_relative_label(root, manifest),
        "sha256": manifest_sha256,
        "dataset_version": manifest_dataset_version,
    }
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
                "role": frozen_public_identity["role"],
                "sha256": frozen_public_identity["sha256"],
            },
            "locked_dataset": {
                "dataset_id": frozen_identity.dataset_id,
                "identity_payload": dict(frozen_identity.dataset_identity),
                "audit": {
                    "dataset_version": manifest_dataset_version,
                    "manifest_sha256": manifest_sha256,
                    "reference_text_sha256": {
                        sample.row["sample_id"]: hashlib.sha256(
                            sample.row["reference_text"].encode("utf-8")
                        ).hexdigest()
                        for sample in sorted(
                            prepared_samples,
                            key=lambda item: item.row["sample_id"],
                        )
                    },
                },
            },
            "manifest": manifest_identity,
            "runtime": {
                "git_commit": current_runtime.git_commit,
                "config_role": "pipeline_config",
                "config_sha256": current_runtime.config_sha256,
            },
            "output_role": "evaluation_output",
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
                "input_path": root_relative_label(root, input_path),
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
            "dataset_root": "<dataset_root>",
            "dataset_spec": {
                "sha256": loaded_dataset_spec.sha256,
            },
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
    parser.add_argument("--dataset-spec", type=Path)
    parser.add_argument("--split", required=True)
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
            dataset_spec=args.dataset_spec,
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
        print("Evaluation refused: EVALUATION_INPUT_INVALID")
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
