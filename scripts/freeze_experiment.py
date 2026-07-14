"""Create an immutable record before the one-shot locked-test evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import subprocess
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

import yaml

try:
    from scripts.validate_dataset import validate_dataset
except ModuleNotFoundError:  # Support ``python scripts/freeze_experiment.py``.
    from validate_dataset import validate_dataset


FREEZE_SCHEMA_VERSION = "audiorescue-experiment-freeze-v2"
DATASET_VALIDATOR_ID = "scripts.validate_dataset.validate_dataset"
DATASET_VALIDATOR_VERSION = "audiorescue-dataset-validator-v1"
EXPECTED_SPLITS = {
    "dev": 18,
    "locked_test": 9,
    "clean_control": 3,
    "real": 3,
}
EXPECTED_DATASET_WAVS = 42
EXPECTED_MANIFEST_ROWS = sum(EXPECTED_SPLITS.values())
EXPECTED_HASHES_VERIFIED = EXPECTED_MANIFEST_ROWS
APPROVED_CONSENT_TOKEN = "team-approved-for-competition-evaluation"
_TRUE_VALUES = {"1", "true", "yes"}
_FALSE_VALUES = {"0", "false", "no"}


class FreezeError(RuntimeError):
    """Raised when the experiment is not ready to be frozen safely."""


@dataclass(frozen=True, slots=True)
class GitSnapshot:
    commit: str
    dirty_entries: tuple[str, ...]


class DatasetValidationReport(Protocol):
    """Narrow result contract required by the freeze gate."""

    checked_wavs: int
    expected_wavs: int
    manifest_rows: int
    hashes_verified: int

    @property
    def errors(self) -> Sequence[Any]: ...

    @property
    def warnings(self) -> Sequence[Any]: ...


DatasetValidator = Callable[[str | Path, str | Path | None], DatasetValidationReport]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_full_git_sha(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def _read_file_bytes(path: Path, label: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise FreezeError(f"cannot read {label}: {error}") from error


def _require_unchanged(path: Path, expected: bytes, label: str) -> None:
    current = _read_file_bytes(path, label)
    if current != expected:
        raise FreezeError(f"{label} changed during freeze; retry from a stable snapshot")


def inspect_git(project_root: str | Path) -> GitSnapshot:
    root = Path(project_root).expanduser().resolve()
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError) as error:
        raise FreezeError(f"cannot inspect Git state: {error}") from error
    if not _is_full_git_sha(commit):
        raise FreezeError("Git HEAD is not a full 40-character hexadecimal commit SHA")
    return GitSnapshot(commit=commit, dirty_entries=tuple(status))


def _load_config(path: Path, *, payload: bytes | None = None) -> dict[str, Any]:
    try:
        source = _read_file_bytes(path, "app config") if payload is None else payload
        parsed = yaml.safe_load(source.decode("utf-8-sig"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise FreezeError(f"cannot read app config: {error}") from error
    if not isinstance(parsed, dict):
        raise FreezeError("app config must be a mapping")
    for section in ("contract_version", "audio", "enhancement", "asr"):
        if section not in parsed:
            raise FreezeError(f"app config is missing {section}")

    asr = parsed["asr"]
    enhancement = parsed["enhancement"]
    if not isinstance(asr, dict) or not isinstance(enhancement, dict):
        raise FreezeError("asr and enhancement config sections must be mappings")
    if asr.get("initial_prompt") is not None:
        raise FreezeError("asr.initial_prompt must remain null for a fair evaluation")
    strength = enhancement.get("default_strength")
    if isinstance(strength, bool) or not isinstance(strength, (int, float)):
        raise FreezeError("enhancement.default_strength must be numeric")
    if not math.isfinite(float(strength)) or not 0.0 <= float(strength) <= 1.0:
        raise FreezeError("enhancement.default_strength must be finite and within 0..1")
    return parsed


def _load_manifest(
    path: Path,
    *,
    payload: bytes | None = None,
) -> tuple[list[dict[str, str]], str, Counter[str]]:
    try:
        source = _read_file_bytes(path, "manifest") if payload is None else payload
        text = source.decode("utf-8-sig")
        with io.StringIO(text, newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise FreezeError("manifest has no header")
            required = {
                "sample_id",
                "dataset_version",
                "split",
                "reference_text",
                "consent_or_license",
                "sha256",
                "is_locked",
            }
            if not required.issubset(reader.fieldnames):
                missing = ", ".join(sorted(required - set(reader.fieldnames)))
                raise FreezeError(f"manifest is missing columns: {missing}")
            rows = list(reader)
    except FreezeError:
        raise
    except (OSError, UnicodeError, csv.Error) as error:
        raise FreezeError(f"cannot read manifest: {error}") from error

    if not rows:
        raise FreezeError("manifest is empty")
    sample_ids = [row["sample_id"].strip() for row in rows]
    if any(not sample_id for sample_id in sample_ids):
        raise FreezeError("manifest contains an empty sample_id")
    if len(sample_ids) != len(set(sample_ids)):
        raise FreezeError("manifest contains duplicate sample_id values")

    versions = {row["dataset_version"].strip() for row in rows}
    if len(versions) != 1 or "" in versions:
        raise FreezeError("manifest must contain one non-empty dataset_version")
    split_counts = Counter(row["split"].strip() for row in rows)
    if dict(split_counts) != EXPECTED_SPLITS:
        raise FreezeError(
            f"manifest split counts must be {EXPECTED_SPLITS}, got {dict(split_counts)}"
        )
    for row in rows:
        split = row["split"].strip()
        locked = row["is_locked"].strip().lower()
        if locked not in _TRUE_VALUES | _FALSE_VALUES:
            raise FreezeError(
                f"invalid is_locked value for sample {row['sample_id']}: "
                f"{row['is_locked']!r}"
            )
        expected_locked = split in {"locked_test", "clean_control"}
        actual_locked = locked in _TRUE_VALUES
        if actual_locked is not expected_locked:
            raise FreezeError(
                f"{split} row is_locked must be {str(expected_locked).lower()}: "
                f"{row['sample_id']}"
            )
        consent = row["consent_or_license"].strip()
        if consent != APPROVED_CONSENT_TOKEN:
            raise FreezeError(
                "consent/license must use the exact formal token "
                f"{APPROVED_CONSENT_TOKEN!r}: {row['sample_id']}"
            )
        digest = row["sha256"].strip().lower()
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise FreezeError(f"invalid sha256 for sample: {row['sample_id']}")
    return rows, next(iter(versions)), split_counts


def _processing_config(config: dict[str, Any]) -> dict[str, Any]:
    selected = {
        "contract_version": config["contract_version"],
        "audio": config["audio"],
        "enhancement": config["enhancement"],
        "asr": config["asr"],
    }
    for optional in ("comparison", "events", "cache"):
        if optional in config:
            selected[optional] = config[optional]
    # Round-trip through strict JSON to reject YAML-only values or NaN/Infinity.
    try:
        return json.loads(
            json.dumps(selected, ensure_ascii=False, allow_nan=False, sort_keys=True)
        )
    except (TypeError, ValueError) as error:
        raise FreezeError(f"processing config is not strict JSON: {error}") from error


def _code_fingerprints(project_root: Path) -> dict[str, str]:
    candidates = list((project_root / "core").glob("*.py"))
    candidates.extend((project_root / "ui").glob("*.py") if (project_root / "ui").is_dir() else [])
    for relative in ("app.py", "configs/app.yaml", "configs/demo.yaml"):
        path = project_root / relative
        if path.is_file():
            candidates.append(path)
    return {
        path.resolve().relative_to(project_root).as_posix(): sha256_file(path)
        for path in sorted(set(candidates))
        if path.is_file()
    }


def _format_validation_issue(issue: Any) -> str:
    """Render real or injected validator issues without widening the interface."""

    code = getattr(issue, "code", None)
    path = getattr(issue, "path", None)
    message = getattr(issue, "message", None)
    if code is None and path is None and message is None:
        return str(issue)
    parts = [str(value) for value in (code, path, message) if value not in (None, "")]
    return " ".join(parts)


def _validate_dataset_for_freeze(
    *,
    dataset_root: Path,
    manifest_path: Path,
    dataset_validator: DatasetValidator,
) -> dict[str, Any]:
    """Run the mandatory semantic/audio gate and return JSON-safe evidence."""

    try:
        report = dataset_validator(dataset_root, manifest_path)
        errors = tuple(report.errors)
        warnings = tuple(report.warnings)
        counters = {
            "checked_wavs": report.checked_wavs,
            "expected_wavs": report.expected_wavs,
            "manifest_rows": report.manifest_rows,
            "hashes_verified": report.hashes_verified,
        }
    except Exception as error:
        raise FreezeError(f"dataset validator could not complete: {error}") from error

    for name, value in counters.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise FreezeError(
                f"dataset validator returned invalid {name}: {value!r}"
            )
    if errors:
        preview = "; ".join(_format_validation_issue(issue) for issue in errors[:5])
        if len(errors) > 5:
            preview += f"; and {len(errors) - 5} more"
        raise FreezeError(
            f"dataset validation reported {len(errors)} error(s); freeze refused: "
            f"{preview}"
        )

    expected_counters = {
        "checked_wavs": EXPECTED_DATASET_WAVS,
        "expected_wavs": EXPECTED_DATASET_WAVS,
        "manifest_rows": EXPECTED_MANIFEST_ROWS,
        "hashes_verified": EXPECTED_HASHES_VERIFIED,
    }
    mismatches = [
        f"{name} must be {expected}, got {counters[name]}"
        for name, expected in expected_counters.items()
        if counters[name] != expected
    ]
    if mismatches:
        raise FreezeError(
            "dataset validation coverage is incomplete: " + "; ".join(mismatches)
        )

    return {
        "validator_id": DATASET_VALIDATOR_ID,
        "validator_version": DATASET_VALIDATOR_VERSION,
        "dataset_root": str(dataset_root),
        **counters,
        "warning_count": len(warnings),
    }


def _write_exclusive(destination: Path, encoded: str) -> None:
    """Create one complete freeze path without ever replacing an existing file."""

    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise FreezeError(f"cannot create freeze directory: {error}") from error

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(destination, flags, 0o600)
    except FileExistsError as error:
        raise FreezeError(
            f"freeze record already exists; use a new versioned filename: {destination}"
        ) from error
    except OSError as error:
        raise FreezeError(f"cannot create freeze record: {error}") from error

    created = True
    open_descriptor: int | None = descriptor
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            open_descriptor = None
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as error:
        if open_descriptor is not None:
            os.close(open_descriptor)
        if created:
            try:
                destination.unlink(missing_ok=True)
            except OSError:
                pass
        raise FreezeError(f"cannot persist freeze record: {error}") from error


def freeze_experiment(
    *,
    config_path: str | Path,
    manifest_path: str | Path,
    output_path: str | Path,
    project_root: str | Path,
    dataset_root: str | Path | None = None,
    allow_dirty: bool = False,
    git_snapshot: GitSnapshot | None = None,
    now: Callable[[], datetime] | None = None,
    dataset_validator: DatasetValidator | None = None,
) -> dict[str, Any]:
    """Validate and atomically persist the immutable evaluation snapshot."""

    root = Path(project_root).expanduser().resolve()
    config_file = Path(config_path).expanduser().resolve()
    manifest_file = Path(manifest_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    dataset_directory = (
        Path(dataset_root).expanduser().resolve()
        if dataset_root is not None
        else manifest_file.parent
    )
    if destination.exists():
        raise FreezeError(
            f"freeze record already exists; use a new versioned filename: {destination}"
        )
    if not config_file.is_file() or not manifest_file.is_file():
        raise FreezeError("config and manifest must both exist")

    config_bytes = _read_file_bytes(config_file, "app config")
    manifest_bytes = _read_file_bytes(manifest_file, "manifest")
    config = _load_config(config_file, payload=config_bytes)
    rows, dataset_version, split_counts = _load_manifest(
        manifest_file,
        payload=manifest_bytes,
    )
    snapshot = git_snapshot or inspect_git(root)
    if not _is_full_git_sha(snapshot.commit):
        raise FreezeError("Git HEAD is not a full 40-character hexadecimal commit SHA")
    if snapshot.dirty_entries and not allow_dirty:
        preview = "; ".join(snapshot.dirty_entries[:5])
        raise FreezeError(f"Git worktree is dirty; commit or revert first: {preview}")
    validation_evidence = _validate_dataset_for_freeze(
        dataset_root=dataset_directory,
        manifest_path=manifest_file,
        dataset_validator=(
            validate_dataset if dataset_validator is None else dataset_validator
        ),
    )

    created = (now or (lambda: datetime.now(timezone.utc)))()
    if created.tzinfo is None:
        raise FreezeError("freeze timestamp must include a timezone")
    record = {
        "freeze_schema_version": FREEZE_SCHEMA_VERSION,
        "created_at_utc": created.astimezone(timezone.utc).isoformat(),
        "dataset_version": dataset_version,
        "contract_version": config["contract_version"],
        "git": {
            "commit": snapshot.commit,
            "dirty": bool(snapshot.dirty_entries),
            "dirty_entries": list(snapshot.dirty_entries),
        },
        "config": {
            "path": str(config_file),
            "sha256": sha256_bytes(config_bytes),
            "processing": _processing_config(config),
        },
        "manifest": {
            "path": str(manifest_file),
            "sha256": sha256_bytes(manifest_bytes),
            "row_count": len(rows),
            "split_counts": dict(split_counts),
            "locked_sample_ids": [
                row["sample_id"].strip()
                for row in rows
                if row["split"].strip() == "locked_test"
            ],
        },
        "dataset_validation": validation_evidence,
        "code_sha256": _code_fingerprints(root),
        "rules": {
            "reference_text_not_used_as_asr_prompt": True,
            "locked_test_is_one_shot": True,
            "config_changes_require_new_freeze_record": True,
        },
    }
    encoded = json.dumps(
        record,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    _require_unchanged(config_file, config_bytes, "app config")
    _require_unchanged(manifest_file, manifest_bytes, "manifest")
    _write_exclusive(destination, encoded)
    return record


def _parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Freeze config, dataset manifest, code hashes, and Git commit."
    )
    parser.add_argument("--config", default=project_root / "configs/app.yaml")
    parser.add_argument("--manifest", default=project_root / "data_local/manifest.csv")
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="dataset root; defaults to the selected manifest's parent directory",
    )
    parser.add_argument(
        "--output", default=project_root / "data_local/config_frozen.json"
    )
    parser.add_argument("--project-root", default=project_root)
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Development only: record dirty entries instead of refusing to freeze.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        record = freeze_experiment(
            config_path=args.config,
            manifest_path=args.manifest,
            output_path=args.output,
            project_root=args.project_root,
            dataset_root=args.dataset_root,
            allow_dirty=args.allow_dirty,
        )
    except FreezeError as error:
        print(f"Freeze refused: {error}")
        return 1
    print(f"Experiment frozen at Git commit {record['git']['commit']}")
    print(f"Manifest rows: {record['manifest']['row_count']}")
    print(f"Freeze record: {Path(args.output).expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
