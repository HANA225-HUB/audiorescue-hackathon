#!/usr/bin/env python3
"""Validate a generic AudioRescue recording dataset and manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import re
import sys
import unicodedata
import wave
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from scripts.dataset_spec import (
    DEFAULT_APPROVED_CONSENT_TOKEN,
    DEFAULT_PENDING_CONSENT_TOKEN,
    DatasetSpec,
    evaluation_compat_spec,
    load_dataset_spec,
)
from scripts.build_dataset import MANIFEST_FIELDS


DATASET_VERSION = "audiorescue-synthetic-template-v1"
PENDING_CONSENT_TOKEN = DEFAULT_PENDING_CONSENT_TOKEN
APPROVED_CONSENT_TOKEN = DEFAULT_APPROVED_CONSENT_TOKEN

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_INTEGER_RE = re.compile(r"^[+-]?\d+$")
_TRUE_VALUES = {"1", "true", "yes"}
_FALSE_VALUES = {"0", "false", "no"}


@dataclass(frozen=True, slots=True)
class ValidationPolicy:
    sample_rate: int = 48_000
    channels: int = 1
    sample_width_bytes: int = 2
    audio_min_seconds: float = 1.0
    audio_max_seconds: float = 60.0
    noise_min_seconds: float = 1.0
    noise_max_seconds: float = 60.0
    near_silence_dbfs: float = -60.0
    mixed_peak_limit: float = 0.981
    clean_recommended_peak: float = 10 ** (-3.0 / 20.0)


DEFAULT_POLICY = ValidationPolicy()


@dataclass(frozen=True, slots=True)
class ArtifactSpec:
    relative_path: Path
    kind: str
    speaker: str | None = None
    sentence: str | None = None
    noise: str | None = None
    snr_code: str | None = None
    snr_db: float | None = None
    split: str | None = None
    sample_id: str | None = None


@dataclass(frozen=True, slots=True)
class WavInfo:
    sample_rate: int
    channels: int
    sample_width_bytes: int
    frames: int
    duration_seconds: float
    peak_abs: float | None
    rms_dbfs: float | None
    clipped_samples: int


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    severity: str
    code: str
    path: str
    message: str


@dataclass(slots=True)
class ValidationReport:
    root: Path
    expected_wavs: int = 0
    checked_wavs: int = 0
    manifest_rows: int = 0
    hashes_verified: int = 0
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def add(self, severity: str, code: str, path: str | Path, message: str) -> None:
        self.issues.append(ValidationIssue(severity, code, str(path), message))

    def render(self) -> str:
        state = "PASS" if self.ok else "FAIL"
        lines = [
            f"AudioRescue dataset validation: {state}",
            f"Root: {self.root}",
            f"Required WAVs checked: {self.checked_wavs}/{self.expected_wavs}",
            f"Manifest rows: {self.manifest_rows}",
            f"SHA-256 values verified: {self.hashes_verified}",
            f"Errors: {len(self.errors)}; Warnings: {len(self.warnings)}",
        ]
        for issue in self.issues:
            lines.append(
                f"[{issue.severity.upper()}] {issue.code} {issue.path}: {issue.message}"
            )
        return "\n".join(lines)


def normalize_reference(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).lower()
    return "".join(character for character in normalized if character.isalnum())


def _coerce_spec(spec: DatasetSpec | str | Path | None) -> DatasetSpec:
    if isinstance(spec, DatasetSpec):
        return spec
    if spec is None:
        return evaluation_compat_spec()
    return load_dataset_spec(spec)


def expected_dataset_artifacts(
    spec: DatasetSpec | str | Path | None = None,
) -> dict[Path, ArtifactSpec]:
    loaded = _coerce_spec(spec)
    clean_by_id = loaded.clean_by_id
    noise_by_id = loaded.noise_by_id
    artifacts: dict[Path, ArtifactSpec] = {}
    for item in loaded.clean_recordings:
        artifacts[item.path] = ArtifactSpec(
            item.path,
            "clean",
            speaker=item.speaker_id,
            sentence=item.sentence_id,
            split=item.split,
        )
    for item in loaded.noise_recordings:
        artifacts[item.path] = ArtifactSpec(item.path, "noise", noise=item.noise_type)
    for item in loaded.mixes:
        clean = clean_by_id[item.clean_id]
        noise = noise_by_id[item.noise_id]
        artifacts[item.path] = ArtifactSpec(
            item.path,
            "mix",
            speaker=clean.speaker_id,
            sentence=clean.sentence_id,
            noise=noise.noise_type,
            snr_code=item.snr_label,
            snr_db=item.snr_db,
            split=item.split,
            sample_id=item.sample_id,
        )
    for item in loaded.real_recordings:
        artifacts[item.path] = ArtifactSpec(
            item.path,
            "real",
            speaker=item.speaker_id,
            sentence=item.sentence_id,
            noise=item.noise_type,
            split=item.split,
            sample_id=item.sample_id,
        )
    return artifacts


def required_manifest_primary_paths(
    spec: DatasetSpec | str | Path | None = None,
) -> set[Path]:
    return _coerce_spec(spec).manifest_primary_paths()


@dataclass(frozen=True, slots=True)
class _ExpectedRow:
    sample_id: str
    primary_path: Path
    source_type: str
    split: str
    speaker_id: str
    sentence_id: str
    reference_text: str
    noise_type: str
    is_locked: bool
    is_demo_candidate: bool
    snr_db: float | None = None


def _expected_manifest_rows(spec: DatasetSpec) -> dict[str, _ExpectedRow]:
    clean_by_id = spec.clean_by_id
    noise_by_id = spec.noise_by_id
    expected: dict[str, _ExpectedRow] = {}
    for item in spec.mixes:
        clean = clean_by_id[item.clean_id]
        noise = noise_by_id[item.noise_id]
        expected[item.sample_id] = _ExpectedRow(
            sample_id=item.sample_id,
            primary_path=item.path,
            source_type="controlled_mix",
            split=item.split,
            speaker_id=clean.speaker_id,
            sentence_id=clean.sentence_id,
            reference_text=clean.reference_text,
            noise_type=noise.noise_type,
            is_locked=item.is_locked,
            is_demo_candidate=item.is_demo_candidate,
            snr_db=item.snr_db,
        )
    for item in spec.clean_controls:
        clean = clean_by_id[item.clean_id]
        expected[item.sample_id] = _ExpectedRow(
            sample_id=item.sample_id,
            primary_path=clean.path,
            source_type="clean_control",
            split=item.split,
            speaker_id=clean.speaker_id,
            sentence_id=clean.sentence_id,
            reference_text=clean.reference_text,
            noise_type="",
            is_locked=item.is_locked,
            is_demo_candidate=item.is_demo_candidate,
        )
    for item in spec.real_recordings:
        expected[item.sample_id] = _ExpectedRow(
            sample_id=item.sample_id,
            primary_path=item.path,
            source_type="real",
            split=item.split,
            speaker_id=item.speaker_id,
            sentence_id=item.sentence_id,
            reference_text=item.reference_text,
            noise_type=item.noise_type,
            is_locked=item.is_locked,
            is_demo_candidate=item.is_demo_candidate,
        )
    return expected


def _parse_manifest_bool(value: str) -> bool | None:
    normalized = value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_wav(
    path: Path,
    spec: ArtifactSpec,
    policy: ValidationPolicy = DEFAULT_POLICY,
) -> tuple[WavInfo | None, list[ValidationIssue]]:
    issues: list[ValidationIssue] = []

    def error(code: str, message: str) -> None:
        issues.append(ValidationIssue("error", code, str(path), message))

    def warning(code: str, message: str) -> None:
        issues.append(ValidationIssue("warning", code, str(path), message))

    try:
        with wave.open(str(path), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            frame_count = wav_file.getnframes()
            compression = wav_file.getcomptype()

            if compression != "NONE":
                error("WAV_COMPRESSION", f"expected uncompressed PCM, got {compression}")
            if sample_rate != policy.sample_rate:
                error("WAV_SAMPLE_RATE", f"expected {policy.sample_rate} Hz, got {sample_rate} Hz")
            if channels != policy.channels:
                error("WAV_CHANNELS", f"expected {policy.channels} channel, got {channels}")
            if sample_width != policy.sample_width_bytes:
                error(
                    "WAV_BIT_DEPTH",
                    f"expected {policy.sample_width_bytes * 8}-bit PCM, got {sample_width * 8}-bit",
                )

            duration = frame_count / sample_rate if sample_rate > 0 else 0.0
            minimum = policy.noise_min_seconds if spec.kind == "noise" else policy.audio_min_seconds
            maximum = policy.noise_max_seconds if spec.kind == "noise" else policy.audio_max_seconds
            if duration < minimum or duration > maximum:
                error(
                    "WAV_DURATION",
                    f"expected {minimum:g}..{maximum:g} seconds for {spec.kind}, got {duration:.3f}",
                )

            peak_abs: float | None = None
            rms_dbfs: float | None = None
            clipped_samples = 0
            sample_count = 0
            square_sum = 0
            peak_sample = 0

            if sample_width == 2 and compression == "NONE":
                while True:
                    raw_frames = wav_file.readframes(16_384)
                    if not raw_frames:
                        break
                    if len(raw_frames) % 2:
                        error("WAV_TRUNCATED", "PCM16 payload has an odd byte count")
                        break
                    samples = array("h")
                    samples.frombytes(raw_frames)
                    if sys.byteorder != "little":
                        samples.byteswap()
                    sample_count += len(samples)
                    for sample in samples:
                        magnitude = abs(sample)
                        if magnitude > peak_sample:
                            peak_sample = magnitude
                        square_sum += sample * sample
                        if sample <= -32_768 or sample >= 32_767:
                            clipped_samples += 1

                expected_samples = frame_count * channels
                if sample_count != expected_samples:
                    error(
                        "WAV_TRUNCATED",
                        f"header declares {expected_samples} samples but read {sample_count}",
                    )

                if sample_count == 0 or peak_sample == 0:
                    error("WAV_SILENT", "audio is digital silence")
                    peak_abs = 0.0
                else:
                    peak_abs = peak_sample / 32_768.0
                    rms = math.sqrt(square_sum / sample_count) / 32_768.0
                    rms_dbfs = 20.0 * math.log10(rms)
                    if rms_dbfs < policy.near_silence_dbfs:
                        error(
                            "WAV_NEAR_SILENT",
                            f"RMS {rms_dbfs:.2f} dBFS is below {policy.near_silence_dbfs:.2f} dBFS",
                        )

                if clipped_samples:
                    error("WAV_CLIPPED", f"contains {clipped_samples} full-scale sample(s)")
                if spec.kind == "mix" and peak_abs is not None and peak_abs > policy.mixed_peak_limit:
                    error(
                        "WAV_MIX_PEAK",
                        f"mixed peak {peak_abs:.6f} exceeds protected limit {policy.mixed_peak_limit:.6f}",
                    )
                if (
                    spec.kind == "clean"
                    and peak_abs is not None
                    and peak_abs > policy.clean_recommended_peak
                    and not clipped_samples
                ):
                    warning(
                        "WAV_CLEAN_PEAK_HIGH",
                        f"clean peak {peak_abs:.6f} is above the recommended -3 dBFS ceiling",
                    )

            info = WavInfo(
                sample_rate=sample_rate,
                channels=channels,
                sample_width_bytes=sample_width,
                frames=frame_count,
                duration_seconds=duration,
                peak_abs=peak_abs,
                rms_dbfs=rms_dbfs,
                clipped_samples=clipped_samples,
            )
            return info, issues
    except (EOFError, OSError, wave.Error) as exc:
        error("WAV_UNREADABLE", f"cannot read WAV: {exc}")
        return None, issues


def _manifest_path(root: Path, raw_path: str) -> tuple[Path, Path] | None:
    value = raw_path.strip()
    if not value:
        return None
    supplied = Path(value.replace("\\", "/"))
    if supplied.is_absolute():
        return None
    if any(part in {"", ".", ".."} for part in supplied.parts):
        return None
    candidate = (root / supplied).resolve()
    try:
        relative = candidate.relative_to(root.resolve())
    except ValueError:
        return None
    return relative, candidate


def _primary_path_field(row: Mapping[str, str]) -> str:
    source_type = (row.get("source_type") or "").strip().lower()
    if source_type == "clean_control":
        return row.get("clean_path", "")
    return (row.get("mixed_path") or row.get("clean_path") or "").strip()


def validate_manifest(
    root: Path,
    manifest_path: Path,
    wav_infos: dict[Path, WavInfo],
    required_primary_paths: Iterable[Path] | None = None,
    *,
    spec: DatasetSpec | str | Path | None = None,
) -> tuple[list[ValidationIssue], int, int]:
    """Validate manifest fields against the selected generic spec."""

    loaded = _coerce_spec(spec)
    expected_rows = _expected_manifest_rows(loaded)
    required_paths = set(required_primary_paths or required_manifest_primary_paths(loaded))
    issues: list[ValidationIssue] = []
    rows_count = 0
    hashes_verified = 0

    def add(code: str, path: str | Path, message: str) -> None:
        issues.append(ValidationIssue("error", code, str(path), message))

    if not manifest_path.is_file():
        add("MANIFEST_MISSING", manifest_path, "manifest.csv does not exist")
        return issues, rows_count, hashes_verified

    try:
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            headers = reader.fieldnames or []
            missing_headers = [field for field in MANIFEST_FIELDS if field not in headers]
            unexpected_headers = [field for field in headers if field not in MANIFEST_FIELDS]
            if missing_headers:
                add("MANIFEST_COLUMNS", manifest_path, "missing required manifest columns")
            if unexpected_headers:
                add("MANIFEST_COLUMNS", manifest_path, "unexpected manifest columns")
            if len(headers) != len(set(headers)):
                add("MANIFEST_COLUMNS", manifest_path, "contains duplicate column names")

            seen_ids: dict[str, int] = {}
            seen_primary_paths: dict[Path, int] = {}
            hash_cache: dict[Path, str] = {}

            for line_number, raw_row in enumerate(reader, start=2):
                rows_count += 1
                row = {
                    str(key): (value or "").strip()
                    for key, value in raw_row.items()
                    if key is not None
                }
                row_label = f"{manifest_path}:{line_number}"
                if None in raw_row:
                    add("MANIFEST_COLUMNS", row_label, "row has more values than the header")

                sample_id = row.get("sample_id", "")
                expected = expected_rows.get(sample_id)
                if not sample_id:
                    add("MANIFEST_REQUIRED", row_label, "sample_id is empty")
                elif sample_id in seen_ids:
                    add("MANIFEST_DUPLICATE_ID", row_label, "sample_id is duplicated")
                else:
                    seen_ids[sample_id] = line_number

                if row.get("dataset_version") != loaded.dataset_version:
                    add("MANIFEST_VERSION", row_label, "dataset_version does not match spec")
                if row.get("consent_or_license") != loaded.consent_tokens.approved:
                    add("MANIFEST_CONSENT", row_label, "consent_or_license is not approved")

                resolved_fields: dict[str, tuple[Path, Path]] = {}
                for field_name in ("clean_path", "noise_path", "mixed_path"):
                    raw_value = row.get(field_name, "")
                    if not raw_value:
                        continue
                    resolved = _manifest_path(root, raw_value)
                    if resolved is None:
                        add("MANIFEST_PATH", row_label, f"{field_name} must stay inside dataset root")
                        continue
                    relative, absolute = resolved
                    resolved_fields[field_name] = resolved
                    if absolute.is_symlink():
                        add("MANIFEST_SYMLINK", row_label, f"{field_name} must not be a symlink")
                    if not absolute.is_file():
                        add("MANIFEST_PATH", row_label, f"{field_name} does not exist")

                primary_raw = _primary_path_field(row)
                primary_resolved = _manifest_path(root, primary_raw) if primary_raw else None
                primary_relative: Path | None = None
                primary_absolute: Path | None = None
                if primary_resolved is None:
                    add("MANIFEST_PRIMARY_PATH", row_label, "cannot determine a primary audio path")
                else:
                    primary_relative, primary_absolute = primary_resolved
                    if primary_relative in seen_primary_paths:
                        add("MANIFEST_DUPLICATE_PATH", row_label, "primary path is duplicated")
                    else:
                        seen_primary_paths[primary_relative] = line_number

                if expected is None:
                    add("MANIFEST_SAMPLE_ID", row_label, "sample_id is not declared by spec")
                else:
                    if primary_relative != expected.primary_path:
                        add("MANIFEST_PRIMARY_PATH", row_label, "primary path does not match spec")
                    expected_values = {
                        "source_type": expected.source_type,
                        "split": expected.split,
                        "speaker_id": expected.speaker_id,
                        "sentence_id": expected.sentence_id,
                        "noise_type": expected.noise_type,
                    }
                    for field_name, expected_value in expected_values.items():
                        code = "MANIFEST_SPLIT" if field_name == "split" else "MANIFEST_SEMANTICS"
                        if row.get(field_name, "") != expected_value:
                            add(code, row_label, f"{field_name} does not match spec")
                    if row.get("reference_text", "") != expected.reference_text:
                        add("MANIFEST_REFERENCE", row_label, "reference_text does not match spec")
                    expected_normalized = normalize_reference(expected.reference_text)
                    if row.get("reference_normalized", "") != expected_normalized:
                        add("MANIFEST_REFERENCE", row_label, "reference_normalized does not match spec")

                    declared_locked = _parse_manifest_bool(row.get("is_locked", ""))
                    if declared_locked is None:
                        add("MANIFEST_BOOLEAN", row_label, "is_locked must be boolean-like")
                    elif declared_locked is not expected.is_locked:
                        add("MANIFEST_LOCKED", row_label, "is_locked does not match spec")
                    declared_demo = _parse_manifest_bool(row.get("is_demo_candidate", ""))
                    if declared_demo is None:
                        add("MANIFEST_BOOLEAN", row_label, "is_demo_candidate must be boolean-like")
                    elif declared_demo is not expected.is_demo_candidate:
                        add("MANIFEST_DEMO_CANDIDATE", row_label, "is_demo_candidate does not match spec")

                declared_hash = row.get("sha256", "")
                if not _SHA256_RE.fullmatch(declared_hash):
                    add("MANIFEST_HASH", row_label, "sha256 must contain 64 hexadecimal characters")
                elif primary_absolute is not None and primary_absolute.is_file():
                    actual_hash = hash_cache.get(primary_absolute)
                    if actual_hash is None:
                        actual_hash = sha256_file(primary_absolute)
                        hash_cache[primary_absolute] = actual_hash
                    if actual_hash != declared_hash.lower():
                        add("MANIFEST_HASH", row_label, "sha256 does not match primary audio")
                    else:
                        hashes_verified += 1

                info = wav_infos.get(primary_relative) if primary_relative is not None else None
                if info is not None:
                    try:
                        declared_rate = int(row.get("sample_rate", ""))
                        declared_channels = int(row.get("channels", ""))
                        declared_duration = float(row.get("duration_seconds", ""))
                    except ValueError:
                        add("MANIFEST_AUDIO_META", row_label, "audio metadata must be numeric")
                    else:
                        if declared_rate != info.sample_rate or declared_channels != info.channels:
                            add("MANIFEST_AUDIO_META", row_label, "audio metadata does not match WAV")
                        tolerance = max(0.001, 1.0 / max(info.sample_rate, 1))
                        if not math.isfinite(declared_duration) or abs(
                            declared_duration - info.duration_seconds
                        ) > tolerance:
                            add("MANIFEST_AUDIO_META", row_label, "duration_seconds does not match WAV")

                distance = row.get("recording_distance_cm", "")
                if distance:
                    try:
                        numeric_distance = float(distance)
                    except ValueError:
                        numeric_distance = math.nan
                    if not math.isfinite(numeric_distance) or numeric_distance < 0:
                        add("MANIFEST_AUDIO_META", row_label, "recording_distance_cm is invalid")

                if expected is None or expected.snr_db is None:
                    for field_name in ("snr_db", "mix_seed", "noise_offset_seconds", "mix_alpha", "final_gain"):
                        if row.get(field_name, ""):
                            add("MANIFEST_MIX_META", row_label, f"{field_name} must be empty")
                    continue

                try:
                    declared_snr = float(row.get("snr_db", ""))
                except ValueError:
                    declared_snr = math.nan
                if not math.isfinite(declared_snr) or declared_snr != expected.snr_db:
                    add("MANIFEST_SNR", row_label, "snr_db does not match spec")
                seed_raw = row.get("mix_seed", "")
                if not _INTEGER_RE.fullmatch(seed_raw):
                    add("MANIFEST_MIX_META", row_label, "mix_seed must be an integer")
                for field_name in ("noise_offset_seconds", "mix_alpha", "final_gain"):
                    try:
                        value = float(row.get(field_name, ""))
                    except ValueError:
                        value = math.nan
                    if not math.isfinite(value):
                        add("MANIFEST_MIX_META", row_label, f"{field_name} must be finite")
                    elif field_name != "noise_offset_seconds" and value <= 0:
                        add("MANIFEST_MIX_META", row_label, f"{field_name} must be positive")
                    elif field_name == "final_gain" and value > 1:
                        add("MANIFEST_MIX_META", row_label, "final_gain must be at most one")
                    elif field_name == "noise_offset_seconds" and value < 0:
                        add("MANIFEST_MIX_META", row_label, "noise_offset_seconds must be non-negative")

            expected_row_count = len(expected_rows)
            if rows_count != expected_row_count:
                add("MANIFEST_ROW_COUNT", manifest_path, "manifest row count does not match spec")
            actual_primary_paths = set(seen_primary_paths)
            for relative in sorted(required_paths - actual_primary_paths, key=lambda item: item.as_posix()):
                add("MANIFEST_COVERAGE", manifest_path, "manifest is missing a required primary path")
            for relative in sorted(actual_primary_paths - required_paths, key=lambda item: item.as_posix()):
                add("MANIFEST_COVERAGE", manifest_path, "manifest contains an unexpected primary path")
    except (csv.Error, OSError, UnicodeError) as exc:
        add("MANIFEST_UNREADABLE", manifest_path, f"cannot read manifest: {exc}")

    return issues, rows_count, hashes_verified


def validate_dataset(
    root: str | Path,
    manifest_path: str | Path | None = None,
    policy: ValidationPolicy = DEFAULT_POLICY,
    *,
    spec_path: str | Path | None = None,
    spec: DatasetSpec | None = None,
) -> ValidationReport:
    dataset_root = Path(root).expanduser().resolve()
    loaded = spec or load_dataset_spec(spec_path)
    report = ValidationReport(root=dataset_root)
    artifacts = expected_dataset_artifacts(loaded)
    report.expected_wavs = len(artifacts)

    if not dataset_root.is_dir():
        report.add("error", "DATASET_ROOT", dataset_root, "dataset root is not a directory")
        return report

    wav_infos: dict[Path, WavInfo] = {}
    for relative, artifact in artifacts.items():
        absolute = dataset_root / relative
        if not absolute.is_file():
            report.add("error", "MISSING_WAV", relative, "required WAV is missing")
            continue
        report.checked_wavs += 1
        info, issues = inspect_wav(absolute, artifact, policy)
        report.issues.extend(issues)
        if info is not None:
            wav_infos[relative] = info

    expected_paths = set(artifacts)
    for directory_name in ("raw", "controlled"):
        directory = dataset_root / directory_name
        if not directory.is_dir():
            continue
        for absolute in directory.rglob("*.wav"):
            relative = absolute.relative_to(dataset_root)
            if relative not in expected_paths:
                report.add(
                    "error",
                    "INVALID_WAV_NAME_OR_LOCATION",
                    relative,
                    "WAV is not declared by the selected spec",
                )

    selected_manifest = (
        Path(manifest_path).expanduser().resolve()
        if manifest_path is not None
        else dataset_root / "manifest.csv"
    )
    manifest_issues, row_count, hash_count = validate_manifest(
        dataset_root,
        selected_manifest,
        wav_infos,
        spec=loaded,
    )
    report.issues.extend(manifest_issues)
    report.manifest_rows = row_count
    report.hashes_verified = hash_count
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate AudioRescue dataset recordings and manifest.csv")
    parser.add_argument("dataset_root", nargs="?", default="data_local")
    parser.add_argument("--manifest", help="optional manifest path; defaults to DATASET_ROOT/manifest.csv")
    parser.add_argument("--spec", help="local dataset spec JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = validate_dataset(args.dataset_root, args.manifest, spec_path=args.spec)
    print(report.render())
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
