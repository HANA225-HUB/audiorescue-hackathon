#!/usr/bin/env python3
"""Validate the frozen AudioRescue-CN-Mini-v1 recording dataset.

The validator intentionally uses only the Python standard library so it can be
run on the recording laptop before the model environment is installed.
"""

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
from typing import Iterable, Sequence


DATASET_VERSION = "AudioRescue-CN-Mini-v1"
PENDING_CONSENT_TOKEN = "pending_team_confirmation"
APPROVED_CONSENT_TOKEN = "team-approved-for-competition-evaluation"
SPEAKERS = ("A", "B", "C")
SENTENCES = ("s01", "s02", "s03")
REFERENCE_TEXTS = {
    "s01": "今天下午三点，我们在实验室讨论语音处理项目的最终方案。",
    "s02": "请记录会议中的三个重点：数据来源、模型效果和系统稳定性。",
    "s03": "如果现场网络中断，系统仍然可以在本地完成音频增强和文字转写。",
}
SNR_CODES = {"snrp05": 5.0, "snr000": 0.0, "snrm05": -5.0}
NOISE_ASSIGNMENT = {
    ("A", "s01"): "fan",
    ("A", "s02"): "keyboard",
    ("A", "s03"): "traffic",
    ("B", "s01"): "keyboard",
    ("B", "s02"): "traffic",
    ("B", "s03"): "fan",
    ("C", "s01"): "traffic",
    ("C", "s02"): "fan",
    ("C", "s03"): "keyboard",
}
REAL_ASSIGNMENT = {
    "A": ("s03", "traffic"),
    "B": ("s01", "fan"),
    "C": ("s02", "keyboard"),
}
DEMO_SAMPLE_IDS = {
    "mix_spkB_s03_fan_snr000",
    "mix_spkC_s03_keyboard_snrm05",
    "clean_spkA_s03",
    "real_spkA_traffic_r01",
}

MANIFEST_FIELDS = (
    "sample_id",
    "dataset_version",
    "split",
    "speaker_id",
    "sentence_id",
    "reference_text",
    "reference_normalized",
    "source_type",
    "clean_path",
    "noise_path",
    "mixed_path",
    "noise_type",
    "noise_source",
    "consent_or_license",
    "snr_db",
    "mix_seed",
    "noise_offset_seconds",
    "mix_alpha",
    "final_gain",
    "sample_rate",
    "channels",
    "duration_seconds",
    "recording_device",
    "recording_distance_cm",
    "sha256",
    "is_demo_candidate",
    "is_locked",
    "notes",
)

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
    noise_min_seconds: float = 45.0
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
    split: str | None = None


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


def expected_dataset_artifacts() -> dict[Path, ArtifactSpec]:
    """Return the frozen 9 clean + 3 noise + 27 mix + 3 real matrix."""

    artifacts: dict[Path, ArtifactSpec] = {}
    for speaker in SPEAKERS:
        for sentence in SENTENCES:
            path = Path("raw") / "clean" / f"spk{speaker}" / f"clean_spk{speaker}_{sentence}.wav"
            artifacts[path] = ArtifactSpec(path, "clean", speaker, sentence)

    for noise in ("fan", "keyboard", "traffic"):
        path = Path("raw") / "noise" / f"noise_{noise}_take01.wav"
        artifacts[path] = ArtifactSpec(path, "noise", noise=noise)

    for speaker in SPEAKERS:
        for sentence in SENTENCES:
            noise = NOISE_ASSIGNMENT[(speaker, sentence)]
            split = "dev" if sentence in {"s01", "s02"} else "locked_test"
            for snr_code in SNR_CODES:
                filename = f"mix_spk{speaker}_{sentence}_{noise}_{snr_code}.wav"
                path = Path("controlled") / split / filename
                artifacts[path] = ArtifactSpec(
                    path,
                    "mix",
                    speaker,
                    sentence,
                    noise,
                    snr_code,
                    split,
                )

    for speaker, (sentence, noise) in REAL_ASSIGNMENT.items():
        path = Path("raw") / "real" / f"real_spk{speaker}_{noise}_r01.wav"
        artifacts[path] = ArtifactSpec(
            path,
            "real",
            speaker,
            sentence,
            noise,
            split="real",
        )
    return artifacts


def required_manifest_primary_paths() -> set[Path]:
    """Return the 27 mix, 3 real, and 3 S03 clean-control sample paths."""

    artifacts = expected_dataset_artifacts()
    paths = {
        path
        for path, spec in artifacts.items()
        if spec.kind in {"mix", "real"}
        or (spec.kind == "clean" and spec.sentence == "s03")
    }
    return paths


def normalize_reference(text: str) -> str:
    """Apply the same frozen normalization used by the CER implementation."""

    normalized = unicodedata.normalize("NFKC", text).lower()
    return "".join(character for character in normalized if character.isalnum())


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
    """Inspect one WAV without numpy or soundfile."""

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
    supplied = Path(value)
    if supplied.is_absolute():
        candidate = supplied.resolve()
    else:
        parts = supplied.parts
        if parts and parts[0] == root.name:
            supplied = Path(*parts[1:])
        candidate = (root / supplied).resolve()
    try:
        relative = candidate.relative_to(root.resolve())
    except ValueError:
        return None
    return relative, candidate


def _primary_path_field(row: dict[str, str]) -> str:
    source_type = (row.get("source_type") or "").strip().lower()
    if source_type in {"clean", "clean_control"}:
        return row.get("clean_path") or ""
    if source_type == "noise":
        return row.get("noise_path") or ""
    return (row.get("mixed_path") or row.get("clean_path") or "").strip()


def validate_manifest(
    root: Path,
    manifest_path: Path,
    wav_infos: dict[Path, WavInfo],
    required_primary_paths: Iterable[Path] | None = None,
) -> tuple[list[ValidationIssue], int, int]:
    """Validate the exact 33-row runnable matrix and every frozen row field."""

    issues: list[ValidationIssue] = []
    rows_count = 0
    hashes_verified = 0
    required_paths = set(required_primary_paths or required_manifest_primary_paths())
    all_artifacts = expected_dataset_artifacts()
    entries_by_id = {
        path.stem: (path, all_artifacts[path])
        for path in required_paths
        if path in all_artifacts
    }

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
                add(
                    "MANIFEST_COLUMNS",
                    manifest_path,
                    "missing columns: " + ", ".join(missing_headers),
                )
            if unexpected_headers:
                add(
                    "MANIFEST_COLUMNS",
                    manifest_path,
                    "unexpected columns: " + ", ".join(unexpected_headers),
                )
            if len(headers) != len(set(headers)):
                add("MANIFEST_COLUMNS", manifest_path, "contains duplicate column names")

            seen_ids: dict[str, int] = {}
            seen_primary_paths: dict[Path, int] = {}
            hash_cache: dict[Path, str] = {}
            mix_cell_values: dict[
                tuple[str, str], list[tuple[int, float, str]]
            ] = {}

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
                id_entry = entries_by_id.get(sample_id)
                if not sample_id:
                    add("MANIFEST_REQUIRED", row_label, "sample_id is empty")
                elif sample_id in seen_ids:
                    add(
                        "MANIFEST_DUPLICATE_ID",
                        row_label,
                        f"sample_id {sample_id!r} already appeared on line {seen_ids[sample_id]}",
                    )
                else:
                    seen_ids[sample_id] = line_number

                if row.get("dataset_version") != DATASET_VERSION:
                    add(
                        "MANIFEST_VERSION",
                        row_label,
                        f"dataset_version must be {DATASET_VERSION!r}",
                    )
                consent_or_license = row.get("consent_or_license", "")
                if consent_or_license != APPROVED_CONSENT_TOKEN:
                    add(
                        "MANIFEST_CONSENT",
                        row_label,
                        "consent_or_license must be the exact formal token "
                        f"{APPROVED_CONSENT_TOKEN!r}; "
                        f"{PENDING_CONSENT_TOKEN!r} is development-only",
                    )

                resolved_fields: dict[str, tuple[Path, Path]] = {}
                for field_name in ("clean_path", "noise_path", "mixed_path"):
                    raw_value = row.get(field_name, "")
                    if not raw_value:
                        continue
                    resolved = _manifest_path(root, raw_value)
                    if resolved is None:
                        add(
                            "MANIFEST_PATH",
                            row_label,
                            f"{field_name} must stay inside dataset root: {raw_value!r}",
                        )
                        continue
                    relative, absolute = resolved
                    resolved_fields[field_name] = resolved
                    if not absolute.is_file():
                        add(
                            "MANIFEST_PATH",
                            row_label,
                            f"{field_name} does not exist: {relative.as_posix()}",
                        )

                if id_entry is not None:
                    _id_path, id_spec = id_entry
                    primary_field = "clean_path" if id_spec.kind == "clean" else "mixed_path"
                    primary_raw = row.get(primary_field, "")
                else:
                    primary_raw = _primary_path_field(row)
                primary_resolved = _manifest_path(root, primary_raw) if primary_raw else None
                primary_relative: Path | None = None
                primary_absolute: Path | None = None
                if primary_resolved is None:
                    add(
                        "MANIFEST_PRIMARY_PATH",
                        row_label,
                        "cannot determine an in-root primary audio path",
                    )
                else:
                    primary_relative, primary_absolute = primary_resolved
                    if primary_relative in seen_primary_paths:
                        add(
                            "MANIFEST_DUPLICATE_PATH",
                            row_label,
                            "primary path already appeared on line "
                            f"{seen_primary_paths[primary_relative]}: "
                            f"{primary_relative.as_posix()}",
                        )
                    else:
                        seen_primary_paths[primary_relative] = line_number

                if id_entry is not None:
                    semantic_path, spec = id_entry
                elif primary_relative in required_paths:
                    semantic_path = primary_relative
                    spec = all_artifacts[primary_relative]
                else:
                    semantic_path = None
                    spec = None

                if semantic_path is not None:
                    if sample_id != semantic_path.stem:
                        add(
                            "MANIFEST_SAMPLE_ID",
                            row_label,
                            f"sample_id must be {semantic_path.stem!r} for "
                            f"{semantic_path.as_posix()}",
                        )
                    if primary_relative != semantic_path:
                        actual = (
                            primary_relative.as_posix()
                            if primary_relative is not None
                            else "<missing>"
                        )
                        add(
                            "MANIFEST_PRIMARY_PATH",
                            row_label,
                            f"primary path must be {semantic_path.as_posix()}, got {actual}",
                        )

                declared_hash = row.get("sha256", "")
                if not _SHA256_RE.fullmatch(declared_hash):
                    add(
                        "MANIFEST_HASH",
                        row_label,
                        "sha256 must contain exactly 64 hexadecimal characters",
                    )
                elif primary_absolute is not None and primary_absolute.is_file():
                    actual_hash = hash_cache.get(primary_absolute)
                    if actual_hash is None:
                        actual_hash = sha256_file(primary_absolute)
                        hash_cache[primary_absolute] = actual_hash
                    if actual_hash != declared_hash.lower():
                        add(
                            "MANIFEST_HASH",
                            row_label,
                            f"sha256 mismatch for {primary_relative.as_posix()}",
                        )
                    else:
                        hashes_verified += 1

                info = wav_infos.get(primary_relative) if primary_relative is not None else None
                if info is not None:
                    try:
                        declared_rate = int(row.get("sample_rate", ""))
                        declared_channels = int(row.get("channels", ""))
                        declared_duration = float(row.get("duration_seconds", ""))
                    except ValueError:
                        add(
                            "MANIFEST_AUDIO_META",
                            row_label,
                            "sample_rate, channels, and duration_seconds must be numeric",
                        )
                    else:
                        if declared_rate != info.sample_rate or declared_channels != info.channels:
                            add(
                                "MANIFEST_AUDIO_META",
                                row_label,
                                "declared sample_rate/channels do not match WAV header",
                            )
                        tolerance = max(0.001, 1.0 / max(info.sample_rate, 1))
                        if not math.isfinite(declared_duration) or abs(
                            declared_duration - info.duration_seconds
                        ) > tolerance:
                            add(
                                "MANIFEST_AUDIO_META",
                                row_label,
                                f"declared duration {declared_duration!r} does not match "
                                f"{info.duration_seconds:.6f}",
                            )

                distance = row.get("recording_distance_cm", "")
                if distance:
                    try:
                        numeric_distance = float(distance)
                    except ValueError:
                        numeric_distance = math.nan
                    if not math.isfinite(numeric_distance) or numeric_distance < 0:
                        add(
                            "MANIFEST_AUDIO_META",
                            row_label,
                            "recording_distance_cm must be empty or a finite non-negative number",
                        )

                if spec is None or semantic_path is None:
                    continue

                expected_source_type = {
                    "mix": "controlled_mix",
                    "clean": "clean_control",
                    "real": "real",
                }[spec.kind]
                expected_split = (
                    spec.split
                    if spec.kind == "mix"
                    else "clean_control" if spec.kind == "clean" else "real"
                )
                expected_locked = spec.kind == "clean" or (
                    spec.kind == "mix" and spec.split == "locked_test"
                )
                expected_speaker = spec.speaker or ""
                expected_sentence = (spec.sentence or "").upper()
                expected_noise = spec.noise or ""

                expected_values = {
                    "source_type": expected_source_type,
                    "split": expected_split,
                    "speaker_id": expected_speaker,
                    "sentence_id": expected_sentence,
                    "noise_type": expected_noise,
                }
                for field_name, expected_value in expected_values.items():
                    if row.get(field_name, "") != expected_value:
                        add(
                            "MANIFEST_SEMANTICS",
                            row_label,
                            f"{field_name} must be {expected_value!r} for "
                            f"{semantic_path.name}",
                        )

                sentence_key = spec.sentence or ""
                expected_reference = REFERENCE_TEXTS.get(sentence_key, "")
                if row.get("reference_text", "") != expected_reference:
                    add(
                        "MANIFEST_REFERENCE",
                        row_label,
                        f"reference_text does not match frozen {expected_sentence}",
                    )
                expected_normalized = normalize_reference(expected_reference)
                if row.get("reference_normalized", "") != expected_normalized:
                    add(
                        "MANIFEST_REFERENCE",
                        row_label,
                        f"reference_normalized does not match frozen {expected_sentence}",
                    )

                declared_locked = _parse_manifest_bool(row.get("is_locked", ""))
                if declared_locked is None:
                    add(
                        "MANIFEST_BOOLEAN",
                        row_label,
                        "is_locked must be true/false (or 1/0, yes/no)",
                    )
                elif declared_locked is not expected_locked:
                    add(
                        "MANIFEST_LOCKED",
                        row_label,
                        f"is_locked must be {str(expected_locked).lower()} for "
                        f"{semantic_path.name}",
                    )

                expected_demo = semantic_path.stem in DEMO_SAMPLE_IDS
                declared_demo = _parse_manifest_bool(row.get("is_demo_candidate", ""))
                if declared_demo is None:
                    add(
                        "MANIFEST_BOOLEAN",
                        row_label,
                        "is_demo_candidate must be true/false (or 1/0, yes/no)",
                    )
                elif declared_demo is not expected_demo:
                    add(
                        "MANIFEST_DEMO_CANDIDATE",
                        row_label,
                        f"is_demo_candidate must be {str(expected_demo).lower()} for "
                        f"{semantic_path.name}",
                    )

                expected_path_fields: dict[str, Path] = {}
                forbidden_path_fields: tuple[str, ...]
                if spec.kind == "mix":
                    expected_path_fields = {
                        "clean_path": Path("raw")
                        / "clean"
                        / f"spk{spec.speaker}"
                        / f"clean_spk{spec.speaker}_{spec.sentence}.wav",
                        "noise_path": Path("raw")
                        / "noise"
                        / f"noise_{spec.noise}_take01.wav",
                        "mixed_path": semantic_path,
                    }
                    forbidden_path_fields = ()
                elif spec.kind == "clean":
                    expected_path_fields = {"clean_path": semantic_path}
                    forbidden_path_fields = ("noise_path", "mixed_path")
                else:
                    expected_path_fields = {"mixed_path": semantic_path}
                    forbidden_path_fields = ("clean_path", "noise_path")

                for field_name, expected_path in expected_path_fields.items():
                    resolved = resolved_fields.get(field_name)
                    if resolved is None:
                        add(
                            "MANIFEST_PATH_SEMANTICS",
                            row_label,
                            f"{field_name} is required for {semantic_path.name}",
                        )
                    elif resolved[0] != expected_path:
                        add(
                            "MANIFEST_PATH_SEMANTICS",
                            row_label,
                            f"{field_name} must point to {expected_path.as_posix()}",
                        )
                for field_name in forbidden_path_fields:
                    if row.get(field_name, ""):
                        add(
                            "MANIFEST_PATH_SEMANTICS",
                            row_label,
                            f"{field_name} must be empty for {expected_source_type}",
                        )

                mix_only_fields = (
                    "snr_db",
                    "mix_seed",
                    "noise_offset_seconds",
                    "mix_alpha",
                    "final_gain",
                )
                if spec.kind != "mix":
                    for field_name in mix_only_fields:
                        if row.get(field_name, ""):
                            add(
                                "MANIFEST_MIX_META",
                                row_label,
                                f"{field_name} must be empty for {expected_source_type}",
                            )
                    continue

                try:
                    declared_snr = float(row.get("snr_db", ""))
                except ValueError:
                    declared_snr = math.nan
                expected_snr = SNR_CODES[spec.snr_code or ""]
                if not math.isfinite(declared_snr) or declared_snr != expected_snr:
                    add(
                        "MANIFEST_SNR",
                        row_label,
                        f"expected finite snr_db {expected_snr:g}, got {row.get('snr_db', '')!r}",
                    )

                seed_raw = row.get("mix_seed", "")
                seed_value: int | None = None
                if not _INTEGER_RE.fullmatch(seed_raw):
                    add("MANIFEST_MIX_META", row_label, "mix_seed must be an integer")
                else:
                    seed_value = int(seed_raw)

                def finite_mix_value(field_name: str) -> float | None:
                    try:
                        value = float(row.get(field_name, ""))
                    except ValueError:
                        value = math.nan
                    if not math.isfinite(value):
                        add(
                            "MANIFEST_MIX_META",
                            row_label,
                            f"{field_name} must be a finite number",
                        )
                        return None
                    return value

                offset = finite_mix_value("noise_offset_seconds")
                alpha = finite_mix_value("mix_alpha")
                final_gain = finite_mix_value("final_gain")
                if offset is not None and offset < 0:
                    add(
                        "MANIFEST_MIX_META",
                        row_label,
                        "noise_offset_seconds must be non-negative",
                    )
                if alpha is not None and alpha <= 0:
                    add("MANIFEST_MIX_META", row_label, "mix_alpha must be greater than zero")
                if final_gain is not None and not 0 < final_gain <= 1:
                    add(
                        "MANIFEST_MIX_META",
                        row_label,
                        "final_gain must be greater than zero and at most one",
                    )
                if (
                    seed_value is not None
                    and offset is not None
                    and offset >= 0
                    and spec.speaker is not None
                    and spec.sentence is not None
                ):
                    mix_cell_values.setdefault(
                        (spec.speaker, spec.sentence), []
                    ).append((seed_value, offset, row_label))

            expected_row_count = len(required_paths)
            if rows_count != expected_row_count:
                add(
                    "MANIFEST_ROW_COUNT",
                    manifest_path,
                    f"manifest must contain exactly {expected_row_count} rows, got {rows_count}",
                )

            actual_primary_paths = set(seen_primary_paths)
            missing_primary = sorted(
                required_paths - actual_primary_paths,
                key=lambda item: item.as_posix(),
            )
            unexpected_primary = sorted(
                actual_primary_paths - required_paths,
                key=lambda item: item.as_posix(),
            )
            for relative in missing_primary:
                add(
                    "MANIFEST_COVERAGE",
                    manifest_path,
                    f"missing runnable sample row for {relative.as_posix()}",
                )
            for relative in unexpected_primary:
                add(
                    "MANIFEST_COVERAGE",
                    manifest_path,
                    f"unexpected primary sample path {relative.as_posix()}",
                )

            for (speaker, sentence), values in sorted(mix_cell_values.items()):
                seeds = {seed for seed, _offset, _label in values}
                offsets = {offset for _seed, offset, _label in values}
                if len(seeds) > 1 or len(offsets) > 1:
                    add(
                        "MANIFEST_MIX_CELL",
                        manifest_path,
                        f"spk{speaker}/{sentence} must share one mix_seed and one "
                        "noise_offset_seconds across its three SNR variants",
                    )
    except (csv.Error, OSError, UnicodeError) as exc:
        add("MANIFEST_UNREADABLE", manifest_path, f"cannot read manifest: {exc}")

    return issues, rows_count, hashes_verified


def validate_dataset(
    root: str | Path,
    manifest_path: str | Path | None = None,
    policy: ValidationPolicy = DEFAULT_POLICY,
) -> ValidationReport:
    dataset_root = Path(root).expanduser().resolve()
    report = ValidationReport(root=dataset_root)
    artifacts = expected_dataset_artifacts()
    report.expected_wavs = len(artifacts)

    if not dataset_root.is_dir():
        report.add("error", "DATASET_ROOT", dataset_root, "dataset root is not a directory")
        return report

    wav_infos: dict[Path, WavInfo] = {}
    for relative, spec in artifacts.items():
        absolute = dataset_root / relative
        if not absolute.is_file():
            report.add("error", "MISSING_WAV", relative, f"required {spec.kind} recording is missing")
            continue
        report.checked_wavs += 1
        info, issues = inspect_wav(absolute, spec, policy)
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
                    "WAV does not match the frozen filename and directory matrix",
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
    )
    report.issues.extend(manifest_issues)
    report.manifest_rows = row_count
    report.hashes_verified = hash_count
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate AudioRescue-CN-Mini-v1 recordings and manifest.csv",
    )
    parser.add_argument(
        "dataset_root",
        nargs="?",
        default="data_local",
        help="dataset root containing raw/, controlled/, and manifest.csv",
    )
    parser.add_argument(
        "--manifest",
        help="optional manifest path; defaults to DATASET_ROOT/manifest.csv",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = validate_dataset(args.dataset_root, args.manifest)
    print(report.render())
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
