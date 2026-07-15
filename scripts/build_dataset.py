"""Build deterministic controlled mixtures from a generic dataset spec.

The script uses only Python's standard library. Source WAVs must already match
the declared spec audio format; format conversion belongs to a separate
normalization step and is never performed silently here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import tempfile
import unicodedata
import wave
from array import array
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.dataset_spec import (
    DEFAULT_APPROVED_CONSENT_TOKEN,
    DEFAULT_PENDING_CONSENT_TOKEN,
    DatasetSpec,
    DatasetSpecError,
    ensure_private_repo_root,
    load_dataset_spec,
    safe_join,
)


PENDING_CONSENT_TOKEN = DEFAULT_PENDING_CONSENT_TOKEN
APPROVED_CONSENT_TOKEN = DEFAULT_APPROVED_CONSENT_TOKEN
ALLOWED_CONSENT_TOKENS = frozenset({PENDING_CONSENT_TOKEN, APPROVED_CONSENT_TOKEN})
SAMPLE_RATE = 48_000
CHANNELS = 1
SAMPLE_WIDTH_BYTES = 2
PEAK_LIMIT = 0.98
DEFAULT_BASE_SEED = 20_260_714

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


class DatasetBuildError(RuntimeError):
    """Raised when source material cannot be mixed without changing the contract."""


@dataclass(frozen=True, slots=True)
class WavData:
    path: Path
    samples: array
    sample_rate: int


@dataclass(frozen=True, slots=True)
class MixResult:
    samples: array
    alpha: float
    final_gain: float
    clean_power: float
    noise_power: float


def _allowed_tokens(spec: DatasetSpec) -> frozenset[str]:
    return frozenset({spec.consent_tokens.pending, spec.consent_tokens.approved})


def _load_spec(path: str | Path | None, *, example_mode: bool = False) -> DatasetSpec:
    try:
        return load_dataset_spec(path, allow_example=example_mode)
    except DatasetSpecError as exc:
        if "explicit dataset spec" in str(exc):
            raise DatasetBuildError("explicit dataset spec is required") from exc
        raise DatasetBuildError("dataset spec is invalid") from exc
    except ValueError as exc:
        raise DatasetBuildError("dataset spec is invalid") from exc


def _require_private_root(root: Path) -> None:
    try:
        ensure_private_repo_root(root)
    except DatasetSpecError as exc:
        raise DatasetBuildError(
            "dataset root inside the repository must be ignored by Git"
        ) from exc


def _safe_path(root: Path, relative: str | Path, *, must_exist: bool = False) -> Path:
    try:
        return safe_join(root, relative, must_exist=must_exist)
    except DatasetSpecError as exc:
        raise DatasetBuildError("unsafe path declared by dataset spec") from exc


def _verify_formal_consent_ledger(
    root: Path,
    spec: DatasetSpec,
    real_audio: dict[str, WavData],
) -> None:
    """Bind the formal token to every required master and its hashes."""

    ledger = root / "recording_metadata.csv"
    required_fields = {
        "asset_id",
        "source_original_path",
        "relative_path",
        "consent_status",
        "source_original_sha256",
        "standardized_sha256",
    }
    try:
        with ledger.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or [])
            if not required_fields.issubset(fields):
                raise DatasetBuildError("recording metadata is missing required columns")
            rows = list(reader)
    except DatasetBuildError:
        raise
    except (OSError, UnicodeError, csv.Error) as exc:
        raise DatasetBuildError("recording metadata could not be read") from exc

    expected = spec.master_audio_paths()
    expected = {
        asset_id: path
        for asset_id, path in expected.items()
        if asset_id in real_audio or not path.parts[:2] == ("raw", "real")
    }
    by_id: dict[str, dict[str, str]] = {}
    for row in rows:
        asset_id = (row.get("asset_id") or "").strip()
        if not asset_id or asset_id in by_id:
            raise DatasetBuildError("recording metadata contains duplicate or empty asset ids")
        by_id[asset_id] = row
    if set(by_id) != set(expected):
        raise DatasetBuildError("recording metadata must cover every required master")

    source_root = (root / "source_original").resolve()
    for asset_id, relative_path in expected.items():
        row = by_id[asset_id]
        if (row.get("consent_status") or "").strip().lower() != "yes":
            raise DatasetBuildError("recording metadata consent_status must be yes")
        entered_relative = (row.get("relative_path") or "").strip()
        standardized_path = _safe_path(root, entered_relative, must_exist=True)
        expected_standardized = _safe_path(root, relative_path, must_exist=True)
        if standardized_path != expected_standardized:
            raise DatasetBuildError("recording metadata relative_path does not match spec")

        original_value = (row.get("source_original_path") or "").strip()
        if not original_value:
            raise DatasetBuildError("recording metadata is missing source_original_path")
        original_entered = Path(original_value).expanduser()
        if original_entered.is_absolute():
            raise DatasetBuildError("source originals must stay under source_original")
        original_path = _safe_path(root, original_entered, must_exist=True)
        try:
            original_path.relative_to(source_root)
        except ValueError as exc:
            raise DatasetBuildError("source originals must stay under source_original") from exc
        if not original_path.is_file():
            raise DatasetBuildError("source original file is missing")

        original_hash = (row.get("source_original_sha256") or "").strip().lower()
        standardized_hash = (row.get("standardized_sha256") or "").strip().lower()
        for digest in (original_hash, standardized_hash):
            if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
                raise DatasetBuildError("recording metadata contains an invalid SHA-256")
        if sha256_file(original_path) != original_hash:
            raise DatasetBuildError("source original SHA-256 does not match")
        if sha256_file(standardized_path) != standardized_hash:
            raise DatasetBuildError("standardized WAV SHA-256 does not match")


def normalize_reference(text: str) -> str:
    """Match the CER normalization: NFKC, lowercase, alphanumerics only."""

    normalized = unicodedata.normalize("NFKC", text).lower()
    return "".join(character for character in normalized if character.isalnum())


def read_pcm16_mono(path: str | Path) -> WavData:
    """Read a contract-compliant WAV or fail with a fixed safe message."""

    resolved = Path(path)
    if not resolved.is_file():
        raise DatasetBuildError("source WAV is missing")

    try:
        with wave.open(str(resolved), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            compression = wav_file.getcomptype()
            frame_count = wav_file.getnframes()
            raw_frames = wav_file.readframes(frame_count)
    except (EOFError, OSError, wave.Error) as exc:
        raise DatasetBuildError("source WAV could not be read") from exc

    problems = []
    if sample_rate != SAMPLE_RATE:
        problems.append("sample_rate")
    if channels != CHANNELS:
        problems.append("channels")
    if sample_width != SAMPLE_WIDTH_BYTES:
        problems.append("bit_depth")
    if compression != "NONE":
        problems.append("compression")
    if problems:
        raise DatasetBuildError(
            "source WAV does not match declared PCM16 mono format; "
            "explicit normalization or resampling is required"
        )

    samples = array("h")
    samples.frombytes(raw_frames)
    if sys.byteorder == "big":
        samples.byteswap()
    if len(samples) != frame_count:
        raise DatasetBuildError("source WAV frame count is incomplete")
    if not samples:
        raise DatasetBuildError("source WAV is empty")
    return WavData(path=resolved, samples=samples, sample_rate=sample_rate)


def _center_and_power(samples: Sequence[int]) -> tuple[array, float]:
    if not samples:
        raise DatasetBuildError("cannot mix an empty signal")
    mean = math.fsum(float(value) for value in samples) / len(samples)
    centered = array("d", (float(value) - mean for value in samples))
    power = math.fsum(value * value for value in centered) / len(centered)
    return centered, power


def _mix_centered(
    clean: Sequence[float],
    noise: Sequence[float],
    clean_power: float,
    noise_power: float,
    snr_db: float,
    peak_limit: float,
) -> MixResult:
    if len(clean) != len(noise):
        raise DatasetBuildError("clean and noise segments must have the same length")
    if clean_power <= 1e-12:
        raise DatasetBuildError("clean signal is too close to silence")
    if noise_power <= 1e-12:
        raise DatasetBuildError("noise signal is too close to silence")
    if not math.isfinite(snr_db):
        raise DatasetBuildError("snr_db must be finite")
    if not 0 < peak_limit <= 1:
        raise DatasetBuildError("peak_limit must be in (0, 1]")

    alpha = math.sqrt(clean_power / (noise_power * (10.0 ** (snr_db / 10.0))))
    peak_pcm = max(
        abs(clean_value + alpha * noise_value)
        for clean_value, noise_value in zip(clean, noise)
    )
    peak = peak_pcm / 32768.0
    final_gain = 1.0 if peak <= peak_limit else peak_limit / peak
    quantized_limit = math.floor(peak_limit * 32768.0)
    output = array("h")
    for clean_value, noise_value in zip(clean, noise):
        value = round((clean_value + alpha * noise_value) * final_gain)
        value = max(-quantized_limit, min(quantized_limit, value))
        output.append(value)

    return MixResult(
        samples=output,
        alpha=alpha,
        final_gain=final_gain,
        clean_power=clean_power,
        noise_power=noise_power,
    )


def mix_pcm16(
    clean_samples: Sequence[int],
    noise_samples: Sequence[int],
    snr_db: float,
    *,
    noise_offset_samples: int = 0,
    peak_limit: float = PEAK_LIMIT,
) -> MixResult:
    """Mix PCM16 arrays at an exact component SNR with shared peak protection."""

    if noise_offset_samples < 0:
        raise DatasetBuildError("noise_offset_samples cannot be negative")
    stop = noise_offset_samples + len(clean_samples)
    if stop > len(noise_samples):
        raise DatasetBuildError("noise is too short; loop filling is not allowed")

    clean_centered, clean_power = _center_and_power(clean_samples)
    noise_centered, noise_power = _center_and_power(noise_samples[noise_offset_samples:stop])
    return _mix_centered(clean_centered, noise_centered, clean_power, noise_power, float(snr_db), peak_limit)


def derive_mix_seed(base_seed: int, clean_id: str, noise_id: str, sample_id: str) -> int:
    key = f"{base_seed}|{clean_id}|{noise_id}|{sample_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big")


def choose_noise_offset(mix_seed: int, max_offset_samples: int) -> int:
    if max_offset_samples < 0:
        raise DatasetBuildError("noise is shorter than clean")
    if max_offset_samples == 0:
        return 0
    digest = hashlib.sha256(f"{mix_seed}|noise-offset".encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big") % (max_offset_samples + 1)


def _relative(path: Path, dataset_root: Path) -> str:
    return path.resolve().relative_to(dataset_root.resolve()).as_posix()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_pcm16_mono(path: str | Path, samples: Sequence[int]) -> None:
    """Atomically write a mono PCM16 WAV."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    encoded = array("h", samples)
    if sys.byteorder == "big":
        encoded.byteswap()
    try:
        with wave.open(str(temporary_path), "wb") as wav_file:
            wav_file.setnchannels(CHANNELS)
            wav_file.setsampwidth(SAMPLE_WIDTH_BYTES)
            wav_file.setframerate(SAMPLE_RATE)
            wav_file.writeframes(encoded.tobytes())
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_manifest_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        with temporary_path.open("w", encoding="utf-8", newline="") as file_handle:
            writer = csv.DictWriter(
                file_handle,
                fieldnames=MANIFEST_FIELDS,
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_manifest_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as file_handle:
            json.dump(rows, file_handle, ensure_ascii=False, indent=2)
            file_handle.write("\n")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_and_validate_sources(
    dataset_root: str | Path,
    *,
    spec_path: str | Path | None = None,
    spec: DatasetSpec | None = None,
    example_mode: bool = False,
    allow_missing_real: bool = False,
) -> tuple[dict[str, WavData], dict[str, WavData], dict[str, WavData]]:
    """Load clean, noise, and real recordings declared by the spec."""

    root = Path(dataset_root).expanduser().resolve()
    _require_private_root(root)
    loaded_spec = spec or _load_spec(spec_path, example_mode=example_mode)
    clean_audio = {
        item.id: read_pcm16_mono(_safe_path(root, item.path))
        for item in loaded_spec.clean_recordings
    }
    noise_audio = {
        item.id: read_pcm16_mono(_safe_path(root, item.path))
        for item in loaded_spec.noise_recordings
    }
    real_audio: dict[str, WavData] = {}
    for item in loaded_spec.real_recordings:
        try:
            path = _safe_path(root, item.path)
        except DatasetBuildError:
            if allow_missing_real:
                continue
            raise
        if allow_missing_real and not path.is_file():
            continue
        try:
            real_audio[item.sample_id] = read_pcm16_mono(path)
        except DatasetBuildError as exc:
            raise DatasetBuildError("real recording is missing or invalid") from exc

    for item in loaded_spec.mixes:
        clean = clean_audio[item.clean_id]
        noise = noise_audio[item.noise_id]
        if len(noise.samples) < len(clean.samples):
            raise DatasetBuildError("noise is shorter than a referenced clean signal")
    return clean_audio, noise_audio, real_audio


def build_dataset(
    dataset_root: str | Path,
    *,
    spec_path: str | Path | None = None,
    base_seed: int = DEFAULT_BASE_SEED,
    overwrite: bool = False,
    consent_or_license: str | None = None,
    recording_device: str = "",
    recording_distance_cm: float | None = None,
    example_mode: bool = False,
    allow_missing_real: bool = False,
) -> list[dict[str, Any]]:
    """Generate mixtures and manifests for every runnable spec row."""

    root = Path(dataset_root).expanduser().resolve()
    _require_private_root(root)
    spec = _load_spec(spec_path, example_mode=example_mode)
    if consent_or_license is None:
        consent_or_license = spec.consent_tokens.pending
    allowed_tokens = _allowed_tokens(spec)
    if not isinstance(consent_or_license, str) or consent_or_license not in allowed_tokens:
        raise DatasetBuildError("consent_or_license must use one exact workflow token")

    clean_audio, noise_audio, real_audio = load_and_validate_sources(
        root,
        spec=spec,
        allow_missing_real=allow_missing_real,
    )
    if consent_or_license == spec.consent_tokens.approved:
        _verify_formal_consent_ledger(root, spec, real_audio)

    manifest_csv = _safe_path(root, "manifest.csv")
    manifest_json = _safe_path(root, "manifest.json")
    output_paths = [_safe_path(root, item.path) for item in spec.mixes]
    collisions = [path for path in (*output_paths, manifest_csv, manifest_json) if path.exists()]
    if collisions and not overwrite:
        raise DatasetBuildError("target files already exist; use --overwrite")

    rows: list[dict[str, Any]] = []
    clean_by_id = spec.clean_by_id
    noise_by_id = spec.noise_by_id
    centered_cache: dict[str, tuple[array, float]] = {}
    offset_cache: dict[str, tuple[int, int]] = {}

    for item in spec.mixes:
        clean_spec = clean_by_id[item.clean_id]
        noise_spec = noise_by_id[item.noise_id]
        clean = clean_audio[item.clean_id]
        noise = noise_audio[item.noise_id]
        mix_seed = derive_mix_seed(base_seed, item.clean_id, item.noise_id, item.sample_id)
        max_offset = len(noise.samples) - len(clean.samples)
        noise_offset_samples = choose_noise_offset(mix_seed, max_offset)
        clean_centered, clean_power = centered_cache.setdefault(
            item.clean_id,
            _center_and_power(clean.samples),
        )
        noise_stop = noise_offset_samples + len(clean.samples)
        noise_cache_key = f"{item.noise_id}|{noise_offset_samples}|{len(clean.samples)}"
        if noise_cache_key in centered_cache:
            noise_centered, noise_power = centered_cache[noise_cache_key]
        else:
            noise_centered, noise_power = _center_and_power(
                noise.samples[noise_offset_samples:noise_stop]
            )
            centered_cache[noise_cache_key] = (noise_centered, noise_power)
        offset_cache[item.sample_id] = (mix_seed, noise_offset_samples)
        mixed_path = _safe_path(root, item.path)
        result = _mix_centered(
            clean_centered,
            noise_centered,
            clean_power,
            noise_power,
            item.snr_db,
            PEAK_LIMIT,
        )
        write_pcm16_mono(mixed_path, result.samples)
        rows.append(
            {
                "sample_id": item.sample_id,
                "dataset_version": spec.dataset_version,
                "split": item.split,
                "speaker_id": clean_spec.speaker_id,
                "sentence_id": clean_spec.sentence_id,
                "reference_text": clean_spec.reference_text,
                "reference_normalized": normalize_reference(clean_spec.reference_text),
                "source_type": "controlled_mix",
                "clean_path": _relative(clean.path, root),
                "noise_path": _relative(noise.path, root),
                "mixed_path": _relative(mixed_path, root),
                "noise_type": noise_spec.noise_type,
                "noise_source": "local_spec_recording",
                "consent_or_license": consent_or_license,
                "snr_db": int(item.snr_db) if item.snr_db.is_integer() else item.snr_db,
                "mix_seed": mix_seed,
                "noise_offset_seconds": noise_offset_samples / SAMPLE_RATE,
                "mix_alpha": result.alpha,
                "final_gain": result.final_gain,
                "sample_rate": SAMPLE_RATE,
                "channels": CHANNELS,
                "duration_seconds": len(result.samples) / SAMPLE_RATE,
                "recording_device": recording_device,
                "recording_distance_cm": "" if recording_distance_cm is None else recording_distance_cm,
                "sha256": sha256_file(mixed_path),
                "is_demo_candidate": item.is_demo_candidate,
                "is_locked": item.is_locked,
                "notes": "deterministic controlled mix from local spec",
            }
        )

    for item in spec.clean_controls:
        clean_spec = clean_by_id[item.clean_id]
        clean = clean_audio[item.clean_id]
        rows.append(
            {
                "sample_id": item.sample_id,
                "dataset_version": spec.dataset_version,
                "split": item.split,
                "speaker_id": clean_spec.speaker_id,
                "sentence_id": clean_spec.sentence_id,
                "reference_text": clean_spec.reference_text,
                "reference_normalized": normalize_reference(clean_spec.reference_text),
                "source_type": "clean_control",
                "clean_path": _relative(clean.path, root),
                "noise_path": "",
                "mixed_path": "",
                "noise_type": "",
                "noise_source": "",
                "consent_or_license": consent_or_license,
                "snr_db": "",
                "mix_seed": "",
                "noise_offset_seconds": "",
                "mix_alpha": "",
                "final_gain": "",
                "sample_rate": SAMPLE_RATE,
                "channels": CHANNELS,
                "duration_seconds": len(clean.samples) / SAMPLE_RATE,
                "recording_device": recording_device,
                "recording_distance_cm": "" if recording_distance_cm is None else recording_distance_cm,
                "sha256": sha256_file(clean.path),
                "is_demo_candidate": item.is_demo_candidate,
                "is_locked": item.is_locked,
                "notes": "clean control reused in place",
            }
        )

    for item in spec.real_recordings:
        real = real_audio.get(item.sample_id)
        if real is None:
            continue
        rows.append(
            {
                "sample_id": item.sample_id,
                "dataset_version": spec.dataset_version,
                "split": item.split,
                "speaker_id": item.speaker_id,
                "sentence_id": item.sentence_id,
                "reference_text": item.reference_text,
                "reference_normalized": normalize_reference(item.reference_text),
                "source_type": "real",
                "clean_path": "",
                "noise_path": "",
                "mixed_path": _relative(real.path, root),
                "noise_type": item.noise_type,
                "noise_source": "local_spec_recording",
                "consent_or_license": consent_or_license,
                "snr_db": "",
                "mix_seed": "",
                "noise_offset_seconds": "",
                "mix_alpha": "",
                "final_gain": "",
                "sample_rate": SAMPLE_RATE,
                "channels": CHANNELS,
                "duration_seconds": len(real.samples) / SAMPLE_RATE,
                "recording_device": recording_device,
                "recording_distance_cm": "" if recording_distance_cm is None else recording_distance_cm,
                "sha256": sha256_file(real.path),
                "is_demo_candidate": item.is_demo_candidate,
                "is_locked": item.is_locked,
                "notes": "real recording without aligned clean reference",
            }
        )

    _write_manifest_csv(manifest_csv, rows)
    _write_manifest_json(manifest_json, rows)
    return rows


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build generic AudioRescue dataset mixtures from a local spec.",
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("data_local"))
    parser.add_argument("--spec", type=Path, default=None, help="Local dataset spec JSON")
    parser.add_argument(
        "--example-spec",
        action="store_true",
        help="explicitly use the tracked synthetic example spec",
    )
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--allow-missing-real", action="store_true")
    parser.add_argument(
        "--consent-or-license",
        default=None,
        help="Consent/license token declared by the selected spec",
    )
    parser.add_argument("--recording-device", default="")
    parser.add_argument("--recording-distance-cm", type=float, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    try:
        spec = _load_spec(args.spec, example_mode=args.example_spec)
        if args.validate_only:
            _clean, _noise, real = load_and_validate_sources(
                args.dataset_root,
                spec=spec,
                example_mode=args.example_spec,
                allow_missing_real=args.allow_missing_real,
            )
            print(f"Validation passed for declared sources; real recordings loaded: {len(real)}")
            return 0

        rows = build_dataset(
            args.dataset_root,
            spec_path=args.spec,
            base_seed=args.base_seed,
            overwrite=args.overwrite,
            consent_or_license=args.consent_or_license,
            recording_device=args.recording_device,
            recording_distance_cm=args.recording_distance_cm,
            example_mode=args.example_spec,
            allow_missing_real=args.allow_missing_real,
        )
    except DatasetBuildError as exc:
        print("Dataset build refused: DATASET_INPUT_INVALID")
        return 2

    mixture_rows = [row for row in rows if row["source_type"] == "controlled_mix"]
    print(
        f"Build complete: mixtures={len(mixture_rows)}, manifest rows={len(rows)}."
    )
    print("CSV manifest: manifest.csv")
    print("JSON manifest: manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
