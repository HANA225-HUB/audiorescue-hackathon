"""Build the deterministic AudioRescue-CN-Mini-v1 controlled mixtures.

The script intentionally uses only Python's standard library.  Source WAVs
must already be 48 kHz, mono, uncompressed PCM16; format conversion belongs to
the normalization stage and is never performed silently here.
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


DATASET_VERSION = "AudioRescue-CN-Mini-v1"
PENDING_CONSENT_TOKEN = "pending_team_confirmation"
APPROVED_CONSENT_TOKEN = "team-approved-for-competition-evaluation"
ALLOWED_CONSENT_TOKENS = frozenset(
    {PENDING_CONSENT_TOKEN, APPROVED_CONSENT_TOKEN}
)
SAMPLE_RATE = 48_000
CHANNELS = 1
SAMPLE_WIDTH_BYTES = 2
PEAK_LIMIT = 0.98
DEFAULT_BASE_SEED = 20_260_714

SENTENCES = {
    "S01": "今天下午三点，我们在实验室讨论语音处理项目的最终方案。",
    "S02": "请记录会议中的三个重点：数据来源、模型效果和系统稳定性。",
    "S03": "如果现场网络中断，系统仍然可以在本地完成音频增强和文字转写。",
}

LATIN_SQUARE = {
    ("A", "S01"): "fan",
    ("A", "S02"): "keyboard",
    ("A", "S03"): "traffic",
    ("B", "S01"): "keyboard",
    ("B", "S02"): "traffic",
    ("B", "S03"): "fan",
    ("C", "S01"): "traffic",
    ("C", "S02"): "fan",
    ("C", "S03"): "keyboard",
}

SNR_VARIANTS = ((5, "snrp05"), (0, "snr000"), (-5, "snrm05"))
DEMO_SAMPLE_IDS = {
    "mix_spkB_s03_fan_snr000",
    "mix_spkC_s03_keyboard_snrm05",
    "clean_spkA_s03",
    "real_spkC_traffic_r01",
}

REAL_RECORDINGS = {
    "A": ("S01", "fan"),
    "B": ("S02", "keyboard"),
    "C": ("S03", "traffic"),
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


def _verify_formal_consent_ledger(
    root: Path,
    clean_audio: dict[tuple[str, str], WavData],
    noise_audio: dict[str, WavData],
    real_audio: dict[str, WavData],
) -> None:
    """Bind the formal token to all 15 approved masters and their hashes."""

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
                missing = ", ".join(sorted(required_fields - fields))
                raise DatasetBuildError(
                    f"正式授权台账缺少列：{missing}（{ledger}）"
                )
            rows = list(reader)
    except DatasetBuildError:
        raise
    except (OSError, UnicodeError, csv.Error) as exc:
        raise DatasetBuildError(f"无法读取正式授权台账 {ledger}：{exc}") from exc

    expected: dict[str, Path] = {}
    for (speaker, sentence), wav in clean_audio.items():
        expected[f"clean_spk{speaker}_{sentence.lower()}"] = wav.path.resolve()
    for noise_type, wav in noise_audio.items():
        expected[f"noise_{noise_type}_take01"] = wav.path.resolve()
    for speaker, wav in real_audio.items():
        _sentence, noise_type = REAL_RECORDINGS[speaker]
        expected[f"real_spk{speaker}_{noise_type}_r01"] = wav.path.resolve()

    by_id: dict[str, dict[str, str]] = {}
    for line_number, row in enumerate(rows, start=2):
        asset_id = (row.get("asset_id") or "").strip()
        if not asset_id or asset_id in by_id:
            raise DatasetBuildError(
                f"正式授权台账第 {line_number} 行 asset_id 为空或重复：{asset_id!r}"
            )
        by_id[asset_id] = row
    if set(by_id) != set(expected):
        missing = sorted(set(expected) - set(by_id))
        extra = sorted(set(by_id) - set(expected))
        raise DatasetBuildError(
            "正式授权台账必须精确覆盖 15 条母带；"
            f"缺少={missing}，多余={extra}"
        )

    source_root = (root / "source_original").resolve()
    for asset_id, standardized_path in expected.items():
        row = by_id[asset_id]
        if (row.get("consent_status") or "").strip().lower() != "yes":
            raise DatasetBuildError(
                f"{asset_id} 未明确授权比赛评测；recording_metadata.csv "
                "的 consent_status 必须为 yes"
            )
        relative_path = (row.get("relative_path") or "").strip()
        try:
            ledger_standardized = (root / relative_path).resolve()
            ledger_standardized.relative_to(root.resolve())
        except (OSError, RuntimeError, ValueError) as exc:
            raise DatasetBuildError(f"{asset_id} 的 relative_path 非法") from exc
        if ledger_standardized != standardized_path:
            raise DatasetBuildError(
                f"{asset_id} 的 relative_path 与固定数据路径不一致"
            )

        original_value = (row.get("source_original_path") or "").strip()
        if not original_value:
            raise DatasetBuildError(f"{asset_id} 缺少 source_original_path")
        entered = Path(original_value).expanduser()
        original_path = (entered if entered.is_absolute() else root / entered).resolve()
        try:
            original_path.relative_to(source_root)
        except ValueError as exc:
            raise DatasetBuildError(
                f"{asset_id} 的母带必须位于 {source_root}"
            ) from exc
        if not original_path.is_file():
            raise DatasetBuildError(f"{asset_id} 的母带不存在：{original_path}")

        original_hash = (row.get("source_original_sha256") or "").strip().lower()
        standardized_hash = (row.get("standardized_sha256") or "").strip().lower()
        for field_name, digest in (
            ("source_original_sha256", original_hash),
            ("standardized_sha256", standardized_hash),
        ):
            if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
                raise DatasetBuildError(f"{asset_id} 的 {field_name} 不是有效 SHA-256")
        if sha256_file(original_path) != original_hash:
            raise DatasetBuildError(f"{asset_id} 的原始母带 SHA-256 不匹配")
        if sha256_file(standardized_path) != standardized_hash:
            raise DatasetBuildError(f"{asset_id} 的标准化 WAV SHA-256 不匹配")


def normalize_reference(text: str) -> str:
    """Match the frozen CER normalization: NFKC, lowercase, alphanumerics only."""

    normalized = unicodedata.normalize("NFKC", text).lower()
    return "".join(character for character in normalized if character.isalnum())


def read_pcm16_mono(path: str | Path) -> WavData:
    """Read a contract-compliant WAV or fail with an actionable message."""

    resolved = Path(path)
    if not resolved.is_file():
        raise DatasetBuildError(f"缺少源文件：{resolved}")

    try:
        with wave.open(str(resolved), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            compression = wav_file.getcomptype()
            frame_count = wav_file.getnframes()
            raw_frames = wav_file.readframes(frame_count)
    except (EOFError, OSError, wave.Error) as exc:
        raise DatasetBuildError(f"无法读取 WAV：{resolved}（{exc}）") from exc

    problems = []
    if sample_rate != SAMPLE_RATE:
        problems.append(f"采样率 {sample_rate} Hz")
    if channels != CHANNELS:
        problems.append(f"声道数 {channels}")
    if sample_width != SAMPLE_WIDTH_BYTES:
        problems.append(f"位深 {sample_width * 8} bit")
    if compression != "NONE":
        problems.append(f"压缩类型 {compression}")
    if problems:
        actual = "、".join(problems)
        raise DatasetBuildError(
            f"{resolved} 不符合 48kHz/mono/PCM16（实际：{actual}）。"
            "请先显式标准化；构建脚本不会静默重采样或下混。"
        )

    samples = array("h")
    samples.frombytes(raw_frames)
    if sys.byteorder == "big":  # WAV PCM is little-endian.
        samples.byteswap()
    if len(samples) != frame_count:
        raise DatasetBuildError(f"WAV 帧数不完整：{resolved}")
    if not samples:
        raise DatasetBuildError(f"WAV 为空：{resolved}")
    return WavData(path=resolved, samples=samples, sample_rate=sample_rate)


def _center_and_power(samples: Sequence[int]) -> tuple[array, float]:
    if not samples:
        raise DatasetBuildError("不能对空信号混音")
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
        raise DatasetBuildError("clean 与 noise 片段长度必须一致")
    if clean_power <= 1e-12:
        raise DatasetBuildError("clean 近似静音，无法定义 SNR；请重新录制")
    if noise_power <= 1e-12:
        raise DatasetBuildError("noise 近似静音，无法定义 SNR；请重新录制")
    if not math.isfinite(snr_db):
        raise DatasetBuildError("SNR 必须是有限数值")
    if not 0 < peak_limit <= 1:
        raise DatasetBuildError("peak_limit 必须在 (0, 1] 范围内")

    alpha = math.sqrt(clean_power / (noise_power * (10.0 ** (snr_db / 10.0))))
    peak_pcm = max(
        abs(clean_value + alpha * noise_value)
        for clean_value, noise_value in zip(clean, noise)
    )
    peak = peak_pcm / 32768.0
    final_gain = 1.0 if peak <= peak_limit else peak_limit / peak

    # Round to nearest PCM count, then enforce the continuous 0.98 limit after
    # quantization.  The same final_gain multiplies both components, so SNR is
    # unchanged by peak protection.
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
        raise DatasetBuildError("noise_offset_samples 不能为负数")
    stop = noise_offset_samples + len(clean_samples)
    if stop > len(noise_samples):
        raise DatasetBuildError(
            "noise 长度不足，不能循环填充；请录制更长噪声或选择新的片段"
        )

    clean_centered, clean_power = _center_and_power(clean_samples)
    noise_centered, noise_power = _center_and_power(
        noise_samples[noise_offset_samples:stop]
    )
    return _mix_centered(
        clean_centered,
        noise_centered,
        clean_power,
        noise_power,
        float(snr_db),
        peak_limit,
    )


def derive_mix_seed(
    base_seed: int,
    speaker_id: str,
    sentence_id: str,
    noise_type: str,
) -> int:
    """Derive a stable per-cell seed without Python's randomized hash()."""

    key = f"{base_seed}|{speaker_id}|{sentence_id}|{noise_type}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big")


def choose_noise_offset(mix_seed: int, max_offset_samples: int) -> int:
    if max_offset_samples < 0:
        raise DatasetBuildError("noise 比 clean 短，不能生成非循环片段")
    if max_offset_samples == 0:
        return 0
    digest = hashlib.sha256(f"{mix_seed}|noise-offset".encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big") % (max_offset_samples + 1)


def _clean_path(dataset_root: Path, speaker_id: str, sentence_id: str) -> Path:
    return (
        dataset_root
        / "raw"
        / "clean"
        / f"spk{speaker_id}"
        / f"clean_spk{speaker_id}_{sentence_id.lower()}.wav"
    )


def _noise_path(dataset_root: Path, noise_type: str) -> Path:
    return dataset_root / "raw" / "noise" / f"noise_{noise_type}_take01.wav"


def _real_path(dataset_root: Path, speaker_id: str, noise_type: str) -> Path:
    return dataset_root / "raw" / "real" / f"real_spk{speaker_id}_{noise_type}_r01.wav"


def _mixed_path(
    dataset_root: Path,
    split: str,
    speaker_id: str,
    sentence_id: str,
    noise_type: str,
    snr_suffix: str,
) -> Path:
    filename = (
        f"mix_spk{speaker_id}_{sentence_id.lower()}_"
        f"{noise_type}_{snr_suffix}.wav"
    )
    return dataset_root / "controlled" / split / filename


def _relative(path: Path, dataset_root: Path) -> str:
    return path.resolve().relative_to(dataset_root.resolve()).as_posix()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_pcm16_mono(path: str | Path, samples: Sequence[int]) -> None:
    """Atomically write a 48 kHz mono PCM16 WAV."""

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
    allow_missing_real: bool = False,
) -> tuple[
    dict[tuple[str, str], WavData],
    dict[str, WavData],
    dict[str, WavData],
]:
    """Load 9 clean, 3 noise and normally all 3 real recordings."""

    root = Path(dataset_root)
    clean_audio = {}
    for speaker_id in ("A", "B", "C"):
        for sentence_id in SENTENCES:
            path = _clean_path(root, speaker_id, sentence_id)
            clean_audio[(speaker_id, sentence_id)] = read_pcm16_mono(path)

    noise_audio = {}
    for noise_type in ("fan", "keyboard", "traffic"):
        path = _noise_path(root, noise_type)
        noise_audio[noise_type] = read_pcm16_mono(path)

    real_audio = {}
    for speaker_id, (_sentence_id, noise_type) in REAL_RECORDINGS.items():
        path = _real_path(root, speaker_id, noise_type)
        if allow_missing_real and not path.is_file():
            continue
        real_audio[speaker_id] = read_pcm16_mono(path)

    for (speaker_id, sentence_id), clean in clean_audio.items():
        noise_type = LATIN_SQUARE[(speaker_id, sentence_id)]
        noise = noise_audio[noise_type]
        if len(noise.samples) < len(clean.samples):
            raise DatasetBuildError(
                f"{noise.path} 比 {clean.path} 短；手册禁止循环噪声，请录制更长片段"
            )
    return clean_audio, noise_audio, real_audio


def build_dataset(
    dataset_root: str | Path,
    *,
    base_seed: int = DEFAULT_BASE_SEED,
    overwrite: bool = False,
    consent_or_license: str = PENDING_CONSENT_TOKEN,
    recording_device: str = "",
    recording_distance_cm: float | None = None,
    allow_missing_real: bool = False,
) -> list[dict[str, Any]]:
    """Generate 27 mixtures and manifests for every available runnable case.

    By default, the frozen manifest is complete: 27 controlled mixtures, three
    S03 clean controls, and three real noisy recordings.  The explicit
    ``allow_missing_real`` escape hatch exists only for the handbook's early
    mixing phase before the three real recordings have been captured.
    """

    if (
        not isinstance(consent_or_license, str)
        or consent_or_license not in ALLOWED_CONSENT_TOKENS
    ):
        allowed = ", ".join(sorted(ALLOWED_CONSENT_TOKENS))
        raise DatasetBuildError(
            "consent_or_license must use one exact workflow token: " + allowed
        )

    root = Path(dataset_root)
    clean_audio, noise_audio, real_audio = load_and_validate_sources(
        root,
        allow_missing_real=allow_missing_real,
    )
    if consent_or_license == APPROVED_CONSENT_TOKEN:
        _verify_formal_consent_ledger(
            root,
            clean_audio,
            noise_audio,
            real_audio,
        )

    output_paths = []
    for speaker_id in ("A", "B", "C"):
        for sentence_id in SENTENCES:
            split = "locked_test" if sentence_id == "S03" else "dev"
            noise_type = LATIN_SQUARE[(speaker_id, sentence_id)]
            for _snr_db, snr_suffix in SNR_VARIANTS:
                output_paths.append(
                    _mixed_path(
                        root,
                        split,
                        speaker_id,
                        sentence_id,
                        noise_type,
                        snr_suffix,
                    )
                )
    manifest_csv = root / "manifest.csv"
    manifest_json = root / "manifest.json"
    collisions = [
        path for path in (*output_paths, manifest_csv, manifest_json) if path.exists()
    ]
    if collisions and not overwrite:
        preview = ", ".join(str(path) for path in collisions[:3])
        extra = " …" if len(collisions) > 3 else ""
        raise DatasetBuildError(
            f"目标文件已存在：{preview}{extra}。如确认重建，请显式使用 --overwrite。"
        )

    rows: list[dict[str, Any]] = []
    for speaker_id in ("A", "B", "C"):
        for sentence_id, reference_text in SENTENCES.items():
            split = "locked_test" if sentence_id == "S03" else "dev"
            noise_type = LATIN_SQUARE[(speaker_id, sentence_id)]
            clean = clean_audio[(speaker_id, sentence_id)]
            noise = noise_audio[noise_type]
            mix_seed = derive_mix_seed(
                base_seed,
                speaker_id,
                sentence_id,
                noise_type,
            )
            max_offset = len(noise.samples) - len(clean.samples)
            noise_offset_samples = choose_noise_offset(mix_seed, max_offset)

            clean_centered, clean_power = _center_and_power(clean.samples)
            noise_stop = noise_offset_samples + len(clean.samples)
            noise_centered, noise_power = _center_and_power(
                noise.samples[noise_offset_samples:noise_stop]
            )

            for snr_db, snr_suffix in SNR_VARIANTS:
                mixed_path = _mixed_path(
                    root,
                    split,
                    speaker_id,
                    sentence_id,
                    noise_type,
                    snr_suffix,
                )
                result = _mix_centered(
                    clean_centered,
                    noise_centered,
                    clean_power,
                    noise_power,
                    snr_db,
                    PEAK_LIMIT,
                )
                write_pcm16_mono(mixed_path, result.samples)

                sample_id = mixed_path.stem
                rows.append(
                    {
                        "sample_id": sample_id,
                        "dataset_version": DATASET_VERSION,
                        "split": split,
                        "speaker_id": speaker_id,
                        "sentence_id": sentence_id,
                        "reference_text": reference_text,
                        "reference_normalized": normalize_reference(reference_text),
                        "source_type": "controlled_mix",
                        "clean_path": _relative(clean.path, root),
                        "noise_path": _relative(noise.path, root),
                        "mixed_path": _relative(mixed_path, root),
                        "noise_type": noise_type,
                        "noise_source": "team_recorded_take01",
                        "consent_or_license": consent_or_license,
                        "snr_db": snr_db,
                        "mix_seed": mix_seed,
                        "noise_offset_seconds": noise_offset_samples / SAMPLE_RATE,
                        "mix_alpha": result.alpha,
                        "final_gain": result.final_gain,
                        "sample_rate": SAMPLE_RATE,
                        "channels": CHANNELS,
                        "duration_seconds": len(result.samples) / SAMPLE_RATE,
                        "recording_device": recording_device,
                        "recording_distance_cm": (
                            "" if recording_distance_cm is None else recording_distance_cm
                        ),
                        "sha256": sha256_file(mixed_path),
                        "is_demo_candidate": sample_id in DEMO_SAMPLE_IDS,
                        "is_locked": split == "locked_test",
                        "notes": (
                            "deterministic controlled mix; one shared noise segment "
                            "for all SNR variants"
                        ),
                    }
                )

    # S03 clean controls are runnable cases but reuse the masters in-place.
    for speaker_id in ("A", "B", "C"):
        sentence_id = "S03"
        reference_text = SENTENCES[sentence_id]
        clean = clean_audio[(speaker_id, sentence_id)]
        sample_id = clean.path.stem
        rows.append(
            {
                "sample_id": sample_id,
                "dataset_version": DATASET_VERSION,
                "split": "clean_control",
                "speaker_id": speaker_id,
                "sentence_id": sentence_id,
                "reference_text": reference_text,
                "reference_normalized": normalize_reference(reference_text),
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
                "recording_distance_cm": (
                    "" if recording_distance_cm is None else recording_distance_cm
                ),
                "sha256": sha256_file(clean.path),
                "is_demo_candidate": sample_id in DEMO_SAMPLE_IDS,
                "is_locked": True,
                "notes": "S03 clean control; reused in place and never mixed",
            }
        )

    # Real recordings have a reference transcript but no aligned clean/noise
    # pair, so SNR and waveform-reference metrics remain deliberately blank.
    for speaker_id in ("A", "B", "C"):
        real = real_audio.get(speaker_id)
        if real is None:
            continue
        sentence_id, noise_type = REAL_RECORDINGS[speaker_id]
        reference_text = SENTENCES[sentence_id]
        sample_id = real.path.stem
        rows.append(
            {
                "sample_id": sample_id,
                "dataset_version": DATASET_VERSION,
                "split": "real",
                "speaker_id": speaker_id,
                "sentence_id": sentence_id,
                "reference_text": reference_text,
                "reference_normalized": normalize_reference(reference_text),
                "source_type": "real",
                "clean_path": "",
                "noise_path": "",
                "mixed_path": _relative(real.path, root),
                "noise_type": noise_type,
                "noise_source": "simultaneous_team_recording",
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
                "recording_distance_cm": (
                    "" if recording_distance_cm is None else recording_distance_cm
                ),
                "sha256": sha256_file(real.path),
                "is_demo_candidate": sample_id in DEMO_SAMPLE_IDS,
                "is_locked": False,
                "notes": (
                    "real noisy recording without aligned clean reference; "
                    "do not compute PESQ, STOI, or measured SNR"
                ),
            }
        )

    _write_manifest_csv(manifest_csv, rows)
    _write_manifest_json(manifest_json, rows)
    return rows


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "从 9 条 clean 和 3 条 noise 构建 AudioRescue-CN-Mini-v1 的 "
            "27 条确定性混合音频。"
        ),
        epilog=(
            "源文件必须已是 48kHz/mono/PCM16。默认不会覆盖已有混音或 manifest。"
        ),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data_local"),
        help="数据集根目录（默认：data_local）",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=DEFAULT_BASE_SEED,
        help=f"确定性种子（默认：{DEFAULT_BASE_SEED}）",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="显式覆盖脚本生成的 27 条混音和两个 manifest",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="只检查 15 个源 WAV 的名称、格式和长度，不生成文件",
    )
    parser.add_argument(
        "--allow-missing-real",
        action="store_true",
        help=(
            "仅用于先混音后录 real 的中间阶段；最终冻结 manifest 前必须去掉此开关重建"
        ),
    )
    parser.add_argument(
        "--consent-or-license",
        choices=sorted(ALLOWED_CONSENT_TOKENS),
        default=PENDING_CONSENT_TOKEN,
        help=(
            "写入所有生成行的授权 token；默认 pending 仅供开发，"
            "正式 validator 只接受已确认的比赛评测 token"
        ),
    )
    parser.add_argument(
        "--recording-device",
        default="",
        help="可选：写入 manifest 的统一录音设备说明",
    )
    parser.add_argument(
        "--recording-distance-cm",
        type=float,
        default=None,
        help="可选：写入 manifest 的统一麦克风距离",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    try:
        if args.validate_only:
            _clean, _noise, real = load_and_validate_sources(
                args.dataset_root,
                allow_missing_real=args.allow_missing_real,
            )
            print(
                "验证通过：9 条 clean、3 条 noise"
                f" 与 {len(real)} 条 real 均为 48kHz/mono/PCM16。"
            )
            return 0

        rows = build_dataset(
            args.dataset_root,
            base_seed=args.base_seed,
            overwrite=args.overwrite,
            consent_or_license=args.consent_or_license,
            recording_device=args.recording_device,
            recording_distance_cm=args.recording_distance_cm,
            allow_missing_real=args.allow_missing_real,
        )
    except DatasetBuildError as exc:
        parser.exit(2, f"数据集构建失败：{exc}\n")

    mixture_rows = [row for row in rows if row["source_type"] == "controlled_mix"]
    dev_count = sum(row["split"] == "dev" for row in mixture_rows)
    locked_count = sum(row["split"] == "locked_test" for row in mixture_rows)
    print(
        f"构建完成：{len(mixture_rows)} 条混音（dev={dev_count}, "
        f"locked_test={locked_count}），manifest 共 {len(rows)} 行。"
    )
    print(f"CSV manifest：{args.dataset_root / 'manifest.csv'}")
    print(f"JSON manifest：{args.dataset_root / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
