"""Audio decoding, inspection, and normalization for the A-owned backend."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Any

import numpy as np

from core.schemas import (
    AudioMeta,
    ErrorCode,
    InputAudioError,
    InputTooLongError,
    WarningItem,
)

MAX_DURATION_SECONDS = 60.0
MIN_DURATION_SECONDS = 1.0
MAX_INPUT_BYTES = 128 * 1024 * 1024
FFPROBE_TIMEOUT_SECONDS = 15.0
FFMPEG_TIMEOUT_SECONDS = 120.0
CLIP_THRESHOLD = 0.99
SILENCE_THRESHOLD = 1e-4
NEAR_SILENT_RMS_DBFS = -50.0


def normalize_audio(
    input_path: str,
    output_path: str,
    target_sr: int = 48_000,
    mono: bool = True,
) -> AudioMeta:
    """Normalize input audio to WAV / target sample rate / optional mono / PCM16."""

    source = Path(input_path)
    target = Path(output_path)
    if not source.exists() or not source.is_file():
        raise InputAudioError("输入音频不存在或不是文件", details={"path": str(source)})
    _validate_input_size(source)

    source_format = source.suffix.lower().lstrip(".") or None
    if source_format == "wav":
        header_duration = _wav_header_duration(source)
        _validate_duration(header_duration, source)
        samples, sample_rate, channels = _read_wav(source)
        original_sample_rate = sample_rate
        original_channels = channels
    else:
        probed_duration = _probe_non_wav(source)
        _validate_duration(probed_duration, source)
        samples, sample_rate, channels = _decode_with_ffmpeg(source, target_sr, mono)
        original_sample_rate = None
        original_channels = None

    duration_seconds = _duration_seconds(samples, sample_rate)
    _validate_duration(duration_seconds, source)

    normalized = np.asarray(samples, dtype=np.float32)
    if mono and normalized.ndim == 2:
        normalized = normalized.mean(axis=1)
        channels = 1
    if sample_rate != target_sr:
        normalized = _resample_linear(normalized, sample_rate, target_sr)
        sample_rate = target_sr
    elif mono:
        channels = 1

    _write_wav(target, normalized, sample_rate)
    written, written_sr, written_channels = _read_wav(target)
    report = inspect_audio(written, written_sr)

    return AudioMeta(
        source_name=source.name,
        sample_rate=written_sr,
        channels=written_channels,
        duration_seconds=report["duration_seconds"],
        peak_abs=report["peak_abs"],
        rms_dbfs=report["rms_dbfs"],
        clipped_ratio=report["clipped_ratio"],
        silent_ratio=report["silent_ratio"],
        normalized_path=str(target.resolve()),
        original_sample_rate=original_sample_rate,
        original_channels=original_channels,
        source_format=source_format,
    )


def inspect_audio(samples, sample_rate: int) -> dict[str, Any]:
    """Return basic audio diagnostics and recoverable warning items."""

    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim == 0:
        audio = audio.reshape(1)
    channels = int(audio.shape[1]) if audio.ndim == 2 else 1
    flat = audio.reshape(-1)
    duration_seconds = float(audio.shape[0] / sample_rate) if sample_rate > 0 else 0.0

    if flat.size == 0:
        peak_abs = 0.0
        rms_dbfs = None
        clipped_ratio = 0.0
        silent_ratio = None
    else:
        abs_flat = np.abs(flat)
        peak_abs = float(np.max(abs_flat))
        clipped_ratio = float(np.mean(abs_flat >= CLIP_THRESHOLD))
        silent_ratio = float(np.mean(abs_flat <= SILENCE_THRESHOLD))
        rms = float(np.sqrt(np.mean(np.square(flat, dtype=np.float64))))
        rms_dbfs = 20.0 * math.log10(rms) if rms > 0 else None

    warnings: list[WarningItem] = []
    if clipped_ratio > 0:
        warnings.append(
            _warning(
                ErrorCode.INPUT_CLIPPED,
                "输入音频存在接近满幅的样本，可能已经削波",
                details={"clipped_ratio": clipped_ratio},
            )
        )
    if rms_dbfs is None or rms_dbfs < NEAR_SILENT_RMS_DBFS or peak_abs <= SILENCE_THRESHOLD:
        warnings.append(
            _warning(
                ErrorCode.INPUT_NEAR_SILENT,
                "输入音频接近静音，转写可能为空",
                details={"rms_dbfs": rms_dbfs, "peak_abs": peak_abs},
            )
        )

    return {
        "sample_rate": int(sample_rate),
        "channels": channels,
        "duration_seconds": duration_seconds,
        "peak_abs": peak_abs,
        "rms_dbfs": rms_dbfs,
        "clipped_ratio": clipped_ratio,
        "silent_ratio": silent_ratio,
        "warnings": warnings,
    }


def _warning(
    code: ErrorCode,
    message: str,
    *,
    details: dict[str, Any] | None = None,
) -> WarningItem:
    return WarningItem(
        code=code,
        message=message,
        module="audio_io",
        recoverable=True,
        details=details or {},
    )


def _decode_with_ffmpeg(source: Path, target_sr: int, mono: bool) -> tuple[np.ndarray, int, int]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise InputAudioError(
            "当前环境缺少 ffmpeg，暂不能解码非 WAV 输入",
            details={"path": str(source), "source_format": source.suffix.lower()},
        )

    with tempfile.TemporaryDirectory() as temp_dir:
        decoded = Path(temp_dir) / "decoded.wav"
        command = [
            ffmpeg,
            "-nostdin",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-ar",
            str(target_sr),
            "-sample_fmt",
            "s16",
        ]
        if mono:
            command.extend(["-ac", "1"])
        command.append(str(decoded))
        try:
            completed = subprocess.run(
                command,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                check=False,
                timeout=FFMPEG_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            raise InputAudioError(
                "输入音频解码超时",
                details={
                    "path": str(source),
                    "timeout_seconds": FFMPEG_TIMEOUT_SECONDS,
                },
            ) from exc
        except OSError as exc:
            raise InputAudioError(
                "无法启动 ffmpeg 解码输入音频",
                detail=str(exc),
                details={"path": str(source)},
            ) from exc
        if completed.returncode != 0:
            raise InputAudioError(
                "输入音频无法解码",
                detail=_bounded_detail(completed.stderr),
                details={"path": str(source)},
            )
        return _read_wav(decoded)


def _validate_input_size(source: Path) -> None:
    try:
        size_bytes = source.stat().st_size
    except OSError as exc:
        raise InputAudioError(
            "无法读取输入音频文件信息",
            detail=str(exc),
            details={"path": str(source)},
        ) from exc
    if size_bytes == 0:
        raise InputAudioError("输入音频为空文件", details={"path": str(source)})
    if size_bytes > MAX_INPUT_BYTES:
        raise InputAudioError(
            "输入音频文件过大",
            details={
                "path": str(source),
                "size_bytes": size_bytes,
                "max_size_bytes": MAX_INPUT_BYTES,
            },
        )


def _validate_duration(duration_seconds: float, source: Path) -> None:
    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise InputAudioError(
            "输入音频时长无效，无法处理",
            details={"path": str(source), "duration_seconds": duration_seconds},
        )
    if duration_seconds < MIN_DURATION_SECONDS:
        raise InputAudioError(
            "输入音频短于 1 秒，无法稳定处理",
            details={"path": str(source), "duration_seconds": duration_seconds},
        )
    if duration_seconds > MAX_DURATION_SECONDS:
        raise InputTooLongError(
            "输入音频超过 60 秒，请裁剪后重试",
            details={"path": str(source), "duration_seconds": duration_seconds},
        )


def _wav_header_duration(source: Path) -> float:
    """Read only the WAV header so oversized audio is rejected before allocation."""

    try:
        with wave.open(str(source), "rb") as handle:
            if handle.getcomptype() != "NONE":
                raise InputAudioError(
                    "暂不支持压缩 WAV 输入", details={"path": str(source)}
                )
            sample_rate = handle.getframerate()
            channels = handle.getnchannels()
            frame_count = handle.getnframes()
    except InputAudioError:
        raise
    except (EOFError, OSError, wave.Error) as exc:
        raise InputAudioError(
            "输入 WAV 头无法读取",
            detail=str(exc),
            details={"path": str(source)},
        ) from exc

    if sample_rate <= 0 or channels <= 0 or frame_count <= 0:
        raise InputAudioError(
            "输入 WAV 头包含无效参数",
            details={
                "path": str(source),
                "sample_rate": sample_rate,
                "channels": channels,
                "frames": frame_count,
            },
        )
    return float(frame_count / sample_rate)


def _probe_non_wav(source: Path) -> float:
    """Probe duration before decoding a non-WAV upload."""

    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise InputAudioError(
            "当前环境缺少 ffprobe，无法安全预检非 WAV 输入",
            details={"path": str(source), "source_format": source.suffix.lower()},
        )
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "format=duration,size:stream=duration",
        "-of",
        "json",
        str(source),
    ]
    try:
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=False,
            timeout=FFPROBE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise InputAudioError(
            "输入音频预检超时",
            details={
                "path": str(source),
                "timeout_seconds": FFPROBE_TIMEOUT_SECONDS,
            },
        ) from exc
    except OSError as exc:
        raise InputAudioError(
            "无法启动 ffprobe 预检输入音频",
            detail=str(exc),
            details={"path": str(source)},
        ) from exc

    if completed.returncode != 0:
        raise InputAudioError(
            "输入音频无法读取或不包含音频流",
            detail=_bounded_detail(completed.stderr),
            details={"path": str(source)},
        )
    try:
        payload = json.loads(completed.stdout)
    except (TypeError, json.JSONDecodeError) as exc:
        raise InputAudioError(
            "ffprobe 未返回可用的音频元数据",
            detail=str(exc),
            details={"path": str(source)},
        ) from exc

    durations: list[float] = []
    format_payload = payload.get("format", {}) if isinstance(payload, dict) else {}
    streams_payload = payload.get("streams", []) if isinstance(payload, dict) else []
    if not isinstance(streams_payload, list) or not any(
        isinstance(stream, dict) for stream in streams_payload
    ):
        raise InputAudioError(
            "输入文件不包含可用的音频流",
            details={"path": str(source)},
        )
    if isinstance(format_payload, dict):
        _append_duration(durations, format_payload.get("duration"))
        reported_size = _positive_finite_number(format_payload.get("size"))
        if reported_size is not None and reported_size > MAX_INPUT_BYTES:
            raise InputAudioError(
                "输入音频文件过大",
                details={
                    "path": str(source),
                    "probed_size_bytes": int(reported_size),
                    "max_size_bytes": MAX_INPUT_BYTES,
                },
            )
    if isinstance(streams_payload, list):
        for stream in streams_payload:
            if isinstance(stream, dict):
                _append_duration(durations, stream.get("duration"))
    if not durations:
        raise InputAudioError(
            "ffprobe 无法确定输入音频时长",
            details={"path": str(source)},
        )
    return max(durations)


def _append_duration(durations: list[float], raw_value: Any) -> None:
    value = _positive_finite_number(raw_value)
    if value is not None:
        durations.append(value)


def _positive_finite_number(raw_value: Any) -> float | None:
    try:
        value = float(raw_value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def _bounded_detail(value: Any, limit: int = 500) -> str | None:
    detail = str(value or "").strip()
    if not detail:
        return None
    return detail[:limit]


def _read_wav(path: str | Path) -> tuple[np.ndarray, int, int]:
    try:
        with wave.open(str(path), "rb") as handle:
            if handle.getcomptype() != "NONE":
                raise InputAudioError("暂不支持压缩 WAV 输入", details={"path": str(path)})
            channels = handle.getnchannels()
            sample_rate = handle.getframerate()
            sample_width = handle.getsampwidth()
            frame_count = handle.getnframes()
            raw = handle.readframes(frame_count)
    except InputAudioError:
        raise
    except Exception as exc:
        raise InputAudioError(
            "输入音频无法读取",
            detail=str(exc),
            details={"path": str(path)},
        ) from exc

    if sample_rate <= 0 or channels <= 0:
        raise InputAudioError(
            "输入 WAV 包含无效参数",
            details={
                "path": str(path),
                "sample_rate": sample_rate,
                "channels": channels,
            },
        )
    if frame_count == 0 or not raw:
        raise InputAudioError("输入音频为空或没有可读帧", details={"path": str(path)})
    expected_bytes = frame_count * channels * sample_width
    if len(raw) != expected_bytes:
        raise InputAudioError(
            "输入 WAV 帧数据不完整",
            details={
                "path": str(path),
                "expected_bytes": expected_bytes,
                "actual_bytes": len(raw),
            },
        )

    try:
        if sample_width == 1:
            data = (
                np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0
            ) / 128.0
        elif sample_width == 2:
            data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        elif sample_width == 4:
            data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
        else:
            raise InputAudioError(
                "暂不支持该 WAV 位深",
                details={"path": str(path), "sample_width": sample_width},
            )

        if channels > 1:
            data = data.reshape(-1, channels)
    except InputAudioError:
        raise
    except (TypeError, ValueError) as exc:
        raise InputAudioError(
            "输入 WAV 帧数据无法解码",
            detail=str(exc),
            details={"path": str(path)},
        ) from exc
    return np.asarray(data, dtype=np.float32), int(sample_rate), int(channels)


def _write_wav(path: str | Path, samples: np.ndarray, sample_rate: int) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    audio = np.asarray(samples, dtype=np.float32)
    if audio.size == 0:
        raise InputAudioError("输出音频为空", details={"path": str(path)})
    if not np.all(np.isfinite(audio)):
        raise InputAudioError("输出音频包含 NaN 或 Inf", details={"path": str(path)})

    if audio.ndim == 1:
        channels = 1
        interleaved = audio
    elif audio.ndim == 2:
        channels = audio.shape[1]
        interleaved = audio.reshape(-1)
    else:
        raise InputAudioError("音频数组维度不受支持", details={"shape": list(audio.shape)})

    pcm = np.clip(interleaved, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    with wave.open(str(target), "wb") as handle:
        handle.setnchannels(int(channels))
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(pcm.tobytes())


def _duration_seconds(samples: np.ndarray, sample_rate: int) -> float:
    if sample_rate <= 0:
        return 0.0
    return float(np.asarray(samples).shape[0] / sample_rate)


def _resample_linear(samples: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    audio = np.asarray(samples, dtype=np.float32)
    if source_sr == target_sr:
        return audio.copy()
    target_frames = max(1, int(round(audio.shape[0] * target_sr / source_sr)))
    source_positions = np.linspace(0.0, audio.shape[0] - 1, num=audio.shape[0])
    target_positions = np.linspace(0.0, audio.shape[0] - 1, num=target_frames)
    if audio.ndim == 1:
        return np.interp(target_positions, source_positions, audio).astype(np.float32)
    channels = [
        np.interp(target_positions, source_positions, audio[:, index])
        for index in range(audio.shape[1])
    ]
    return np.stack(channels, axis=1).astype(np.float32)


__all__ = ["inspect_audio", "normalize_audio"]
