"""Whisper transcription wrapper for the A-owned backend."""

from __future__ import annotations

import hashlib
import os
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from core.schemas import ASRInferenceError, TranscriptResult, TranscriptSegment

DEFAULT_MODEL_NAME = "base"
DEFAULT_DEVICE = "auto"

_ASR_MODEL: Any | None = None
_ASR_KEY: tuple[str, str] | None = None
_ASR_FINGERPRINT_KEY: tuple[tuple[str, str], ...] | None = None
_ASR_LOCK = threading.Lock()


def load_asr(
    model_name: str,
    device: str,
    *,
    expected_fingerprint: Mapping[str, str] | None = None,
):
    """Load and memoize one Whisper model for this process."""

    global _ASR_MODEL, _ASR_KEY, _ASR_FINGERPRINT_KEY
    normalized_model = _validated_text(model_name, "model_name")
    resolved_device = _resolve_device(device)
    key = (normalized_model, resolved_device)
    expected_key = _fingerprint_key(expected_fingerprint)
    with _ASR_LOCK:
        if expected_fingerprint is not None:
            verify_asr_model_fingerprint(
                expected_fingerprint, normalized_model, resolved_device
            )
        if _ASR_MODEL is not None and _ASR_KEY == key:
            return _ASR_MODEL
        try:
            import whisper
        except Exception as exc:  # pragma: no cover - depends on optional runtime deps.
            raise ASRInferenceError(
                "Whisper 依赖不可用，无法加载转写模型",
                detail=str(exc),
            ) from exc
        try:
            loaded_model = whisper.load_model(
                normalized_model, device=resolved_device
            )
        except Exception as exc:  # pragma: no cover - model availability is environment-specific.
            raise ASRInferenceError("Whisper 模型加载失败", detail=str(exc)) from exc
        _ASR_MODEL = loaded_model
        _ASR_KEY = key
        _ASR_FINGERPRINT_KEY = expected_key
        return loaded_model


def transcribe_audio(
    audio_path: str,
    language: str = "zh",
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    device: str = DEFAULT_DEVICE,
) -> TranscriptResult:
    """Transcribe with the model/device identity supplied by the pipeline."""

    path = Path(audio_path)
    if not path.exists() or not path.is_file() or path.stat().st_size == 0:
        raise ASRInferenceError("转写输入音频不存在或为空", details={"path": str(path)})

    normalized_model = _validated_text(model_name, "model_name")
    resolved_device = _resolve_device(device)
    model = load_asr(normalized_model, resolved_device)
    start = time.perf_counter()
    try:
        payload = model.transcribe(
            str(path),
            language=language,
            task="transcribe",
            temperature=0.0,
            condition_on_previous_text=False,
            initial_prompt=None,
            fp16=resolved_device.startswith("cuda"),
        )
    except ASRInferenceError:
        raise
    except Exception as exc:
        raise ASRInferenceError("ASR 转写失败", detail=str(exc)) from exc

    return TranscriptResult(
        text=str(payload.get("text", "")),
        language=payload.get("language") or language,
        segments=_serialize_segments(payload.get("segments", [])),
        runtime_seconds=time.perf_counter() - start,
        model_name=f"whisper-{normalized_model}",
        error=None,
    )


def _serialize_segments(raw_segments: Any) -> list[TranscriptSegment]:
    segments: list[TranscriptSegment] = []
    for item in raw_segments or []:
        try:
            start = float(item.get("start", 0.0))
            end = float(item.get("end", start))
            text = str(item.get("text", ""))
        except AttributeError:
            continue
        segments.append({"start": start, "end": end, "text": text})
    return segments


def get_asr_model_fingerprint(model_name: str, device: str = DEFAULT_DEVICE) -> dict[str, str]:
    """Return sanitized Whisper checkpoint SHA-256 evidence."""

    normalized_model = _validated_text(model_name, "model_name")
    resolved_device = _resolve_device(device)
    checkpoint_path = _resolve_whisper_checkpoint_path(normalized_model)
    model_label = (
        checkpoint_path.stem
        if Path(normalized_model).expanduser().is_file()
        else normalized_model
    )
    if not checkpoint_path.is_file():
        raise ASRInferenceError(
            "Whisper checkpoint is missing",
            details={"checkpoint_path": checkpoint_path.name},
        )
    return {
        "model_name": f"whisper-{model_label}",
        "device": resolved_device,
        "checkpoint_path": checkpoint_path.name,
        "checkpoint_sha256": _sha256_file(checkpoint_path),
    }


def verify_asr_model_fingerprint(
    expected_fingerprint: Mapping[str, str],
    model_name: str,
    device: str = DEFAULT_DEVICE,
) -> dict[str, str]:
    """Return actual fingerprint or fail closed on expected hash mismatch."""

    fingerprint = get_asr_model_fingerprint(model_name, device)
    _assert_expected_fingerprint(
        fingerprint,
        expected_fingerprint,
        required_keys=("checkpoint_sha256",),
    )
    return fingerprint


def _resolve_device(device: str) -> str:
    normalized_device = _validated_text(device, "device").lower()
    if normalized_device != "auto":
        return normalized_device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _validated_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ASRInferenceError(f"{field_name} 必须是非空字符串")
    return value.strip()


def _resolve_whisper_checkpoint_path(model_name: str) -> Path:
    candidate = Path(model_name).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    try:
        import whisper
    except Exception as exc:  # pragma: no cover - depends on optional runtime deps.
        raise ASRInferenceError(
            "Whisper dependency is unavailable",
            detail=str(exc),
        ) from exc
    if model_name not in getattr(whisper, "_MODELS", {}):
        raise ASRInferenceError(
            "Whisper model name is not recognized",
            details={"model_name": model_name},
        )
    default_cache = Path.home() / ".cache"
    cache_root = Path(os.getenv("XDG_CACHE_HOME", str(default_cache))) / "whisper"
    return (cache_root / f"{model_name}.pt").expanduser().resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_expected_fingerprint(
    actual: Mapping[str, str],
    expected: Mapping[str, str],
    *,
    required_keys: tuple[str, ...],
) -> None:
    for key in required_keys:
        wanted = str(expected.get(key, "")).strip().lower()
        if not wanted:
            continue
        got = str(actual.get(key, "")).strip().lower()
        if got != wanted:
            raise ASRInferenceError(
                "Model fingerprint mismatch",
                details={
                    "field": key,
                    "expected_sha256": wanted,
                    "actual_sha256": got,
                },
            )


def _fingerprint_key(
    expected_fingerprint: Mapping[str, str] | None,
) -> tuple[tuple[str, str], ...] | None:
    if expected_fingerprint is None:
        return None
    return tuple(
        sorted((str(key), str(value)) for key, value in expected_fingerprint.items())
    )


__all__ = [
    "get_asr_model_fingerprint",
    "load_asr",
    "transcribe_audio",
    "verify_asr_model_fingerprint",
]
