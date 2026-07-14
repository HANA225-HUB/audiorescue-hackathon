"""Whisper transcription wrapper for the A-owned backend."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

from core.schemas import ASRInferenceError, TranscriptResult, TranscriptSegment

DEFAULT_MODEL_NAME = "base"
DEFAULT_DEVICE = "auto"

_ASR_MODEL: Any | None = None
_ASR_KEY: tuple[str, str] | None = None
_ASR_LOCK = threading.Lock()


def load_asr(model_name: str, device: str):
    """Load and memoize one Whisper model for this process."""

    global _ASR_MODEL, _ASR_KEY
    normalized_model = _validated_text(model_name, "model_name")
    resolved_device = _resolve_device(device)
    key = (normalized_model, resolved_device)
    with _ASR_LOCK:
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


__all__ = ["load_asr", "transcribe_audio"]
