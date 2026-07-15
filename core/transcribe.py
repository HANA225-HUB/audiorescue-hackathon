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

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "configs" / "app.yaml"
_KNOWN_WHISPER_MODELS = {
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small",
    "small.en",
    "medium",
    "medium.en",
    "large",
    "large-v1",
    "large-v2",
    "large-v3",
    "large-v3-turbo",
    "turbo",
}

_ASR_MODEL: Any | None = None
_ASR_KEY: tuple[str, str] | None = None
_ASR_FINGERPRINT_KEY: tuple[tuple[str, str], ...] | None = None
_ASR_CACHE_IDENTITY: tuple[Any, ...] | None = None
_ASR_LOCK = threading.Lock()
_ASR_INFERENCE_LOCK = threading.Lock()


def load_asr(
    model_name: str,
    device: str,
    *,
    expected_fingerprint: Mapping[str, str] | None = None,
):
    """Load and memoize one Whisper model for this process."""

    global _ASR_MODEL, _ASR_KEY, _ASR_FINGERPRINT_KEY, _ASR_CACHE_IDENTITY

    normalized_model = _validated_text(model_name, "model_name")
    resolved_device = _resolve_device(device)
    effective_expected = _effective_expected_fingerprint(
        expected_fingerprint,
        model_name=normalized_model,
    )

    with _ASR_LOCK:
        actual_fingerprint: dict[str, str] | None = None
        if effective_expected is not None:
            actual_fingerprint = verify_asr_model_fingerprint(
                effective_expected,
                normalized_model,
                resolved_device,
            )
        else:
            try:
                actual_fingerprint = get_asr_model_fingerprint(
                    normalized_model,
                    resolved_device,
                )
            except ASRInferenceError:
                actual_fingerprint = None

        key = (normalized_model, resolved_device)
        expected_key = _fingerprint_key(effective_expected)
        actual_key = _fingerprint_key(actual_fingerprint)
        cache_identity = (
            normalized_model,
            resolved_device,
            actual_key,
            expected_key,
        )
        if _ASR_MODEL is not None and _ASR_CACHE_IDENTITY == cache_identity:
            return _ASR_MODEL

        try:
            import whisper
        except Exception as exc:  # pragma: no cover - optional runtime deps.
            raise ASRInferenceError(
                "Whisper dependency is unavailable",
                detail=str(exc),
            ) from exc
        try:
            loaded_model = whisper.load_model(normalized_model, device=resolved_device)
        except Exception as exc:  # pragma: no cover - environment-specific.
            raise ASRInferenceError("Whisper model load failed", detail=str(exc)) from exc
        _ASR_MODEL = loaded_model
        _ASR_KEY = key
        _ASR_FINGERPRINT_KEY = expected_key
        _ASR_CACHE_IDENTITY = cache_identity
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
        raise ASRInferenceError(
            "Transcription input audio is missing or empty",
            details={"filename": path.name},
        )

    normalized_model = _validated_text(model_name, "model_name")
    resolved_device = _resolve_device(device)
    model = load_asr(normalized_model, resolved_device)
    with _ASR_INFERENCE_LOCK:
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
            raise ASRInferenceError("ASR transcription failed", detail=str(exc)) from exc
        runtime_seconds = time.perf_counter() - start

    return TranscriptResult(
        text=str(payload.get("text", "")),
        language=payload.get("language") or language,
        segments=_serialize_segments(payload.get("segments", [])),
        runtime_seconds=runtime_seconds,
        model_name=f"whisper-{_model_label(normalized_model)}",
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


def get_asr_model_fingerprint(
    model_name: str,
    device: str = DEFAULT_DEVICE,
) -> dict[str, str]:
    """Return sanitized Whisper checkpoint SHA-256 evidence."""

    normalized_model = _validated_text(model_name, "model_name")
    resolved_device = _resolve_device(device)
    checkpoint_path = _resolve_whisper_checkpoint_path(normalized_model)
    if not checkpoint_path.is_file():
        raise ASRInferenceError(
            "Whisper checkpoint is missing",
            details={"checkpoint_path": checkpoint_path.name},
        )
    return {
        "model_name": f"whisper-{_model_label(normalized_model)}",
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
        raise ASRInferenceError(f"{field_name} must be a non-empty string")
    return value.strip()


def _resolve_whisper_checkpoint_path(model_name: str) -> Path:
    candidate = Path(model_name).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    if not _is_known_whisper_model(model_name):
        raise ASRInferenceError(
            "Whisper model name is not recognized",
            details={"model_name": model_name},
        )
    default_cache = Path.home() / ".cache"
    cache_root = Path(os.getenv("XDG_CACHE_HOME", str(default_cache))) / "whisper"
    return (cache_root / f"{model_name}.pt").expanduser().resolve()


def _is_known_whisper_model(model_name: str) -> bool:
    if model_name in _KNOWN_WHISPER_MODELS:
        return True
    try:
        import whisper

        return model_name in getattr(whisper, "_MODELS", {})
    except Exception:
        return False


def _model_label(model_name: str) -> str:
    candidate = Path(model_name).expanduser()
    if candidate.is_file():
        return candidate.stem
    return model_name


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
            raise ASRInferenceError(
                "Model fingerprint expectation is empty",
                details={"field": key},
            )
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
    fingerprint: Mapping[str, str] | None,
) -> tuple[tuple[str, str], ...] | None:
    if fingerprint is None:
        return None
    return tuple(sorted((str(key), str(value)) for key, value in fingerprint.items()))


def _effective_expected_fingerprint(
    expected_fingerprint: Mapping[str, str] | None,
    *,
    model_name: str,
) -> dict[str, str] | None:
    if expected_fingerprint is not None:
        return {key: str(value) for key, value in expected_fingerprint.items()}
    if Path(model_name).expanduser().is_file():
        return None
    config = _configured_asr_fingerprint()
    if config is None:
        return None
    if str(config.get("model", "")).strip() != model_name:
        return None
    return {"checkpoint_sha256": str(config.get("checkpoint_sha256", ""))}


def _configured_asr_fingerprint() -> dict[str, str] | None:
    config_path = Path(os.getenv("AUDIORESCUE_CONFIG", str(_DEFAULT_CONFIG_PATH)))
    if not config_path.is_file():
        return None
    try:
        payload = _read_yaml_mapping(config_path)
    except Exception as exc:
        raise ASRInferenceError(
            "Configured model fingerprint cannot be read",
            detail=str(exc),
        ) from exc
    if not isinstance(payload, Mapping):
        return None
    asr_payload = payload.get("asr")
    if not isinstance(asr_payload, Mapping):
        return None
    if "checkpoint_sha256" not in asr_payload:
        return None
    return {
        "model": str(asr_payload.get("model", "")),
        "checkpoint_sha256": str(asr_payload.get("checkpoint_sha256", "")),
    }


def _read_yaml_mapping(config_path: Path) -> Mapping[str, Any]:
    try:
        import yaml

        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if isinstance(payload, Mapping):
            return payload
        return {}
    except ModuleNotFoundError:
        return _read_simple_yaml_mapping(config_path)


def _read_simple_yaml_mapping(config_path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    current_section: dict[str, str] | None = None
    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not raw_line.startswith((" ", "\t")) and line.endswith(":"):
            section_name = line[:-1].strip()
            current_section = {}
            payload[section_name] = current_section
            continue
        if current_section is None or ":" not in line:
            continue
        key, value = line.split(":", 1)
        current_section[key.strip()] = value.strip().strip('"').strip("'")
    return payload


__all__ = [
    "get_asr_model_fingerprint",
    "load_asr",
    "transcribe_audio",
    "verify_asr_model_fingerprint",
]
