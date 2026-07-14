"""DeepFilterNet enhancement wrapper for the A-owned backend."""

from __future__ import annotations

import math
import threading
import time
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np

from core.audio_io import _read_wav, _resample_linear, _write_wav
from core.schemas import (
    EnhancementError,
    EnhancementOutput,
    ErrorCode,
    OutputValidationError,
    WarningItem,
)

_ENHANCER_BACKEND: Any | None = None
_ENHANCER_MODEL_DIR: str | None = None
_ENHANCER_LOCK = threading.Lock()


class _DeepFilterNetBackend:
    model_name = "DeepFilterNet3"

    def __init__(self, model_dir: str | None = None) -> None:
        try:
            from df.enhance import enhance, init_df, load_audio, save_audio
        except Exception as exc:  # pragma: no cover - depends on optional runtime deps.
            raise EnhancementError(
                "DeepFilterNet 依赖不可用，无法加载增强模型",
                detail=str(exc),
            ) from exc

        try:
            if model_dir:
                model, df_state, _ = init_df(model_base_dir=model_dir)
            else:
                model, df_state, _ = init_df()
        except Exception as exc:  # pragma: no cover - model availability is environment-specific.
            raise EnhancementError("DeepFilterNet 模型加载失败", detail=str(exc)) from exc

        self._enhance = enhance
        self._load_audio = load_audio
        self._save_audio = save_audio
        self._model = model
        self._df_state = df_state
        self._sample_rate = int(df_state.sr())

    def enhance_file(self, input_wav: str, output_full_wav: str) -> None:
        audio, _ = self._load_audio(input_wav, sr=self._sample_rate)
        enhanced = self._enhance(self._model, self._df_state, audio)
        self._save_audio(output_full_wav, enhanced, self._sample_rate)


def load_enhancer(model_dir: str | None = None):
    """Load and memoize the DeepFilterNet backend for this process."""

    global _ENHANCER_BACKEND, _ENHANCER_MODEL_DIR
    with _ENHANCER_LOCK:
        if _ENHANCER_BACKEND is not None and _ENHANCER_MODEL_DIR == model_dir:
            return _ENHANCER_BACKEND
        loaded_backend = _DeepFilterNetBackend(model_dir)
        _ENHANCER_BACKEND = loaded_backend
        _ENHANCER_MODEL_DIR = model_dir
        return loaded_backend


def enhance_audio(
    input_wav: str,
    output_full_wav: str,
    output_mix_wav: str,
    strength: float = 0.75,
) -> EnhancementOutput:
    """Run enhancement and create the dry/wet mixed output track."""

    if isinstance(strength, bool) or not isinstance(strength, Real):
        raise EnhancementError(
            "增强强度必须是 0.0 到 1.0 之间的有限数字",
            details={"strength": repr(strength)},
        )
    normalized_strength = float(strength)
    if not math.isfinite(normalized_strength) or not 0.0 <= normalized_strength <= 1.0:
        raise EnhancementError(
            "增强强度必须在 0.0 到 1.0 之间",
            details={"strength": repr(strength)},
        )

    input_path = Path(input_wav)
    full_path = Path(output_full_wav)
    mix_path = Path(output_mix_wav)
    full_path.parent.mkdir(parents=True, exist_ok=True)
    mix_path.parent.mkdir(parents=True, exist_ok=True)

    warnings: list[WarningItem] = []
    try:
        backend = load_enhancer()
        start = time.perf_counter()
        backend.enhance_file(str(input_path), str(full_path))
        dry, dry_sr, _ = _read_wav(input_path)
        wet, wet_sr, _ = _read_wav(full_path)
        dry = _mono(dry)
        wet = _mono(wet)
        if wet_sr != dry_sr:
            wet = _resample_linear(wet, wet_sr, dry_sr)
        wet = _match_length(wet, dry.shape[0])
        mixed = (1.0 - normalized_strength) * dry + normalized_strength * wet
        if not np.all(np.isfinite(mixed)):
            raise OutputValidationError("增强输出包含 NaN 或 Inf")

        if mixed.size == 0:
            raise OutputValidationError("增强输出为空")
        peak = float(np.max(np.abs(mixed)))
        if peak > 0.99:
            scale = 0.99 / peak
            mixed = mixed * scale
            warnings.append(
                WarningItem(
                    code=ErrorCode.OUTPUT_PEAK_PROTECTED,
                    message="增强混合轨峰值过高，已做安全缩放",
                    module="enhance",
                    recoverable=True,
                    details={"original_peak": peak, "scale": scale},
                )
            )
        _write_wav(mix_path, mixed, dry_sr)
        _validate_output(full_path)
        _validate_output(mix_path)
    except OutputValidationError:
        raise
    except EnhancementError:
        raise
    except Exception as exc:
        raise EnhancementError("音频增强失败", detail=str(exc)) from exc

    return EnhancementOutput(
        full_output_path=str(full_path.resolve()),
        mixed_output_path=str(mix_path.resolve()),
        strength=normalized_strength,
        runtime_seconds=time.perf_counter() - start,
        model_name=getattr(backend, "model_name", "DeepFilterNet3"),
        warnings=warnings,
    )


def _mono(samples: np.ndarray) -> np.ndarray:
    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim == 2:
        return audio.mean(axis=1)
    return audio


def _match_length(samples: np.ndarray, target_frames: int) -> np.ndarray:
    audio = np.asarray(samples, dtype=np.float32)
    if audio.shape[0] == target_frames:
        return audio
    if audio.shape[0] > target_frames:
        return audio[:target_frames]
    return np.pad(audio, (0, target_frames - audio.shape[0]))


def _validate_output(path: Path) -> None:
    try:
        samples, _, _ = _read_wav(path)
    except Exception as exc:
        raise OutputValidationError(
            "增强输出不可播放或无法读取",
            detail=str(exc),
            details={"path": str(path)},
        ) from exc
    if samples.size == 0 or not np.all(np.isfinite(samples)):
        raise OutputValidationError(
            "增强输出为空或包含非法数值",
            details={"path": str(path)},
        )


__all__ = ["enhance_audio", "load_enhancer"]
