"""DeepFilterNet enhancement wrapper for the A-owned backend."""

from __future__ import annotations

import hashlib
import math
import os
import threading
import time
from collections.abc import Mapping
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

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "configs" / "app.yaml"

_ENHANCER_BACKEND: Any | None = None
_ENHANCER_MODEL_DIR: str | None = None
_ENHANCER_FINGERPRINT_KEY: tuple[tuple[str, str], ...] | None = None
_ENHANCER_CACHE_IDENTITY: tuple[Any, ...] | None = None
_ENHANCER_LOCK = threading.Lock()
_ENHANCER_INFERENCE_LOCK = threading.Lock()


class _DeepFilterNetBackend:
    model_name = "DeepFilterNet3"

    def __init__(
        self,
        model_dir: str | None = None,
        *,
        fingerprint: Mapping[str, str] | None = None,
    ) -> None:
        try:
            from df.enhance import enhance, init_df, load_audio, save_audio
        except Exception as exc:  # pragma: no cover - optional runtime deps.
            raise EnhancementError(
                "DeepFilterNet dependency is unavailable",
                detail=str(exc),
            ) from exc

        try:
            resolved_model_dir = _resolve_deepfilternet_model_dir(model_dir)
            model, df_state, _ = init_df(model_base_dir=str(resolved_model_dir))
        except Exception as exc:  # pragma: no cover - environment-specific.
            raise EnhancementError("DeepFilterNet model load failed", detail=str(exc)) from exc

        self._enhance = enhance
        self._load_audio = load_audio
        self._save_audio = save_audio
        self._model = model
        self._df_state = df_state
        self._sample_rate = int(df_state.sr())
        self.fingerprint = dict(fingerprint or get_enhancer_model_fingerprint(str(resolved_model_dir)))

    def enhance_file(self, input_wav: str, output_full_wav: str) -> None:
        audio, _ = self._load_audio(input_wav, sr=self._sample_rate)
        enhanced = self._enhance(self._model, self._df_state, audio)
        self._save_audio(output_full_wav, enhanced, self._sample_rate)


def load_enhancer(
    model_dir: str | None = None,
    *,
    expected_fingerprint: Mapping[str, str] | None = None,
):
    """Load and memoize the DeepFilterNet backend for this process."""

    global _ENHANCER_BACKEND, _ENHANCER_MODEL_DIR, _ENHANCER_FINGERPRINT_KEY
    global _ENHANCER_CACHE_IDENTITY

    with _ENHANCER_LOCK:
        effective_expected = _effective_expected_fingerprint(
            expected_fingerprint,
            model_dir=model_dir,
            section="enhancement",
            keys=("config_sha256", "checkpoint_sha256"),
        )
        actual_fingerprint = get_enhancer_model_fingerprint(model_dir)
        if effective_expected is not None:
            _assert_expected_fingerprint(
                actual_fingerprint,
                effective_expected,
                required_keys=("config_sha256", "checkpoint_sha256"),
            )

        expected_key = _fingerprint_key(effective_expected)
        actual_key = _fingerprint_key(actual_fingerprint)
        cache_identity = (
            "DeepFilterNet3",
            _model_dir_cache_token(model_dir),
            actual_key,
            expected_key,
        )
        if (
            _ENHANCER_BACKEND is not None
            and _ENHANCER_CACHE_IDENTITY == cache_identity
        ):
            return _ENHANCER_BACKEND

        loaded_backend = _DeepFilterNetBackend(
            model_dir,
            fingerprint=actual_fingerprint,
        )
        _ENHANCER_BACKEND = loaded_backend
        _ENHANCER_MODEL_DIR = model_dir
        _ENHANCER_FINGERPRINT_KEY = expected_key
        _ENHANCER_CACHE_IDENTITY = cache_identity
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
            "Enhancement strength must be a finite number between 0.0 and 1.0",
            details={"strength": repr(strength)},
        )
    normalized_strength = float(strength)
    if not math.isfinite(normalized_strength) or not 0.0 <= normalized_strength <= 1.0:
        raise EnhancementError(
            "Enhancement strength must be between 0.0 and 1.0",
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
        with _ENHANCER_INFERENCE_LOCK:
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
                raise OutputValidationError("Enhanced output contains NaN or Inf")

            if mixed.size == 0:
                raise OutputValidationError("Enhanced output is empty")
            peak = float(np.max(np.abs(mixed)))
            if peak > 0.99:
                scale = 0.99 / peak
                mixed = mixed * scale
                warnings.append(
                    WarningItem(
                        code=ErrorCode.OUTPUT_PEAK_PROTECTED,
                        message="Enhanced mix peak was safely attenuated",
                        module="enhance",
                        recoverable=True,
                        details={"original_peak": peak, "scale": scale},
                    )
                )
            _write_wav(mix_path, mixed, dry_sr)
            _validate_output(full_path)
            _validate_output(mix_path)
            runtime_seconds = time.perf_counter() - start
    except OutputValidationError:
        raise
    except EnhancementError:
        raise
    except Exception as exc:
        raise EnhancementError("Audio enhancement failed", detail=str(exc)) from exc

    return EnhancementOutput(
        full_output_path=str(full_path.resolve()),
        mixed_output_path=str(mix_path.resolve()),
        strength=normalized_strength,
        runtime_seconds=runtime_seconds,
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


def get_enhancer_model_fingerprint(model_dir: str | None = None) -> dict[str, str]:
    """Return sanitized DeepFilterNet config/checkpoint SHA-256 evidence."""

    try:
        resolved_model_dir = _resolve_deepfilternet_model_dir(model_dir)
        config_path = resolved_model_dir / "config.ini"
        checkpoint_path = _select_deepfilternet_checkpoint(
            resolved_model_dir / "checkpoints"
        )
        if not config_path.is_file():
            raise EnhancementError(
                "DeepFilterNet config.ini is missing",
                details={"model_dir_name": resolved_model_dir.name},
            )
        return {
            "model_name": resolved_model_dir.name,
            "model_dir_name": resolved_model_dir.name,
            "config_path": "config.ini",
            "config_sha256": _sha256_file(config_path),
            "checkpoint_path": checkpoint_path.relative_to(
                resolved_model_dir
            ).as_posix(),
            "checkpoint_sha256": _sha256_file(checkpoint_path),
        }
    except EnhancementError:
        raise
    except Exception as exc:
        raise EnhancementError(
            "DeepFilterNet fingerprint cannot be resolved",
            detail=str(exc),
        ) from exc


def verify_enhancer_model_fingerprint(
    expected_fingerprint: Mapping[str, str],
    model_dir: str | None = None,
) -> dict[str, str]:
    """Return actual fingerprint or fail closed on expected hash mismatch."""

    fingerprint = get_enhancer_model_fingerprint(model_dir)
    _assert_expected_fingerprint(
        fingerprint,
        expected_fingerprint,
        required_keys=("config_sha256", "checkpoint_sha256"),
    )
    return fingerprint


def _validate_output(path: Path) -> None:
    try:
        samples, _, _ = _read_wav(path)
    except Exception as exc:
        raise OutputValidationError(
            "Enhanced output is not playable",
            detail=str(exc),
            details={"filename": path.name},
        ) from exc
    if samples.size == 0 or not np.all(np.isfinite(samples)):
        raise OutputValidationError(
            "Enhanced output is empty or non-finite",
            details={"filename": path.name},
        )


def _resolve_deepfilternet_model_dir(model_dir: str | None = None) -> Path:
    if model_dir:
        return Path(model_dir).expanduser().resolve()
    try:
        from df.enhance import get_model_basedir
    except Exception as exc:  # pragma: no cover - optional runtime deps.
        raise EnhancementError(
            "DeepFilterNet dependency is unavailable",
            detail=str(exc),
        ) from exc
    return Path(get_model_basedir("DeepFilterNet3")).expanduser().resolve()


def _select_deepfilternet_checkpoint(checkpoint_dir: Path) -> Path:
    if not checkpoint_dir.is_dir():
        raise EnhancementError(
            "DeepFilterNet checkpoints directory is missing",
            details={"checkpoint_dir": "checkpoints"},
        )
    candidates = list(checkpoint_dir.glob("model*.ckpt.best"))
    if not candidates:
        candidates = list(checkpoint_dir.glob("model*.ckpt"))
    if not candidates:
        raise EnhancementError(
            "DeepFilterNet checkpoint is missing",
            details={"checkpoint_dir": "checkpoints"},
        )
    return max(candidates, key=_checkpoint_epoch)


def _checkpoint_epoch(path: Path) -> int:
    try:
        return int(path.name.split(".")[0].split("_")[-1])
    except ValueError as exc:
        raise EnhancementError(
            "DeepFilterNet checkpoint name does not contain an epoch",
            details={"checkpoint_name": path.name},
        ) from exc


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
            raise EnhancementError(
                "Model fingerprint expectation is empty",
                details={"field": key},
            )
        got = str(actual.get(key, "")).strip().lower()
        if got != wanted:
            raise EnhancementError(
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


def _model_dir_cache_token(model_dir: str | None) -> str:
    if model_dir is None:
        return "__default__"
    return str(Path(model_dir).expanduser().resolve())


def _effective_expected_fingerprint(
    expected_fingerprint: Mapping[str, str] | None,
    *,
    model_dir: str | None,
    section: str,
    keys: tuple[str, ...],
) -> dict[str, str] | None:
    if expected_fingerprint is not None:
        return {key: str(value) for key, value in expected_fingerprint.items()}
    if model_dir is not None:
        return None
    return _configured_expected_fingerprint(section, keys)


def _configured_expected_fingerprint(
    section: str,
    keys: tuple[str, ...],
) -> dict[str, str] | None:
    config_path = Path(os.getenv("AUDIORESCUE_CONFIG", str(_DEFAULT_CONFIG_PATH)))
    if not config_path.is_file():
        return None
    try:
        payload = _read_yaml_mapping(config_path)
    except Exception as exc:
        raise EnhancementError(
            "Configured model fingerprint cannot be read",
            detail=str(exc),
        ) from exc
    if not isinstance(payload, Mapping):
        return None
    section_payload = payload.get(section)
    if not isinstance(section_payload, Mapping):
        return None
    if not any(key in section_payload for key in keys):
        return None
    return {key: str(section_payload.get(key, "")) for key in keys}


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
    "enhance_audio",
    "get_enhancer_model_fingerprint",
    "load_enhancer",
    "verify_enhancer_model_fingerprint",
]
