"""Run an A-layer audio backend smoke test.

Default mode runs normalize -> enhance -> before/after ASR when model
dependencies are installed. Use --normalize-only for dependency-limited
machines to verify the file contract without claiming model inference.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import logging
import os
import platform
import shutil
import sys
import time
import warnings
import wave
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, TextIO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.audio_io import inspect_audio, normalize_audio
from core.enhance import enhance_audio, get_enhancer_model_fingerprint
from core.schemas import AudioMeta, AudioRescueError, TranscriptResult
from core.transcribe import get_asr_model_fingerprint, transcribe_audio


def main() -> int:
    parser = _build_parser()
    if _is_help_request(sys.argv[1:]):
        parser.parse_args()
        return 0

    output_stream = sys.stdout
    with _CliLogCapture() as log_capture:
        try:
            args = parser.parse_args()
            if args.runs < 1:
                raise _ArgumentParseError("invalid_range")
            summary, exit_code = _run_smoke(args)
        except _ArgumentParseError as exc:
            summary = _argument_parse_error_summary(exc)
            exit_code = 2
        except Exception as exc:
            summary = _pre_run_error_summary(exc)
            exit_code = 2

    summary["cli_log_capture"] = log_capture.summary()
    _print_json(summary, stream=output_stream)
    return exit_code


def _build_parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(description="AudioRescue A-layer smoke test")
    parser.add_argument(
        "--input",
        default=str(ROOT / "tests" / "fixtures" / "dev_smoke_s01_fan.wav"),
        help="Input audio path",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "outputs" / f"a_smoke_{time.strftime('%Y%m%d_%H%M%S')}"),
        help="Directory where smoke artifacts are written",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of smoke runs. Use 2+ to capture cold/warm evidence.",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.75,
        help="Dry/wet strength for enhanced_mix.wav",
    )
    parser.add_argument(
        "--normalize-only",
        action="store_true",
        help="Only verify normalization; do not load DeepFilterNet or Whisper",
    )
    parser.add_argument(
        "--asr-model",
        default="base",
        help="Whisper model name passed through the A-layer ASR contract",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Whisper device passed through the A-layer ASR contract",
    )
    return parser


class _ArgumentParseError(Exception):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__("argument parse failed")


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise _ArgumentParseError(_classify_argparse_error(message))


def _classify_argparse_error(message: str) -> str:
    normalized = str(message).lower()
    if "unrecognized arguments" in normalized:
        return "unknown_argument"
    if "expected one argument" in normalized or "requires an argument" in normalized:
        return "missing_value"
    if "invalid" in normalized:
        return "invalid_value"
    return "parse_error"


def _is_help_request(argv: list[str]) -> bool:
    return any(item in {"-h", "--help"} for item in argv)


def _pre_run_error_summary(exc: Exception) -> dict[str, Any]:
    return {
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "runs": [
            {
                "run_index": 0,
                "run_kind": "pre_run",
                "error": _unexpected_error_evidence(exc),
                "cuda_after_run": _cuda_after_run(),
            }
        ],
    }


def _argument_parse_error_summary(exc: _ArgumentParseError) -> dict[str, Any]:
    return {
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "runs": [
            {
                "run_index": 0,
                "run_kind": "pre_run",
                "error": {
                    "code": "INPUT_INVALID",
                    "message": _SAFE_ERROR_MESSAGES["INPUT_INVALID"],
                    "module": "smoke_audio_core",
                    "details": {
                        "category": "argument_parse",
                        "reason": exc.reason,
                    },
                },
                "cuda_after_run": _cuda_after_run(),
            }
        ],
    }


def _run_smoke(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    input_path = Path(args.input)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "environment": _probe_environment(),
        "input": {
            "sha256": _optional_sha256_file(input_path),
            "bytes": input_path.stat().st_size if input_path.is_file() else None,
        },
        "output_layout": {
            "runs": int(args.runs),
            "per_run_subdirectories": args.runs > 1,
        },
        "normalize_only": bool(args.normalize_only),
        "asr_request": {
            "model": _safe_asr_model_label(args.asr_model),
            "device": _safe_device_label(args.device),
        },
        "runs": [],
    }

    exit_code = 0
    for run_index in range(1, args.runs + 1):
        run_dir = output_dir if args.runs == 1 else output_dir / f"run_{run_index:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        try:
            run_payload = _run_once(
                input_path=input_path,
                run_dir=run_dir,
                run_index=run_index,
                strength=float(args.strength),
                normalize_only=bool(args.normalize_only),
                asr_model=str(args.asr_model),
                device=str(args.device),
            )
            summary["runs"].append(run_payload)
        except AudioRescueError as exc:
            summary["runs"].append(
                {
                    "run_index": run_index,
                    "run_kind": _run_kind(run_index),
                    "error": _audio_rescue_error_evidence(exc),
                    "cuda_after_run": _cuda_after_run(),
                }
            )
            exit_code = 2
            break
        except Exception as exc:
            summary["runs"].append(
                {
                    "run_index": run_index,
                    "run_kind": _run_kind(run_index),
                    "error": _unexpected_error_evidence(exc),
                    "cuda_after_run": _cuda_after_run(),
                }
            )
            exit_code = 2
            break

    return summary, exit_code


def _run_once(
    *,
    input_path: Path,
    run_dir: Path,
    run_index: int,
    strength: float,
    normalize_only: bool,
    asr_model: str,
    device: str,
) -> dict[str, Any]:
    original = run_dir / "original.wav"
    enhanced_full = run_dir / "enhanced_full.wav"
    enhanced_mix = run_dir / "enhanced_mix.wav"

    run_started = time.perf_counter()
    _reset_cuda_peak_memory()
    normalize_start = time.perf_counter()
    meta = normalize_audio(str(input_path), str(original))
    run_payload: dict[str, Any] = {
        "run_index": run_index,
        "run_kind": _run_kind(run_index),
        "normalize_seconds": time.perf_counter() - normalize_start,
        "input_meta": _audio_meta_evidence(meta),
        "output_files": {"original": _wav_file_evidence(original)},
    }

    if normalize_only:
        run_payload["cuda_after_run"] = _cuda_after_run()
        run_payload["run_seconds"] = time.perf_counter() - run_started
        return run_payload

    enhancement = enhance_audio(
        str(original),
        str(enhanced_full),
        str(enhanced_mix),
        strength=strength,
    )
    run_payload["enhancement"] = {
        "strength": enhancement["strength"],
        "runtime_seconds": enhancement["runtime_seconds"],
        "model_name": enhancement["model_name"],
        "warnings": [_warning_to_dict(item) for item in enhancement["warnings"]],
    }
    run_payload["model_fingerprints"] = {
        "enhancer": get_enhancer_model_fingerprint(),
    }
    run_payload["output_files"] = {
        "original": _wav_file_evidence(original),
        "enhanced_full": _wav_file_evidence(enhanced_full),
        "enhanced_mix": _wav_file_evidence(enhanced_mix),
    }

    before = transcribe_audio(
        str(original),
        language="zh",
        model_name=asr_model,
        device=device,
    )
    after = transcribe_audio(
        str(enhanced_mix),
        language="zh",
        model_name=asr_model,
        device=device,
    )
    model_label = _safe_asr_model_label(asr_model)
    device_label = _safe_device_label(device)
    run_payload["model_fingerprints"]["asr"] = _safe_asr_fingerprint(
        get_asr_model_fingerprint(
            asr_model,
            device,
        ),
        model_label=model_label,
        device_label=device_label,
    )
    run_payload["transcript_before"] = _transcript_evidence(
        before,
        model_label=model_label,
    )
    run_payload["transcript_after"] = _transcript_evidence(
        after,
        model_label=model_label,
    )
    run_payload["mixed_inspection"] = inspect_audio(*_read_for_inspection(enhanced_mix))
    run_payload["cuda_after_run"] = _cuda_after_run()
    run_payload["run_seconds"] = time.perf_counter() - run_started
    return run_payload


def _run_kind(run_index: int) -> str:
    return "cold" if run_index == 1 else "warm"


_KNOWN_WHISPER_MODEL_LABELS = frozenset(
    {
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
)
_HEX_CHARS = frozenset("0123456789abcdefABCDEF")


def _safe_asr_model_label(value: Any) -> str:
    if not isinstance(value, str):
        return "custom"
    normalized = value.strip()
    if normalized in _KNOWN_WHISPER_MODEL_LABELS:
        return normalized
    return "custom"


def _safe_transcript_model_name(value: Any, model_label: str | None) -> str:
    if model_label is not None:
        return f"whisper-{model_label}" if model_label != "custom" else "whisper-custom"
    if not isinstance(value, str):
        return "whisper-custom"
    normalized = value.strip()
    prefix = "whisper-"
    if normalized.startswith(prefix):
        label = normalized[len(prefix) :]
        if label in _KNOWN_WHISPER_MODEL_LABELS:
            return normalized
    if normalized in _KNOWN_WHISPER_MODEL_LABELS:
        return f"whisper-{normalized}"
    return "whisper-custom"


def _safe_device_label(value: Any) -> str:
    if not isinstance(value, str):
        return "custom"
    normalized = value.strip().lower()
    if normalized in {"auto", "cpu", "cuda", "mps"}:
        return normalized
    if normalized.startswith("cuda:") and normalized[5:].isdigit():
        return "cuda"
    return "custom"


def _safe_asr_fingerprint(
    value: Mapping[str, Any],
    *,
    model_label: str,
    device_label: str,
) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    if not isinstance(value, Mapping):
        return safe
    if "model_name" in value:
        safe["model_name"] = (
            f"whisper-{model_label}" if model_label != "custom" else "whisper-custom"
        )
    if "device" in value:
        safe["device"] = device_label
    if "checkpoint_path" in value:
        safe["checkpoint_path"] = (
            f"{model_label}.pt" if model_label != "custom" else "custom"
        )
    checkpoint_sha256 = value.get("checkpoint_sha256")
    if _is_sha256_text(checkpoint_sha256):
        safe["checkpoint_sha256"] = str(checkpoint_sha256).lower()
    return safe


def _is_sha256_text(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in _HEX_CHARS for char in value)
    )


def _probe_environment() -> dict[str, Any]:
    payload = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "ffmpeg_available": shutil.which("ffmpeg") is not None,
        "deepfilternet_importable": _can_import("df.enhance"),
        "whisper_importable": _can_import("whisper"),
        "cache_env": {
            "XDG_CACHE_HOME_set": bool(os.environ.get("XDG_CACHE_HOME")),
            "TORCH_HOME_set": bool(os.environ.get("TORCH_HOME")),
            "HF_HOME_set": bool(os.environ.get("HF_HOME")),
        },
    }
    payload.update(_torch_environment())
    return payload


def _torch_environment() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {"torch_importable": False, "torch_error": type(exc).__name__}
    cuda_available = bool(torch.cuda.is_available())
    payload: dict[str, Any] = {
        "torch_importable": True,
        "torch_version": getattr(torch, "__version__", None),
        "torch_cuda_available": cuda_available,
        "torch_cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
        "cuda_device_count": int(torch.cuda.device_count()) if cuda_available else 0,
    }
    if cuda_available:
        payload["cuda_device_name"] = torch.cuda.get_device_name(0)
    return payload


def _can_import(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


def _read_for_inspection(path: Path):
    from core.audio_io import _read_wav

    samples, sample_rate, _ = _read_wav(path)
    return samples, sample_rate


def _audio_meta_evidence(meta: AudioMeta) -> dict[str, Any]:
    return {
        "sample_rate": meta.sample_rate,
        "channels": meta.channels,
        "duration_seconds": meta.duration_seconds,
        "peak_abs": meta.peak_abs,
        "rms_dbfs": meta.rms_dbfs,
        "clipped_ratio": meta.clipped_ratio,
        "silent_ratio": meta.silent_ratio,
        "original_sample_rate": meta.original_sample_rate,
        "original_channels": meta.original_channels,
        "source_format": meta.source_format,
    }


def _transcript_evidence(
    transcript: TranscriptResult,
    *,
    model_label: str | None = None,
) -> dict[str, Any]:
    return {
        "text_length": len(transcript.text),
        "text_nonempty": bool(transcript.text.strip()),
        "language": transcript.language,
        "segments_count": len(transcript.segments),
        "runtime_seconds": transcript.runtime_seconds,
        "model_name": _safe_transcript_model_name(transcript.model_name, model_label),
        "error": transcript.error,
    }


def _wav_file_evidence(path: Path) -> dict[str, Any]:
    with wave.open(str(path), "rb") as handle:
        frames = handle.getnframes()
        sample_rate = handle.getframerate()
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        comptype = handle.getcomptype()
    return {
        "filename": path.name,
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
        "sample_rate": sample_rate,
        "channels": channels,
        "sample_width_bits": sample_width * 8,
        "format": "PCM" if comptype == "NONE" else comptype,
        "duration_seconds": frames / sample_rate if sample_rate else 0.0,
        "frames": frames,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _optional_sha256_file(path: Path) -> str | None:
    candidate = Path(path)
    if not candidate.is_file():
        return None
    return _sha256_file(candidate)


def _reset_cuda_peak_memory() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        return


def _cuda_after_run() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {"torch_importable": False, "torch_error": type(exc).__name__}
    if not torch.cuda.is_available():
        return {
            "torch_importable": True,
            "torch_cuda_available": False,
            "max_memory_allocated_bytes": 0,
            "max_memory_reserved_bytes": 0,
        }
    return {
        "torch_importable": True,
        "torch_cuda_available": True,
        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def _warning_to_dict(item) -> dict[str, Any]:
    return {
        "code": item.code.value,
        "message": item.message,
        "module": item.module,
        "recoverable": item.recoverable,
        "details": _safe_details(item.details),
    }


def _audio_rescue_error_evidence(exc: AudioRescueError) -> dict[str, Any]:
    return {
        "code": exc.code.value,
        "message": _safe_error_message(exc),
        "module": exc.module,
        "details": _safe_details(exc.details),
    }


def _unexpected_error_evidence(exc: Exception) -> dict[str, Any]:
    return {
        "code": "INTERNAL_ERROR",
        "message": "internal error",
        "module": "smoke_audio_core",
        "details": _safe_details({"exception_type": type(exc).__name__}),
    }


class _CapturedLogHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.NOTSET)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


class _CliLogCapture:
    """Capture Python-level CLI noise and expose only fixed-count evidence."""

    def __init__(self) -> None:
        self._stdout = io.StringIO()
        self._stderr = io.StringIO()
        self._stack: contextlib.ExitStack | None = None
        self._warning_records: list[warnings.WarningMessage] = []
        self._log_handler = _CapturedLogHandler()

    def __enter__(self) -> "_CliLogCapture":
        stack = contextlib.ExitStack()
        stack.enter_context(contextlib.redirect_stdout(self._stdout))
        stack.enter_context(contextlib.redirect_stderr(self._stderr))
        self._warning_records = stack.enter_context(warnings.catch_warnings(record=True))
        warnings.simplefilter("always")

        original_call_handlers = logging.Logger.callHandlers

        def safe_call_handlers(_logger, record: logging.LogRecord) -> None:
            self._log_handler.handle(record)

        logging.Logger.callHandlers = safe_call_handlers
        stack.callback(setattr, logging.Logger, "callHandlers", original_call_handlers)

        self._stack = stack
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if self._stack is None:
            return False
        result = self._stack.__exit__(exc_type, exc, traceback)
        self._stack = None
        return bool(result)

    def summary(self) -> dict[str, Any]:
        warning_categories = Counter(
            _safe_counter_key(item.category.__name__)
            for item in self._warning_records
        )
        logging_levels = Counter(
            _safe_counter_key(record.levelname)
            for record in self._log_handler.records
        )
        return {
            "captured": True,
            "capture_scope": "python_stdout_stderr_warnings_logging",
            "stdout_lines": _count_captured_lines(self._stdout.getvalue()),
            "stderr_lines": _count_captured_lines(self._stderr.getvalue()),
            "stdout_chars": len(self._stdout.getvalue()),
            "stderr_chars": len(self._stderr.getvalue()),
            "warning_count": len(self._warning_records),
            "warning_categories": dict(sorted(warning_categories.items())),
            "logging_count": len(self._log_handler.records),
            "logging_levels": dict(sorted(logging_levels.items())),
        }


def _count_captured_lines(value: str) -> int:
    if not value:
        return 0
    return len(value.splitlines())


def _safe_counter_key(value: Any) -> str:
    if not isinstance(value, str):
        return "unknown"
    if 1 <= len(value) <= 80 and all(
        char.isascii() and (char.isalnum() or char in "._:-") for char in value
    ):
        return value
    return "unknown"


_PATH_DETAIL_TOKENS = ("path", "dir", "cache", "file", "url", "uri")
_SAFE_DETAIL_STRING_KEYS = {
    "code",
    "component",
    "exception_type",
    "module",
    "stage",
    "status",
}
_SAFE_DETAIL_BOOL_KEYS = {"recoverable", "retryable"}
_SAFE_DETAIL_NUMBER_KEYS = {
    "attempt",
    "attempts",
    "count",
    "exit_code",
    "run_index",
}
_SAFE_ERROR_MESSAGES = {
    "INPUT_INVALID": "input audio is invalid",
    "INPUT_TOO_LONG": "input audio is too long",
    "ENHANCE_FAILED": "enhancement failed",
    "OUTPUT_INVALID": "output audio is invalid",
    "ASR_FAILED": "asr failed",
    "VIS_FAILED": "visualization failed",
    "INTERNAL_ERROR": "internal error",
}
_REDACTED = "<redacted>"


def _safe_error_message(exc: AudioRescueError) -> str:
    return _SAFE_ERROR_MESSAGES.get(exc.code.value, "audio processing failed")


def _safe_details(value: Any) -> Any:
    if isinstance(value, Mapping):
        safe: dict[str, Any] = {}
        for key, item in value.items():
            normalized_key = str(key)
            lower_key = normalized_key.lower()
            if any(token in lower_key for token in _PATH_DETAIL_TOKENS):
                safe[normalized_key] = _REDACTED if item is not None else None
            elif lower_key in _SAFE_DETAIL_STRING_KEYS:
                safe[normalized_key] = _safe_detail_string(item)
            elif lower_key in _SAFE_DETAIL_BOOL_KEYS:
                safe[normalized_key] = item if isinstance(item, bool) else _REDACTED
            elif lower_key in _SAFE_DETAIL_NUMBER_KEYS:
                safe[normalized_key] = item if _is_safe_number(item) else _REDACTED
            elif isinstance(item, (Mapping, list, tuple)):
                safe[normalized_key] = _safe_details(item)
            elif item is None:
                safe[normalized_key] = None
            else:
                safe[normalized_key] = _REDACTED
        return safe
    if isinstance(value, list):
        return [_safe_details(item) for item in value]
    if isinstance(value, tuple):
        return [_safe_details(item) for item in value]
    if isinstance(value, bool) or value is None:
        return value
    if _is_safe_number(value):
        return value
    if isinstance(value, str):
        return _REDACTED
    return type(value).__name__


def _safe_detail_string(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return _REDACTED
    if len(value) > 80:
        return _REDACTED
    if any(token in value.lower() for token in ("://", "\\", "/", "token=", "cache", "outputs")):
        return _REDACTED
    if not all(char.isascii() and (char.isalnum() or char in "._:-") for char in value):
        return _REDACTED
    return value


def _is_safe_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and value == value
        and value not in (float("inf"), float("-inf"))
    )


def _print_json(payload: dict[str, Any], *, stream: TextIO | None = None) -> None:
    print(json.dumps(_json_ready(payload), ensure_ascii=False, indent=2), file=stream)


def _json_ready(value: Any) -> Any:
    if is_dataclass(value):
        return _json_ready(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if hasattr(value, "value"):
        return value.value
    return value


if __name__ == "__main__":
    raise SystemExit(main())
