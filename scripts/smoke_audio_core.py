"""Run an A-layer audio backend smoke test.

Default mode runs normalize -> enhance -> before/after ASR when model
dependencies are installed. Use --normalize-only for dependency-limited
machines to verify the file contract without claiming model inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import time
import wave
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.audio_io import inspect_audio, normalize_audio
from core.enhance import enhance_audio, get_enhancer_model_fingerprint
from core.schemas import AudioMeta, AudioRescueError, TranscriptResult
from core.transcribe import get_asr_model_fingerprint, transcribe_audio


def main() -> int:
    parser = argparse.ArgumentParser(description="AudioRescue A-layer smoke test")
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
    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be at least 1")

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
        "asr_request": {"model": args.asr_model, "device": args.device},
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

    _print_json(summary)
    return exit_code


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
    run_payload["model_fingerprints"]["asr"] = get_asr_model_fingerprint(
        asr_model,
        device,
    )
    run_payload["transcript_before"] = _transcript_evidence(before)
    run_payload["transcript_after"] = _transcript_evidence(after)
    run_payload["mixed_inspection"] = inspect_audio(*_read_for_inspection(enhanced_mix))
    run_payload["cuda_after_run"] = _cuda_after_run()
    run_payload["run_seconds"] = time.perf_counter() - run_started
    return run_payload


def _run_kind(run_index: int) -> str:
    return "cold" if run_index == 1 else "warm"


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


def _transcript_evidence(transcript: TranscriptResult) -> dict[str, Any]:
    return {
        "text_length": len(transcript.text),
        "text_nonempty": bool(transcript.text.strip()),
        "language": transcript.language,
        "segments_count": len(transcript.segments),
        "runtime_seconds": transcript.runtime_seconds,
        "model_name": transcript.model_name,
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
        "message": exc.public_message,
        "module": exc.module,
        "details": _safe_details(exc.details),
    }


def _safe_details(value: Any) -> Any:
    if isinstance(value, Mapping):
        safe: dict[str, Any] = {}
        for key, item in value.items():
            normalized_key = str(key)
            if any(token in normalized_key.lower() for token in ("path", "dir", "cache")):
                safe[normalized_key] = Path(str(item)).name if item is not None else None
            else:
                safe[normalized_key] = _safe_details(item)
        return safe
    if isinstance(value, list):
        return [_safe_details(item) for item in value]
    if isinstance(value, tuple):
        return [_safe_details(item) for item in value]
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    return type(value).__name__


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(_json_ready(payload), ensure_ascii=False, indent=2))


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
