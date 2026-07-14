"""Run an A-layer audio backend smoke test.

Default mode runs normalize -> enhance -> before/after ASR when model
dependencies are installed. Use --normalize-only for dependency-limited
machines to verify the file contract without claiming model inference.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.audio_io import inspect_audio, normalize_audio
from core.enhance import enhance_audio
from core.schemas import AudioRescueError
from core.transcribe import transcribe_audio


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
        help="Directory where original/enhanced files are written",
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
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    original = output_dir / "original.wav"
    enhanced_full = output_dir / "enhanced_full.wav"
    enhanced_mix = output_dir / "enhanced_mix.wav"

    summary: dict[str, Any] = {
        "environment": _probe_environment(),
        "input": str(Path(args.input).resolve()),
        "output_dir": str(output_dir),
        "normalize_only": bool(args.normalize_only),
    }

    try:
        normalize_start = time.perf_counter()
        meta = normalize_audio(args.input, str(original))
        summary["normalize_seconds"] = time.perf_counter() - normalize_start
        summary["input_meta"] = {
            "sample_rate": meta.sample_rate,
            "channels": meta.channels,
            "duration_seconds": meta.duration_seconds,
            "peak_abs": meta.peak_abs,
            "rms_dbfs": meta.rms_dbfs,
            "clipped_ratio": meta.clipped_ratio,
            "silent_ratio": meta.silent_ratio,
            "normalized_path": meta.normalized_path,
        }

        if args.normalize_only:
            _print_json(summary)
            return 0

        enhancement = enhance_audio(
            str(original),
            str(enhanced_full),
            str(enhanced_mix),
            strength=args.strength,
        )
        summary["enhancement"] = {
            "full_output_path": enhancement["full_output_path"],
            "mixed_output_path": enhancement["mixed_output_path"],
            "strength": enhancement["strength"],
            "runtime_seconds": enhancement["runtime_seconds"],
            "model_name": enhancement["model_name"],
            "warnings": [_warning_to_dict(item) for item in enhancement["warnings"]],
        }

        before = transcribe_audio(str(original), language="zh")
        after = transcribe_audio(str(enhanced_mix), language="zh")
        summary["transcript_before"] = asdict(before)
        summary["transcript_after"] = asdict(after)
        summary["mixed_inspection"] = inspect_audio(*_read_for_inspection(enhanced_mix))
        _print_json(summary)
        return 0
    except AudioRescueError as exc:
        summary["error"] = {
            "code": exc.code.value,
            "message": exc.public_message,
            "module": exc.module,
            "details": exc.details,
        }
        _print_json(summary)
        return 2


def _probe_environment() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "ffmpeg": shutil.which("ffmpeg"),
        "deepfilternet_importable": _can_import("df.enhance"),
        "whisper_importable": _can_import("whisper"),
    }


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


def _warning_to_dict(item) -> dict[str, Any]:
    return {
        "code": item.code.value,
        "message": item.message,
        "module": item.module,
        "recoverable": item.recoverable,
        "details": item.details,
    }


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
