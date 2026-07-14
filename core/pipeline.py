"""C-owned orchestration boundary.

Only the job-directory contract is implemented in v0.1. A and B must not edit
this file; the full pipeline will be added after their independent modules run.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from core.schemas import ProcessResult


_SAFE_JOB_ID = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass(frozen=True, slots=True)
class JobPaths:
    root: Path
    original: Path
    enhanced_full: Path
    enhanced_mix: Path
    transcript_before: Path
    transcript_after: Path
    waveform: Path
    spectrogram: Path
    result_json: Path
    run_log: Path


def new_job_id() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{uuid4().hex[:8]}"


def create_job_paths(
    output_root: str | Path = "outputs",
    job_id: str | None = None,
) -> JobPaths:
    """Create one non-overwriting job directory and return absolute paths."""

    resolved_job_id = job_id or new_job_id()
    if not _SAFE_JOB_ID.fullmatch(resolved_job_id):
        raise ValueError("job_id may contain only letters, numbers, '_' and '-'")

    root = (Path(output_root) / resolved_job_id).resolve()
    root.mkdir(parents=True, exist_ok=False)
    return JobPaths(
        root=root,
        original=root / "original.wav",
        enhanced_full=root / "enhanced_full.wav",
        enhanced_mix=root / "enhanced_mix.wav",
        transcript_before=root / "transcript_before.json",
        transcript_after=root / "transcript_after.json",
        waveform=root / "waveform_compare.png",
        spectrogram=root / "spectrogram_compare.png",
        result_json=root / "result.json",
        run_log=root / "run.log",
    )


def process_audio(
    input_path: str,
    strength: float = 0.75,
    enable_events: bool = False,
    reference_text: str | None = None,
    force_recompute: bool = False,
) -> ProcessResult:
    """Frozen public entry point; implementation follows first A/B integration."""

    raise NotImplementedError("pipeline implementation starts after A/B module smoke tests")


__all__ = ["JobPaths", "create_job_paths", "new_job_id", "process_audio"]
