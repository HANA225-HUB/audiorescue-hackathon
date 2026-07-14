"""Frozen public contracts shared by A, B, and C.

Contract version: v0.1-contract
Only C may change this file after team review.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from math import isfinite
from typing import Any, TypedDict


class ProcessStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


class ErrorCode(str, Enum):
    # Input and normalization
    INPUT_INVALID = "INPUT_INVALID"
    INPUT_TOO_LONG = "INPUT_TOO_LONG"
    INPUT_CLIPPED = "INPUT_CLIPPED"
    INPUT_NEAR_SILENT = "INPUT_NEAR_SILENT"
    INPUT_STEREO_DOWNMIXED = "INPUT_STEREO_DOWNMIXED"
    INPUT_RESAMPLED = "INPUT_RESAMPLED"

    # Enhancement and output validation
    ENHANCE_FAILED = "ENHANCE_FAILED"
    OUTPUT_INVALID = "OUTPUT_INVALID"
    OUTPUT_PEAK_PROTECTED = "OUTPUT_PEAK_PROTECTED"

    # ASR. ASR_FAILED is internal; pipeline maps it to before/after.
    ASR_FAILED = "ASR_FAILED"
    ASR_BEFORE_FAILED = "ASR_BEFORE_FAILED"
    ASR_AFTER_FAILED = "ASR_AFTER_FAILED"
    ASR_EMPTY = "ASR_EMPTY"

    # Optional or downstream stages
    VIS_FAILED = "VIS_FAILED"
    EVENTS_SKIPPED = "EVENTS_SKIPPED"
    CACHE_USED = "CACHE_USED"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class TranscriptSegment(TypedDict):
    start: float
    end: float
    text: str


@dataclass(slots=True)
class AudioMeta:
    """Metadata for the normalized track.

    Optional original_* fields preserve the source properties before resampling
    or downmixing. rms_dbfs/silent_ratio may be None when inspection could not
    compute them reliably, but all other core fields are required.
    """

    source_name: str
    sample_rate: int
    channels: int
    duration_seconds: float
    peak_abs: float
    rms_dbfs: float | None
    clipped_ratio: float
    silent_ratio: float | None
    normalized_path: str
    original_sample_rate: int | None = None
    original_channels: int | None = None
    source_format: str | None = None


@dataclass(slots=True)
class AudioLevelMetrics:
    """Comparable level evidence for one persisted audio track.

    ``peak_abs`` uses normalized full scale, so valid values are finite and in
    the inclusive range 0..1. ``rms_dbfs`` is finite for non-silent audio and
    ``None`` for digital silence; infinity is never used as a JSON sentinel.
    """

    peak_abs: float
    rms_dbfs: float | None

    def __post_init__(self) -> None:
        self.peak_abs = _validated_finite_float(self.peak_abs, "peak_abs")
        if not 0.0 <= self.peak_abs <= 1.0:
            raise ValueError("peak_abs must be between 0.0 and 1.0")

        if self.rms_dbfs is not None:
            self.rms_dbfs = _validated_finite_float(self.rms_dbfs, "rms_dbfs")


@dataclass(slots=True)
class TranscriptResult:
    text: str
    language: str | None
    segments: list[TranscriptSegment]
    runtime_seconds: float
    model_name: str
    error: str | None = None


@dataclass(slots=True)
class CerResult:
    reference: str
    normalized_reference: str
    normalized_hypothesis: str
    cer: float
    substitutions: int
    deletions: int
    insertions: int


@dataclass(slots=True)
class EventResult:
    label: str
    score: float
    start_seconds: float
    end_seconds: float


@dataclass(slots=True)
class WarningItem:
    code: ErrorCode
    message: str
    module: str
    recoverable: bool
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RuntimeStats:
    decode_seconds: float = 0.0
    enhancer_load_seconds: float = 0.0
    enhancement_seconds: float = 0.0
    asr_load_seconds: float = 0.0
    asr_before_seconds: float = 0.0
    asr_after_seconds: float = 0.0
    visualization_seconds: float = 0.0
    event_seconds: float = 0.0
    persistence_seconds: float = 0.0
    total_seconds: float = 0.0
    cache_hit: bool = False
    cold_start: bool = False
    device: str = "cpu"


@dataclass(slots=True)
class ProcessResult:
    job_id: str
    status: ProcessStatus
    runtime: RuntimeStats = field(default_factory=RuntimeStats)

    input_meta: AudioMeta | None = None
    original_audio_path: str | None = None

    # Compatibility alias: enhanced_audio_path always points to mixed_output_path.
    enhanced_audio_path: str | None = None
    full_output_path: str | None = None
    mixed_output_path: str | None = None
    original_levels: AudioLevelMetrics | None = None
    mixed_levels: AudioLevelMetrics | None = None

    transcript_before: TranscriptResult | None = None
    transcript_after: TranscriptResult | None = None
    cer_before: CerResult | None = None
    cer_after: CerResult | None = None
    text_diff_html: str | None = None
    waveform_path: str | None = None
    spectrogram_path: str | None = None
    events: list[EventResult] = field(default_factory=list)
    warnings: list[WarningItem] = field(default_factory=list)
    config_snapshot: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.enhanced_audio_path is None and self.mixed_output_path is not None:
            self.enhanced_audio_path = self.mixed_output_path
        elif self.mixed_output_path is None and self.enhanced_audio_path is not None:
            self.mixed_output_path = self.enhanced_audio_path
        elif (
            self.enhanced_audio_path is not None
            and self.mixed_output_path is not None
            and self.enhanced_audio_path != self.mixed_output_path
        ):
            raise ValueError(
                "enhanced_audio_path must be identical to mixed_output_path"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready dictionary, including enum values as strings."""

        return _json_ready(asdict(self))


class EnhancementOutput(TypedDict):
    full_output_path: str
    mixed_output_path: str
    strength: float
    runtime_seconds: float
    model_name: str
    warnings: list[WarningItem]


def _validated_finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, (bool, str, bytes)):
        raise TypeError(f"{field_name} must be a real number")
    try:
        numeric_value = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError(f"{field_name} must be a real number") from error
    if not isfinite(numeric_value):
        raise ValueError(f"{field_name} must be finite")
    return numeric_value


def _json_ready(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("JSON numbers must be finite")
        return value
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("JSON object keys must be strings")
            normalized[key] = _json_ready(item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    raise TypeError(f"value is not JSON-compatible: {type(value).__name__}")


class AudioRescueError(Exception):
    code = ErrorCode.INTERNAL_ERROR
    module = "core"
    recoverable = False

    def __init__(
        self,
        public_message: str,
        *,
        detail: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.public_message = public_message
        self.detail = detail
        self.details = details or {}
        super().__init__(detail or public_message)

    def to_warning(
        self,
        *,
        code: ErrorCode | None = None,
        module: str | None = None,
    ) -> WarningItem:
        return WarningItem(
            code=code or self.code,
            message=self.public_message,
            module=module or self.module,
            recoverable=self.recoverable,
            details=self.details,
        )


class InputAudioError(AudioRescueError):
    code = ErrorCode.INPUT_INVALID
    module = "audio_io"


class InputTooLongError(AudioRescueError):
    code = ErrorCode.INPUT_TOO_LONG
    module = "audio_io"


class EnhancementError(AudioRescueError):
    code = ErrorCode.ENHANCE_FAILED
    module = "enhance"


class OutputValidationError(AudioRescueError):
    code = ErrorCode.OUTPUT_INVALID
    module = "enhance"


class ASRInferenceError(AudioRescueError):
    code = ErrorCode.ASR_FAILED
    module = "transcribe"
    recoverable = True


class VisualizationError(AudioRescueError):
    code = ErrorCode.VIS_FAILED
    module = "visualize"
    recoverable = True


__all__ = [
    "ASRInferenceError",
    "AudioLevelMetrics",
    "AudioMeta",
    "AudioRescueError",
    "CerResult",
    "EnhancementError",
    "EnhancementOutput",
    "ErrorCode",
    "EventResult",
    "InputAudioError",
    "InputTooLongError",
    "OutputValidationError",
    "ProcessResult",
    "ProcessStatus",
    "RuntimeStats",
    "TranscriptResult",
    "TranscriptSegment",
    "VisualizationError",
    "WarningItem",
]
