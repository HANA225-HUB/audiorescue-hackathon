"""C-owned orchestration boundary for the AudioRescue P0 pipeline.

The public ``process_audio`` signature is frozen. Model and UI modules are
imported lazily so importing this module never downloads weights or starts a
GPU workload. Tests replace the dependency loader with deterministic fakes;
production uses the A/B-owned modules through the same narrow contracts.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import threading
import time
import traceback
import wave
from array import array
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

from core.cache import LocalTaskCache, build_cache_key, sha256_file
from core.metrics import compute_cer
from core.schemas import (
    ASRInferenceError,
    AudioLevelMetrics,
    AudioMeta,
    AudioRescueError,
    CerResult,
    EnhancementError,
    ErrorCode,
    EventResult,
    InputAudioError,
    InputTooLongError,
    OutputValidationError,
    ProcessResult,
    ProcessStatus,
    RuntimeStats,
    TranscriptResult,
    VisualizationError,
    WarningItem,
)


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "configs" / "app.yaml"
_DEFAULT_DEMO_CONFIG_PATH = _PROJECT_ROOT / "configs" / "demo.yaml"
_DEFAULT_LABELS_CONFIG_PATH = _PROJECT_ROOT / "configs" / "labels.yaml"
_PIPELINE_CACHE_VERSION = "pipeline-v0.1.0"
_SAFE_JOB_ID = re.compile(r"^[A-Za-z0-9_-]+$")
_PROCESS_LOCK = threading.Lock()
_MODEL_STATE_LOCK = threading.Lock()
_ENHANCER_WARMED_KEY: tuple[int, str] | None = None
_ASR_WARMED_KEY: tuple[int, str, str] | None = None


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


@dataclass(frozen=True, slots=True)
class PipelineDependencies:
    """Callables owned by A/B and loaded only when a request runs."""

    normalize_audio: Callable[..., AudioMeta]
    enhance_audio: Callable[..., Mapping[str, Any]]
    transcribe_audio: Callable[..., TranscriptResult]
    create_waveform_comparison: Callable[..., str]
    create_spectrogram_comparison: Callable[..., str]
    build_text_diff: Callable[[str, str], str]
    detect_events: Callable[..., list[EventResult]]
    load_enhancer: Callable[..., Any] | None = None
    load_asr: Callable[..., Any] | None = None


class _RunLog:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def add(self, stage: str, message: str) -> None:
        self.lines.append(f"[{stage}] {message}")

    def exception(self, stage: str, error: BaseException) -> None:
        self.add(stage, f"{type(error).__name__}: {error}")
        self.lines.append(traceback.format_exc().rstrip())

    def render(self) -> str:
        return "\n".join(self.lines).rstrip() + "\n"


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
    """Run one isolated P0 job and return a stable, JSON-ready result object."""

    # DeepFilterNet state and GPU memory are intentionally single-request in P0.
    with _PROCESS_LOCK:
        try:
            return _process_audio_locked(
                input_path=input_path,
                strength=strength,
                enable_events=enable_events,
                reference_text=reference_text,
                force_recompute=force_recompute,
            )
        except Exception as error:
            # A broken local config or unwritable output directory must still
            # become a renderable failed result instead of crashing Gradio.
            return ProcessResult(
                job_id=new_job_id(),
                status=ProcessStatus.FAILED,
                runtime=RuntimeStats(),
                warnings=[
                    WarningItem(
                        code=ErrorCode.INTERNAL_ERROR,
                        message="系统无法创建本次任务，请检查本地配置和输出目录。",
                        module="pipeline",
                        recoverable=False,
                        details={"exception_type": type(error).__name__},
                    )
                ],
                config_snapshot={"pipeline_version": _PIPELINE_CACHE_VERSION},
            )


def _process_audio_locked(
    *,
    input_path: str,
    strength: float,
    enable_events: bool,
    reference_text: str | None,
    force_recompute: bool,
) -> ProcessResult:
    started = time.monotonic()
    log = _RunLog()
    config, config_path = _load_app_config()
    output_root = _resolve_output_root(config)
    paths = create_job_paths(output_root)
    normalized_strength = _normalize_request_strength(strength)
    valid_reference = reference_text is None or isinstance(reference_text, str)
    valid_flags = isinstance(enable_events, bool) and isinstance(force_recompute, bool)
    snapshot = _config_snapshot(
        config,
        normalized_strength,
        enable_events if isinstance(enable_events, bool) else False,
        reference_text if isinstance(reference_text, str) else None,
    )
    device = _resolve_device(config.get("asr", {}).get("device", "auto"))
    result = ProcessResult(
        job_id=paths.root.name,
        status=ProcessStatus.FAILED,
        runtime=RuntimeStats(device=device),
        config_snapshot=snapshot,
    )

    if normalized_strength is None or not valid_reference or not valid_flags:
        result.warnings.append(
            WarningItem(
                code=ErrorCode.INPUT_INVALID,
                message="请求参数格式无效，请检查增强强度、开关和参考文本。",
                module="pipeline",
                recoverable=False,
                details={
                    "strength_type": type(strength).__name__,
                    "reference_type": type(reference_text).__name__,
                    "enable_events_type": type(enable_events).__name__,
                    "force_recompute_type": type(force_recompute).__name__,
                },
            )
        )
        log.add("request", "request parameter types or strength value are invalid")
        return _finalize(result, paths, log, started)

    input_file = Path(input_path).expanduser()
    if not input_file.is_file() or input_file.stat().st_size == 0:
        result.warnings.append(
            WarningItem(
                code=ErrorCode.INPUT_INVALID,
                message="输入音频不存在或为空，请重新选择可读取的音频文件。",
                module="pipeline",
                recoverable=False,
                details={},
            )
        )
        log.add("request", "input path is missing, not a file, or empty")
        return _finalize(result, paths, log, started)

    cache_context: tuple[LocalTaskCache, str] | None = None
    if not force_recompute:
        try:
            cache_context = _prepare_cache(
                input_file=input_file,
                strength=normalized_strength,
                enable_events=enable_events,
                reference_text=reference_text,
                resolved_device=device,
                config=config,
                config_path=config_path,
            )
            cached_payload = (
                cache_context[0].load(cache_context[1])
                if cache_context is not None
                else None
            )
            if cached_payload is not None and _cached_artifacts_exist(cached_payload):
                cached = _process_result_from_dict(cached_payload)
                lookup_seconds = time.monotonic() - started
                cached.config_snapshot = dict(cached.config_snapshot)
                cached.config_snapshot["cached_runtime"] = {
                    field: getattr(cached.runtime, field)
                    for field in RuntimeStats.__dataclass_fields__
                }
                cached.runtime = RuntimeStats(
                    total_seconds=lookup_seconds,
                    cache_hit=True,
                    cold_start=False,
                    device=cached.runtime.device,
                )
                cached.warnings.append(
                    WarningItem(
                        code=ErrorCode.CACHE_USED,
                        message="已使用本地缓存结果，本次没有重新执行模型推理。",
                        module="cache",
                        recoverable=True,
                        details={"lookup_seconds": lookup_seconds},
                    )
                )
                log.add("cache", f"hit {cache_context[1]}")
                # The cached result keeps its original job identity and paths.
                # Remove the still-empty probe directory so a hit cannot leave
                # an unrelated orphan job behind.
                try:
                    paths.root.rmdir()
                except OSError:
                    pass
                return cached
        except Exception as error:
            # Cache is an optimization. A permission, hash, corruption, or
            # fingerprint error is always a miss, never a failed P0 request.
            log.exception("cache_lookup", error)

    try:
        dependencies = _load_default_dependencies()
    except Exception as error:
        log.exception("dependencies", error)
        result.warnings.append(
            WarningItem(
                code=ErrorCode.INTERNAL_ERROR,
                message="处理模块尚未完整安装或加载失败，请检查 A/B 模块和依赖。",
                module="pipeline",
                recoverable=False,
                details={"exception_type": type(error).__name__},
            )
        )
        return _finalize(result, paths, log, started)

    # Normalize and inspect. Failure here is terminal because all later stages
    # require the standardized original track.
    stage_started = time.monotonic()
    try:
        meta = dependencies.normalize_audio(
            str(input_file.resolve()),
            str(paths.original),
            target_sr=int(config["audio"]["target_sample_rate"]),
            mono=bool(config["audio"]["channels"] == 1),
        )
        meta = _validated_audio_meta(meta)
        target_sample_rate = int(config["audio"]["target_sample_rate"])
        target_channels = int(config["audio"]["channels"])
        original_frames = _validate_pcm_wav(
            paths.original,
            sample_rate=target_sample_rate,
            channels=target_channels,
        )
        actual_duration = original_frames / target_sample_rate
        minimum_duration = float(config["audio"].get("min_duration_seconds", 0))
        maximum_duration = float(config["audio"].get("max_duration_seconds", 60))
        if actual_duration < minimum_duration:
            raise InputAudioError(
                f"音频时长不足 {minimum_duration:g} 秒，请选择更完整的片段。"
            )
        if actual_duration > maximum_duration:
            raise InputTooLongError(
                f"音频超过 {maximum_duration:g} 秒，请先截取短片段。"
            )
        if (
            meta.sample_rate != target_sample_rate
            or meta.channels != target_channels
            or Path(meta.normalized_path).resolve() != paths.original
            or abs(meta.duration_seconds - actual_duration) > 0.02
        ):
            raise InputAudioError("标准化元数据与实际 WAV 文件不一致。")
        result.input_meta = meta
        result.original_audio_path = str(paths.original)
        result.original_levels = _compute_audio_levels(paths.original)
        _append_input_warnings(result, meta)
        log.add("normalize", f"ok {actual_duration:.3f}s")
    except (InputAudioError, InputTooLongError) as error:
        log.exception("normalize", error)
        result.warnings.append(error.to_warning())
        result.runtime.decode_seconds = time.monotonic() - stage_started
        return _finalize(result, paths, log, started)
    except Exception as error:
        log.exception("normalize", error)
        result.warnings.append(
            _unexpected_warning(
                code=ErrorCode.INPUT_INVALID,
                module="audio_io",
                message="输入音频无法完成标准化，请更换文件后重试。",
                error=error,
                recoverable=False,
            )
        )
        result.runtime.decode_seconds = time.monotonic() - stage_started
        return _finalize(result, paths, log, started)
    result.runtime.decode_seconds = time.monotonic() - stage_started

    _warm_models(result, dependencies, config, device, log)

    enhancement_succeeded = False
    stage_started = time.monotonic()
    try:
        enhancement = dependencies.enhance_audio(
            str(paths.original),
            str(paths.enhanced_full),
            str(paths.enhanced_mix),
            strength=normalized_strength,
        )
        if not isinstance(enhancement, Mapping):
            raise OutputValidationError("增强模块返回结构无效。")
        enhancement_warnings = _coerce_warnings(enhancement.get("warnings", []))
        _validate_enhancement_contract(
            enhancement,
            paths,
            config,
            requested_strength=normalized_strength,
        )
        result.full_output_path = str(paths.enhanced_full)
        result.mixed_output_path = str(paths.enhanced_mix)
        result.enhanced_audio_path = str(paths.enhanced_mix)
        result.mixed_levels = _compute_audio_levels(paths.enhanced_mix)
        result.warnings.extend(enhancement_warnings)
        enhancement_succeeded = True
        log.add("enhancement", "full and mixed tracks validated")
    except (EnhancementError, OutputValidationError) as error:
        log.exception("enhancement", error)
        result.warnings.append(error.to_warning())
    except Exception as error:
        log.exception("enhancement", error)
        result.warnings.append(
            _unexpected_warning(
                code=ErrorCode.ENHANCE_FAILED,
                module="enhance",
                message="音频增强失败，已保留能够生成的原轨结果。",
                error=error,
                recoverable=False,
            )
        )
    result.runtime.enhancement_seconds = time.monotonic() - stage_started

    # The two ASR calls are independent. Original ASR is still useful when
    # enhancement fails, while an after-track failure must not discard before.
    result.transcript_before = _run_asr(
        dependencies=dependencies,
        audio_path=paths.original,
        language=str(config["asr"]["language"]),
        expected_model=str(config["asr"]["model"]),
        device=device,
        result=result,
        runtime_field="asr_before_seconds",
        failure_code=ErrorCode.ASR_BEFORE_FAILED,
        track_name="原轨",
        log=log,
    )
    if enhancement_succeeded:
        result.transcript_after = _run_asr(
            dependencies=dependencies,
            audio_path=paths.enhanced_mix,
            language=str(config["asr"]["language"]),
            expected_model=str(config["asr"]["model"]),
            device=device,
            result=result,
            runtime_field="asr_after_seconds",
            failure_code=ErrorCode.ASR_AFTER_FAILED,
            track_name="增强轨",
            log=log,
        )

    _persist_transcript(paths.transcript_before, result.transcript_before, log)
    _persist_transcript(paths.transcript_after, result.transcript_after, log)

    if reference_text is not None and reference_text.strip():
        try:
            if result.transcript_before is not None:
                result.cer_before = compute_cer(reference_text, result.transcript_before.text)
            if result.transcript_after is not None:
                result.cer_after = compute_cer(reference_text, result.transcript_after.text)
        except ValueError as error:
            log.exception("metrics", error)
            result.warnings.append(
                _unexpected_warning(
                    code=ErrorCode.INTERNAL_ERROR,
                    module="metrics",
                    message="参考文本无法用于 CER，已跳过该指标。",
                    error=error,
                    recoverable=True,
                )
            )

    if result.transcript_before is not None and result.transcript_after is not None:
        try:
            text_diff_html = dependencies.build_text_diff(
                result.transcript_before.text,
                result.transcript_after.text,
            )
            if not isinstance(text_diff_html, str):
                raise TypeError("text diff must be a string")
            result.text_diff_html = text_diff_html
        except Exception as error:
            log.exception("text_diff", error)
            result.warnings.append(
                _unexpected_warning(
                    code=ErrorCode.INTERNAL_ERROR,
                    module="text_diff",
                    message="文本差异暂时无法生成，转写文本仍可正常查看。",
                    error=error,
                    recoverable=True,
                )
            )

    if enhancement_succeeded:
        _run_visualizations(result, paths, dependencies, log)

    if enable_events and result.original_audio_path is not None:
        stage_started = time.monotonic()
        try:
            labels = _load_event_labels()
            result.events = _validated_event_results(
                dependencies.detect_events(
                    result.original_audio_path,
                    labels=labels,
                    enabled=True,
                )
            )
        except Exception as error:
            log.exception("events", error)
            result.warnings.append(
                _unexpected_warning(
                    code=ErrorCode.EVENTS_SKIPPED,
                    module="events",
                    message="环境事件检测未完成，P0 音频结果不受影响。",
                    error=error,
                    recoverable=True,
                )
            )
        result.runtime.event_seconds = time.monotonic() - stage_started

    result.status = _derive_status(result, enhancement_succeeded)
    finalized = _finalize(result, paths, log, started)

    if cache_context is None:
        try:
            cache_context = _prepare_cache(
                input_file=input_file,
                strength=normalized_strength,
                enable_events=enable_events,
                reference_text=reference_text,
                resolved_device=device,
                config=config,
                config_path=config_path,
            )
        except Exception as error:
            log.exception("cache_prepare_save", error)

    if (
        cache_context is not None
        and finalized.status in {ProcessStatus.SUCCESS, ProcessStatus.PARTIAL}
        and _cached_artifacts_exist(finalized.to_dict())
    ):
        cache, cache_key = cache_context
        try:
            cache.save(cache_key, finalized.to_dict())
            log.add("cache", f"saved {cache_key}")
            _write_log(paths.run_log, log)
        except Exception as error:
            # Cache is an optimization and cannot downgrade a completed P0 job.
            log.exception("cache", error)
            _write_log(paths.run_log, log)

    return finalized


def _load_default_dependencies() -> PipelineDependencies:
    from core.audio_io import normalize_audio
    from core.enhance import enhance_audio, load_enhancer
    from core.events import detect_events
    from core.text_diff import build_text_diff
    from core.transcribe import load_asr, transcribe_audio
    from core.visualize import (
        create_spectrogram_comparison,
        create_waveform_comparison,
    )

    return PipelineDependencies(
        normalize_audio=normalize_audio,
        enhance_audio=enhance_audio,
        transcribe_audio=transcribe_audio,
        create_waveform_comparison=create_waveform_comparison,
        create_spectrogram_comparison=create_spectrogram_comparison,
        build_text_diff=build_text_diff,
        detect_events=detect_events,
        load_enhancer=load_enhancer,
        load_asr=load_asr,
    )


def _load_app_config(path: str | Path | None = None) -> tuple[dict[str, Any], Path]:
    configured_path = path or os.environ.get("AUDIORESCUE_CONFIG") or _DEFAULT_CONFIG_PATH
    config_path = Path(configured_path).expanduser().resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("app config must be a mapping")
    payload = _json_safe_copy(payload)
    for required in ("contract_version", "app", "audio", "enhancement", "asr"):
        if required not in payload:
            raise ValueError(f"app config is missing {required}")
    return payload, config_path


def _resolve_output_root(config: Mapping[str, Any]) -> Path:
    configured = Path(str(config["app"]["output_root"])).expanduser()
    if configured.is_absolute():
        return configured.resolve()
    return (_PROJECT_ROOT / configured).resolve()


def _config_snapshot(
    config: Mapping[str, Any],
    strength: float | None,
    enable_events: bool,
    reference_text: str | None,
) -> dict[str, Any]:
    reference_hash = None
    if reference_text is not None:
        reference_hash = hashlib.sha256(reference_text.encode("utf-8")).hexdigest()
    public_config = _public_processing_config(config)
    return {
        "contract_version": config["contract_version"],
        **public_config,
        "request": {
            "strength": strength,
            "enable_events": bool(enable_events),
            "reference_text_sha256": reference_hash,
        },
        "pipeline_version": _PIPELINE_CACHE_VERSION,
    }


def _normalize_request_strength(value: Any) -> float | None:
    """Return one finite 0..1 float, rejecting bools and implicit strings."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        return None
    return numeric


def _prepare_cache(
    *,
    input_file: Path,
    strength: float,
    enable_events: bool,
    reference_text: str | None,
    resolved_device: str,
    config: Mapping[str, Any],
    config_path: Path,
) -> tuple[LocalTaskCache, str] | None:
    cache_config = config.get("cache", {})
    if not cache_config.get("enabled", False):
        return None
    if cache_config.get("demo_only", True):
        if not _is_frozen_demo(input_file):
            return None
    elif not cache_config.get("allow_user_audio", False):
        # User audio is never cached merely because demo_only was toggled.
        # It requires a second explicit local-only consent switch.
        return None

    reference_hash = None
    if reference_text is not None:
        reference_hash = hashlib.sha256(reference_text.encode("utf-8")).hexdigest()
    model_name = f"{config['enhancement']['model']}+whisper-{config['asr']['model']}"
    public_config = _public_processing_config(config)
    event_labels_sha256 = None
    if enable_events:
        event_labels_sha256 = sha256_file(_DEFAULT_LABELS_CONFIG_PATH)
    key = build_cache_key(
        input_file,
        strength=strength,
        model_name=model_name,
        language=str(config["asr"]["language"]),
        contract_version=str(config["contract_version"]),
        config_version=sha256_file(config_path),
        code_version=_code_fingerprint(),
        processing_config={
            "enable_events": enable_events,
            "reference_text_sha256": reference_hash,
            "resolved_device": resolved_device,
            "event_labels_sha256": event_labels_sha256,
            **public_config,
        },
    )
    return LocalTaskCache(_PROJECT_ROOT / ".cache" / "task-results"), key


def _is_frozen_demo(input_file: Path) -> bool:
    try:
        payload = yaml.safe_load(_DEFAULT_DEMO_CONFIG_PATH.read_text(encoding="utf-8"))
        samples = payload.get("samples", []) if isinstance(payload, dict) else []
        resolved_input = input_file.resolve()
        allowed_uses = {
            "development_only",
            "main_demo",
            "boundary_demo",
            "p1_demo",
        }
        for sample in samples:
            if (
                not isinstance(sample, dict)
                or not sample.get("file")
                or sample.get("intended_use") not in allowed_uses
                or not sample.get("sha256")
            ):
                continue
            configured = (_PROJECT_ROOT / str(sample["file"])).resolve()
            if configured != resolved_input:
                continue
            return sha256_file(resolved_input) == str(sample["sha256"]).lower()
    except (OSError, ValueError, yaml.YAMLError):
        return False
    return False


def _public_processing_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return only non-secret, result-affecting fields safe for result.json."""

    allowed = {
        "audio": {
            "target_sample_rate",
            "channels",
            "pcm_subtype",
            "min_duration_seconds",
            "max_duration_seconds",
            "peak_limit",
        },
        "enhancement": {
            "backend",
            "model",
            "checkpoint_sha256",
            "config_sha256",
            "default_strength",
            "strengths",
            "default_playback_track",
            "default_asr_after_track",
        },
        "asr": {
            "backend",
            "model",
            "checkpoint_sha256",
            "language",
            "task",
            "device",
            "temperature",
            "condition_on_previous_text",
            "initial_prompt",
            "fp16_on_cuda",
        },
        "comparison": {
            "loudness_dsp_matching",
            "same_player_volume",
            "report_rms_dbfs",
            "report_peak_abs",
        },
    }
    return {
        section: {
            key: value
            for key, value in dict(config.get(section, {})).items()
            if key in keys
        }
        for section, keys in allowed.items()
    }


def _code_fingerprint() -> str:
    """Hash every implementation file that can change a cached P0 result."""

    relative_paths = (
        "core/schemas.py",
        "core/pipeline.py",
        "core/cache.py",
        "core/metrics.py",
        "core/events.py",
        "core/audio_io.py",
        "core/enhance.py",
        "core/transcribe.py",
        "core/visualize.py",
        "core/text_diff.py",
    )
    digest = hashlib.sha256()
    for relative in relative_paths:
        path = _PROJECT_ROOT / relative
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        if path.is_file():
            digest.update(path.read_bytes())
        else:
            digest.update(b"<missing>")
        digest.update(b"\0")
    return digest.hexdigest()


def _resolve_device(configured: Any) -> str:
    value = str(configured).lower()
    if value != "auto":
        return value
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _warm_models(
    result: ProcessResult,
    dependencies: PipelineDependencies,
    config: Mapping[str, Any],
    device: str,
    log: _RunLog,
) -> None:
    global _ASR_WARMED_KEY, _ENHANCER_WARMED_KEY

    with _MODEL_STATE_LOCK:
        enhancer_key = (
            id(dependencies.load_enhancer),
            str(config["enhancement"]["model"]),
        )
        if (
            dependencies.load_enhancer is not None
            and enhancer_key != _ENHANCER_WARMED_KEY
        ):
            started = time.monotonic()
            try:
                dependencies.load_enhancer()
                _ENHANCER_WARMED_KEY = enhancer_key
                result.runtime.cold_start = True
                log.add("warmup", "enhancer loaded")
            except Exception as error:
                log.exception("warmup_enhancer", error)
            result.runtime.enhancer_load_seconds = time.monotonic() - started

        asr_key = (
            id(dependencies.load_asr),
            str(config["asr"]["model"]),
            device,
        )
        if dependencies.load_asr is not None and asr_key != _ASR_WARMED_KEY:
            started = time.monotonic()
            try:
                dependencies.load_asr(str(config["asr"]["model"]), device)
                _ASR_WARMED_KEY = asr_key
                result.runtime.cold_start = True
                log.add("warmup", "asr loaded")
            except Exception as error:
                log.exception("warmup_asr", error)
            result.runtime.asr_load_seconds = time.monotonic() - started


def _validated_transcript_result(value: Any, track_name: str) -> TranscriptResult:
    """Validate and rebuild one ASR result before it reaches persistence."""

    if not isinstance(value, TranscriptResult):
        raise ASRInferenceError(f"{track_name}转写模块返回结构无效。")
    if not isinstance(value.text, str):
        raise ASRInferenceError(f"{track_name}转写 text 必须是字符串。")
    if value.language is not None and not isinstance(value.language, str):
        raise ASRInferenceError(f"{track_name}转写 language 必须是字符串或空值。")
    if not isinstance(value.model_name, str) or not value.model_name.strip():
        raise ASRInferenceError(f"{track_name}转写 model_name 无效。")
    if value.error is not None and not isinstance(value.error, str):
        raise ASRInferenceError(f"{track_name}转写 error 必须是字符串或空值。")
    if isinstance(value.runtime_seconds, (bool, str, bytes)):
        raise ASRInferenceError(f"{track_name}转写 runtime_seconds 无效。")
    try:
        runtime_seconds = float(value.runtime_seconds)
    except (TypeError, ValueError, OverflowError) as error:
        raise ASRInferenceError(
            f"{track_name}转写 runtime_seconds 无效。"
        ) from error
    if not math.isfinite(runtime_seconds) or runtime_seconds < 0:
        raise ASRInferenceError(f"{track_name}转写 runtime_seconds 无效。")
    if not isinstance(value.segments, list):
        raise ASRInferenceError(f"{track_name}转写 segments 必须是列表。")

    normalized_segments: list[dict[str, Any]] = []
    for index, segment in enumerate(value.segments):
        if not isinstance(segment, Mapping):
            raise ASRInferenceError(f"{track_name}第 {index + 1} 个分段结构无效。")
        if set(segment) != {"start", "end", "text"}:
            raise ASRInferenceError(f"{track_name}第 {index + 1} 个分段字段无效。")
        if not isinstance(segment["text"], str):
            raise ASRInferenceError(f"{track_name}第 {index + 1} 个分段文本无效。")
        start = _finite_external_number(
            segment["start"], f"{track_name}第 {index + 1} 个分段 start"
        )
        end = _finite_external_number(
            segment["end"], f"{track_name}第 {index + 1} 个分段 end"
        )
        if start < 0 or end < start:
            raise ASRInferenceError(f"{track_name}第 {index + 1} 个分段时间无效。")
        normalized_segments.append(
            {"start": start, "end": end, "text": str(segment["text"])}
        )

    return TranscriptResult(
        text=str(value.text),
        language=None if value.language is None else str(value.language),
        segments=normalized_segments,
        runtime_seconds=runtime_seconds,
        model_name=str(value.model_name),
        error=None if value.error is None else str(value.error),
    )


def _finite_external_number(value: Any, field_name: str) -> float:
    if isinstance(value, (bool, str, bytes)):
        raise ASRInferenceError(f"{field_name} 必须是有限数字。")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ASRInferenceError(f"{field_name} 必须是有限数字。") from error
    if not math.isfinite(numeric):
        raise ASRInferenceError(f"{field_name} 必须是有限数字。")
    return numeric


def _run_asr(
    *,
    dependencies: PipelineDependencies,
    audio_path: Path,
    language: str,
    expected_model: str,
    device: str,
    result: ProcessResult,
    runtime_field: str,
    failure_code: ErrorCode,
    track_name: str,
    log: _RunLog,
) -> TranscriptResult | None:
    started = time.monotonic()
    try:
        transcript = dependencies.transcribe_audio(
            str(audio_path),
            language=language,
            model_name=expected_model,
            device=device,
        )
        transcript = _validated_transcript_result(transcript, track_name)
        if transcript.error is not None:
            raise ASRInferenceError(
                f"{track_name}转写返回了 error 字段，不能标记为成功。",
                detail=transcript.error,
            )
        if not math.isfinite(transcript.runtime_seconds) or transcript.runtime_seconds < 0:
            raise ASRInferenceError(f"{track_name}转写耗时字段无效。")
        normalized_model = transcript.model_name.strip().lower()
        allowed_models = {
            expected_model.strip().lower(),
            f"whisper-{expected_model.strip().lower()}",
            f"openai-whisper-{expected_model.strip().lower()}",
        }
        if normalized_model not in allowed_models:
            raise ASRInferenceError(f"{track_name}使用的 Whisper 模型与冻结配置不一致。")
        if transcript.language is not None and transcript.language.lower() != language.lower():
            raise ASRInferenceError(f"{track_name}转写语言与冻结配置不一致。")
        if not transcript.text.strip():
            result.warnings.append(
                WarningItem(
                    code=ErrorCode.ASR_EMPTY,
                    message=f"{track_name}转写结果为空。",
                    module="transcribe",
                    recoverable=True,
                    details={"track": track_name},
                )
            )
        log.add("asr", f"{track_name} ok, {len(transcript.text)} characters")
        return transcript
    except ASRInferenceError as error:
        log.exception("asr", error)
        result.warnings.append(error.to_warning(code=failure_code))
    except Exception as error:
        log.exception("asr", error)
        result.warnings.append(
            _unexpected_warning(
                code=failure_code,
                module="transcribe",
                message=f"{track_name}转写失败，其他已完成结果仍然保留。",
                error=error,
                recoverable=True,
            )
        )
    finally:
        setattr(result.runtime, runtime_field, time.monotonic() - started)
    return None


def _run_visualizations(
    result: ProcessResult,
    paths: JobPaths,
    dependencies: PipelineDependencies,
    log: _RunLog,
) -> None:
    started = time.monotonic()
    failed = False
    try:
        waveform_path = dependencies.create_waveform_comparison(
            str(paths.original), str(paths.enhanced_mix), str(paths.waveform)
        )
        result.waveform_path = _validated_file_path(waveform_path, paths.waveform)
    except Exception as error:
        failed = True
        log.exception("waveform", error)

    try:
        spectrogram_path = dependencies.create_spectrogram_comparison(
            str(paths.original), str(paths.enhanced_mix), str(paths.spectrogram)
        )
        result.spectrogram_path = _validated_file_path(
            spectrogram_path, paths.spectrogram
        )
    except Exception as error:
        failed = True
        log.exception("spectrogram", error)

    if failed:
        result.warnings.append(
            VisualizationError(
                "部分或全部对照图生成失败，音频与转写结果仍可使用。"
            ).to_warning()
        )
    result.runtime.visualization_seconds = time.monotonic() - started


def _derive_status(result: ProcessResult, enhancement_succeeded: bool) -> ProcessStatus:
    if not enhancement_succeeded or result.mixed_output_path is None:
        return ProcessStatus.FAILED
    degraders = {
        ErrorCode.ASR_BEFORE_FAILED,
        ErrorCode.ASR_AFTER_FAILED,
        ErrorCode.ASR_EMPTY,
        ErrorCode.VIS_FAILED,
    }
    if any(warning.code in degraders for warning in result.warnings):
        return ProcessStatus.PARTIAL
    if result.transcript_before is None or result.transcript_after is None:
        return ProcessStatus.PARTIAL
    if result.waveform_path is None or result.spectrogram_path is None:
        return ProcessStatus.PARTIAL
    return ProcessStatus.SUCCESS


def _validate_enhancement_contract(
    enhancement: Mapping[str, Any],
    paths: JobPaths,
    config: Mapping[str, Any],
    *,
    requested_strength: float,
) -> None:
    try:
        returned_strength = float(enhancement["strength"])
        runtime_seconds = float(enhancement["runtime_seconds"])
        model_name = str(enhancement["model_name"]).strip()
    except (KeyError, TypeError, ValueError) as error:
        raise OutputValidationError("增强模块缺少必需的返回字段。") from error
    if not math.isfinite(returned_strength) or not math.isclose(
        returned_strength,
        requested_strength,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise OutputValidationError("增强模块返回的 strength 与本次请求不一致。")
    if not math.isfinite(runtime_seconds) or runtime_seconds < 0:
        raise OutputValidationError("增强模块返回的 runtime_seconds 无效。")
    if model_name.lower() != str(config["enhancement"]["model"]).strip().lower():
        raise OutputValidationError("增强模块使用的模型与冻结配置不一致。")

    returned_full = Path(str(enhancement.get("full_output_path", ""))).resolve()
    returned_mix = Path(str(enhancement.get("mixed_output_path", ""))).resolve()
    if returned_full != paths.enhanced_full or returned_mix != paths.enhanced_mix:
        raise OutputValidationError("增强模块写入了契约之外的输出路径。")
    sample_rate = int(config["audio"]["target_sample_rate"])
    channels = int(config["audio"]["channels"])
    full_frames = _validate_pcm_wav(paths.enhanced_full, sample_rate, channels)
    mixed_frames = _validate_pcm_wav(paths.enhanced_mix, sample_rate, channels)
    original_frames = _validate_pcm_wav(paths.original, sample_rate, channels)
    tolerance_frames = int(sample_rate * 0.2)
    if (
        abs(full_frames - original_frames) > tolerance_frames
        or abs(mixed_frames - original_frames) > tolerance_frames
    ):
        raise OutputValidationError("增强输出与原轨时长差异超过 0.2 秒。")


def _validated_audio_meta(value: Any) -> AudioMeta:
    """Rebuild A's metadata from finite, JSON-safe built-in scalars."""

    if not isinstance(value, AudioMeta):
        raise InputAudioError("音频标准化模块返回了无效元数据。")

    def required_text(item: Any, field_name: str) -> str:
        if not isinstance(item, str) or not item.strip():
            raise InputAudioError(f"音频元数据 {field_name} 无效。")
        return str(item)

    def optional_text(item: Any, field_name: str) -> str | None:
        if item is None:
            return None
        return required_text(item, field_name)

    def finite_number(item: Any, field_name: str) -> float:
        if isinstance(item, (bool, str, bytes)):
            raise InputAudioError(f"音频元数据 {field_name} 必须是有限数字。")
        try:
            numeric = float(item)
        except (TypeError, ValueError, OverflowError) as error:
            raise InputAudioError(
                f"音频元数据 {field_name} 必须是有限数字。"
            ) from error
        if not math.isfinite(numeric):
            raise InputAudioError(f"音频元数据 {field_name} 必须是有限数字。")
        return numeric

    def positive_int(item: Any, field_name: str) -> int:
        numeric = finite_number(item, field_name)
        integer = int(numeric)
        if numeric != integer or integer <= 0:
            raise InputAudioError(f"音频元数据 {field_name} 必须是正整数。")
        return integer

    def optional_positive_int(item: Any, field_name: str) -> int | None:
        return None if item is None else positive_int(item, field_name)

    duration = finite_number(value.duration_seconds, "duration_seconds")
    peak = finite_number(value.peak_abs, "peak_abs")
    clipped_ratio = finite_number(value.clipped_ratio, "clipped_ratio")
    rms = None if value.rms_dbfs is None else finite_number(value.rms_dbfs, "rms_dbfs")
    silent_ratio = (
        None
        if value.silent_ratio is None
        else finite_number(value.silent_ratio, "silent_ratio")
    )
    if duration <= 0:
        raise InputAudioError("音频元数据 duration_seconds 必须大于 0。")
    if not 0.0 <= peak <= 1.0:
        raise InputAudioError("音频元数据 peak_abs 必须在 0 到 1 之间。")
    if not 0.0 <= clipped_ratio <= 1.0:
        raise InputAudioError("音频元数据 clipped_ratio 必须在 0 到 1 之间。")
    if silent_ratio is not None and not 0.0 <= silent_ratio <= 1.0:
        raise InputAudioError("音频元数据 silent_ratio 必须在 0 到 1 之间。")

    return AudioMeta(
        source_name=required_text(value.source_name, "source_name"),
        sample_rate=positive_int(value.sample_rate, "sample_rate"),
        channels=positive_int(value.channels, "channels"),
        duration_seconds=duration,
        peak_abs=peak,
        rms_dbfs=rms,
        clipped_ratio=clipped_ratio,
        silent_ratio=silent_ratio,
        normalized_path=required_text(value.normalized_path, "normalized_path"),
        original_sample_rate=optional_positive_int(
            value.original_sample_rate, "original_sample_rate"
        ),
        original_channels=optional_positive_int(
            value.original_channels, "original_channels"
        ),
        source_format=optional_text(value.source_format, "source_format"),
    )


def _compute_audio_levels(path: Path) -> AudioLevelMetrics:
    """Measure peak and RMS from the persisted PCM16 track without extra DSP."""

    try:
        with wave.open(str(path), "rb") as audio:
            if audio.getsampwidth() != 2 or audio.getcomptype() != "NONE":
                raise OutputValidationError("响度体检仅接受 PCM16 WAV。")
            peak_sample = 0
            square_sum = 0
            sample_count = 0
            while raw := audio.readframes(65_536):
                samples = array("h")
                samples.frombytes(raw)
                if sys.byteorder != "little":
                    samples.byteswap()
                if samples:
                    peak_sample = max(peak_sample, max(abs(sample) for sample in samples))
                    square_sum += sum(sample * sample for sample in samples)
                    sample_count += len(samples)
    except OutputValidationError:
        raise
    except (OSError, EOFError, wave.Error, ValueError) as error:
        raise OutputValidationError("无法计算音轨响度证据。", detail=str(error)) from error

    if sample_count <= 0:
        raise OutputValidationError("无法从空音轨计算响度证据。")
    peak_abs = peak_sample / 32_768.0
    rms_abs = math.sqrt(square_sum / sample_count) / 32_768.0
    rms_dbfs = None if rms_abs == 0.0 else 20.0 * math.log10(rms_abs)
    return AudioLevelMetrics(peak_abs=peak_abs, rms_dbfs=rms_dbfs)


def _validate_pcm_wav(path: Path, sample_rate: int, channels: int) -> int:
    if not path.is_file() or path.stat().st_size <= 44:
        raise OutputValidationError("输出 WAV 不存在或为空。")
    try:
        with wave.open(str(path), "rb") as audio:
            if audio.getframerate() != sample_rate:
                raise OutputValidationError("输出 WAV 采样率不符合契约。")
            if audio.getnchannels() != channels:
                raise OutputValidationError("输出 WAV 声道数不符合契约。")
            if audio.getsampwidth() != 2 or audio.getcomptype() != "NONE":
                raise OutputValidationError("输出 WAV 必须为 PCM16。")
            frames = audio.getnframes()
            if frames <= 0:
                raise OutputValidationError("输出 WAV 没有音频帧。")
            return frames
    except (wave.Error, EOFError, OSError) as error:
        raise OutputValidationError("输出 WAV 无法解码。", detail=str(error)) from error


def _append_input_warnings(result: ProcessResult, meta: AudioMeta) -> None:
    if meta.original_sample_rate is not None and meta.original_sample_rate != meta.sample_rate:
        result.warnings.append(
            WarningItem(
                ErrorCode.INPUT_RESAMPLED,
                "输入音频已重采样为 48kHz。",
                "audio_io",
                True,
                {"original_sample_rate": meta.original_sample_rate},
            )
        )
    if meta.original_channels is not None and meta.original_channels != meta.channels:
        result.warnings.append(
            WarningItem(
                ErrorCode.INPUT_STEREO_DOWNMIXED,
                "输入音频已转换为单声道。",
                "audio_io",
                True,
                {"original_channels": meta.original_channels},
            )
        )
    if meta.clipped_ratio > 0:
        result.warnings.append(
            WarningItem(
                ErrorCode.INPUT_CLIPPED,
                "输入音频存在削波，增强无法恢复已经丢失的细节。",
                "audio_io",
                True,
                {"clipped_ratio": meta.clipped_ratio},
            )
        )
    if (meta.rms_dbfs is not None and meta.rms_dbfs <= -60.0) or (
        meta.silent_ratio is not None and meta.silent_ratio >= 0.98
    ):
        result.warnings.append(
            WarningItem(
                ErrorCode.INPUT_NEAR_SILENT,
                "输入音频接近静音，转写结果可能为空。",
                "audio_io",
                True,
                {"rms_dbfs": meta.rms_dbfs, "silent_ratio": meta.silent_ratio},
            )
        )


def _coerce_warnings(values: Any) -> list[WarningItem]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise OutputValidationError("增强模块 warnings 必须为列表。")
    warnings: list[WarningItem] = []
    for value in values:
        if isinstance(value, WarningItem):
            warning = value
        elif isinstance(value, Mapping):
            try:
                warning = WarningItem(
                    code=ErrorCode(value["code"]),
                    message=value["message"],
                    module=value["module"],
                    recoverable=value["recoverable"],
                    details=value.get("details", {}),
                )
            except (KeyError, TypeError, ValueError) as error:
                raise OutputValidationError(
                    "增强模块返回了无效 warning。"
                ) from error
        else:
            raise OutputValidationError("增强模块返回了无效 warning。")
        warnings.append(_validated_warning(warning))
    return warnings


def _validated_warning(value: Any) -> WarningItem:
    if not isinstance(value, WarningItem):
        raise OutputValidationError("warning 必须使用冻结的 WarningItem 结构。")
    if not isinstance(value.code, ErrorCode):
        try:
            code = ErrorCode(value.code)
        except (TypeError, ValueError) as error:
            raise OutputValidationError("warning code 无效。") from error
    else:
        code = value.code
    if not isinstance(value.message, str) or not value.message.strip():
        raise OutputValidationError("warning message 无效。")
    if not isinstance(value.module, str) or not value.module.strip():
        raise OutputValidationError("warning module 无效。")
    if not isinstance(value.recoverable, bool):
        raise OutputValidationError("warning recoverable 必须是布尔值。")
    if not isinstance(value.details, Mapping):
        raise OutputValidationError("warning details 必须是 JSON 对象。")
    try:
        details = _json_safe_copy(dict(value.details))
    except (TypeError, ValueError) as error:
        raise OutputValidationError("warning details 包含非 JSON 数据。") from error
    return WarningItem(
        code=code,
        message=str(value.message),
        module=str(value.module),
        recoverable=value.recoverable,
        details=details,
    )


def _json_safe_copy(value: Any) -> Any:
    """Return built-in JSON values only, rejecting NaN and custom objects."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON numbers must be finite")
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("JSON object keys must be strings")
            normalized[str(key)] = _json_safe_copy(item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_json_safe_copy(item) for item in value]
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def _safe_warning_for_persistence(value: Any) -> WarningItem:
    try:
        return _validated_warning(value)
    except Exception as error:
        return WarningItem(
            code=ErrorCode.INTERNAL_ERROR,
            message="某个模块返回了无效告警详情，已安全省略该详情。",
            module="pipeline",
            recoverable=True,
            details={"exception_type": type(error).__name__},
        )


def _unexpected_warning(
    *,
    code: ErrorCode,
    module: str,
    message: str,
    error: BaseException,
    recoverable: bool,
) -> WarningItem:
    return WarningItem(
        code=code,
        message=message,
        module=module,
        recoverable=recoverable,
        details={"exception_type": type(error).__name__},
    )


def _persist_transcript(
    path: Path,
    transcript: TranscriptResult | None,
    log: _RunLog,
) -> None:
    if transcript is None:
        return
    try:
        _atomic_json_write(
            path,
            {
                "text": transcript.text,
                "language": transcript.language,
                "segments": transcript.segments,
                "runtime_seconds": transcript.runtime_seconds,
                "model_name": transcript.model_name,
                "error": transcript.error,
            },
        )
    except Exception as error:
        log.exception("transcript_persistence", error)


def _finalize(
    result: ProcessResult,
    paths: JobPaths,
    log: _RunLog,
    started: float,
) -> ProcessResult:
    persistence_started = time.monotonic()
    result.warnings = [
        _safe_warning_for_persistence(warning) for warning in result.warnings
    ]
    try:
        result.config_snapshot = _json_safe_copy(result.config_snapshot)
    except (TypeError, ValueError) as error:
        result.config_snapshot = {
            "pipeline_version": _PIPELINE_CACHE_VERSION,
            "snapshot_invalid": True,
        }
        result.warnings.append(
            _unexpected_warning(
                code=ErrorCode.INTERNAL_ERROR,
                module="pipeline",
                message="运行配置快照包含无效值，已使用安全的最小快照。",
                error=error,
                recoverable=True,
            )
        )
    try:
        _write_log(paths.run_log, log)
        result.runtime.persistence_seconds = time.monotonic() - persistence_started
        result.runtime.total_seconds = time.monotonic() - started
        # Commit result.json exactly once. If this atomic write fails there is
        # no older success JSON left behind that disagrees with the return value.
        _atomic_json_write(paths.result_json, result.to_dict())
    except Exception as error:
        log.exception("persistence", error)
        if result.status == ProcessStatus.SUCCESS:
            result.status = ProcessStatus.PARTIAL
        result.warnings.append(
            _unexpected_warning(
                code=ErrorCode.INTERNAL_ERROR,
                module="persistence",
                message="结果已返回，但本地结果文件未能完整保存。",
                error=error,
                recoverable=True,
            )
        )
        result.runtime.persistence_seconds = time.monotonic() - persistence_started
        result.runtime.total_seconds = time.monotonic() - started
        try:
            _write_log(paths.run_log, log)
            _atomic_json_write(paths.result_json, result.to_dict())
        except Exception:
            pass
    return result


def _atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_log(path: Path, log: _RunLog) -> None:
    path.write_text(log.render(), encoding="utf-8")


def _validated_file_path(returned: Any, expected: Path) -> str:
    path = Path(str(returned)).resolve()
    if path != expected or not path.is_file() or path.stat().st_size == 0:
        raise VisualizationError("可视化模块没有生成契约路径中的有效文件。")
    return str(path)


def _validated_event_results(values: Any) -> list[EventResult]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError("event results must be a sequence")
    normalized: list[EventResult] = []
    for value in values:
        if not isinstance(value, EventResult):
            raise TypeError("event detector returned a non-EventResult value")
        if not isinstance(value.label, str) or not value.label.strip():
            raise ValueError("event label must be a non-empty string")
        score = _finite_event_number(value.score, "score")
        start = _finite_event_number(value.start_seconds, "start_seconds")
        end = _finite_event_number(value.end_seconds, "end_seconds")
        if not 0.0 <= score <= 1.0 or start < 0.0 or end < start:
            raise ValueError("event score or timestamps are outside the valid range")
        normalized.append(
            EventResult(
                label=str(value.label),
                score=score,
                start_seconds=start,
                end_seconds=end,
            )
        )
    return normalized


def _finite_event_number(value: Any, field_name: str) -> float:
    if isinstance(value, (bool, str, bytes)):
        raise TypeError(f"event {field_name} must be a finite number")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError(f"event {field_name} must be a finite number") from error
    if not math.isfinite(numeric):
        raise ValueError(f"event {field_name} must be finite")
    return numeric


def _load_event_labels() -> list[str]:
    payload = yaml.safe_load(_DEFAULT_LABELS_CONFIG_PATH.read_text(encoding="utf-8"))
    labels = payload.get("labels", []) if isinstance(payload, dict) else []
    normalized = [str(label).strip() for label in labels if str(label).strip()]
    if not normalized:
        raise ValueError("event label list is empty")
    return normalized


def _cached_artifacts_exist(payload: Mapping[str, Any]) -> bool:
    try:
        status = payload.get("status")
        if status not in {ProcessStatus.SUCCESS.value, ProcessStatus.PARTIAL.value}:
            return False
        cached_result = _process_result_from_dict(payload)
        cached_result.to_dict()
        _validated_event_results(cached_result.events)
        if (
            cached_result.input_meta is None
            or cached_result.original_levels is None
            or cached_result.mixed_levels is None
        ):
            return False
        input_meta = _validated_audio_meta(cached_result.input_meta)
        sample_rate = input_meta.sample_rate
        channels = input_meta.channels

        audio_fields = (
            "original_audio_path",
            "full_output_path",
            "mixed_output_path",
        )
        artifact_parents: set[Path] = set()
        for field in audio_fields:
            value = payload.get(field)
            if not isinstance(value, str) or not value:
                return False
            path = Path(value).resolve()
            _validate_pcm_wav(path, sample_rate, channels)
            artifact_parents.add(path.parent)

        # Validate every optional path that claims to exist, including partial
        # results. A stale non-null visualization may never survive a cache hit.
        for field in ("waveform_path", "spectrogram_path"):
            value = payload.get(field)
            if value is None:
                if status == ProcessStatus.SUCCESS.value:
                    return False
                continue
            if not isinstance(value, str) or not value:
                return False
            path = Path(value).resolve()
            if not path.is_file() or path.stat().st_size == 0:
                return False
            artifact_parents.add(path.parent)

        if len(artifact_parents) != 1:
            return False
        job_root = next(iter(artifact_parents))
        if job_root.name != str(payload.get("job_id", "")):
            return False

        transcript_files = {
            "transcript_before": job_root / "transcript_before.json",
            "transcript_after": job_root / "transcript_after.json",
        }
        for field, transcript_path in transcript_files.items():
            transcript = payload.get(field)
            if transcript is None:
                if status == ProcessStatus.SUCCESS.value:
                    return False
                continue
            if not isinstance(transcript, Mapping):
                return False
            candidate = _validated_transcript_result(
                TranscriptResult(**dict(transcript)), field
            )
            if status == ProcessStatus.SUCCESS.value and not candidate.text.strip():
                return False
            if not transcript_path.is_file() or transcript_path.stat().st_size == 0:
                return False
            persisted_transcript = json.loads(
                transcript_path.read_text(encoding="utf-8")
            )
            if persisted_transcript != transcript:
                return False

        result_path = job_root / "result.json"
        run_log_path = job_root / "run.log"
        if not result_path.is_file() or not run_log_path.is_file():
            return False
        persisted_result = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            not isinstance(persisted_result, Mapping)
            or persisted_result.get("job_id") != payload.get("job_id")
            or persisted_result.get("status") != status
        ):
            return False
        return True
    except (
        AudioRescueError,
        KeyError,
        OSError,
        OutputValidationError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ):
        return False


def _process_result_from_dict(payload: Mapping[str, Any]) -> ProcessResult:
    def make_optional(model: type[Any], value: Any) -> Any:
        return None if value is None else model(**value)

    warnings = [
        WarningItem(
            code=ErrorCode(item["code"]),
            message=item["message"],
            module=item["module"],
            recoverable=item["recoverable"],
            details=dict(item.get("details", {})),
        )
        for item in payload.get("warnings", [])
    ]
    events = [EventResult(**item) for item in payload.get("events", [])]
    return ProcessResult(
        job_id=str(payload["job_id"]),
        status=ProcessStatus(payload["status"]),
        runtime=RuntimeStats(**payload.get("runtime", {})),
        input_meta=make_optional(AudioMeta, payload.get("input_meta")),
        original_audio_path=payload.get("original_audio_path"),
        enhanced_audio_path=payload.get("enhanced_audio_path"),
        full_output_path=payload.get("full_output_path"),
        mixed_output_path=payload.get("mixed_output_path"),
        original_levels=make_optional(
            AudioLevelMetrics, payload.get("original_levels")
        ),
        mixed_levels=make_optional(AudioLevelMetrics, payload.get("mixed_levels")),
        transcript_before=make_optional(
            TranscriptResult, payload.get("transcript_before")
        ),
        transcript_after=make_optional(
            TranscriptResult, payload.get("transcript_after")
        ),
        cer_before=make_optional(CerResult, payload.get("cer_before")),
        cer_after=make_optional(CerResult, payload.get("cer_after")),
        text_diff_html=payload.get("text_diff_html"),
        waveform_path=payload.get("waveform_path"),
        spectrogram_path=payload.get("spectrogram_path"),
        events=events,
        warnings=warnings,
        config_snapshot=dict(payload.get("config_snapshot", {})),
    )


__all__ = [
    "JobPaths",
    "PipelineDependencies",
    "create_job_paths",
    "new_job_id",
    "process_audio",
]
