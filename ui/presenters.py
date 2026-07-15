"""ProcessResult-to-UI presentation mapping owned by B.

The presenter accepts either frozen schema dataclasses or JSON-ready fixture
dicts. It does not call audio models, does not compute CER, and does not
recompute text diffs in the UI path.
"""

from __future__ import annotations

import html
import json
import re
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from .file_delivery import register_files_for_delivery, unregister_delivery_url
from .file_staging import (
    default_staging_root,
    resolve_allowed_file,
    stage_files_for_gradio,
)
from .media_validation import media_kind_for_role, validate_media_path

ROOT_DIR = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT_DIR / "tests" / "fixtures"

STATUS_LABELS = {
    "success": "急救完成",
    "partial": "音频已完成，部分结果不可用",
    "failed": "急救未完成",
}

STATUS_CLASSES = {
    "success": "ar-status-success",
    "partial": "ar-status-partial",
    "failed": "ar-status-failed",
}

STRENGTH_LABELS = {
    "轻度": 0.50,
    "标准": 0.75,
    "强力": 1.00,
}

UI_TUPLE_KEYS = (
    "status_md",
    "input_md",
    "playback_note_md",
    "original_audio",
    "enhanced_audio",
    "transcript_before_md",
    "transcript_after_md",
    "diff_html",
    "cer_md",
    "spectrogram_image",
    "waveform_image",
    "runtime_md",
    "warnings_html",
    "mixed_download",
    "full_download",
    "transcript_download",
    "result_download",
)

DELIVERY_FILENAMES = {
    "original_audio": "original.wav",
    "mixed_audio": "mixed.wav",
    "full_audio": "full.wav",
    "spectrogram_image": "spectrogram.png",
    "waveform_image": "waveform.png",
}

CORE_WARNING_CODES = {
    "INPUT_INVALID",
    "INPUT_TOO_LONG",
    "INPUT_CLIPPED",
    "INPUT_NEAR_SILENT",
    "INPUT_STEREO_DOWNMIXED",
    "INPUT_RESAMPLED",
    "ENHANCE_FAILED",
    "OUTPUT_INVALID",
    "OUTPUT_PEAK_PROTECTED",
    "ASR_FAILED",
    "ASR_BEFORE_FAILED",
    "ASR_AFTER_FAILED",
    "ASR_EMPTY",
    "VIS_FAILED",
    "EVENTS_SKIPPED",
    "CACHE_USED",
    "INTERNAL_ERROR",
}

UI_WARNING_CODES = {
    "UI_ERROR",
    "UI_FIXTURE_DISABLED",
    "UI_FIXTURE_LOAD_FAILED",
    "UI_PIPELINE_UNAVAILABLE",
    "UI_PIPELINE_FAILED",
    "UI_FILE_DELIVERY_FAILED",
    "UI_FILE_DELIVERY_OPTIONAL",
    "UI_RESULT_INCOMPLETE",
    "UI_MEDIA_INVALID",
    "UI_WAV_INVALID",
}

WARNING_CODES = CORE_WARNING_CODES | UI_WARNING_CODES | {"UNKNOWN"}

WARNING_MESSAGES = {
    "INPUT_INVALID": "输入文件无效或无法读取。",
    "INPUT_TOO_LONG": "输入超过 60 秒限制。",
    "INPUT_CLIPPED": "输入存在削波，结果可能不稳定。",
    "INPUT_NEAR_SILENT": "输入接近静音，结果可能不稳定。",
    "INPUT_STEREO_DOWNMIXED": "输入已下混为单声道。",
    "INPUT_RESAMPLED": "输入已重采样为 48 kHz。",
    "ENHANCE_FAILED": "增强阶段失败，请更换样例或使用可用缓存。",
    "OUTPUT_INVALID": "输出音频无效，相关结果不可用。",
    "OUTPUT_PEAK_PROTECTED": "输出已进行峰值保护。",
    "ASR_FAILED": "转写阶段失败。",
    "ASR_BEFORE_FAILED": "处理前转写不可用，其他可用结果已保留。",
    "ASR_AFTER_FAILED": "增强后转写不可用，其他可用结果已保留。",
    "ASR_EMPTY": "转写为空文本，可能是静音或语音过弱。",
    "VIS_FAILED": "可视化生成失败，音频和转写结果仍可查看。",
    "EVENTS_SKIPPED": "事件识别已跳过，不影响核心链路。",
    "CACHE_USED": "使用本地缓存结果。",
    "INTERNAL_ERROR": "内部处理异常，细节已隐藏。",
    "UI_ERROR": "页面展示异常，细节已隐藏。",
    "UI_FIXTURE_DISABLED": "正式模式未启用开发 fixture，请上传音频。",
    "UI_FIXTURE_LOAD_FAILED": "开发 fixture 读取失败，细节已隐藏。",
    "UI_PIPELINE_UNAVAILABLE": "真实处理管线暂不可用，请检查后端接入状态。",
    "UI_PIPELINE_FAILED": "真实处理管线调用失败，细节已隐藏。",
    "UI_FILE_DELIVERY_FAILED": "页面文件投递失败，相关播放器或下载已关闭。",
    "UI_FILE_DELIVERY_OPTIONAL": "调试文件投递失败，核心播放结果不受影响。",
    "UI_RESULT_INCOMPLETE": "页面结果不完整，已按可用内容降级展示。",
    "UI_MEDIA_INVALID": "媒体文件不可展示，相关播放器或图片已关闭。",
    "UI_WAV_INVALID": "音频文件不可播放，相关播放器或下载已关闭。",
    "UNKNOWN": "出现未分类警告，细节已隐藏。",
}

WARNING_MODULES = {
    "audio_io",
    "enhance",
    "transcribe",
    "asr",
    "visualization",
    "pipeline",
    "cache",
    "events",
    "core",
    "metrics",
    "persistence",
    "text_diff",
    "ui",
    "visualize",
    "unknown",
}

DETAIL_NUMBER_KEYS = {
    "attempt",
    "attempts",
    "count",
    "runs",
    "sample_rate",
    "channels",
    "duration_seconds",
    "runtime_seconds",
    "total_seconds",
    "strength",
    "peak_abs",
    "rms_dbfs",
    "clipped_ratio",
    "silent_ratio",
    "exit_code",
}

DETAIL_BOOL_KEYS = {"recoverable", "required", "cache_hit", "cold_start"}

DETAIL_ENUM_VALUES = {
    "track": {"before", "after", "original", "mixed", "full"},
    "role": {
        "original",
        "mixed",
        "full",
        "spectrogram",
        "waveform",
        "transcript_before",
        "transcript_after",
    },
    "stage": {
        "input",
        "normalize",
        "enhance",
        "asr_before",
        "asr_after",
        "visualization",
        "events",
        "cache",
        "pipeline",
        "presentation",
        "delivery",
        "staging",
        "ui",
    },
    "reason": {
        "missing",
        "unavailable",
        "invalid_media",
        "invalid_wav",
        "result_incomplete",
        "transcript_missing",
        "visual_missing",
        "visual_delivery_failed",
        "delivery_failed",
        "staging_failed",
        "registration_failed",
        "not_allowed",
        "source_unavailable",
    },
    "device": {"auto", "cpu", "cuda", "mps", "custom", "unknown"},
}

DELIVERY_ROLES = (
    "original_audio",
    "mixed_audio",
    "full_audio",
    "spectrogram_image",
    "waveform_image",
)

REQUIRED_DELIVERY_ROLES = ("original_audio", "mixed_audio")

DELIVERY_ROLE_LABELS = {
    "original_audio": "original",
    "mixed_audio": "mixed",
    "full_audio": "full",
    "spectrogram_image": "spectrogram",
    "waveform_image": "waveform",
}


def json_ready(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return json_ready(asdict(value))
    if hasattr(value, "to_dict"):
        return json_ready(value.to_dict())
    if hasattr(value, "model_dump"):
        return json_ready(value.model_dump())
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def as_dict(value: Any) -> dict[str, Any]:
    converted = json_ready(value)
    return converted if isinstance(converted, dict) else {}


def list_fixture_names() -> list[str]:
    if not FIXTURE_DIR.exists():
        return []
    return sorted(path.stem for path in FIXTURE_DIR.glob("process_result_*.json"))


def load_fixture(name: str) -> dict[str, Any]:
    safe_name = Path(name).stem
    fixture_path = FIXTURE_DIR / f"{safe_name}.json"
    if not fixture_path.exists():
        raise FileNotFoundError(f"找不到 fixture：{fixture_path}")
    with fixture_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    data["_fixture_path"] = str(fixture_path)
    return data


def strength_value(label_or_value: Any) -> float:
    if isinstance(label_or_value, (int, float)):
        value = float(label_or_value)
    else:
        value = STRENGTH_LABELS.get(str(label_or_value), 0.75)
    return max(0.0, min(1.0, value))


def file_if_exists(
    path_value: Any,
    *,
    allowed_roots: tuple[Path, ...] = (),
) -> str | None:
    # File-bearing Gradio components can make a server-side path downloadable.
    # No trusted result root therefore means no file, even when the path exists.
    resolved = resolve_allowed_file(
        path_value,
        allowed_roots=allowed_roots,
        base_dir=ROOT_DIR,
    )
    return str(resolved) if resolved is not None else None


def _result_file_roots(result: dict[str, Any]) -> tuple[Path, ...]:
    roots: list[Path] = []
    job_id = str(result.get("job_id") or "")
    if re.fullmatch(r"[A-Za-z0-9_-]+", job_id):
        roots.append(ROOT_DIR / "outputs" / job_id)
    fixture_path = result.get("_fixture_path")
    if fixture_path:
        try:
            Path(str(fixture_path)).resolve().relative_to(FIXTURE_DIR.resolve())
            roots.append(FIXTURE_DIR)
        except (OSError, RuntimeError, ValueError):
            pass
    return tuple(roots)


def format_seconds(value: Any) -> str:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return "未记录"
    return f"{seconds:.2f} 秒"


def format_number(value: Any, digits: int = 2) -> str:
    if value is None:
        return "未记录"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "未记录"
    return f"{number:.{digits}f}"


def format_percent(value: Any) -> str:
    if value is None:
        return "未记录"
    try:
        number = float(value) * 100
    except (TypeError, ValueError):
        return "未记录"
    return f"{number:.1f}%"


def warning_codes(result: dict[str, Any]) -> set[str]:
    codes: set[str] = set()
    for item in result.get("warnings") or []:
        code = as_dict(item).get("code")
        if code:
            codes.add(_normalize_warning_code(code))
    return codes


def is_cache_result(result: dict[str, Any]) -> bool:
    runtime = as_dict(result.get("runtime"))
    return bool(runtime.get("cache_hit"))


def display_text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def transcript_text(transcript: Any) -> str:
    data = as_dict(transcript)
    text = display_text(data.get("text"))
    if text:
        return text
    error = display_text(data.get("error"))
    if error:
        return "转写不可用：请查看警告状态。"
    return ""


def transcript_markdown(title: str, transcript: Any) -> str:
    data = as_dict(transcript)
    text = transcript_text(data)
    model = data.get("model_name") or "未记录"
    language = data.get("language") or "未记录"
    runtime = format_seconds(data.get("runtime_seconds"))
    if not text:
        text = "未生成"
    return (
        f"### {title}\n\n"
        f"{text}\n\n"
        f"- 模型：`{model}`\n"
        f"- 语言：`{language}`\n"
        f"- 耗时：{runtime}"
    )


def input_markdown(result: dict[str, Any]) -> str:
    meta = as_dict(result.get("input_meta"))
    if not meta:
        return "### 输入信息\n\n未生成输入元数据。"
    lines = [
        "### 输入信息",
        "",
        f"- 文件：`{'已隐藏文件名' if meta.get('source_name') else '未记录'}`",
        f"- 原始格式：`{meta.get('source_format') or '未记录'}`",
        f"- 原始采样率：`{meta.get('original_sample_rate') or '未记录'}` Hz",
        f"- 原始声道：`{meta.get('original_channels') or '未记录'}`",
        f"- 标准化时长：{format_seconds(meta.get('duration_seconds'))}",
        f"- 标准化采样率：`{meta.get('sample_rate') or '未记录'}` Hz",
        f"- 标准化声道：`{meta.get('channels') or '未记录'}`",
        f"- 原轨峰值：{format_number(meta.get('peak_abs'), 3)}",
        f"- 原轨 RMS：{format_number(meta.get('rms_dbfs'), 1)} dBFS",
        f"- 原轨削波比例：{format_percent(meta.get('clipped_ratio'))}",
        f"- 静音比例：{format_percent(meta.get('silent_ratio'))}",
    ]
    return "\n".join(lines)


def status_markdown(result: dict[str, Any]) -> str:
    raw_status = str(result.get("status") or "failed")
    status = raw_status if raw_status in STATUS_LABELS else "failed"
    label = STATUS_LABELS.get(raw_status, "未知状态")
    cache_badge = "<span class='ar-cache-badge'>本地缓存</span>" if is_cache_result(result) else ""
    runtime = as_dict(result.get("runtime"))
    config = as_dict(result.get("config_snapshot"))
    request = as_dict(config.get("request"))
    enhancement = as_dict(config.get("enhancement"))
    asr = as_dict(config.get("asr"))
    strength = request.get(
        "strength",
        config.get("strength", enhancement.get("default_strength", "未记录")),
    )
    asr_model = config.get("asr_model") or asr.get("model") or "未记录"
    sample_notice = config.get("sample_notice")

    facts = [
        ("任务", result.get("job_id") or "未记录"),
        ("总耗时", format_seconds(runtime.get("total_seconds"))),
        ("增强强度", strength),
        ("Whisper", asr_model),
    ]
    if sample_notice:
        facts.append(("样例说明", sample_notice))

    class_name = STATUS_CLASSES[status]
    parts = [
        (
            f"<section class='ar-status-card {class_name}' "
            f"data-status='{html.escape(status)}' aria-label='处理状态 {html.escape(status)}'>"
        ),
        "<div class='ar-status-header'>",
        f"<span class='ar-status-machine'>状态：{html.escape(status)}</span>",
        f"<h2>{html.escape(label)}</h2>",
        cache_badge,
        "</div>",
        "<dl class='ar-status-facts'>",
    ]
    for key, value in facts:
        parts.append(
            f"<div><dt>{html.escape(str(key))}</dt><dd>{html.escape(str(value))}</dd></div>"
        )
    parts.extend(["</dl>", "</section>"])
    return "".join(parts)


def playback_note_markdown(result: dict[str, Any]) -> str:
    full_path = result.get("full_output_path")
    mixed_path = result.get("mixed_output_path") or result.get("enhanced_audio_path")
    lines = [
        "### A/B 播放说明",
        "",
        "- 处理前播放器使用标准化原轨。",
        "- 增强后播放器使用 dry/wet 混合轨，也就是 `mixed_output_path`。",
        "- 同一播放设置，未做响度匹配；请勿把音量差异直接等同于清晰度提升。",
    ]
    if mixed_path:
        lines.append("- 当前混合轨：`mixed_output_path` 已返回，文件名已隐藏。")
    if full_path:
        lines.append("- 100% DeepFilterNet 输出仅用于调试或下载，不作为默认 After 播放轨。")
    original_levels = as_dict(result.get("original_levels"))
    mixed_levels = as_dict(result.get("mixed_levels"))
    if original_levels and mixed_levels:
        lines.extend(
            [
                "",
                "| 轨道 | 峰值 | RMS |",
                "|---|---:|---:|",
                f"| 原轨 | {format_number(original_levels.get('peak_abs'), 3)} | "
                f"{format_number(original_levels.get('rms_dbfs'), 1)} dBFS |",
                f"| 混合增强轨 | {format_number(mixed_levels.get('peak_abs'), 3)} | "
                f"{format_number(mixed_levels.get('rms_dbfs'), 1)} dBFS |",
            ]
        )
    else:
        lines.append("- 响度证据尚未完整返回，当前不能据此宣称 A/B 响度公平。")
    return "\n".join(lines)


def runtime_markdown(result: dict[str, Any]) -> str:
    runtime = as_dict(result.get("runtime"))
    rows = [
        ("解码/规范化", "decode_seconds"),
        ("增强模型加载", "enhancer_load_seconds"),
        ("增强", "enhancement_seconds"),
        ("ASR 模型加载", "asr_load_seconds"),
        ("原轨 ASR", "asr_before_seconds"),
        ("增强轨 ASR", "asr_after_seconds"),
        ("可视化", "visualization_seconds"),
        ("事件识别", "event_seconds"),
        ("结果持久化", "persistence_seconds"),
        ("总计", "total_seconds"),
    ]
    lines = ["### 运行证据", "", "| 阶段 | 耗时 |", "|---|---:|"]
    for label, key in rows:
        lines.append(f"| {label} | {format_seconds(runtime.get(key))} |")
    lines.extend(
        [
            "",
            f"- 设备：`{runtime.get('device') or '未记录'}`",
            f"- 冷启动：`{'是' if runtime.get('cold_start') else '否'}`",
            f"- 缓存命中：`{'是' if runtime.get('cache_hit') else '否'}`",
        ]
    )
    if is_cache_result(result):
        lines.append("- 当前结果来自本地缓存，不标记为现场新推理。")
    return "\n".join(lines)


def cer_markdown(result: dict[str, Any]) -> str:
    before = as_dict(result.get("cer_before"))
    after = as_dict(result.get("cer_after"))
    if not before or not after:
        return "### CER\n\n未提供人工参考文本或 C 未返回 CER，本页不显示 CER。"

    before_cer = before.get("cer")
    after_cer = after.get("cer")
    try:
        delta = (float(before_cer) - float(after_cer)) * 100
        if delta > 1e-12:
            delta_text = f"下降 {delta:.1f} 个百分点（改善）"
        elif delta < -1e-12:
            delta_text = f"上升 {abs(delta):.1f} 个百分点（变差）"
        else:
            delta_text = "持平"
    except (TypeError, ValueError):
        delta_text = "变化未记录"
    ref_len = len(str(before.get("normalized_reference") or before.get("reference") or ""))
    return (
        "### CER\n\n"
        f"- 原始轨 CER：{format_percent(before_cer)}\n"
        f"- 增强轨 CER：{format_percent(after_cer)}\n"
        f"- 变化：{delta_text}\n"
        f"- 参考字符数：{ref_len}"
    )


def unsafe_html(fragment: str) -> bool:
    lowered = fragment.lower()
    return (
        "<script" in lowered
        or "javascript:" in lowered
        or bool(re.search(r"\son[a-z]+\s*=", lowered))
    )


def diff_html(result: dict[str, Any]) -> str:
    supplied = result.get("text_diff_html")
    if not supplied:
        return "<div class='empty-state'>暂无文本差异。UI 不重复计算 diff，等待 pipeline 填充。</div>"
    supplied_text = str(supplied)
    if unsafe_html(supplied_text):
        return (
            "<div class='diff-warning'>差异 HTML 含不安全内容，已转义显示。</div>"
            f"<pre>{html.escape(supplied_text)}</pre>"
        )
    return supplied_text


def _normalize_warning_code(value: Any) -> str:
    if isinstance(value, Enum):
        value = value.value
    code = str(value or "").strip().upper()
    return code if code in WARNING_CODES else "UNKNOWN"


def _normalize_warning_module(value: Any) -> str:
    module = str(value or "").strip().lower()
    if re.fullmatch(r"[a-z_]{2,32}", module) and module in WARNING_MODULES:
        return module
    return "unknown"


def _safe_detail_value(key: str, value: Any) -> Any:
    if key in DETAIL_BOOL_KEYS and isinstance(value, bool):
        return value
    if key in DETAIL_NUMBER_KEYS and not isinstance(value, bool):
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if number != number or number in {float("inf"), float("-inf")}:
            return None
        return int(number) if number.is_integer() else number
    allowed_values = DETAIL_ENUM_VALUES.get(key)
    if allowed_values is not None:
        normalized = str(value or "").strip().lower()
        return normalized if normalized in allowed_values else None
    return None


def _safe_details(details: Any) -> dict[str, Any]:
    raw = as_dict(details)
    safe: dict[str, Any] = {}
    for key, value in raw.items():
        safe_key = str(key).strip().lower()
        if not re.fullmatch(r"[a-z_]{2,32}", safe_key):
            continue
        safe_value = _safe_detail_value(safe_key, value)
        if safe_value is not None:
            safe[safe_key] = safe_value
    return safe


def warnings_html(result: dict[str, Any]) -> str:
    warnings = result.get("warnings") or []
    if not warnings:
        return "<div class='empty-state'>暂无警告。</div>"
    parts = ["<div class='warning-list'>"]
    for item in warnings:
        data = as_dict(item)
        code = _normalize_warning_code(data.get("code"))
        message = WARNING_MESSAGES.get(code, WARNING_MESSAGES["UNKNOWN"])
        module = _normalize_warning_module(data.get("module"))
        recoverable = "可恢复" if data.get("recoverable", True) else "需处理"
        details = _safe_details(data.get("details"))
        parts.append("<div class='warning-item'>")
        parts.append(f"<strong>{html.escape(code)}</strong>")
        parts.append(f"<span>{html.escape(message)}</span>")
        parts.append(f"<em>{html.escape(module)} / {recoverable}</em>")
        if details:
            details_json = html.escape(json.dumps(details, ensure_ascii=False, indent=2))
            parts.append(f"<details><summary>工程细节</summary><pre>{details_json}</pre></details>")
        parts.append("</div>")
    parts.append("</div>")
    return "".join(parts)


def _empty_delivery_html(label: str) -> str:
    return (
        "<div class='empty-state ar-delivery-empty' "
        f"data-delivery-role='{html.escape(label, quote=True)}'>未生成</div>"
    )


def _audio_delivery_html(
    *,
    title: str,
    url: str | None,
    filename: str,
    role: str,
    allow_download: bool = True,
) -> str:
    if not url:
        return _empty_delivery_html(role)
    safe_title = html.escape(title)
    safe_url = html.escape(url, quote=True)
    safe_filename = html.escape(filename, quote=True)
    download = ""
    if allow_download:
        download = (
            "<a class='ar-delivery-link' "
            f"href='{safe_url}' download='{safe_filename}'>下载 {safe_filename}</a>"
        )
    return (
        "<section class='ar-delivery-card ar-audio-card' "
        f"data-delivery-role='{html.escape(role, quote=True)}'>"
        f"<h3>{safe_title}</h3>"
        f"<audio controls preload='metadata' src='{safe_url}'></audio>"
        f"{download}"
        "</section>"
    )


def _image_delivery_html(*, title: str, url: str | None, role: str) -> str:
    if not url:
        return _empty_delivery_html(role)
    safe_title = html.escape(title)
    safe_url = html.escape(url, quote=True)
    return (
        "<figure class='ar-delivery-card ar-visual-card' "
        f"data-delivery-role='{html.escape(role, quote=True)}'>"
        f"<img src='{safe_url}' alt='{safe_title}' loading='lazy'>"
        f"<figcaption>{safe_title}</figcaption>"
        "</figure>"
    )


def _download_delivery_html(
    *,
    title: str,
    url: str | None,
    filename: str,
    role: str,
    audio_preview: bool = False,
) -> str:
    if not url:
        return _empty_delivery_html(role)
    safe_title = html.escape(title)
    safe_url = html.escape(url, quote=True)
    safe_filename = html.escape(filename, quote=True)
    audio = (
        f"<audio controls preload='metadata' src='{safe_url}'></audio>"
        if audio_preview
        else ""
    )
    return (
        "<section class='ar-delivery-card ar-download-card' "
        f"data-delivery-role='{html.escape(role, quote=True)}'>"
        f"<h3>{safe_title}</h3>"
        f"{audio}"
        "<a class='ar-delivery-link' "
        f"href='{safe_url}' download='{safe_filename}'>下载 {safe_filename}</a>"
        "</section>"
    )


def _normalized_status(value: Any) -> str:
    status = str(value or "failed").strip().lower()
    return status if status in STATUS_LABELS else "failed"


def _empty_delivery_mapping() -> dict[str, str | None]:
    return {role: None for role in DELIVERY_ROLES}


def _stage_result_files(
    paths_by_role: dict[str, Any],
    *,
    allowed_roots: tuple[Path, ...],
) -> dict[str, str | None]:
    try:
        staged = stage_files_for_gradio(
            paths_by_role,
            allowed_roots=allowed_roots,
            base_dir=ROOT_DIR,
        )
    except Exception:
        return _empty_delivery_mapping()
    return {role: staged.get(role) for role in DELIVERY_ROLES}


def _register_delivery_urls(staged_files: dict[str, str | None]) -> dict[str, str | None]:
    try:
        urls = register_files_for_delivery(
            staged_files,
            filenames_by_role=DELIVERY_FILENAMES,
            allowed_roots=(default_staging_root(),),
        )
    except Exception:
        return _empty_delivery_mapping()
    return {role: urls.get(role) for role in DELIVERY_ROLES}


def _invalid_media_roles(
    paths_by_role: dict[str, Any],
    staged_files: dict[str, str | None],
    *,
    allowed_roots: tuple[Path, ...],
) -> set[str]:
    invalid: set[str] = set()
    for role in DELIVERY_ROLES:
        kind = media_kind_for_role(role)
        if kind is None:
            continue
        staged = staged_files.get(role)
        try:
            if staged and not validate_media_path(Path(staged), kind=kind):
                invalid.add(role)
                continue
            raw_path = paths_by_role.get(role)
            if raw_path and staged is None:
                resolved = resolve_allowed_file(
                    raw_path,
                    allowed_roots=allowed_roots,
                    base_dir=ROOT_DIR,
                )
                if resolved is not None and not validate_media_path(resolved, kind=kind):
                    invalid.add(role)
        except Exception:
            continue
    return invalid


def _effective_delivery_urls(
    delivery_urls: dict[str, str | None],
    staged_files: dict[str, str | None],
    invalid_media_roles: set[str],
) -> dict[str, str | None]:
    effective = {role: delivery_urls.get(role) for role in DELIVERY_ROLES}
    for role, url in list(effective.items()):
        if not url:
            continue
        if role in invalid_media_roles or not staged_files.get(role):
            unregister_delivery_url(url)
            effective[role] = None
    return effective


def _delivery_warning(
    role: str,
    *,
    required: bool,
    invalid_media: bool = False,
) -> dict[str, Any]:
    track = DELIVERY_ROLE_LABELS.get(role, "original")
    if invalid_media and media_kind_for_role(role) == "wav":
        code = "UI_WAV_INVALID"
    elif invalid_media:
        code = "UI_MEDIA_INVALID"
    else:
        code = (
            "UI_FILE_DELIVERY_FAILED" if required else "UI_FILE_DELIVERY_OPTIONAL"
        )
    return {
        "code": code,
        "message": "",
        "module": "ui",
        "recoverable": True,
        "details": {
            "track": track,
            "role": track,
            "required": required,
            "reason": (
                "invalid_wav"
                if code == "UI_WAV_INVALID"
                else "invalid_media"
                if code == "UI_MEDIA_INVALID"
                else "delivery_failed"
            ),
            "stage": "delivery",
        },
    }


def _result_incomplete_warning(role: str, *, reason: str) -> dict[str, Any]:
    return {
        "code": "UI_RESULT_INCOMPLETE",
        "message": "",
        "module": "ui",
        "recoverable": True,
        "details": {
            "role": role,
            "required": True,
            "reason": reason,
            "stage": "presentation",
        },
    }


def _warning_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _finite_non_negative_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if number != number or number in {float("inf"), float("-inf")} or number < 0:
        return None
    return number


def _transcript_segment_is_complete(segment: Any) -> bool:
    item = as_dict(segment)
    if set(item) != {"start", "end", "text"}:
        return False
    if not isinstance(item.get("text"), str):
        return False
    start = _finite_non_negative_number(item.get("start"))
    end = _finite_non_negative_number(item.get("end"))
    if start is None or end is None or end < start:
        return False
    return True


def _transcript_is_complete(value: Any) -> bool:
    data = as_dict(value)
    if not data:
        return False
    text = data.get("text")
    language = data.get("language")
    segments = data.get("segments")
    model_name = data.get("model_name")
    error = data.get("error")
    if not isinstance(text, str):
        return False
    if language is not None and not isinstance(language, str):
        return False
    if not isinstance(segments, list):
        return False
    if _finite_non_negative_number(data.get("runtime_seconds")) is None:
        return False
    if not isinstance(model_name, str) or not model_name.strip():
        return False
    if error is not None:
        return False
    for segment in segments:
        if not _transcript_segment_is_complete(segment):
            return False
    return True


def _presentation_status_and_warnings(
    result: dict[str, Any],
    paths_by_role: dict[str, Any],
    delivery_urls: dict[str, str | None],
    invalid_media_roles: set[str],
) -> tuple[str, list[dict[str, Any]]]:
    raw_status = _normalized_status(result.get("status"))
    ui_warnings: list[dict[str, Any]] = []

    required_ok = {
        role: bool(delivery_urls.get(role))
        for role in REQUIRED_DELIVERY_ROLES
    }
    if raw_status in {"success", "partial"}:
        for role, delivered in required_ok.items():
            if not delivered:
                ui_warnings.append(
                    _delivery_warning(
                        role,
                        required=True,
                        invalid_media=role in invalid_media_roles,
                    )
                )
        delivered_count = sum(1 for delivered in required_ok.values() if delivered)
        if delivered_count == len(REQUIRED_DELIVERY_ROLES):
            presentation_status = raw_status
        elif delivered_count > 0:
            presentation_status = "partial"
        else:
            presentation_status = "failed"
    else:
        presentation_status = "failed"
        for role in REQUIRED_DELIVERY_ROLES:
            if paths_by_role.get(role) and not delivery_urls.get(role):
                ui_warnings.append(
                    _delivery_warning(
                        role,
                        required=True,
                        invalid_media=role in invalid_media_roles,
                    )
                )

    if raw_status in {"success", "partial"} and not delivery_urls.get("full_audio"):
        ui_warnings.append(
            _delivery_warning(
                "full_audio",
                required=False,
                invalid_media="full_audio" in invalid_media_roles,
            )
        )

    if raw_status in {"success", "partial"}:
        for role in ("spectrogram_image", "waveform_image"):
            if role in invalid_media_roles and not delivery_urls.get(role):
                ui_warnings.append(
                    _delivery_warning(
                        role,
                        required=False,
                        invalid_media=True,
                    )
                )

    if raw_status == "success":
        incomplete = False
        if not _transcript_is_complete(result.get("transcript_before")):
            incomplete = True
            ui_warnings.append(
                _result_incomplete_warning(
                    "transcript_before",
                    reason="transcript_missing",
                )
            )
        if not _transcript_is_complete(result.get("transcript_after")):
            incomplete = True
            ui_warnings.append(
                _result_incomplete_warning(
                    "transcript_after",
                    reason="transcript_missing",
                )
            )
        for role in ("spectrogram_image", "waveform_image"):
            if not delivery_urls.get(role):
                incomplete = True
                ui_warnings.append(
                    _result_incomplete_warning(
                        DELIVERY_ROLE_LABELS[role],
                        reason=(
                            "visual_delivery_failed"
                            if paths_by_role.get(role) or role in invalid_media_roles
                            else "visual_missing"
                        ),
                    )
                )
        if incomplete and presentation_status == "success":
            presentation_status = "partial"

    return presentation_status, ui_warnings


def result_to_view(raw_result: Any) -> dict[str, Any]:
    result = as_dict(raw_result)
    if not result:
        result = failed_result("", "UI_ERROR")

    mixed_path = result.get("mixed_output_path") or result.get("enhanced_audio_path")
    allowed_roots = _result_file_roots(result)
    paths_by_role = {
        "original_audio": result.get("original_audio_path"),
        "mixed_audio": mixed_path,
        "full_audio": result.get("full_output_path"),
        "spectrogram_image": result.get("spectrogram_path"),
        "waveform_image": result.get("waveform_path"),
    }
    staged_files = _stage_result_files(paths_by_role, allowed_roots=allowed_roots)
    delivery_urls = _register_delivery_urls(staged_files)
    invalid_media_roles = _invalid_media_roles(
        paths_by_role,
        staged_files,
        allowed_roots=allowed_roots,
    )
    effective_delivery_urls = _effective_delivery_urls(
        delivery_urls,
        staged_files,
        invalid_media_roles,
    )
    presentation_status, ui_warnings = _presentation_status_and_warnings(
        result,
        paths_by_role,
        effective_delivery_urls,
        invalid_media_roles,
    )
    display_result = {
        **result,
        "status": presentation_status,
        "warnings": _warning_list(result.get("warnings")) + ui_warnings,
    }
    view = {
        "status_md": status_markdown(display_result),
        "input_md": input_markdown(display_result),
        "playback_note_md": playback_note_markdown(display_result),
        "original_audio": _audio_delivery_html(
            title="处理前：标准化原轨",
            url=effective_delivery_urls["original_audio"],
            filename=DELIVERY_FILENAMES["original_audio"],
            role="original_audio",
        ),
        "enhanced_audio": _audio_delivery_html(
            title="增强后：混合增强轨",
            url=effective_delivery_urls["mixed_audio"],
            filename=DELIVERY_FILENAMES["mixed_audio"],
            role="mixed_audio",
        ),
        "transcript_before_md": transcript_markdown("处理前转写", result.get("transcript_before")),
        "transcript_after_md": transcript_markdown("增强后转写", result.get("transcript_after")),
        "diff_html": diff_html(result),
        "cer_md": cer_markdown(result),
        "spectrogram_image": _image_delivery_html(
            title="声谱图对照",
            url=effective_delivery_urls["spectrogram_image"],
            role="spectrogram_image",
        ),
        "waveform_image": _image_delivery_html(
            title="波形对照",
            url=effective_delivery_urls["waveform_image"],
            role="waveform_image",
        ),
        "runtime_md": runtime_markdown(display_result),
        "warnings_html": warnings_html(display_result),
        "mixed_download": _download_delivery_html(
            title="混合增强 WAV",
            url=effective_delivery_urls["mixed_audio"],
            filename=DELIVERY_FILENAMES["mixed_audio"],
            role="mixed_download",
        ),
        "full_download": _download_delivery_html(
            title="100% 增强调试 WAV",
            url=effective_delivery_urls["full_audio"],
            filename=DELIVERY_FILENAMES["full_audio"],
            role="full_download",
            audio_preview=True,
        ),
        "transcript_download": _empty_delivery_html("transcript_download"),
        "result_download": _empty_delivery_html("result_download"),
    }
    return view


def result_to_ui_tuple(raw_result: Any) -> tuple[Any, ...]:
    view = result_to_view(raw_result)
    return tuple(view[key] for key in UI_TUPLE_KEYS)


def failed_result(message: str, code: str = "INTERNAL_ERROR") -> dict[str, Any]:
    return {
        "job_id": "ui_error",
        "status": "failed",
        "runtime": {
            "decode_seconds": 0.0,
            "enhancer_load_seconds": 0.0,
            "enhancement_seconds": 0.0,
            "asr_load_seconds": 0.0,
            "asr_before_seconds": 0.0,
            "asr_after_seconds": 0.0,
            "visualization_seconds": 0.0,
            "event_seconds": 0.0,
            "persistence_seconds": 0.0,
            "total_seconds": 0.0,
            "cache_hit": False,
            "cold_start": False,
            "device": "unknown",
        },
        "input_meta": None,
        "original_audio_path": None,
        "enhanced_audio_path": None,
        "full_output_path": None,
        "mixed_output_path": None,
        "transcript_before": None,
        "transcript_after": None,
        "cer_before": None,
        "cer_after": None,
        "text_diff_html": None,
        "waveform_path": None,
        "spectrogram_path": None,
        "events": [],
        "warnings": [
            {
                "code": code,
                "message": message,
                "module": "ui",
                "recoverable": True,
                "details": {},
            }
        ],
        "config_snapshot": {},
    }
