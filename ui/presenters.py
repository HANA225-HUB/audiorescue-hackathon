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

ROOT_DIR = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT_DIR / "tests" / "fixtures"

STATUS_LABELS = {
    "success": "急救完成",
    "partial": "音频已完成，部分结果不可用",
    "failed": "急救未完成",
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
    if not allowed_roots:
        return None
    if not path_value:
        return None
    path = Path(str(path_value)).expanduser()
    if not path.is_absolute():
        path = ROOT_DIR / path
    try:
        resolved = path.resolve()
    except (OSError, RuntimeError):
        return None
    if not resolved.is_file():
        return None
    allowed = False
    for root in allowed_roots:
        try:
            resolved.relative_to(root.resolve())
            allowed = True
            break
        except (OSError, RuntimeError, ValueError):
            continue
    if not allowed:
        return None
    return str(resolved)


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
            codes.add(str(code))
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
        return f"转写不可用：{error}"
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
        f"- 文件：`{meta.get('source_name') or '未记录'}`",
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
    status = str(result.get("status") or "failed")
    label = STATUS_LABELS.get(status, "未知状态")
    cache_badge = " | 本地缓存" if is_cache_result(result) else ""
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

    lines = [
        f"## {label}{cache_badge}",
        "",
        f"- 任务：`{result.get('job_id') or '未记录'}`",
        f"- 总耗时：{format_seconds(runtime.get('total_seconds'))}",
        f"- 增强强度：`{strength}`",
        f"- Whisper：`{asr_model}`",
    ]
    if sample_notice:
        lines.append(f"- 样例说明：{sample_notice}")
    return "\n".join(lines)


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
        lines.append(f"- 当前混合轨：`{Path(str(mixed_path)).name}`")
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


def _safe_details(details: Any) -> dict[str, Any]:
    blocked = {"traceback", "stack", "stacktrace", "exception"}

    def scrub(value: Any) -> Any:
        if isinstance(value, dict):
            safe: dict[str, Any] = {}
            for key, item in value.items():
                lowered = str(key).lower()
                if (
                    lowered in blocked
                    or lowered == "path"
                    or lowered.endswith("_path")
                    or lowered.endswith("path")
                ):
                    continue
                safe[str(key)] = scrub(item)
            return safe
        if isinstance(value, (list, tuple)):
            return [scrub(item) for item in value]
        return value

    sanitized = scrub(as_dict(details))
    return sanitized if isinstance(sanitized, dict) else {}


def warnings_html(result: dict[str, Any]) -> str:
    warnings = result.get("warnings") or []
    if not warnings:
        return "<div class='empty-state'>暂无警告。</div>"
    parts = ["<div class='warning-list'>"]
    for item in warnings:
        data = as_dict(item)
        code = html.escape(str(data.get("code") or "UNKNOWN"))
        message = html.escape(str(data.get("message") or "未提供说明"))
        module = html.escape(str(data.get("module") or "unknown"))
        recoverable = "可恢复" if data.get("recoverable", True) else "需处理"
        details = _safe_details(data.get("details"))
        parts.append("<div class='warning-item'>")
        parts.append(f"<strong>{code}</strong>")
        parts.append(f"<span>{message}</span>")
        parts.append(f"<em>{module} / {recoverable}</em>")
        if details:
            details_json = html.escape(json.dumps(details, ensure_ascii=False, indent=2))
            parts.append(f"<details><summary>工程细节</summary><pre>{details_json}</pre></details>")
        parts.append("</div>")
    parts.append("</div>")
    return "".join(parts)


def result_to_view(raw_result: Any) -> dict[str, Any]:
    result = as_dict(raw_result)
    if not result:
        result = failed_result("未收到可展示的处理结果。", "INTERNAL_ERROR")

    mixed_path = result.get("mixed_output_path") or result.get("enhanced_audio_path")
    allowed_roots = _result_file_roots(result)
    view = {
        "status_md": status_markdown(result),
        "input_md": input_markdown(result),
        "playback_note_md": playback_note_markdown(result),
        "original_audio": file_if_exists(
            result.get("original_audio_path"), allowed_roots=allowed_roots
        ),
        "enhanced_audio": file_if_exists(mixed_path, allowed_roots=allowed_roots),
        "transcript_before_md": transcript_markdown("处理前转写", result.get("transcript_before")),
        "transcript_after_md": transcript_markdown("增强后转写", result.get("transcript_after")),
        "diff_html": diff_html(result),
        "cer_md": cer_markdown(result),
        "spectrogram_image": file_if_exists(
            result.get("spectrogram_path"), allowed_roots=allowed_roots
        ),
        "waveform_image": file_if_exists(
            result.get("waveform_path"), allowed_roots=allowed_roots
        ),
        "runtime_md": runtime_markdown(result),
        "warnings_html": warnings_html(result),
        "mixed_download": file_if_exists(mixed_path, allowed_roots=allowed_roots),
        "full_download": file_if_exists(
            result.get("full_output_path"), allowed_roots=allowed_roots
        ),
        "transcript_download": None,
        "result_download": None,
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
