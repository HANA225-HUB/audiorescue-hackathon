"""Gradio layout owned by B."""

from __future__ import annotations

import os
import re
import html
from pathlib import Path
from typing import Any

from .live_controller import (
    DEFAULT_INPUT_CHOICE,
    DEFAULT_OUTPUT_CHOICE,
    LIVE_CONTROLLER,
    VIRTUAL_OUTPUT_CHOICE,
    render_floating_html,
)
from .presenters import (
    failed_result,
    list_fixture_names,
    load_fixture,
    result_to_ui_tuple,
    strength_value,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
CSS_PATH = ROOT_DIR / "ui" / "styles.css"

OPEN_FLOATING_JS = """
() => {
  window.open(
    "/audiorescue/live/floating",
    "audiorescue_meeting_floating",
    "popup=yes,width=420,height=560,menubar=no,toolbar=no,location=no,status=no"
  );
}
"""

REMOTE_HTML_PATTERNS = (
    re.compile(
        r"<script\b[^>]*\bsrc=[\"']https://cdnjs\.cloudflare\.com/ajax/libs/iframe-resizer/[^\"']+[\"'][^>]*>\s*</script>",
        re.I,
    ),
    re.compile(
        r"<link\b[^>]*\bhref=[\"']https://fonts\.(?:googleapis|gstatic)\.com[^\"']*[\"'][^>]*>",
        re.I,
    ),
    re.compile(
        r"<link\b[^>]*\bhref=[\"']https://fonts\.googleapis\.com/css2\?[^\"']*[\"'][^>]*>",
        re.I,
    ),
)


def _read_css() -> str:
    return CSS_PATH.read_text(encoding="utf-8") if CSS_PATH.exists() else ""


def strip_remote_html_resources(html_text: str) -> str:
    cleaned = html_text
    for pattern in REMOTE_HTML_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    return cleaned


def _escape_md(value: object) -> str:
    return html.escape(str(value), quote=False)


def _format_transcript(lines: tuple[str, ...]) -> str:
    if not lines:
        return "暂无确认字幕。"
    return "\n".join(f"{index + 1}. {_escape_md(line)}" for index, line in enumerate(lines[-8:]))


def _live_ui_values(snapshot=None) -> tuple[str, str, str, str, str, str, str, str, str]:
    snap = snapshot or LIVE_CONTROLLER.snapshot()
    audio_error = f"错误：{_escape_md(snap.audio_error)}" if snap.audio_error else "错误：无"
    meeting_error = f"错误：{_escape_md(snap.meeting_error)}" if snap.meeting_error else "错误：无"
    audio_status = (
        "### 音频引擎\n"
        f"**{_escape_md(snap.audio_status)}**\n\n"
        f"模式：`{_escape_md(snap.audio_mode)}` · RTF：`{snap.realtime_factor:.3f}` · {audio_error}"
    )
    meeting_status = (
        "### 会议助手\n"
        f"**{_escape_md(snap.meeting_status)}**\n\n"
        f"实时字幕：`{1 if snap.partial_text else 0}` · "
        f"正式记录：`{len(snap.transcript)}` · ASR丢包：`{snap.asr_dropped_packets}` · {meeting_error}"
    )
    floating_status = (
        "### 悬浮提示窗\n"
        "**同源状态**\n\n"
        "主页面与悬浮窗共用同一个后端控制器和快照接口。"
    )
    meter = (
        f"输入块：`{snap.input_blocks}` · 增强块：`{snap.enhanced_blocks}` · "
        f"输出块：`{snap.output_blocks}` · 丢帧：`{snap.input_drops + snap.output_drops}` · "
        f"欠载：`{snap.output_underruns}` · 重同步：`{snap.resyncs}` · "
        f"P95：`{snap.inference_p95_ms:.2f} ms` · 最大：`{snap.inference_max_ms:.2f} ms`"
    )
    live_text = (
        "#### 实时未定稿字幕\n"
        f"{_escape_md(snap.partial_text) if snap.partial_text else '等待增强后的 16 kHz 音频进入转写流。'}"
    )
    transcript = "#### 正式字幕记录\n" + _format_transcript(snap.transcript)
    suggestion = (
        "#### 最新建议\n"
        f"{_escape_md(snap.suggestion) if snap.suggestion else '启动会议助手后，这里显示最新大模型建议。'}"
    )
    floating = render_floating_html(snap)
    action = f"状态：{_escape_md(snap.last_action)}"
    return (
        audio_status,
        meeting_status,
        floating_status,
        meter,
        live_text,
        transcript,
        suggestion,
        floating,
        action,
    )


class OfflineHtmlResourceMiddleware:
    """Remove Gradio default remote tags from HTML responses for offline demos."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        start_message = None
        should_filter = False
        body_parts: list[bytes] = []

        async def send_wrapper(message):
            nonlocal start_message, should_filter
            if message["type"] == "http.response.start":
                start_message = dict(message)
                headers = [
                    (key.lower(), value)
                    for key, value in start_message.get("headers", [])
                ]
                should_filter = any(
                    key == b"content-type" and b"text/html" in value.lower()
                    for key, value in headers
                )
                if not should_filter:
                    await send(message)
                return

            if message["type"] != "http.response.body" or not should_filter:
                await send(message)
                return

            body_parts.append(message.get("body", b""))
            if message.get("more_body", False):
                return

            raw_body = b"".join(body_parts)
            try:
                filtered_text = strip_remote_html_resources(raw_body.decode("utf-8"))
                filtered_body = filtered_text.encode("utf-8")
            except UnicodeDecodeError:
                filtered_body = raw_body

            headers = [
                (key, value)
                for key, value in start_message.get("headers", [])
                if key.lower() != b"content-length"
            ]
            headers.append((b"content-length", str(len(filtered_body)).encode("ascii")))
            start_message["headers"] = headers
            await send(start_message)
            await send({**message, "body": filtered_body, "more_body": False})

        await self.app(scope, receive, send_wrapper)


def offline_launch_app_kwargs() -> dict[str, Any]:
    from starlette.middleware import Middleware
    from .file_delivery import FileDeliveryMiddleware

    return {
        "middleware": [
            Middleware(FileDeliveryMiddleware),
            Middleware(OfflineHtmlResourceMiddleware),
        ]
    }


def _first_fixture() -> str:
    names = list_fixture_names()
    return names[0] if names else ""


def _fixture_mode_default() -> bool:
    """Keep competition startup on the real pipeline unless explicitly opted in."""

    return os.environ.get("AUDIORESCUE_UI_FIXTURE", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def _run_fixture(fixture_name: str) -> tuple[Any, ...]:
    try:
        result = load_fixture(fixture_name)
    except Exception:
        result = failed_result("", "UI_FIXTURE_LOAD_FAILED")
    return result_to_ui_tuple(result)


def _run_real_pipeline(
    input_file: Any,
    strength_label: str,
    reference_text: str,
    force_recompute: bool,
) -> tuple[Any, ...]:
    if input_file is None:
        return result_to_ui_tuple(failed_result("请选择上传文件，或切回 fixture 模式。", "INPUT_INVALID"))

    input_path = getattr(input_file, "name", input_file)
    try:
        from core.pipeline import process_audio
    except Exception:
        return result_to_ui_tuple(failed_result("", "UI_PIPELINE_UNAVAILABLE"))

    try:
        result = process_audio(
            input_path=str(input_path),
            strength=strength_value(strength_label),
            enable_events=False,
            reference_text=reference_text or None,
            force_recompute=bool(force_recompute),
        )
    except NotImplementedError:
        result = failed_result("", "UI_PIPELINE_UNAVAILABLE")
    except Exception:
        result = failed_result("", "UI_PIPELINE_FAILED")
    return result_to_ui_tuple(result)


def _run(
    use_fixture: bool,
    fixture_name: str,
    input_file: Any,
    strength_label: str,
    reference_text: str,
    force_recompute: bool,
) -> tuple[Any, ...]:
    if use_fixture:
        if not _fixture_mode_default():
            return result_to_ui_tuple(
                failed_result(
                    "",
                    "UI_FIXTURE_DISABLED",
                )
            )
        return _run_fixture(fixture_name)
    return _run_real_pipeline(input_file, strength_label, reference_text, force_recompute)


def build_demo():
    try:
        import gradio as gr
    except Exception as exc:  # pragma: no cover - manual environment check.
        raise RuntimeError(
            "Gradio 尚不可用。B 侧 presenter 和 fixture 测试仍可运行；"
            "页面启动前请安装 requirements.txt。"
        ) from exc

    fixture_enabled = _fixture_mode_default()
    fixture_names = list_fixture_names() if fixture_enabled else []
    default_fixture = _first_fixture() if fixture_enabled else ""
    theme = gr.themes.Base(
        font=["system-ui", "-apple-system", "BlinkMacSystemFont", "Segoe UI", "sans-serif"],
        font_mono=["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
    )

    with gr.Blocks(
        css=_read_css(),
        title="听清又听懂·智能音频急救台",
        theme=theme,
        analytics_enabled=False,
        elem_classes=["ar-app"],
    ) as demo:
        gr.Markdown(
            "# 听清又听懂·智能音频急救台\n"
            "输入与设置 → 运行状态 → A/B 听感 → 双路转写 → 可视化 → 工程证据。",
            elem_classes=["ar-hero"],
        )

        with gr.Row(equal_height=True, elem_classes=["ar-control-grid"]):
            with gr.Column(scale=5, elem_classes=["ar-panel", "ar-upload-panel"]):
                if fixture_enabled:
                    use_fixture = gr.Checkbox(
                        value=True,
                        label="开发专用：使用前端 fixture 假数据",
                    )
                    fixture = gr.Dropdown(
                        choices=fixture_names,
                        value=default_fixture,
                        label="ProcessResult fixture",
                        interactive=True,
                    )
                    gr.Markdown(
                        "开发联调样例仅验证链路，不作为比赛效果证据。"
                        "正式演示请重新以默认环境启动。",
                        elem_classes=["ar-notice"],
                    )
                else:
                    # Keep the callback signature stable without exposing a
                    # production-page switch capable of fabricating results.
                    use_fixture = gr.State(False)
                    fixture = gr.State("")
                input_file = gr.Audio(label="上传音频", type="filepath")
            with gr.Column(scale=4, elem_classes=["ar-panel", "ar-settings-panel"]):
                strength = gr.Radio(
                    choices=["轻度", "标准", "强力"],
                    value="标准",
                    label="增强强度",
                )
                reference_text = gr.Textbox(
                    label="参考台词（可选）",
                    lines=2,
                    placeholder="参考文本只传给 pipeline 计算 CER，禁止进入 Whisper prompt。",
                )
                with gr.Accordion("工程选项", open=False, elem_classes=["ar-engineering"]):
                    force_recompute = gr.Checkbox(value=False, label="强制重算")
                run_button = gr.Button("开始急救", variant="primary")

        gr.Markdown(
            "正在处理时：规范化 → 增强 → 双轨转写 → 可视化。当前没有实时阶段回调。",
            elem_classes=["ar-process-note"],
        )

        with gr.Row(equal_height=True, elem_classes=["ar-result-grid"]):
            status = gr.HTML(elem_classes=["ar-panel", "ar-status-panel"])
            input_info = gr.Markdown(elem_classes=["ar-panel", "ar-meta-panel"])

        playback_note = gr.Markdown(elem_classes=["ar-panel", "ar-playback-note"])

        with gr.Row(equal_height=True, elem_classes=["ar-audio-grid"]):
            original_audio = gr.HTML(
                elem_classes=["ar-audio"],
            )
            enhanced_audio = gr.HTML(
                elem_classes=["ar-audio"],
            )

        with gr.Row(equal_height=True, elem_classes=["ar-transcript-grid"]):
            transcript_before = gr.Markdown(elem_classes=["ar-panel", "ar-transcript"])
            transcript_after = gr.Markdown(elem_classes=["ar-panel", "ar-transcript"])

        with gr.Row(equal_height=True, elem_classes=["ar-analysis-grid"]):
            with gr.Column(scale=5, elem_classes=["ar-panel", "ar-diff-panel"]):
                gr.Markdown("### 文本差异", elem_classes=["ar-section-title"])
                diff = gr.HTML()
            cer = gr.Markdown(elem_classes=["ar-panel", "ar-cer-panel"])

        with gr.Row(equal_height=True, elem_classes=["ar-visual-grid"]):
            spectrogram = gr.HTML(
                elem_classes=["ar-visual"],
            )
            waveform = gr.HTML(
                elem_classes=["ar-visual"],
            )

        with gr.Row(equal_height=True, elem_classes=["ar-evidence-grid"]):
            runtime = gr.Markdown(elem_classes=["ar-panel", "ar-runtime-panel"])
            with gr.Column(scale=1, elem_classes=["ar-panel", "ar-warning-panel"]):
                gr.Markdown("### 警告与工程细节", elem_classes=["ar-section-title"])
                warnings = gr.HTML()

        with gr.Row(elem_classes=["ar-download-grid"]):
            mixed_download = gr.HTML(
                elem_classes=["ar-download"],
            )
            full_download = gr.HTML(
                elem_classes=["ar-download"],
            )
            transcript_download = gr.HTML(
                elem_classes=["ar-download"],
            )
            result_download = gr.HTML(
                elem_classes=["ar-download"],
            )

        outputs = [
            status,
            input_info,
            playback_note,
            original_audio,
            enhanced_audio,
            transcript_before,
            transcript_after,
            diff,
            cer,
            spectrogram,
            waveform,
            runtime,
            warnings,
            mixed_download,
            full_download,
            transcript_download,
            result_download,
        ]

        run_button.click(
            _run,
            inputs=[use_fixture, fixture, input_file, strength, reference_text, force_recompute],
            outputs=outputs,
            scroll_to_output=True,
            show_progress="full",
        )

        gr.Markdown(
            "## 实时会议输入\n"
            "真实麦克风 → 实时增强 → 监听或 BlackHole 虚拟麦 → 字幕与会议建议。",
            elem_classes=["ar-hero", "ar-meeting-hero"],
        )

        with gr.Column(elem_classes=["ar-meeting-page"]):
            with gr.Row(equal_height=True, elem_classes=["ar-meeting-status-row"]):
                audio_live_status = gr.Markdown(
                    _live_ui_values()[0],
                    elem_classes=["ar-meeting-status-card"],
                )
                meeting_live_status = gr.Markdown(
                    _live_ui_values()[1],
                    elem_classes=["ar-meeting-status-card"],
                )
                floating_live_status = gr.Markdown(
                    _live_ui_values()[2],
                    elem_classes=["ar-meeting-status-card"],
                )

            with gr.Row(equal_height=True, elem_classes=["ar-meeting-grid"]):
                with gr.Column(scale=4, elem_classes=["ar-meeting-panel", "ar-meeting-control-panel"]):
                    gr.Markdown("### 设备与实时引擎", elem_classes=["ar-meeting-panel-title"])
                    input_device = gr.Dropdown(
                        choices=[DEFAULT_INPUT_CHOICE],
                        value=DEFAULT_INPUT_CHOICE,
                        label="音频输入",
                        interactive=True,
                        elem_classes=["ar-meeting-input"],
                    )
                    output_device = gr.Dropdown(
                        choices=[DEFAULT_OUTPUT_CHOICE, VIRTUAL_OUTPUT_CHOICE],
                        value=DEFAULT_OUTPUT_CHOICE,
                        label="监听 / 虚拟麦输出",
                        interactive=True,
                        elem_classes=["ar-meeting-input"],
                    )
                    live_mode = gr.Radio(
                        ["enhanced", "quiet", "raw"],
                        value="enhanced",
                        label="实时模式",
                        interactive=True,
                        elem_classes=["ar-meeting-mode"],
                    )
                    refresh_devices = gr.Button("刷新设备", variant="secondary", elem_classes=["ar-meeting-action", "ar-meeting-action-muted"])
                    with gr.Row(elem_classes=["ar-meeting-button-row"]):
                        start_live_audio = gr.Button("启动实时音频", variant="primary", elem_classes=["ar-meeting-action"])
                        stop_live_audio = gr.Button("停止音频", variant="secondary", elem_classes=["ar-meeting-action", "ar-meeting-action-muted"])
                    apply_live_mode = gr.Button("应用实时模式", variant="secondary", elem_classes=["ar-meeting-action", "ar-meeting-action-muted"])
                    live_meter = gr.Markdown(
                        _live_ui_values()[3],
                        elem_classes=["ar-meeting-meter"],
                    )

                with gr.Column(scale=5, elem_classes=["ar-meeting-panel", "ar-meeting-transcript-panel"]):
                    gr.Markdown("### 字幕与会议预设", elem_classes=["ar-meeting-panel-title"])
                    meeting_preset = gr.Textbox(
                        label="会前会议预设",
                        placeholder="填写会议主题、角色、目标和需要重点关注的表达方式。",
                        lines=4,
                        elem_classes=["ar-meeting-input", "ar-meeting-preset"],
                    )
                    with gr.Row(elem_classes=["ar-meeting-button-row"]):
                        start_meeting = gr.Button("启动实时转写", variant="primary", elem_classes=["ar-meeting-action"])
                        stop_meeting = gr.Button("停止会议助手", variant="secondary", elem_classes=["ar-meeting-action", "ar-meeting-action-muted"])
                    live_partial = gr.Markdown(
                        _live_ui_values()[4],
                        elem_classes=["ar-meeting-live-text"],
                    )
                    live_transcript = gr.Markdown(
                        _live_ui_values()[5],
                        elem_classes=["ar-meeting-log"],
                    )

                with gr.Column(scale=4, elem_classes=["ar-meeting-panel", "ar-meeting-assist-panel"]):
                    gr.Markdown("### 会议建议", elem_classes=["ar-meeting-panel-title"])
                    live_suggestion = gr.Markdown(
                        _live_ui_values()[6],
                        elem_classes=["ar-meeting-suggestion"],
                    )
                    request_next_line = gr.Button("立即生成下一句建议", variant="primary", elem_classes=["ar-meeting-action"])
                    open_floating = gr.Button("打开悬浮提示窗", variant="secondary", elem_classes=["ar-meeting-action", "ar-meeting-action-muted"])
                    floating_preview = gr.HTML(
                        _live_ui_values()[7],
                        elem_classes=["ar-meeting-float-preview"],
                    )
                    refresh_live_status = gr.Button("刷新实时状态", variant="secondary", elem_classes=["ar-meeting-action", "ar-meeting-action-muted"])

            live_action_note = gr.Markdown(
                _live_ui_values()[8],
                elem_classes=["ar-meeting-action-note"],
            )

        live_outputs = [
            audio_live_status,
            meeting_live_status,
            floating_live_status,
            live_meter,
            live_partial,
            live_transcript,
            live_suggestion,
            floating_preview,
            live_action_note,
        ]

        def refresh_live_devices():
            try:
                input_choices, output_choices, _ = LIVE_CONTROLLER.list_devices()
                input_value = input_choices[0] if input_choices else DEFAULT_INPUT_CHOICE
                output_value = output_choices[0] if output_choices else DEFAULT_OUTPUT_CHOICE
                return (
                    gr.update(choices=input_choices, value=input_value),
                    gr.update(choices=output_choices, value=output_value),
                    *_live_ui_values(),
                )
            except Exception as exc:
                LIVE_CONTROLLER.record_ui_error("设备刷新失败", exc)
                return (
                    gr.update(choices=[DEFAULT_INPUT_CHOICE], value=DEFAULT_INPUT_CHOICE),
                    gr.update(choices=[DEFAULT_OUTPUT_CHOICE, VIRTUAL_OUTPUT_CHOICE], value=DEFAULT_OUTPUT_CHOICE),
                    *_live_ui_values(),
                )

        def start_audio_from_ui(selected_input, selected_output, selected_mode):
            return _live_ui_values(
                LIVE_CONTROLLER.start_audio(selected_input, selected_output, selected_mode)
            )

        def stop_audio_from_ui():
            return _live_ui_values(LIVE_CONTROLLER.stop_audio())

        def apply_mode_from_ui(selected_mode):
            return _live_ui_values(LIVE_CONTROLLER.set_mode(selected_mode))

        def start_meeting_from_ui(preset):
            return _live_ui_values(LIVE_CONTROLLER.start_meeting(preset))

        def stop_meeting_from_ui():
            return _live_ui_values(LIVE_CONTROLLER.stop_meeting())

        def request_next_line_from_ui():
            return _live_ui_values(LIVE_CONTROLLER.request_next_line())

        def refresh_live_from_ui():
            return _live_ui_values()

        def open_floating_noop():
            return None

        refresh_devices.click(
            refresh_live_devices,
            outputs=[input_device, output_device, *live_outputs],
        )
        start_live_audio.click(
            start_audio_from_ui,
            inputs=[input_device, output_device, live_mode],
            outputs=live_outputs,
            show_progress="full",
        )
        stop_live_audio.click(stop_audio_from_ui, outputs=live_outputs)
        apply_live_mode.click(apply_mode_from_ui, inputs=[live_mode], outputs=live_outputs)
        start_meeting.click(
            start_meeting_from_ui,
            inputs=[meeting_preset],
            outputs=live_outputs,
            show_progress="full",
        )
        stop_meeting.click(stop_meeting_from_ui, outputs=live_outputs)
        request_next_line.click(request_next_line_from_ui, outputs=live_outputs)
        refresh_live_status.click(refresh_live_from_ui, outputs=live_outputs)
        open_floating.click(open_floating_noop, js=OPEN_FLOATING_JS)

        if hasattr(demo, "unload"):
            demo.unload(lambda: LIVE_CONTROLLER.cleanup())

        if default_fixture and fixture_enabled:
            demo.load(_run_fixture, inputs=[fixture], outputs=outputs)

    return demo
