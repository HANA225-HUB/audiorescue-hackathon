"""Gradio layout owned by B."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from .presenters import (
    failed_result,
    list_fixture_names,
    load_fixture,
    result_to_ui_tuple,
    strength_value,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
CSS_PATH = ROOT_DIR / "ui" / "styles.css"

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
        )

        if default_fixture and fixture_enabled:
            demo.load(_run_fixture, inputs=[fixture], outputs=outputs)

    return demo
