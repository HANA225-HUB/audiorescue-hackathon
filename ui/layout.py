"""Gradio layout owned by B."""

from __future__ import annotations

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


def _read_css() -> str:
    return CSS_PATH.read_text(encoding="utf-8") if CSS_PATH.exists() else ""


def _first_fixture() -> str:
    names = list_fixture_names()
    return names[0] if names else ""


def _run_fixture(fixture_name: str) -> tuple[Any, ...]:
    try:
        result = load_fixture(fixture_name)
    except Exception as exc:
        result = failed_result(f"fixture 读取失败：{exc}", "INTERNAL_ERROR")
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
    except Exception as exc:
        return result_to_ui_tuple(failed_result(f"C 的 process_audio 尚不可用：{exc}", "INTERNAL_ERROR"))

    try:
        result = process_audio(
            input_path=str(input_path),
            strength=strength_value(strength_label),
            enable_events=False,
            reference_text=reference_text or None,
            force_recompute=bool(force_recompute),
        )
    except NotImplementedError as exc:
        result = failed_result(f"真实管线尚未实现：{exc}", "INTERNAL_ERROR")
    except Exception as exc:
        result = failed_result(f"真实管线调用失败：{exc}", "INTERNAL_ERROR")
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

    fixture_names = list_fixture_names()
    default_fixture = _first_fixture()

    with gr.Blocks(
        css=_read_css(),
        title="听清又听懂·智能音频急救台",
        elem_classes=["ar-app"],
    ) as demo:
        gr.Markdown(
            "# 听清又听懂·智能音频急救台\n"
            "输入与设置 → 运行状态 → A/B 听感 → 双路转写 → 可视化 → 工程证据。",
            elem_classes=["ar-hero"],
        )

        with gr.Row(equal_height=True, elem_classes=["ar-control-grid"]):
            with gr.Column(scale=5, elem_classes=["ar-panel", "ar-upload-panel"]):
                use_fixture = gr.Checkbox(value=True, label="使用前端 fixture 模式")
                fixture = gr.Dropdown(
                    choices=fixture_names,
                    value=default_fixture,
                    label="ProcessResult fixture",
                    interactive=True,
                )
                gr.Markdown(
                    "开发联调样例仅验证链路，不作为比赛效果证据。正式样例等待 C 的 `configs/demo.yaml`。",
                    elem_classes=["ar-notice"],
                )
                input_file = gr.Audio(label="上传音频", type="filepath")
            with gr.Column(scale=4, elem_classes=["ar-panel", "ar-settings-panel"]):
                strength = gr.Radio(
                    choices=["轻度", "标准", "强力"],
                    value="标准",
                    label="增强强度",
                )
                reference_text = gr.Textbox(
                    label="参考台词（可选）",
                    lines=4,
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
            status = gr.Markdown(elem_classes=["ar-panel", "ar-status-panel"])
            input_info = gr.Markdown(elem_classes=["ar-panel", "ar-meta-panel"])

        playback_note = gr.Markdown(elem_classes=["ar-panel", "ar-playback-note"])

        with gr.Row(equal_height=True, elem_classes=["ar-audio-grid"]):
            original_audio = gr.Audio(
                label="处理前：标准化原轨",
                type="filepath",
                interactive=False,
                elem_classes=["ar-audio"],
            )
            enhanced_audio = gr.Audio(
                label="增强后：混合增强轨",
                type="filepath",
                interactive=False,
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
            spectrogram = gr.Image(
                label="声谱图对照",
                type="filepath",
                interactive=False,
                elem_classes=["ar-visual"],
            )
            waveform = gr.Image(
                label="波形对照",
                type="filepath",
                interactive=False,
                elem_classes=["ar-visual"],
            )

        with gr.Row(equal_height=True, elem_classes=["ar-evidence-grid"]):
            runtime = gr.Markdown(elem_classes=["ar-panel", "ar-runtime-panel"])
            with gr.Column(scale=1, elem_classes=["ar-panel", "ar-warning-panel"]):
                gr.Markdown("### 警告与工程细节", elem_classes=["ar-section-title"])
                warnings = gr.HTML()

        with gr.Row(elem_classes=["ar-download-grid"]):
            mixed_download = gr.File(
                label="下载混合增强 WAV",
                elem_classes=["ar-download"],
            )
            full_download = gr.File(
                label="下载 100% 增强 WAV",
                elem_classes=["ar-download"],
            )
            transcript_download = gr.File(
                label="下载转写 TXT（待 pipeline 提供）",
                elem_classes=["ar-download"],
            )
            result_download = gr.File(
                label="下载结果 JSON（待 pipeline 提供）",
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

        if default_fixture:
            demo.load(_run_fixture, inputs=[fixture], outputs=outputs)

    return demo
