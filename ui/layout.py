"""Gradio layout owned by B."""

from __future__ import annotations

import os
import re
import base64
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
ASSET_DIR = ROOT_DIR / "ui" / "assets"
BACKGROUND_ASSETS = {
    "--ar-bg-intro": "intro-bg.webp",
    "--ar-bg-feature": "feature-bg.webp",
    "--ar-bg-audio": "audio-bg.webp",
    "--ar-bg-meeting": "meeting-bg.webp",
}
MEETING_SCENARIO_CHOICES = [
    ("普通会议", "general"),
    ("答辩 / 评审", "defense"),
    ("组会 / 研讨", "group_meeting"),
    ("项目 / 比赛汇报", "project_report"),
    ("自定义", "custom"),
]
MEETING_SCENARIO_LABELS = dict((value, label) for label, value in MEETING_SCENARIO_CHOICES)
SUGGESTION_KIND_LABELS = {
    "answer": "问题回答",
    "continue_section": "继续当前内容",
    "next_section": "下一段提示",
    "clarify": "澄清建议",
    "correction": "修正提示",
    "close": "收尾提示",
}
APP_JS = """
() => {
  if (window.__audiorescueFlowReady) {
    return;
  }
  window.__audiorescueFlowReady = true;

  const root = document.documentElement;
  let targetX = 50;
  let targetY = 42;
  let currentX = targetX;
  let currentY = targetY;

  const updateVars = () => {
    currentX += (targetX - currentX) * 0.12;
    currentY += (targetY - currentY) * 0.12;
    const tiltX = ((currentX - 50) / 50) * 4;
    const tiltY = ((50 - currentY) / 50) * 3;
    root.style.setProperty("--ar-pointer-x", `${currentX.toFixed(2)}%`);
    root.style.setProperty("--ar-pointer-y", `${currentY.toFixed(2)}%`);
    root.style.setProperty("--ar-tilt-x", `${tiltX.toFixed(2)}deg`);
    root.style.setProperty("--ar-tilt-y", `${tiltY.toFixed(2)}deg`);
    window.requestAnimationFrame(updateVars);
  };

  window.addEventListener("pointermove", (event) => {
    targetX = Math.max(0, Math.min(100, (event.clientX / window.innerWidth) * 100));
    targetY = Math.max(0, Math.min(100, (event.clientY / window.innerHeight) * 100));
    document.body.classList.add("ar-pointer-active");
  }, { passive: true });

  window.addEventListener("pointerdown", () => {
    document.body.classList.add("ar-pointer-down");
    window.setTimeout(() => document.body.classList.remove("ar-pointer-down"), 180);
  }, { passive: true });

  const initSpectrumCanvas = () => {
    let page = null;
    let canvas = null;
    let ctx = null;
    let width = 0;
    let height = 0;
    let dpr = 1;
    const bands = [0.27, 0.38, 0.5, 0.62, 0.75];
    const phases = bands.map((_, index) => index * 1.73);

    const resize = () => {
      if (!page || !canvas || !ctx) {
        return;
      }
      const rect = page.getBoundingClientRect();
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      width = Math.max(1, Math.floor(rect.width));
      height = Math.max(1, Math.floor(rect.height));
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };

    const ensureCanvas = () => {
      const nextPage = document.querySelector(".ar-poster-page");
      if (!nextPage) {
        return false;
      }

      if (nextPage !== page || !canvas || !nextPage.contains(canvas)) {
        page = nextPage;
        canvas = page.querySelector("canvas.ar-spectrum-canvas");
        if (!canvas) {
          canvas = document.createElement("canvas");
          canvas.className = "ar-spectrum-canvas";
          page.prepend(canvas);
        }
        ctx = canvas.getContext("2d");
        resize();
      }

      return Boolean(ctx);
    };

    const drawWave = (time, band, index) => {
      const baseY = height * band;
      const verticalDistance = Math.abs(currentY - band * 100);
      const verticalEnergy = Math.max(0, 1 - verticalDistance / 23);
      const points = 96;
      ctx.beginPath();
      for (let i = 0; i <= points; i += 1) {
        const ratio = i / points;
        const x = ratio * width;
        const horizontalDistance = Math.abs(currentX - ratio * 100);
        const cursorEnergy = Math.max(0, 1 - Math.hypot(horizontalDistance / 32, verticalDistance / 23));
        const localEnergy = 1 + verticalEnergy * 0.55 + cursorEnergy * 2.2;
        const envelope = 0.35 + 0.65 * Math.sin(Math.PI * ratio);
        const slow = Math.sin(time * (1.1 + index * 0.08) + ratio * Math.PI * (5.2 + index));
        const fast = Math.sin(time * (2.4 + index * 0.18) + ratio * Math.PI * (13.5 + index * 0.8));
        const jitter = Math.sin(time * (4.0 + index * 0.22) + ratio * Math.PI * 31) * cursorEnergy;
        const amplitude = (10 + index * 2.3) * localEnergy;
        const y = baseY + (slow * 0.72 + fast * 0.2 + jitter * 0.32) * amplitude * envelope;
        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }
      const warm = index % 2 === 0;
      ctx.lineWidth = warm ? 2.2 : 1.8;
      ctx.strokeStyle = warm
        ? "rgba(181, 105, 47, 0.62)"
        : "rgba(71, 143, 154, 0.48)";
      ctx.shadowColor = warm ? "rgba(255, 177, 93, 0.28)" : "rgba(119, 211, 206, 0.24)";
      ctx.shadowBlur = 14;
      ctx.stroke();
      ctx.shadowBlur = 0;
    };

    const drawBars = (time) => {
      const count = 34;
      const baseY = height * 0.84;
      const spacing = width / count;
      for (let i = 0; i < count; i += 1) {
        const ratio = (i + 0.5) / count;
        const x = ratio * width;
        const horizontalDistance = Math.abs(currentX - ratio * 100);
        const cursorEnergy = Math.max(0, 1 - horizontalDistance / 18);
        const pulse = Math.sin(time * 2.2 + i * 0.68) * 0.5 + 0.5;
        const heightBoost = (22 + 54 * pulse) * (1 + cursorEnergy * 1.65);
        ctx.fillStyle = i % 3 === 0
          ? "rgba(205, 128, 62, 0.28)"
          : "rgba(93, 154, 151, 0.24)";
        ctx.fillRect(x - 2, baseY - heightBoost, 4, heightBoost);
      }
    };

    const render = (now) => {
      if (!ensureCanvas()) {
        window.requestAnimationFrame(render);
        return;
      }

      const rect = page.getBoundingClientRect();
      if (Math.floor(rect.width) !== width || Math.floor(rect.height) !== height) {
        resize();
      }

      const time = now / 1000;
      ctx.clearRect(0, 0, width, height);
      const gradient = ctx.createLinearGradient(0, 0, width, height);
      gradient.addColorStop(0, "rgba(255, 242, 222, 0.92)");
      gradient.addColorStop(0.55, "rgba(239, 207, 169, 0.78)");
      gradient.addColorStop(1, "rgba(221, 174, 124, 0.72)");
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, width, height);

      ctx.globalCompositeOperation = "multiply";
      ctx.fillStyle = "rgba(123, 80, 39, 0.05)";
      for (let x = 0; x < width; x += 96) {
        ctx.fillRect(x, 0, 1, height);
      }
      for (let y = 0; y < height; y += 88) {
        ctx.fillRect(0, y, width, 1);
      }
      ctx.globalCompositeOperation = "source-over";

      bands.forEach((band, index) => drawWave(time + phases[index], band, index));
      drawBars(time);
      window.requestAnimationFrame(render);
    };

    const observer = new MutationObserver(() => ensureCanvas());
    observer.observe(document.body, { childList: true, subtree: true });
    ensureCanvas();
    window.addEventListener("resize", resize, { passive: true });
    window.requestAnimationFrame(render);
  };

  window.requestAnimationFrame(updateVars);
  initSpectrumCanvas();
}
"""
SCROLL_TOP_JS = """
() => {
  document.body.classList.add("ar-page-switching");
  window.setTimeout(() => document.body.classList.remove("ar-page-switching"), 1040);
  window.scrollTo({ top: 0, left: 0, behavior: "smooth" });
}
"""
OPEN_FLOATING_JS = """
() => {
  const url = new URL("/audiorescue/live/floating", window.location.origin).toString();
  const opened = window.open(
    url,
    "audiorescue_meeting_floating",
    "popup=yes,width=440,height=720,menubar=no,toolbar=no,location=no,status=no"
  );
  if (opened && !opened.closed) {
    try {
      opened.focus();
    } catch (error) {
      // Some embedded browsers expose a restricted window object.
    }
    return;
  }

  let notice = document.getElementById("ar-floating-open-notice");
  if (!notice) {
    notice = document.createElement("aside");
    notice.id = "ar-floating-open-notice";
    notice.setAttribute("role", "status");
    notice.innerHTML = `
      <div>
        <strong>悬浮窗被当前浏览器拦截</strong>
        <span>主界面会保持不变；请手动打开独立提示页。</span>
      </div>
      <a href="${url}" target="_blank" rel="noopener noreferrer">打开悬浮提示窗</a>
      <button type="button" aria-label="关闭悬浮窗提示">×</button>
    `;
    Object.assign(notice.style, {
      position: "fixed",
      right: "22px",
      bottom: "22px",
      zIndex: "9999",
      display: "grid",
      gap: "10px",
      maxWidth: "min(360px, calc(100vw - 32px))",
      padding: "14px",
      border: "1px solid rgba(139, 99, 60, 0.26)",
      borderRadius: "8px",
      background: "rgba(255, 246, 230, 0.94)",
      boxShadow: "0 18px 44px rgba(96, 66, 39, 0.18)",
      color: "#2b2118",
      fontFamily: "Songti SC, STSong, serif"
    });
    const link = notice.querySelector("a");
    Object.assign(link.style, {
      color: "#4e3825",
      fontWeight: "800",
      textDecoration: "underline"
    });
    const close = notice.querySelector("button");
    Object.assign(close.style, {
      position: "absolute",
      top: "8px",
      right: "8px",
      border: "0",
      background: "transparent",
      color: "#6d4b31",
      fontSize: "18px",
      cursor: "pointer"
    });
    close.addEventListener("click", () => notice.remove());
    document.body.appendChild(notice);
  } else {
    notice.style.display = "grid";
  }
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


def _asset_data_url(filename: str) -> str:
    path = ASSET_DIR / filename
    if not path.exists():
        return "none"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'url("data:image/webp;base64,{encoded}")'


def _read_css() -> str:
    css = CSS_PATH.read_text(encoding="utf-8") if CSS_PATH.exists() else ""
    if not css:
        return css

    background_vars = "\n".join(
        f"  {name}: {_asset_data_url(filename)};"
        for name, filename in BACKGROUND_ASSETS.items()
    )
    return f"{css}\n\n:root {{\n{background_vars}\n}}\n"


def strip_remote_html_resources(html_text: str) -> str:
    cleaned = html_text
    for pattern in REMOTE_HTML_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    return cleaned


def _escape_md(value: object) -> str:
    escaped = html.escape(str(value), quote=False)
    for marker in (
        "\\",
        "`",
        "*",
        "_",
        "{",
        "}",
        "[",
        "]",
        "(",
        ")",
        "#",
        "+",
        "-",
        ".",
        "!",
        "|",
        ">",
    ):
        escaped = escaped.replace(marker, f"\\{marker}")
    return escaped


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
        f"正式记录：`{len(snap.transcript)}` · ASR丢包：`{snap.asr_dropped_packets}` · {meeting_error}\n\n"
        f"场景：`{_escape_md(MEETING_SCENARIO_LABELS.get(snap.meeting_scenario, snap.meeting_scenario))}` · "
        f"会议资料：`{len(snap.meeting_material_names)}`"
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
    suggestion_meta: list[str] = []
    if snap.suggestion_kind:
        suggestion_meta.append(
            SUGGESTION_KIND_LABELS.get(snap.suggestion_kind, snap.suggestion_kind)
        )
    if snap.confidence > 0:
        suggestion_meta.append(f"置信度 {snap.confidence:.0%}")
    if snap.needs_verification:
        suggestion_meta.append("⚠ 需要核实")
    source_text = " / ".join(_escape_md(item) for item in snap.suggestion_sources)
    material_text = " / ".join(
        _escape_md(item) for item in snap.meeting_material_names
    )
    warning_text = " / ".join(_escape_md(item) for item in snap.material_warnings)
    suggestion = (
        "#### 最新建议\n"
        f"{_escape_md(snap.suggestion) if snap.suggestion else '启动会议助手后，这里显示最新大模型建议。'}\n\n"
        f"{'·'.join(suggestion_meta) if suggestion_meta else '建议状态：待生成'}\n\n"
        f"依据：{source_text or '未引用会议资料'}\n\n"
        f"已加载：{material_text or '无'}"
        + (f"\n\n资料提示：{warning_text}" if warning_text else "")
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


def _build_audio_processing_page(
    gr: Any,
    *,
    fixture_enabled: bool,
    fixture_names: list[str],
    default_fixture: str,
) -> tuple[Any, list[Any]]:
    gr.Markdown(
        "## 后音频降噪增强处理\n"
        "上传与设置 → 运行状态 → A/B 听感 → 双路转写 → 可视化 → 工程证据。",
        elem_classes=["ar-hero", "ar-audio-hero"],
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
    return fixture, outputs


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
        font=[
            "Avenir Next",
            "DIN Alternate",
            "SF Pro Display",
            "PingFang SC",
            "system-ui",
            "sans-serif",
        ],
        font_mono=["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
    )

    with gr.Blocks(
        css=_read_css(),
        title="听清又听懂·智能音频急救台",
        theme=theme,
        analytics_enabled=False,
        js=APP_JS,
        elem_classes=["ar-app", "ar-flow-app"],
    ) as demo:
        page_names = ("intro", "features", "audio", "meeting")

        def show_page(page_name: str):
            return tuple(
                gr.update(visible=name == page_name)
                for name in page_names
            )

        def show_intro():
            return show_page("intro")

        def show_features():
            return show_page("features")

        def show_audio():
            return show_page("audio")

        def show_meeting():
            return show_page("meeting")

        with gr.Column(
            visible=True,
            elem_classes=["ar-flow-page", "ar-poster-page"],
        ) as intro_page:
            with gr.Column(elem_classes=["ar-flow-inner", "ar-poster-inner"]):
                gr.Markdown(
                    "# AudioRescue\n"
                    "**智能音频急救台**\n\n"
                    "把嘈杂录音变成可听、可读、可交付的语音证据。\n\n"
                    "当前分支：语音文件后的音频降噪增强处理 / 实时会议输入。",
                    elem_classes=["ar-poster-copy"],
                )
                enter_button = gr.Button(
                    "进入",
                    variant="primary",
                    size="lg",
                    elem_classes=["ar-entry-button"],
                )

        with gr.Column(
            visible=False,
            elem_classes=["ar-flow-page", "ar-feature-page"],
        ) as feature_page:
            with gr.Column(elem_classes=["ar-flow-inner", "ar-feature-inner"]):
                back_home = gr.Button(
                    "返回首页",
                    variant="secondary",
                    size="sm",
                    elem_classes=["ar-back-button"],
                )
                gr.Markdown(
                    "## 选择工作流\n"
                    "请选择本次要进入的 AudioRescue 功能。",
                    elem_classes=["ar-page-heading"],
                )
                with gr.Row(equal_height=True, elem_classes=["ar-feature-grid"]):
                    with gr.Column(elem_classes=["ar-feature-card", "ar-feature-card-audio"]):
                        gr.Markdown(
                            "### 后音频降噪增强处理\n"
                            "面向已录制语音文件，保留当前上传、增强、A/B 播放、转写和证据导出流程。",
                            elem_classes=["ar-feature-copy"],
                        )
                        audio_feature_button = gr.Button(
                            "后音频降噪增强处理",
                            variant="primary",
                            elem_classes=["ar-feature-action"],
                        )
                    with gr.Column(elem_classes=["ar-feature-card", "ar-feature-card-meeting"]):
                        gr.Markdown(
                            "### 实时会议输入\n"
                            "实时降噪与小声增强、语音转写、会前资料检索和大模型提词已接入。",
                            elem_classes=["ar-feature-copy"],
                        )
                        meeting_feature_button = gr.Button(
                            "实时会议输入",
                            variant="secondary",
                            elem_classes=["ar-feature-action", "ar-feature-action-muted"],
                        )

        with gr.Column(
            visible=False,
            elem_classes=["ar-flow-page", "ar-audio-page"],
        ) as audio_page:
            with gr.Column(elem_classes=["ar-flow-inner", "ar-audio-workspace"]):
                back_from_audio = gr.Button(
                    "返回功能选择",
                    variant="secondary",
                    size="sm",
                    elem_classes=["ar-back-button"],
                )
                fixture, outputs = _build_audio_processing_page(
                    gr,
                    fixture_enabled=fixture_enabled,
                    fixture_names=fixture_names,
                    default_fixture=default_fixture,
                )

        with gr.Column(
            visible=False,
            elem_classes=["ar-flow-page", "ar-meeting-page"],
        ) as meeting_page:
            with gr.Column(elem_classes=["ar-flow-inner", "ar-meeting-inner"]):
                back_from_meeting = gr.Button(
                    "返回功能选择",
                    variant="secondary",
                    size="sm",
                    elem_classes=["ar-back-button"],
                )
                gr.Markdown(
                    "## 实时会议输入\n"
                    "真实麦克风、虚拟麦、实时字幕和会议建议的统一控制台。",
                    elem_classes=["ar-page-heading", "ar-meeting-heading"],
                )
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
                            value=VIRTUAL_OUTPUT_CHOICE,
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
                        refresh_devices = gr.Button(
                            "刷新设备",
                            variant="secondary",
                            elem_classes=["ar-meeting-action", "ar-meeting-action-muted"],
                        )
                        with gr.Row(elem_classes=["ar-meeting-button-row"]):
                            start_live_audio = gr.Button(
                                "启动实时音频",
                                variant="primary",
                                elem_classes=["ar-meeting-action"],
                            )
                            stop_live_audio = gr.Button(
                                "停止音频",
                                variant="secondary",
                                elem_classes=["ar-meeting-action", "ar-meeting-action-muted"],
                            )
                        apply_live_mode = gr.Button(
                            "应用实时模式",
                            variant="secondary",
                            elem_classes=["ar-meeting-action", "ar-meeting-action-muted"],
                        )
                        live_meter = gr.Markdown(
                            _live_ui_values()[3],
                            elem_classes=["ar-meeting-meter"],
                        )

                    with gr.Column(scale=6, elem_classes=["ar-meeting-panel", "ar-meeting-transcript-panel"]):
                        gr.Markdown("### 会前预设与参考资料", elem_classes=["ar-meeting-panel-title"])
                        meeting_title = gr.Textbox(
                            label="会议名称",
                            placeholder="例如：AudioRescue 项目答辩",
                            elem_classes=["ar-meeting-input"],
                        )
                        meeting_scenario = gr.Dropdown(
                            choices=MEETING_SCENARIO_CHOICES,
                            value="general",
                            label="使用场景",
                            interactive=True,
                            elem_classes=["ar-meeting-input"],
                        )
                        with gr.Row(elem_classes=["ar-meeting-preset-row"]):
                            meeting_role = gr.Textbox(
                                label="我的身份",
                                placeholder="例如：学生答辩人",
                                elem_classes=["ar-meeting-input"],
                            )
                            meeting_audience = gr.Textbox(
                                label="参会者 / 听众",
                                placeholder="例如：导师和评委",
                                elem_classes=["ar-meeting-input"],
                            )
                        meeting_objective = gr.Textbox(
                            label="这次会议的目标",
                            placeholder="说清希望模型帮你完成什么。",
                            lines=2,
                            elem_classes=["ar-meeting-input"],
                        )
                        meeting_agenda = gr.Textbox(
                            label="议程 / 汇报顺序（每行一项）",
                            placeholder="背景\n方案\n实验结果\n总结",
                            lines=4,
                            elem_classes=["ar-meeting-input"],
                        )
                        with gr.Accordion("更详细的提示设置", open=False):
                            meeting_focus = gr.Textbox(
                                label="需要重点关注的内容（每行一项）",
                                placeholder="例如：实验数字必须以资料为准",
                                lines=2,
                                elem_classes=["ar-meeting-input"],
                            )
                            meeting_constraints = gr.Textbox(
                                label="禁止项 / 边界（每行一项）",
                                placeholder="例如：不夸大效果，不编造数字",
                                lines=2,
                                elem_classes=["ar-meeting-input"],
                            )
                            meeting_preset = gr.Textbox(
                                label="其他个性化需求",
                                placeholder="例如：回答简洁，完成一段后再提示下一段。",
                                lines=3,
                                elem_classes=["ar-meeting-input", "ar-meeting-preset"],
                            )
                            meeting_tone = gr.Radio(
                                [("自然", "natural"), ("正式", "formal"), ("简洁专业", "concise")],
                                value="natural",
                                label="表达风格",
                                interactive=True,
                                elem_classes=["ar-meeting-mode"],
                            )
                            meeting_coach_level = gr.Radio(
                                [("保守提示", "conservative"), ("积极提示", "active"), ("只手动请求", "manual")],
                                value="conservative",
                                label="提示频率",
                                interactive=True,
                                elem_classes=["ar-meeting-mode"],
                            )
                        meeting_materials = gr.File(
                            label="会议参考资料（最多 10 份）",
                            file_count="multiple",
                            type="filepath",
                            file_types=[".pdf", ".pptx", ".docx", ".txt", ".md"],
                            elem_classes=["ar-meeting-input", "ar-meeting-materials"],
                        )
                        gr.Markdown(
                            "增强后的麦克风音频会发送至阿里云 Fun-ASR 做实时转写；"
                            "生成建议时会把会议设置、最近转写、资料形成的会前结构化概览"
                            "和命中的资料文字片段发给千问。"
                            "资料在本地解析和检索，不上传原始 PDF/PPT/DOCX 文件；"
                            "短文档可能大部分进入命中片段；扫描 PDF 和 PPT 图片暂不 OCR。",
                            elem_classes=["ar-meeting-privacy-note"],
                        )
                        with gr.Row(elem_classes=["ar-meeting-button-row"]):
                            start_meeting = gr.Button(
                                "开始新会议（同时启动音频）",
                                variant="primary",
                                elem_classes=["ar-meeting-action"],
                            )
                            stop_meeting = gr.Button(
                                "停止会议助手",
                                variant="secondary",
                                elem_classes=["ar-meeting-action", "ar-meeting-action-muted"],
                            )
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
                        request_next_line = gr.Button(
                            "立即生成下一句建议",
                            variant="primary",
                            elem_classes=["ar-meeting-action"],
                        )
                        manual_question = gr.Textbox(
                            label="对方刚刚问了什么？",
                            placeholder="当耳机中的导师声音无法被本机 ASR 听到时，在这里输入问题。",
                            lines=3,
                            elem_classes=["ar-meeting-input", "ar-meeting-question"],
                        )
                        request_answer = gr.Button(
                            "根据会议资料生成回答",
                            variant="primary",
                            elem_classes=["ar-meeting-action"],
                        )
                        open_floating = gr.Button(
                            "打开悬浮提示窗",
                            variant="secondary",
                            elem_classes=["ar-meeting-action", "ar-meeting-action-muted"],
                        )
                        floating_preview = gr.HTML(
                            _live_ui_values()[7],
                            elem_classes=["ar-meeting-float-preview"],
                        )
                        refresh_live_status = gr.Button(
                            "刷新实时状态",
                            variant="secondary",
                            elem_classes=["ar-meeting-action", "ar-meeting-action-muted"],
                        )

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
                output_value = (
                    VIRTUAL_OUTPUT_CHOICE
                    if VIRTUAL_OUTPUT_CHOICE in output_choices
                    else output_choices[0] if output_choices else DEFAULT_OUTPUT_CHOICE
                )
                return (
                    gr.update(choices=input_choices, value=input_value),
                    gr.update(choices=output_choices, value=output_value),
                    *_live_ui_values(),
                )
            except Exception as exc:
                LIVE_CONTROLLER.record_ui_error("设备刷新失败", exc)
                return (
                    gr.update(choices=[DEFAULT_INPUT_CHOICE], value=DEFAULT_INPUT_CHOICE),
                    gr.update(
                        choices=[DEFAULT_OUTPUT_CHOICE, VIRTUAL_OUTPUT_CHOICE],
                        value=DEFAULT_OUTPUT_CHOICE,
                    ),
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

        def start_meeting_from_ui(
            selected_input,
            selected_output,
            selected_mode,
            title,
            scenario,
            role,
            audience,
            objective,
            agenda,
            focus,
            constraints,
            preset,
            tone,
            coach_level,
            materials,
        ):
            return _live_ui_values(
                LIVE_CONTROLLER.start_meeting_with_audio(
                    selected_input,
                    selected_output,
                    selected_mode,
                    preset,
                    title=title,
                    scenario=scenario,
                    user_role=role,
                    audience=audience,
                    objective=objective,
                    agenda=agenda,
                    focus_points=focus,
                    constraints=constraints,
                    tone=tone,
                    coach_level=coach_level,
                    material_files=materials,
                )
            )

        def stop_meeting_from_ui():
            return _live_ui_values(LIVE_CONTROLLER.stop_meeting())

        def request_next_line_from_ui():
            return _live_ui_values(LIVE_CONTROLLER.request_next_line())

        def request_answer_from_ui(question):
            return _live_ui_values(LIVE_CONTROLLER.request_answer(question))

        def refresh_live_from_ui():
            return _live_ui_values()

        def open_floating_noop():
            return None

        pages = [intro_page, feature_page, audio_page, meeting_page]
        enter_button.click(show_features, outputs=pages, js=SCROLL_TOP_JS)
        back_home.click(show_intro, outputs=pages, js=SCROLL_TOP_JS)
        audio_feature_button.click(show_audio, outputs=pages, js=SCROLL_TOP_JS)
        meeting_feature_button.click(show_meeting, outputs=pages, js=SCROLL_TOP_JS)
        back_from_audio.click(show_features, outputs=pages, js=SCROLL_TOP_JS)
        back_from_meeting.click(show_features, outputs=pages, js=SCROLL_TOP_JS)
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
            inputs=[
                input_device,
                output_device,
                live_mode,
                meeting_title,
                meeting_scenario,
                meeting_role,
                meeting_audience,
                meeting_objective,
                meeting_agenda,
                meeting_focus,
                meeting_constraints,
                meeting_preset,
                meeting_tone,
                meeting_coach_level,
                meeting_materials,
            ],
            outputs=live_outputs,
            show_progress="full",
        )
        stop_meeting.click(stop_meeting_from_ui, outputs=live_outputs)
        request_next_line.click(request_next_line_from_ui, outputs=live_outputs)
        request_answer.click(
            request_answer_from_ui,
            inputs=[manual_question],
            outputs=live_outputs,
            show_progress="full",
        )
        refresh_live_status.click(refresh_live_from_ui, outputs=live_outputs)
        open_floating.click(open_floating_noop, js=OPEN_FLOATING_JS)
        if hasattr(gr, "Timer"):
            live_timer = gr.Timer(value=1.0, active=True)
            live_timer.tick(refresh_live_from_ui, outputs=live_outputs)

        if default_fixture and fixture_enabled:
            demo.load(_run_fixture, inputs=[fixture], outputs=outputs)

    return demo
