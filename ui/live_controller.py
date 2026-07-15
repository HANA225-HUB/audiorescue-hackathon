"""Shared runtime controller for live audio and meeting UI."""

from __future__ import annotations

import atexit
import html
import re
import threading
from dataclasses import asdict, dataclass
from typing import Any, Callable

from core.live_denoise import (
    GtcrnDenoiser,
    LiveDenoiseEngine,
    QuietVoiceLeveler,
    VoiceLeveler,
    ensure_gtcrn_model,
)
from core.meeting_assistant import LiveMeetingAssistant, redact_text


DEFAULT_INPUT_CHOICE = "默认输入设备"
DEFAULT_OUTPUT_CHOICE = "默认监听输出"
VIRTUAL_OUTPUT_CHOICE = "自动选择 BlackHole 2ch 虚拟麦"
_NO_DEVICE_CHOICE = "点击“刷新设备”后选择"
_PATH_REDACTIONS = (
    re.compile(r"(?<![\w.-])(?:/[^\s:;,]+){2,}"),
    re.compile(r"(?i)\b[a-z]:\\[^\s:;,]+(?:\\[^\s:;,]+)+"),
)


@dataclass(frozen=True)
class LiveDeviceChoice:
    label: str
    index: int | None
    name: str
    input_channels: int
    output_channels: int
    default_samplerate: int


@dataclass(frozen=True)
class LiveUiSnapshot:
    audio_running: bool
    audio_mode: str
    audio_error: str
    audio_status: str
    input_blocks: int
    enhanced_blocks: int
    output_blocks: int
    input_drops: int
    output_drops: int
    output_underruns: int
    resyncs: int
    sink_drops: int
    sink_errors: int
    inference_p95_ms: float
    inference_max_ms: float
    realtime_factor: float
    meeting_status: str
    meeting_error: str
    partial_text: str
    transcript: tuple[str, ...]
    suggestion: str
    asr_dropped_packets: int
    last_action: str
    selected_input: str
    selected_output: str


def _safe_error(exc: BaseException | str | None) -> str:
    if exc is None:
        return ""
    message = redact_text(str(exc))
    for pattern in _PATH_REDACTIONS:
        message = pattern.sub("[路径]", message)
    return " ".join(message.split())[:240]


def _looks_like_virtual_audio(name: str) -> bool:
    normalized = name.casefold()
    return any(token in normalized for token in ("blackhole", "loopback", "vb-cable"))


def _choice_label(index: int, item: dict[str, Any]) -> str:
    return (
        f"{index}: {item['name']} | in={int(item['max_input_channels'])} "
        f"out={int(item['max_output_channels'])} "
        f"rate={int(float(item['default_samplerate']))}"
    )


def _parse_device_index(label: str | None) -> int | None:
    text = str(label or "").strip()
    if not text or text in {DEFAULT_INPUT_CHOICE, DEFAULT_OUTPUT_CHOICE, _NO_DEVICE_CHOICE}:
        return None
    if text == VIRTUAL_OUTPUT_CHOICE:
        return None
    prefix = text.split(":", 1)[0]
    try:
        return int(prefix)
    except ValueError:
        return None


class LiveUiController:
    """Own exactly one live engine and one meeting assistant for all UI surfaces."""

    def __init__(
        self,
        *,
        sounddevice_loader: Callable[[], Any] | None = None,
        ensure_model: Callable[..., Any] = ensure_gtcrn_model,
        denoiser_factory: Callable[..., Any] = GtcrnDenoiser,
        engine_factory: Callable[..., Any] = LiveDenoiseEngine,
        meeting_factory: Callable[..., Any] = LiveMeetingAssistant,
    ) -> None:
        self._sounddevice_loader = sounddevice_loader or self._load_sounddevice
        self._ensure_model = ensure_model
        self._denoiser_factory = denoiser_factory
        self._engine_factory = engine_factory
        self._meeting_factory = meeting_factory
        self._lock = threading.RLock()
        self._engine = None
        self._meeting = None
        self._audio_error = ""
        self._meeting_error = ""
        self._last_action = "待机"
        self._selected_input = DEFAULT_INPUT_CHOICE
        self._selected_output = DEFAULT_OUTPUT_CHOICE

    @staticmethod
    def _load_sounddevice():
        try:
            import sounddevice as sd
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise RuntimeError("缺少 sounddevice，请先安装 requirements.txt。") from exc
        return sd

    def list_devices(self) -> tuple[list[str], list[str], str]:
        sd = self._sounddevice_loader()
        input_choices = [DEFAULT_INPUT_CHOICE]
        output_choices = [DEFAULT_OUTPUT_CHOICE, VIRTUAL_OUTPUT_CHOICE]
        try:
            devices = list(sd.query_devices())
        except Exception as exc:  # pragma: no cover - hardware-specific
            raise RuntimeError(f"无法读取音频设备：{exc}") from exc
        for index, item in enumerate(devices):
            label = _choice_label(index, item)
            if int(item.get("max_input_channels") or 0) > 0:
                input_choices.append(label)
            if int(item.get("max_output_channels") or 0) > 0:
                output_choices.append(label)
        with self._lock:
            self._last_action = f"已刷新设备：输入 {len(input_choices) - 1} / 输出 {len(output_choices) - 2}"
        return input_choices, output_choices, self._last_action

    def record_ui_error(self, prefix: str, exc: BaseException | str) -> None:
        with self._lock:
            self._last_action = f"{prefix}：{_safe_error(exc)}"

    def _find_virtual_output(self) -> int:
        sd = self._sounddevice_loader()
        for index, item in enumerate(sd.query_devices()):
            if int(item.get("max_output_channels") or 0) >= 2 and _looks_like_virtual_audio(
                str(item.get("name") or "")
            ):
                return index
        raise RuntimeError("没有找到 BlackHole/Loopback 虚拟麦克风。请先安装 BlackHole 2ch 并重启。")

    def _create_engine(self, *, meeting_sink_enabled: bool) -> Any:
        model_path = self._ensure_model()
        denoiser = self._denoiser_factory(model_path, num_threads=1)
        leveler = VoiceLeveler(
            sample_rate=denoiser.sample_rate,
            block_size=denoiser.frame_shift_in_samples,
        )
        stream_rate = 48_000
        quiet_leveler = QuietVoiceLeveler(
            sample_rate=stream_rate,
            block_size=stream_rate * denoiser.frame_shift_in_samples // denoiser.sample_rate,
        )
        return self._engine_factory(
            denoiser,
            stream_sample_rate=stream_rate,
            output_queue_blocks=6,
            output_gain=1.0,
            enhancement_processor=leveler,
            quiet_processor=quiet_leveler,
            enhanced_frame_sink=self._accept_enhanced_audio if meeting_sink_enabled else None,
            sink_queue_blocks=64 if meeting_sink_enabled else 32,
        )

    def _accept_enhanced_audio(self, samples, sample_rate: int) -> bool:
        with self._lock:
            meeting = self._meeting
        if meeting is None:
            return False
        return bool(meeting.accept_audio(samples, sample_rate))

    def start_audio(
        self,
        input_choice: str | None,
        output_choice: str | None,
        mode: str | None,
    ) -> LiveUiSnapshot:
        with self._lock:
            if self._engine is not None and getattr(self._engine, "running", False):
                self._last_action = "实时音频已经在运行，未重复启动。"
                if mode:
                    self._engine.set_mode(str(mode))
                return self.snapshot()
            self._audio_error = ""
            self._selected_input = input_choice or DEFAULT_INPUT_CHOICE
            self._selected_output = output_choice or DEFAULT_OUTPUT_CHOICE

        try:
            output_device = (
                self._find_virtual_output()
                if output_choice == VIRTUAL_OUTPUT_CHOICE
                else _parse_device_index(output_choice)
            )
            input_device = _parse_device_index(input_choice)
            engine = self._create_engine(meeting_sink_enabled=True)
            if mode:
                engine.set_mode(str(mode))
            engine.start(input_device=input_device, output_device=output_device, latency="low")
        except Exception as exc:
            with self._lock:
                self._engine = None
                self._audio_error = _safe_error(exc)
                self._last_action = f"实时音频启动失败：{self._audio_error}"
            return self.snapshot()

        with self._lock:
            self._engine = engine
            self._last_action = "实时音频已启动。"
        return self.snapshot()

    def stop_audio(self) -> LiveUiSnapshot:
        with self._lock:
            engine, self._engine = self._engine, None
        if engine is not None:
            try:
                engine.stop()
            except Exception as exc:
                with self._lock:
                    self._audio_error = _safe_error(exc)
                    self._last_action = f"实时音频停止异常：{self._audio_error}"
                return self.snapshot()
        with self._lock:
            self._last_action = "实时音频已停止。"
        return self.snapshot()

    def set_mode(self, mode: str | None) -> LiveUiSnapshot:
        with self._lock:
            engine = self._engine
        if mode not in LiveDenoiseEngine.MODES:
            with self._lock:
                self._audio_error = "不支持的实时模式。"
                self._last_action = self._audio_error
            return self.snapshot()
        if engine is None:
            with self._lock:
                self._last_action = f"已选择 {mode}，启动后生效。"
            return self.snapshot()
        try:
            engine.set_mode(str(mode))
            with self._lock:
                self._last_action = f"实时模式已切换为 {mode}。"
        except Exception as exc:
            with self._lock:
                self._audio_error = _safe_error(exc)
                self._last_action = f"模式切换失败：{self._audio_error}"
        return self.snapshot()

    def start_meeting(self, preset: str | None) -> LiveUiSnapshot:
        with self._lock:
            if self._meeting is not None and getattr(self._meeting, "running", False):
                self._last_action = "会议助手已经在运行，未重复启动。"
                return self.snapshot()
            self._meeting_error = ""
        try:
            meeting = self._meeting_factory(preset=str(preset or "").strip())
            meeting.start()
        except Exception as exc:
            error = _safe_error(exc)
            with self._lock:
                self._meeting = None
                self._meeting_error = error
                self._last_action = f"会议助手启动失败：{error}"
            return self.snapshot()
        with self._lock:
            self._meeting = meeting
            self._meeting_error = ""
            self._last_action = "会议助手已启动。"
        return self.snapshot()

    def stop_meeting(self) -> LiveUiSnapshot:
        with self._lock:
            meeting, self._meeting = self._meeting, None
        if meeting is not None:
            try:
                meeting.stop()
            except Exception as exc:
                error = _safe_error(exc)
                with self._lock:
                    self._meeting_error = error
                    self._last_action = f"会议助手停止异常：{error}"
                return self.snapshot()
        with self._lock:
            self._meeting_error = ""
            self._last_action = "会议助手已停止。"
        return self.snapshot()

    def request_next_line(self) -> LiveUiSnapshot:
        with self._lock:
            meeting = self._meeting
        if meeting is None:
            with self._lock:
                self._last_action = "会议助手未启动，无法生成建议。"
            return self.snapshot()
        ok = bool(meeting.request_next_line())
        with self._lock:
            self._last_action = "已请求下一句建议。" if ok else "会议助手尚未就绪。"
        return self.snapshot()

    def snapshot(self) -> LiveUiSnapshot:
        with self._lock:
            engine = self._engine
            meeting = self._meeting
            audio_error = self._audio_error
            stored_meeting_error = self._meeting_error
            last_action = self._last_action
            selected_input = self._selected_input
            selected_output = self._selected_output
        audio_running = False
        audio_mode = "enhanced"
        audio_status = "待机"
        stats_values = {
            "input_blocks": 0,
            "enhanced_blocks": 0,
            "output_blocks": 0,
            "input_drops": 0,
            "output_drops": 0,
            "output_underruns": 0,
            "resyncs": 0,
            "sink_drops": 0,
            "sink_errors": 0,
            "inference_p95_ms": 0.0,
            "inference_max_ms": 0.0,
            "realtime_factor": 0.0,
        }
        if engine is not None:
            try:
                engine.raise_if_failed()
                stats = engine.snapshot_stats()
                audio_running = bool(stats.running)
                audio_mode = str(stats.mode)
                audio_status = "运行中" if audio_running else "已停止"
                stats_values.update(
                    {
                        "input_blocks": int(stats.input_blocks),
                        "enhanced_blocks": int(stats.enhanced_blocks),
                        "output_blocks": int(stats.output_blocks),
                        "input_drops": int(stats.input_drops),
                        "output_drops": int(stats.output_drops),
                        "output_underruns": int(stats.output_underruns),
                        "resyncs": int(stats.resyncs),
                        "sink_drops": int(stats.sink_drops),
                        "sink_errors": int(stats.sink_errors),
                        "inference_p95_ms": float(stats.inference_p95_ms),
                        "inference_max_ms": float(stats.inference_max_ms),
                        "realtime_factor": float(stats.realtime_factor),
                    }
                )
            except Exception as exc:
                audio_error = _safe_error(exc)
                audio_status = "异常"
        meeting_status = "stopped"
        meeting_error = stored_meeting_error
        partial_text = ""
        transcript: tuple[str, ...] = ()
        suggestion = ""
        asr_dropped_packets = 0
        if meeting is not None:
            snap = meeting.snapshot()
            meeting_status = str(snap.status)
            meeting_error = _safe_error(snap.error) if snap.error else stored_meeting_error
            partial_text = snap.partial_text
            transcript = tuple(snap.transcript)
            suggestion = snap.suggestion
            asr_dropped_packets = int(snap.asr_dropped_packets)
        return LiveUiSnapshot(
            audio_running=audio_running,
            audio_mode=audio_mode,
            audio_error=audio_error,
            audio_status=audio_status,
            meeting_status=meeting_status,
            meeting_error=meeting_error,
            partial_text=partial_text,
            transcript=transcript,
            suggestion=suggestion,
            asr_dropped_packets=asr_dropped_packets,
            last_action=last_action,
            selected_input=selected_input,
            selected_output=selected_output,
            **stats_values,
        )

    def cleanup(self) -> None:
        self.stop_audio()
        self.stop_meeting()


LIVE_CONTROLLER = LiveUiController()
atexit.register(LIVE_CONTROLLER.cleanup)


def snapshot_to_dict(snapshot: LiveUiSnapshot | None = None) -> dict[str, Any]:
    snap = snapshot or LIVE_CONTROLLER.snapshot()
    return asdict(snap)


def render_floating_html(snapshot: LiveUiSnapshot | None = None) -> str:
    snap = snapshot or LIVE_CONTROLLER.snapshot()
    suggestion = html.escape(snap.suggestion or "暂无建议")
    partial = html.escape(snap.partial_text or "暂无实时字幕")
    transcript = html.escape(" / ".join(snap.transcript[-3:]) or "暂无正式字幕")
    status = html.escape(snap.meeting_status)
    return (
        "<section class='ar-floating-card'>"
        f"<header>会议助手 · {status}</header>"
        f"<h2>{suggestion}</h2>"
        f"<p><strong>实时字幕</strong>{partial}</p>"
        f"<p><strong>正式记录</strong>{transcript}</p>"
        "</section>"
    )
