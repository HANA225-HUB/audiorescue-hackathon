"""Shared runtime controller for live audio and meeting UI."""

from __future__ import annotations

import atexit
import html
import re
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from core.live_denoise import (
    GtcrnDenoiser,
    LiveDenoiseEngine,
    QuietVoiceLeveler,
    VoiceLeveler,
    ensure_gtcrn_model,
)
from core.meeting_assistant import LiveMeetingAssistant, redact_text
from core.meeting_context import (
    MeetingKnowledgeBase,
    MeetingPreset,
    MeetingScenario,
    build_knowledge_base,
)


DEFAULT_INPUT_CHOICE = "默认输入设备"
DEFAULT_OUTPUT_CHOICE = "默认监听输出"
VIRTUAL_OUTPUT_CHOICE = "自动选择 BlackHole 2ch 虚拟麦"
_NO_DEVICE_CHOICE = "点击“刷新设备”后选择"
_PATH_REDACTIONS = (
    re.compile(r"(?<![\w.-])(?:/[^/\r\n,;，；。！？!?'\"<>]+){2,}"),
    re.compile(
        r"(?i)(?<![\w.-])[a-z]:\\(?:[^\\:\r\n,;，；。！？!?'\"<>]+\\)+"
        r"[^\\:\r\n,;，；。！？!?'\"<>]+"
    ),
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
    meeting_session_id: str
    meeting_scenario: str
    meeting_material_names: tuple[str, ...]
    material_warnings: tuple[str, ...]
    suggestion_kind: str
    suggestion_sources: tuple[str, ...]
    needs_verification: bool
    confidence: float
    asr_dropped_packets: int
    last_action: str
    selected_input: str
    selected_output: str


def _safe_error(exc: BaseException | str | None) -> str:
    if exc is None:
        return ""
    message = str(exc)
    for pattern in _PATH_REDACTIONS:
        message = pattern.sub("[路径]", message)
    message = redact_text(message)
    return " ".join(message.split())[:240]


def _looks_like_virtual_audio(name: str) -> bool:
    normalized = name.casefold()
    return any(token in normalized for token in ("blackhole", "loopback", "vb-cable"))


def _looks_like_headphones(name: str) -> bool:
    normalized = name.casefold()
    return any(
        token in normalized
        for token in (
            "headphone",
            "headset",
            "airpods",
            "earbuds",
            "耳机",
            "耳麦",
            "耳塞",
        )
    )


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


def _split_meeting_items(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        raw_items = [str(item) for item in value]
    else:
        raw_items = re.split(r"[\n\r,，;；]+", str(value))
    return tuple(item.strip() for item in raw_items if item.strip())


def _normalize_material_paths(value: Any) -> tuple[Path, ...]:
    if value is None or value == "":
        return ()
    if isinstance(value, (str, Path)):
        items = (value,)
    elif isinstance(value, (list, tuple)):
        items = tuple(value)
    else:
        items = (value,)
    paths: list[Path] = []
    for item in items:
        raw: Any = item
        if isinstance(item, dict):
            raw = item.get("path") or item.get("name")
        elif not isinstance(item, (str, Path)):
            raw = getattr(item, "path", None) or getattr(item, "name", None)
        if raw:
            paths.append(Path(str(raw)))
    return tuple(paths)


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
        self._meeting_flow_lock = threading.Lock()
        self._engine = None
        self._meeting = None
        self._audio_error = ""
        self._meeting_error = ""
        self._material_warnings: tuple[str, ...] = ()
        self._last_action = "待机"
        self._selected_input = DEFAULT_INPUT_CHOICE
        self._selected_output = DEFAULT_OUTPUT_CHOICE
        self._audio_state = "idle"
        self._audio_stop_requested = False
        self._meeting_state = "idle"
        self._meeting_stop_requested = False

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

    @staticmethod
    def _default_device_index(sd, devices: list[dict[str, Any]], kind: str) -> int:
        channel_key = "max_input_channels" if kind == "input" else "max_output_channels"
        try:
            default_pair = getattr(sd, "default").device
            default_index = int(default_pair[0 if kind == "input" else 1])
            if 0 <= default_index < len(devices):
                return default_index
        except Exception:
            pass
        for index, item in enumerate(devices):
            if int(item.get(channel_key) or 0) > 0:
                return index
        raise RuntimeError("没有找到可用音频输入设备。" if kind == "input" else "没有找到可用音频输出设备。")

    def _resolve_device_info(self, sd, device: int | None, kind: str) -> tuple[int, dict[str, Any]]:
        devices = list(sd.query_devices())
        index = device
        if index is None:
            index = self._default_device_index(sd, devices, kind)
        if not 0 <= int(index) < len(devices):
            raise RuntimeError("音频设备编号无效。")
        info = dict(devices[int(index)])
        channel_key = "max_input_channels" if kind == "input" else "max_output_channels"
        if int(info.get(channel_key) or 0) <= 0:
            raise RuntimeError("所选设备不支持音频输入。" if kind == "input" else "所选设备不支持音频输出。")
        return int(index), info

    def _prepare_audio_devices(
        self,
        input_choice: str | None,
        output_choice: str | None,
    ) -> tuple[int | None, int | None]:
        input_device = _parse_device_index(input_choice)
        output_device = (
            self._find_virtual_output()
            if output_choice == VIRTUAL_OUTPUT_CHOICE
            else _parse_device_index(output_choice)
        )
        sd = self._sounddevice_loader()
        input_index, input_info = self._resolve_device_info(sd, input_device, "input")
        output_index, output_info = self._resolve_device_info(sd, output_device, "output")
        input_name = str(input_info.get("name") or "")
        output_name = str(output_info.get("name") or "")
        output_is_virtual = _looks_like_virtual_audio(output_name)

        if input_index == output_index:
            raise RuntimeError("输入和输出不能选择同一个音频设备，避免形成自循环。")
        if output_is_virtual and _looks_like_virtual_audio(input_name):
            raise RuntimeError(
                f"当前输入设备“{input_name}”是虚拟音频设备，会形成自读自写回路。"
                "请改选真实麦克风。"
            )
        if not (_looks_like_headphones(output_name) or output_is_virtual):
            raise RuntimeError(
                f"当前输出设备“{output_name}”未被识别为耳机或虚拟麦，"
                "直接监听可能产生回声或啸叫。请连接耳机或选择 BlackHole 2ch。"
            )
        return input_device, output_device

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
            if self._audio_state == "starting":
                self._last_action = "实时音频正在启动，请稍候。"
                return self.snapshot()
            if self._audio_state == "stopping":
                self._last_action = "实时音频正在停止，请稍候。"
                return self.snapshot()
            if self._engine is not None and getattr(self._engine, "running", False):
                self._last_action = "实时音频已经在运行，未重复启动。"
                if mode:
                    self._engine.set_mode(str(mode))
                return self.snapshot()
            self._audio_error = ""
            self._selected_input = input_choice or DEFAULT_INPUT_CHOICE
            self._selected_output = output_choice or DEFAULT_OUTPUT_CHOICE
            self._audio_state = "starting"
            self._audio_stop_requested = False

        engine = None
        try:
            input_device, output_device = self._prepare_audio_devices(input_choice, output_choice)
            engine = self._create_engine(meeting_sink_enabled=True)
            if mode:
                engine.set_mode(str(mode))
            engine.start(input_device=input_device, output_device=output_device, latency="low")
        except Exception as exc:
            with self._lock:
                self._engine = None
                self._audio_state = "idle"
                self._audio_stop_requested = False
                self._audio_error = _safe_error(exc)
                self._last_action = f"实时音频启动失败：{self._audio_error}"
            return self.snapshot()

        with self._lock:
            if self._audio_stop_requested:
                self._audio_state = "stopping"
                stop_after_start = True
            else:
                stop_after_start = False
                self._engine = engine
                self._audio_state = "running"
                self._last_action = "实时音频已启动。"

        if stop_after_start:
            try:
                engine.stop()
            except Exception as exc:
                with self._lock:
                    self._engine = engine
                    self._audio_state = "running"
                    self._audio_stop_requested = False
                    self._audio_error = _safe_error(exc)
                    self._last_action = f"启动后停止异常：{self._audio_error}"
                return self.snapshot()
            with self._lock:
                if self._engine is engine:
                    self._engine = None
                self._audio_state = "idle"
                self._audio_stop_requested = False
                self._last_action = "启动期间收到停止请求，实时音频已停止。"
            return self.snapshot()

        return self.snapshot()

    def stop_audio(self) -> LiveUiSnapshot:
        with self._lock:
            if self._audio_state == "starting":
                self._audio_stop_requested = True
                self._last_action = "实时音频正在启动，已排队停止。"
                return self.snapshot()
            if self._audio_state == "stopping":
                self._last_action = "实时音频正在停止，请稍候。"
                return self.snapshot()
            engine = self._engine
            if engine is None:
                self._audio_state = "idle"
                self._audio_error = ""
                self._last_action = "实时音频已停止。"
                return self.snapshot()
            self._audio_state = "stopping"
        if engine is not None:
            try:
                engine.stop()
            except Exception as exc:
                with self._lock:
                    self._engine = engine
                    self._audio_state = "running"
                    self._audio_error = _safe_error(exc)
                    self._last_action = f"实时音频停止异常：{self._audio_error}"
                return self.snapshot()
        with self._lock:
            if self._engine is engine:
                self._engine = None
            self._audio_state = "idle"
            self._audio_stop_requested = False
            self._audio_error = ""
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

    def start_meeting(
        self,
        preset: str | None = None,
        *,
        title: str | None = None,
        scenario: MeetingScenario | str = MeetingScenario.GENERAL,
        user_role: str | None = None,
        audience: str | None = None,
        objective: str | None = None,
        agenda: Any = None,
        focus_points: Any = None,
        constraints: Any = None,
        tone: str = "natural",
        coach_level: str = "conservative",
        material_files: Any = None,
    ) -> LiveUiSnapshot:
        with self._lock:
            if self._meeting_state == "starting":
                self._last_action = "会议助手正在启动，请稍候。"
                return self.snapshot()
            if self._meeting_state == "stopping":
                self._last_action = "会议助手正在停止，请稍候。"
                return self.snapshot()
            if self._meeting is not None and getattr(self._meeting, "running", False):
                self._last_action = "会议助手已在运行；请先结束当前会议，再开始新会议。"
                return self.snapshot()
            self._meeting_error = ""
            self._material_warnings = ()
            self._meeting_state = "starting"
            self._meeting_stop_requested = False
        meeting = None
        try:
            scenario_value = (
                scenario.value if isinstance(scenario, MeetingScenario) else str(scenario or "")
            )
            legacy_only = not (
                title is not None
                or material_files is not None
                or any(
                    str(value or "").strip()
                    for value in (
                        user_role,
                        audience,
                        objective,
                        agenda,
                        focus_points,
                        constraints,
                    )
                )
                or scenario_value.strip().casefold() not in {"", "general"}
                or str(tone or "natural").strip().casefold() != "natural"
                or str(coach_level or "conservative").strip().casefold()
                != "conservative"
            )
            if legacy_only:
                meeting = self._meeting_factory(preset=str(preset or "").strip())
                material_count = 0
            else:
                config = MeetingPreset(
                    title=str(title or "").strip() or "新会议",
                    scenario=scenario or MeetingScenario.GENERAL,
                    user_role=str(user_role or "").strip(),
                    audience=str(audience or "").strip(),
                    objective=str(objective or "").strip(),
                    agenda=_split_meeting_items(agenda),
                    focus_points=_split_meeting_items(focus_points),
                    constraints=_split_meeting_items(constraints),
                    custom_requirements=str(preset or "").strip(),
                    tone=str(tone or "natural").strip(),
                    coach_level=str(coach_level or "conservative").strip(),
                )
                paths = _normalize_material_paths(material_files)
                knowledge_base = (
                    build_knowledge_base(paths) if paths else MeetingKnowledgeBase()
                )
                records = knowledge_base.list_documents()
                warnings = tuple(
                    warning
                    for record in records
                    for warning in record.warnings
                    if warning
                )
                with self._lock:
                    self._material_warnings = warnings
                material_count = len(records)
                meeting = self._meeting_factory(
                    preset=str(preset or "").strip(),
                    session_config=config,
                    knowledge_base=knowledge_base,
                )
            meeting.start()
        except Exception as exc:
            error = _safe_error(exc)
            with self._lock:
                self._meeting = None
                self._meeting_state = "idle"
                self._meeting_stop_requested = False
                self._meeting_error = error
                self._material_warnings = ()
                self._last_action = f"会议助手启动失败：{error}"
            return self.snapshot()
        with self._lock:
            if self._meeting_stop_requested:
                self._meeting_state = "stopping"
                stop_after_start = True
            else:
                stop_after_start = False
                self._meeting = meeting
                self._meeting_state = "running"
                self._meeting_error = ""
                loaded = (
                    f"，已加载 {material_count} 份会议资料"
                    if material_count
                    else ""
                )
                audio_ready = self._engine is not None and bool(
                    getattr(self._engine, "running", False)
                )
                audio_note = "" if audio_ready else "；请同时启动实时音频"
                self._last_action = f"会议助手已启动{loaded}{audio_note}。"

        if stop_after_start:
            try:
                meeting.stop()
            except Exception as exc:
                error = _safe_error(exc)
                with self._lock:
                    self._meeting = meeting
                    self._meeting_state = "running"
                    self._meeting_stop_requested = False
                    self._meeting_error = error
                    self._last_action = f"会议助手启动后停止异常：{error}"
                return self.snapshot()
            with self._lock:
                self._meeting = None
                self._meeting_state = "idle"
                self._meeting_stop_requested = False
                self._meeting_error = ""
                self._material_warnings = ()
                self._last_action = "启动期间收到停止请求，会议助手已停止。"
        return self.snapshot()

    def start_meeting_with_audio(
        self,
        input_choice: str | None,
        output_choice: str | None,
        mode: str | None,
        preset: str | None = None,
        **meeting_kwargs: Any,
    ) -> LiveUiSnapshot:
        """Replace the meeting session and keep audio/assistant state consistent."""

        with self._meeting_flow_lock:
            with self._lock:
                audio_was_running = self._engine is not None and bool(
                    getattr(self._engine, "running", False)
                )
                meeting_was_running = self._meeting is not None and bool(
                    getattr(self._meeting, "running", False)
                )

            if meeting_was_running:
                self.stop_meeting()
                with self._lock:
                    if self._meeting is not None and bool(
                        getattr(self._meeting, "running", False)
                    ):
                        return self.snapshot()

            audio_snapshot = self.start_audio(input_choice, output_choice, mode)
            if not audio_snapshot.audio_running:
                return audio_snapshot

            meeting_snapshot = self.start_meeting(preset, **meeting_kwargs)
            with self._lock:
                meeting_started = self._meeting is not None and bool(
                    getattr(self._meeting, "running", False)
                )

            if not meeting_started:
                failure_action = meeting_snapshot.last_action
                failure_error = meeting_snapshot.meeting_error
                if not audio_was_running:
                    self.stop_audio()
                    with self._lock:
                        self._meeting_error = failure_error
                        self._last_action = (
                            f"{failure_action}；本次新启动的实时音频已停止。"
                        )
                return self.snapshot()

            final_snapshot = self.snapshot()
            if final_snapshot.audio_running:
                return final_snapshot

            self.stop_meeting()
            with self._lock:
                self._meeting_error = "实时音频在会议启动期间已停止。"
                self._last_action = "实时音频已停止，会议助手未保持运行。"
            return self.snapshot()

    def stop_meeting(self) -> LiveUiSnapshot:
        with self._lock:
            if self._meeting_state == "starting":
                self._meeting_stop_requested = True
                self._last_action = "会议助手正在启动，已排队停止。"
                return self.snapshot()
            if self._meeting_state == "stopping":
                self._last_action = "会议助手正在停止，请稍候。"
                return self.snapshot()
            meeting = self._meeting
            if meeting is None:
                self._meeting_state = "idle"
                self._meeting_error = ""
                self._material_warnings = ()
                self._last_action = "会议助手已停止。"
                return self.snapshot()
            self._meeting_state = "stopping"
        if meeting is not None:
            try:
                meeting.stop()
            except Exception as exc:
                error = _safe_error(exc)
                with self._lock:
                    self._meeting = meeting
                    self._meeting_state = "running"
                    self._meeting_error = error
                    self._last_action = f"会议助手停止异常：{error}"
                return self.snapshot()
        with self._lock:
            if self._meeting is meeting:
                self._meeting = None
            self._meeting_state = "idle"
            self._meeting_stop_requested = False
            self._meeting_error = ""
            self._material_warnings = ()
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

    def request_answer(self, question: str | None) -> LiveUiSnapshot:
        clean_question = str(question or "").strip()
        if not clean_question:
            with self._lock:
                self._last_action = "请先输入对方的问题。"
            return self.snapshot()
        with self._lock:
            meeting = self._meeting
        if meeting is None:
            with self._lock:
                self._last_action = "会议助手未启动，无法生成回答。"
            return self.snapshot()
        try:
            ok = bool(meeting.request_answer(clean_question))
        except Exception as exc:
            with self._lock:
                self._meeting_error = _safe_error(exc)
                self._last_action = f"问题提交失败：{self._meeting_error}"
            return self.snapshot()
        with self._lock:
            self._last_action = "已提交对方问题，正在生成回答。" if ok else "会议助手尚未就绪。"
        return self.snapshot()

    def snapshot(self) -> LiveUiSnapshot:
        with self._lock:
            engine = self._engine
            meeting = self._meeting
            audio_error = self._audio_error
            stored_meeting_error = self._meeting_error
            material_warnings = self._material_warnings
            last_action = self._last_action
            selected_input = self._selected_input
            selected_output = self._selected_output
            audio_state = self._audio_state
            meeting_state = self._meeting_state
        audio_running = False
        audio_mode = "enhanced"
        audio_status = {
            "starting": "启动中",
            "stopping": "停止中",
        }.get(audio_state, "待机")
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
                if audio_state not in {"starting", "stopping"}:
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
        meeting_status = {
            "starting": "starting",
            "stopping": "stopping",
        }.get(meeting_state, "stopped")
        meeting_error = stored_meeting_error
        partial_text = ""
        transcript: tuple[str, ...] = ()
        suggestion = ""
        meeting_session_id = ""
        meeting_scenario = "general"
        meeting_material_names: tuple[str, ...] = ()
        suggestion_kind = ""
        suggestion_sources: tuple[str, ...] = ()
        needs_verification = False
        confidence = 0.0
        asr_dropped_packets = 0
        if meeting is not None:
            snap = meeting.snapshot()
            if meeting_state not in {"starting", "stopping"}:
                meeting_status = str(snap.status)
            meeting_error = _safe_error(snap.error) if snap.error else stored_meeting_error
            partial_text = snap.partial_text
            transcript = tuple(snap.transcript)
            suggestion = snap.suggestion
            meeting_session_id = str(snap.session_id)
            meeting_scenario = str(snap.scenario)
            meeting_material_names = tuple(snap.material_names)
            suggestion_kind = str(snap.suggestion_kind)
            suggestion_sources = tuple(snap.suggestion_sources)
            needs_verification = bool(snap.needs_verification)
            confidence = float(snap.confidence)
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
            meeting_session_id=meeting_session_id,
            meeting_scenario=meeting_scenario,
            meeting_material_names=meeting_material_names,
            material_warnings=material_warnings,
            suggestion_kind=suggestion_kind,
            suggestion_sources=suggestion_sources,
            needs_verification=needs_verification,
            confidence=confidence,
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
    verification = "需要核实" if snap.needs_verification else "已按当前上下文生成"
    sources = html.escape(" / ".join(snap.suggestion_sources) or "未引用会议资料")
    kind = html.escape(snap.suggestion_kind or "待生成")
    confidence = f"{snap.confidence:.0%}" if snap.confidence > 0 else "--"
    return (
        "<section class='ar-floating-card'>"
        f"<header>会议助手 · {status}</header>"
        f"<h2>{suggestion}</h2>"
        f"<p><strong>{kind}</strong> · 置信度 {confidence}</p>"
        f"<p><strong>{verification}</strong> · {sources}</p>"
        f"<p><strong>实时字幕</strong>{partial}</p>"
        f"<p><strong>正式记录</strong>{transcript}</p>"
        "</section>"
    )
