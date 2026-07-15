"""Realtime Fun-ASR adapter for the enhanced 16 kHz microphone branch."""

from __future__ import annotations

import json
import os
import queue
import threading
import uuid
from dataclasses import dataclass
from typing import Callable

import numpy as np


DEFAULT_DASHSCOPE_WEBSOCKET_URL = (
    "wss://dashscope.aliyuncs.com/api-ws/v1/inference"
)
_AUDIO_SENTINEL = object()


@dataclass(frozen=True)
class RealtimeAsrEvent:
    """One partial or final transcript emitted by Fun-ASR."""

    text: str
    is_final: bool
    sentence_id: int | None = None
    begin_time_ms: int | None = None
    end_time_ms: int | None = None


def float32_to_pcm16(samples: np.ndarray) -> bytes:
    """Convert normalized mono float32 samples to little-endian PCM16 bytes."""

    block = np.asarray(samples, dtype=np.float32).reshape(-1)
    if not np.isfinite(block).all():
        raise ValueError("ASR 输入包含非有限采样值。")
    clipped = np.clip(block, -1.0, 1.0)
    return np.rint(clipped * 32767.0).astype("<i2", copy=False).tobytes()


def build_run_task(
    task_id: str,
    *,
    model: str = "fun-asr-realtime",
    sample_rate: int = 16_000,
    language_hint: str | None = "zh",
    final_silence_ms: int = 600,
) -> dict:
    """Build the documented Fun-ASR duplex run-task event."""

    if not 200 <= int(final_silence_ms) <= 6000:
        raise ValueError("final_silence_ms 必须在 200 到 6000 之间。")
    parameters: dict[str, object] = {
        "format": "pcm",
        "sample_rate": int(sample_rate),
        "semantic_punctuation_enabled": False,
        "max_sentence_silence": int(final_silence_ms),
        "heartbeat": True,
    }
    if language_hint:
        parameters["language_hints"] = [str(language_hint)]
    return {
        "header": {
            "action": "run-task",
            "task_id": task_id,
            "streaming": "duplex",
        },
        "payload": {
            "task_group": "audio",
            "task": "asr",
            "function": "recognition",
            "model": model,
            "parameters": parameters,
            "input": {},
        },
    }


def build_finish_task(task_id: str) -> dict:
    return {
        "header": {
            "action": "finish-task",
            "task_id": task_id,
            "streaming": "duplex",
        },
        "payload": {"input": {}},
    }


class DashscopeRealtimeAsr:
    """Threaded WebSocket client that never performs network I/O in audio code."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        websocket_url: str = DEFAULT_DASHSCOPE_WEBSOCKET_URL,
        model: str = "fun-asr-realtime",
        sample_rate: int = 16_000,
        language_hint: str | None = "zh",
        final_silence_ms: int = 600,
        packet_ms: int = 100,
        queue_packets: int = 30,
        on_event: Callable[[RealtimeAsrEvent], None] | None = None,
        on_status: Callable[[str], None] | None = None,
    ) -> None:
        if sample_rate <= 0 or packet_ms <= 0 or queue_packets <= 0:
            raise ValueError("ASR 采样率、分包时长和队列大小必须为正数。")
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        self.websocket_url = websocket_url
        self.model = model
        self.sample_rate = int(sample_rate)
        self.language_hint = language_hint
        self.final_silence_ms = int(final_silence_ms)
        self.packet_bytes = self.sample_rate * 2 * int(packet_ms) // 1000
        self.on_event = on_event
        self.on_status = on_status

        self._audio_queue: queue.Queue[object] = queue.Queue(queue_packets)
        self._ready = threading.Event()
        self._done = threading.Event()
        self._stop = threading.Event()
        self._ws_thread: threading.Thread | None = None
        self._sender_thread: threading.Thread | None = None
        self._ws = None
        self._websocket = None
        self._task_id = ""
        self._error: RuntimeError | None = None
        self._running = False
        self._seen_finals: set[tuple[object, ...]] = set()
        self.dropped_audio_packets = 0

    @property
    def running(self) -> bool:
        return self._running

    @property
    def error(self) -> RuntimeError | None:
        return self._error

    def _emit_status(self, status: str) -> None:
        if self.on_status is not None:
            try:
                self.on_status(status)
            except Exception:
                pass

    def _emit_event(self, event: RealtimeAsrEvent) -> None:
        if self.on_event is not None:
            try:
                self.on_event(event)
            except Exception:
                pass

    def _send_json(self, payload: dict) -> None:
        if self._ws is None:
            raise RuntimeError("Fun-ASR WebSocket 尚未建立。")
        self._ws.send(json.dumps(payload, ensure_ascii=False))

    def _on_open(self, ws) -> None:
        self._ws = ws
        self._send_json(
            build_run_task(
                self._task_id,
                model=self.model,
                sample_rate=self.sample_rate,
                language_hint=self.language_hint,
                final_silence_ms=self.final_silence_ms,
            )
        )

    def _on_message(self, ws, raw_message: str) -> None:
        try:
            message = json.loads(raw_message)
            header = message.get("header") or {}
            event_name = header.get("event")
            if event_name == "task-started":
                self._ready.set()
                self._emit_status("listening")
                return
            if event_name == "result-generated":
                sentence = (
                    ((message.get("payload") or {}).get("output") or {}).get(
                        "sentence"
                    )
                    or {}
                )
                if sentence.get("heartbeat"):
                    return
                text = str(sentence.get("text") or "").strip()
                if not text:
                    return
                is_final = bool(sentence.get("sentence_end"))
                sentence_id = sentence.get("sentence_id")
                if is_final:
                    dedupe_key = (
                        sentence_id,
                        sentence.get("begin_time"),
                        sentence.get("end_time"),
                        text,
                    )
                    if dedupe_key in self._seen_finals:
                        return
                    self._seen_finals.add(dedupe_key)
                self._emit_event(
                    RealtimeAsrEvent(
                        text=text,
                        is_final=is_final,
                        sentence_id=sentence_id,
                        begin_time_ms=sentence.get("begin_time"),
                        end_time_ms=sentence.get("end_time"),
                    )
                )
                return
            if event_name == "task-finished":
                self._done.set()
                self._emit_status("finished")
                ws.close()
                return
            if event_name == "task-failed":
                detail = str(header.get("error_message") or "Fun-ASR 任务失败。")
                self._error = RuntimeError(detail)
                self._ready.set()
                self._done.set()
                self._emit_status("error")
                ws.close()
        except Exception as exc:
            self._error = RuntimeError("无法解析 Fun-ASR 服务响应。")
            self._error.__cause__ = exc
            self._ready.set()
            self._done.set()
            self._emit_status("error")
            ws.close()

    def _on_error(self, ws, error) -> None:
        del ws
        if self._error is None and not self._stop.is_set():
            self._error = RuntimeError(f"Fun-ASR 连接失败：{error}")
        self._ready.set()
        self._done.set()
        self._emit_status("error")

    def _on_close(self, ws, status_code, message) -> None:
        del ws, status_code, message
        if not self._done.is_set() and not self._stop.is_set():
            self._error = self._error or RuntimeError("Fun-ASR 连接意外关闭。")
            self._emit_status("error")
        self._ready.set()
        self._done.set()

    def _run_websocket(self) -> None:
        assert self._ws is not None
        self._ws.run_forever(ping_interval=20, ping_timeout=10)

    def _cleanup_transport(self, *, join_timeout: float = 2.0) -> None:
        """Close a partially started connection without leaving daemon work behind."""

        self._stop.set()
        if self._ws is not None:
            self._ws.close()
        if self._ws_thread is not None:
            self._ws_thread.join(timeout=join_timeout)
            if not self._ws_thread.is_alive():
                self._ws_thread = None
        self._ws = None
        self._running = False

    def _send_binary(self, payload: bytes) -> None:
        if not payload or self._ws is None or self._done.is_set():
            return
        self._ws.send(payload, opcode=self._websocket.ABNF.OPCODE_BINARY)

    def _run_sender(self) -> None:
        pending = bytearray()
        try:
            while not self._stop.is_set() or not self._audio_queue.empty():
                try:
                    item = self._audio_queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                if item is _AUDIO_SENTINEL:
                    break
                pending.extend(item)
                while len(pending) >= self.packet_bytes:
                    self._send_binary(bytes(pending[: self.packet_bytes]))
                    del pending[: self.packet_bytes]
            if pending:
                self._send_binary(bytes(pending))
            if self._ready.is_set() and not self._done.is_set():
                self._send_json(build_finish_task(self._task_id))
        except Exception as exc:
            self._error = RuntimeError("发送 Fun-ASR 音频失败。")
            self._error.__cause__ = exc
            self._done.set()
            self._emit_status("error")
            if self._ws is not None:
                self._ws.close()

    def start(self, *, timeout: float = 12.0) -> None:
        if self._running:
            raise RuntimeError("Fun-ASR 已经在运行。")
        if (
            self._ws_thread is not None
            and self._ws_thread.is_alive()
            or self._sender_thread is not None
            and self._sender_thread.is_alive()
        ):
            raise RuntimeError("Fun-ASR 上一次连接仍在收尾。")
        if not self.api_key:
            raise RuntimeError("未找到 DASHSCOPE_API_KEY。")
        try:
            import websocket
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise RuntimeError(
                "实时转写需要 websocket-client；请安装 requirements.txt。"
            ) from exc

        self._websocket = websocket
        self._ready.clear()
        self._done.clear()
        self._stop.clear()
        self._error = None
        self._seen_finals.clear()
        self._task_id = str(uuid.uuid4())
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break
        self._ws = websocket.WebSocketApp(
            self.websocket_url,
            header=[f"Authorization: Bearer {self.api_key}"],
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        self._emit_status("connecting")
        self._ws_thread = threading.Thread(
            target=self._run_websocket, name="audiorescue-fun-asr", daemon=True
        )
        self._ws_thread.start()
        if not self._ready.wait(timeout):
            self._cleanup_transport()
            raise TimeoutError("连接 Fun-ASR 超时。")
        if self._error is not None:
            startup_error = self._error
            self._cleanup_transport()
            raise RuntimeError("Fun-ASR 启动失败。") from startup_error
        self._sender_thread = threading.Thread(
            target=self._run_sender, name="audiorescue-asr-sender", daemon=True
        )
        self._sender_thread.start()
        self._running = True

    def accept_audio(self, samples: np.ndarray, sample_rate: int) -> bool:
        """Queue one enhanced frame without waiting for the network."""

        if int(sample_rate) != self.sample_rate:
            raise ValueError(
                f"Fun-ASR 需要 {self.sample_rate} Hz，收到 {sample_rate} Hz。"
            )
        if not self._running or self._error is not None:
            return False
        payload = float32_to_pcm16(samples)
        try:
            self._audio_queue.put_nowait(payload)
            return True
        except queue.Full:
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                pass
            self.dropped_audio_packets += 1
            try:
                self._audio_queue.put_nowait(payload)
            except queue.Full:
                self.dropped_audio_packets += 1
                return False
            return True

    def stop(self, *, timeout: float = 8.0) -> None:
        if self._ws is None and self._ws_thread is None:
            self._running = False
            return
        self._running = False
        self._stop.set()
        if self._sender_thread is not None:
            try:
                self._audio_queue.put_nowait(_AUDIO_SENTINEL)
            except queue.Full:
                pass
            self._sender_thread.join(timeout=timeout)
            if not self._sender_thread.is_alive():
                self._sender_thread = None
        self._done.wait(timeout)
        self._cleanup_transport()
