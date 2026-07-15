"""Low-latency microphone denoising for local headphone monitoring.

The audio callback intentionally performs only bounded, non-blocking queue I/O.
GTCRN inference and optional downstream consumers run on worker threads.
"""

from __future__ import annotations

import hashlib
import math
import os
import queue
import shutil
import threading
import time
import urllib.request
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import numpy as np


GTCRN_MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speech-enhancement-models/gtcrn_simple.onnx"
)
GTCRN_MODEL_SHA256 = (
    "e77603ac0c23dac3227dd2d7135b3a585cbee2679048aecfa886657d3ae1b534"
)
DEFAULT_GTCRN_MODEL_PATH = (
    Path.home() / ".cache" / "audiorescue" / "models" / "gtcrn" / "gtcrn_simple.onnx"
)


class StreamingDenoiser(Protocol):
    sample_rate: int
    frame_shift_in_samples: int

    def process(
        self, samples: np.ndarray, input_sample_rate: int | None = None
    ) -> np.ndarray: ...

    def flush(self) -> np.ndarray: ...

    def reset(self) -> None: ...


class FrameProcessor(Protocol):
    def process(self, samples: np.ndarray) -> np.ndarray: ...

    def reset(self) -> None: ...


EnhancedFrameSink = Callable[[np.ndarray, int], None]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_gtcrn_model(
    model_path: str | Path = DEFAULT_GTCRN_MODEL_PATH,
    *,
    download: bool = True,
    url: str = GTCRN_MODEL_URL,
) -> Path:
    """Return a checksum-verified GTCRN model, downloading it when requested."""

    path = Path(model_path).expanduser().resolve()
    if path.is_file():
        actual = _sha256_file(path)
        if actual != GTCRN_MODEL_SHA256:
            raise RuntimeError(
                "GTCRN 模型校验失败；请删除损坏文件后重新下载。"
                f" expected={GTCRN_MODEL_SHA256[:12]} actual={actual[:12]}"
            )
        return path

    if not download:
        raise FileNotFoundError(f"GTCRN 模型不存在：{path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.download")
    request = urllib.request.Request(url, headers={"User-Agent": "AudioRescue/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
            with temporary.open("wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
        actual = _sha256_file(temporary)
        if actual != GTCRN_MODEL_SHA256:
            raise RuntimeError(
                "下载的 GTCRN 模型校验失败。"
                f" expected={GTCRN_MODEL_SHA256[:12]} actual={actual[:12]}"
            )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


class GtcrnDenoiser:
    """Small wrapper around sherpa-onnx's stateful online GTCRN stream."""

    def __init__(self, model_path: str | Path, *, num_threads: int = 1) -> None:
        try:
            import sherpa_onnx
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise RuntimeError(
                "实时降噪需要 sherpa-onnx；请先安装 requirements.txt。"
            ) from exc

        path = Path(model_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"GTCRN 模型不存在：{path}")

        config = sherpa_onnx.OnlineSpeechDenoiserConfig(
            model=sherpa_onnx.OfflineSpeechDenoiserModelConfig(
                gtcrn=sherpa_onnx.OfflineSpeechDenoiserGtcrnModelConfig(
                    model=str(path)
                ),
                num_threads=int(num_threads),
                debug=False,
                provider="cpu",
            )
        )
        if not config.validate():
            raise RuntimeError("GTCRN 配置无效。")

        self._backend = sherpa_onnx.OnlineSpeechDenoiser(config)
        self.sample_rate = int(self._backend.sample_rate)
        self.frame_shift_in_samples = int(self._backend.frame_shift_in_samples)

    def process(
        self, samples: np.ndarray, input_sample_rate: int | None = None
    ) -> np.ndarray:
        block = np.ascontiguousarray(samples, dtype=np.float32).reshape(-1)
        rate = self.sample_rate if input_sample_rate is None else int(input_sample_rate)
        result = self._backend.run(block, rate)
        return np.asarray(result.samples, dtype=np.float32).reshape(-1).copy()

    def flush(self) -> np.ndarray:
        result = self._backend.flush()
        return np.asarray(result.samples, dtype=np.float32).reshape(-1).copy()

    def reset(self) -> None:
        self._backend.reset()


class VoiceLeveler:
    """Stateful automatic gain and peak limiting for quiet meeting speech."""

    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        block_size: int = 256,
        target_rms_dbfs: float = -20.0,
        gate_dbfs: float = -48.0,
        max_gain_db: float = 12.0,
        min_gain_db: float = -6.0,
        attack_ms: float = 40.0,
        release_ms: float = 350.0,
        gate_release_ms: float | None = None,
        peak_limit: float = 0.95,
    ) -> None:
        if sample_rate <= 0 or block_size <= 0:
            raise ValueError("sample_rate and block_size must be positive")
        if attack_ms <= 0 or release_ms <= 0:
            raise ValueError("attack and release must be positive")
        if gate_release_ms is not None and gate_release_ms <= 0:
            raise ValueError("gate release must be positive")
        if not 0 < peak_limit <= 1:
            raise ValueError("peak_limit must be in (0, 1]")
        self.sample_rate = int(sample_rate)
        self.block_size = int(block_size)
        self.target_rms = 10.0 ** (float(target_rms_dbfs) / 20.0)
        self.gate_rms = 10.0 ** (float(gate_dbfs) / 20.0)
        self.max_gain = 10.0 ** (float(max_gain_db) / 20.0)
        self.min_gain = 10.0 ** (float(min_gain_db) / 20.0)
        self.peak_limit = float(peak_limit)
        block_seconds = self.block_size / self.sample_rate
        self._attack_alpha = 1.0 - math.exp(-block_seconds / (attack_ms / 1000.0))
        self._release_alpha = 1.0 - math.exp(-block_seconds / (release_ms / 1000.0))
        self._gate_release_alpha = (
            self._attack_alpha
            if gate_release_ms is None
            else 1.0
            - math.exp(-block_seconds / (float(gate_release_ms) / 1000.0))
        )
        self._gain = 1.0

    @property
    def gain(self) -> float:
        return self._gain

    def reset(self) -> None:
        self._gain = 1.0

    def process(self, samples: np.ndarray) -> np.ndarray:
        block = np.ascontiguousarray(samples, dtype=np.float32).reshape(-1)
        if block.size != self.block_size:
            raise ValueError(
                f"expected {self.block_size} samples, received {block.size}"
            )
        rms = float(np.sqrt(np.mean(np.square(block, dtype=np.float64))))
        if rms < self.gate_rms:
            desired = min(self._gain, 1.0)
            alpha = self._gate_release_alpha
        else:
            desired = float(
                np.clip(
                    self.target_rms / max(rms, 1e-8),
                    self.min_gain,
                    self.max_gain,
                )
            )
            alpha = (
                self._attack_alpha
                if desired < self._gain
                else self._release_alpha
            )
        self._gain += alpha * (desired - self._gain)
        output = block * self._gain
        peak = float(np.max(np.abs(output))) if output.size else 0.0
        if peak > self.peak_limit:
            output *= self.peak_limit / peak
        return np.asarray(output, dtype=np.float32)


class QuietVoiceLeveler(VoiceLeveler):
    """Stronger leveling for quiet rooms without passing speech through GTCRN."""

    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        block_size: int = 256,
    ) -> None:
        super().__init__(
            sample_rate=sample_rate,
            block_size=block_size,
            target_rms_dbfs=-18.0,
            gate_dbfs=-60.0,
            max_gain_db=18.0,
            min_gain_db=0.0,
            attack_ms=35.0,
            release_ms=120.0,
            gate_release_ms=1500.0,
            peak_limit=0.95,
        )


class StreamingOutputResampler:
    """Low-delay stateful resampling from the model rate to the audio device."""

    def __init__(self, input_sample_rate: int, output_sample_rate: int) -> None:
        if input_sample_rate <= 0 or output_sample_rate <= 0:
            raise ValueError("sample rates must be positive")
        self.input_sample_rate = int(input_sample_rate)
        self.output_sample_rate = int(output_sample_rate)
        self._backend = None
        self._input_samples = 0
        self._output_samples = 0
        self.reset()

    def reset(self) -> None:
        self._input_samples = 0
        self._output_samples = 0
        if self.input_sample_rate == self.output_sample_rate:
            self._backend = None
            return
        try:
            import soxr
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise RuntimeError(
                "48 kHz 实时输出需要 soxr；请先安装 requirements.txt。"
            ) from exc
        self._backend = soxr.ResampleStream(
            self.input_sample_rate,
            self.output_sample_rate,
            1,
            dtype="float32",
            quality="QQ",
        )

    def process(self, samples: np.ndarray) -> np.ndarray:
        block = np.ascontiguousarray(samples, dtype=np.float32).reshape(-1)
        self._input_samples += block.size
        expected_total = round(
            self._input_samples * self.output_sample_rate / self.input_sample_rate
        )
        if self._backend is None:
            output = block.copy()
        else:
            output = np.asarray(
                self._backend.resample_chunk(block, last=False), dtype=np.float32
            ).reshape(-1)
        deficit = expected_total - (self._output_samples + output.size)
        if deficit > 0:
            output = np.concatenate((np.zeros(deficit, dtype=np.float32), output))
        self._output_samples += output.size
        return np.ascontiguousarray(output, dtype=np.float32)


@dataclass(frozen=True)
class LiveDenoiseStats:
    input_blocks: int
    enhanced_blocks: int
    output_blocks: int
    input_drops: int
    output_drops: int
    output_underruns: int
    resyncs: int
    callback_statuses: int
    sink_drops: int
    sink_errors: int
    max_input_queue_depth: int
    max_output_queue_depth: int
    inference_mean_ms: float
    inference_p95_ms: float
    inference_max_ms: float
    realtime_factor: float
    mode: str
    running: bool


@dataclass(frozen=True)
class _OutputPair:
    raw: np.ndarray
    enhanced: np.ndarray
    quiet: np.ndarray | None = None


_SINK_SENTINEL = object()


class LiveDenoiseEngine:
    """Full-duplex audio engine with aligned raw/enhanced monitoring."""

    MODES = frozenset({"enhanced", "quiet", "raw"})

    def __init__(
        self,
        denoiser: StreamingDenoiser,
        *,
        stream_sample_rate: int | None = None,
        input_queue_blocks: int = 4,
        output_queue_blocks: int = 6,
        startup_silence_blocks: int | None = None,
        output_gain: float = 1.0,
        crossfade_samples: int = 80,
        enhancement_processor: FrameProcessor | None = None,
        quiet_processor: FrameProcessor | None = None,
        enhanced_frame_sink: EnhancedFrameSink | None = None,
        sink_queue_blocks: int = 32,
        stream_factory=None,
    ) -> None:
        self.denoiser = denoiser
        self.model_sample_rate = int(denoiser.sample_rate)
        self.model_block_size = int(denoiser.frame_shift_in_samples)
        self.sample_rate = int(stream_sample_rate or self.model_sample_rate)
        if (
            self.model_sample_rate <= 0
            or self.model_block_size <= 0
            or self.sample_rate <= 0
        ):
            raise ValueError("denoiser sample rate and frame size must be positive")
        block_numerator = self.sample_rate * self.model_block_size
        if block_numerator % self.model_sample_rate:
            raise ValueError("stream sample rate must map exactly to one model frame")
        self.block_size = block_numerator // self.model_sample_rate
        if self.block_size <= 0:
            raise ValueError("denoiser sample rate and frame size must be positive")
        if startup_silence_blocks is None:
            startup_silence_blocks = 2 if self.sample_rate == self.model_sample_rate else 3
        if input_queue_blocks <= 0 or output_queue_blocks <= 0:
            raise ValueError("queue sizes must be positive")
        if not 0 <= startup_silence_blocks <= output_queue_blocks:
            raise ValueError("startup_silence_blocks must fit in the output queue")
        if not math.isfinite(output_gain) or output_gain < 0:
            raise ValueError("output_gain must be finite and non-negative")
        if crossfade_samples < 0:
            raise ValueError("crossfade_samples must be non-negative")
        if sink_queue_blocks <= 0:
            raise ValueError("sink_queue_blocks must be positive")

        self.output_gain = float(output_gain)
        self.crossfade_samples = int(crossfade_samples)
        self.startup_silence_blocks = int(startup_silence_blocks)
        self.enhancement_processor = enhancement_processor
        self.quiet_processor = quiet_processor
        self.enhanced_frame_sink = enhanced_frame_sink
        self._stream_factory = stream_factory
        self._output_resampler = StreamingOutputResampler(
            self.model_sample_rate, self.sample_rate
        )

        self._input_queue: queue.Queue[np.ndarray] = queue.Queue(input_queue_blocks)
        self._output_queue: queue.Queue[_OutputPair] = queue.Queue(output_queue_blocks)
        self._sink_queue: queue.Queue[object] = queue.Queue(sink_queue_blocks)
        self._stop_event = threading.Event()
        self._resync_event = threading.Event()
        self._mode = "enhanced"
        self._active_mode = "enhanced"
        self._bypass_return_mode = "enhanced"
        self._worker: threading.Thread | None = None
        self._sink_worker: threading.Thread | None = None
        self._stream = None
        self._worker_error: BaseException | None = None
        self._running = False
        self._timings_lock = threading.Lock()
        self._inference_seconds: deque[float] = deque(maxlen=4096)
        self._silence = np.zeros(self.block_size, dtype=np.float32)
        self._crossfade_weights = np.linspace(
            0.0,
            1.0,
            min(self.crossfade_samples, self.block_size),
            dtype=np.float32,
        )
        self._reset_stats()

    def _reset_stats(self) -> None:
        self._input_blocks = 0
        self._enhanced_blocks = 0
        self._output_blocks = 0
        self._input_drops = 0
        self._output_drops = 0
        self._output_underruns = 0
        self._resyncs = 0
        self._callback_statuses = 0
        self._sink_drops = 0
        self._sink_errors = 0
        self._max_input_queue_depth = 0
        self._max_output_queue_depth = 0
        with self._timings_lock:
            self._inference_seconds.clear()

    @staticmethod
    def _drain(target: queue.Queue) -> None:
        while True:
            try:
                target.get_nowait()
            except queue.Empty:
                return

    def _prefill_output(self) -> None:
        for _ in range(self.startup_silence_blocks):
            self._output_queue.put_nowait(_OutputPair(self._silence, self._silence))
        self._max_output_queue_depth = max(
            self._max_output_queue_depth, self._output_queue.qsize()
        )

    def set_mode(self, mode: str) -> None:
        if mode not in self.MODES:
            raise ValueError(f"unsupported monitoring mode: {mode}")
        self._mode = mode
        if mode != "raw":
            self._bypass_return_mode = mode

    def set_bypass(self, enabled: bool) -> None:
        self.set_mode("raw" if enabled else self._bypass_return_mode)

    def toggle_bypass(self) -> bool:
        if self._mode == "raw":
            self.set_bypass(False)
            return False
        self._bypass_return_mode = self._mode
        self.set_bypass(True)
        return True

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def running(self) -> bool:
        return self._running

    def submit_input(self, samples: np.ndarray) -> bool:
        """Copy one callback block into the bounded input queue without waiting."""

        block = np.asarray(samples, dtype=np.float32).reshape(-1)
        if block.size != self.block_size:
            raise ValueError(
                f"expected {self.block_size} input samples, received {block.size}"
            )
        block = np.ascontiguousarray(block).copy()
        try:
            self._input_queue.put_nowait(block)
        except queue.Full:
            self._input_drops += 1
            self._resync_event.set()
            return False
        self._input_blocks += 1
        self._max_input_queue_depth = max(
            self._max_input_queue_depth, self._input_queue.qsize()
        )
        return True

    @staticmethod
    def _path_for_mode(pair: _OutputPair, mode: str) -> np.ndarray:
        if mode == "raw":
            return pair.raw
        if mode == "quiet":
            return pair.quiet if pair.quiet is not None else pair.raw
        return pair.enhanced

    def _select_pair(self, pair: _OutputPair) -> np.ndarray:
        target = self._path_for_mode(pair, self._mode)
        if self._active_mode == self._mode or self.crossfade_samples == 0:
            mono = target.copy()
        else:
            source = self._path_for_mode(pair, self._active_mode)
            mono = target.copy()
            count = min(self.crossfade_samples, self.block_size)
            weights = self._crossfade_weights[:count]
            mono[:count] = source[:count] * (1.0 - weights) + target[:count] * weights
        self._active_mode = self._mode
        if self.output_gain != 1.0:
            mono *= self.output_gain
        np.clip(mono, -1.0, 1.0, out=mono)
        return mono

    def render_output(self) -> np.ndarray:
        """Return one mono output block immediately, or silence on underrun."""

        if self._worker_error is not None:
            self._output_underruns += 1
            return self._silence
        try:
            pair = self._output_queue.get_nowait()
        except queue.Empty:
            self._output_underruns += 1
            return self._silence
        self._output_blocks += 1
        return self._select_pair(pair)

    def audio_callback(self, indata, outdata, frames, time_info, status) -> None:
        """PortAudio callback: bounded queue operations and channel copy only."""

        del time_info
        outdata.fill(0.0)
        if status:
            self._callback_statuses += 1
        if frames != self.block_size or indata.shape[0] != self.block_size:
            self._input_drops += 1
            return
        self.submit_input(indata[:, 0])
        mono = self.render_output()
        channels = outdata.shape[1] if outdata.ndim == 2 else 1
        if channels == 1:
            outdata[:, 0] = mono
        else:
            outdata[:, :] = mono[:, np.newaxis]

    def _put_output_pair(self, pair: _OutputPair) -> None:
        try:
            self._output_queue.put_nowait(pair)
        except queue.Full:
            try:
                self._output_queue.get_nowait()
            except queue.Empty:
                pass
            else:
                self._output_drops += 1
            try:
                self._output_queue.put_nowait(pair)
            except queue.Full:
                self._output_drops += 1
                return
        self._max_output_queue_depth = max(
            self._max_output_queue_depth, self._output_queue.qsize()
        )

    def _offer_sink(self, block: np.ndarray) -> None:
        if self.enhanced_frame_sink is None:
            return
        try:
            self._sink_queue.put_nowait(block.copy())
        except queue.Full:
            self._sink_drops += 1

    def _run_worker(self) -> None:
        raw_pending: deque[tuple[np.ndarray, np.ndarray]] = deque()
        enhanced_model_pending: deque[np.ndarray] = deque()
        model_buffer = np.empty(0, dtype=np.float32)
        output_buffer = np.empty(0, dtype=np.float32)
        try:
            while not self._stop_event.is_set():
                if self._resync_event.is_set():
                    self.denoiser.reset()
                    self._output_resampler.reset()
                    if self.enhancement_processor is not None:
                        self.enhancement_processor.reset()
                    if self.quiet_processor is not None:
                        self.quiet_processor.reset()
                    raw_pending.clear()
                    enhanced_model_pending.clear()
                    model_buffer = np.empty(0, dtype=np.float32)
                    output_buffer = np.empty(0, dtype=np.float32)
                    self._drain(self._input_queue)
                    self._drain(self._output_queue)
                    self._prefill_output()
                    self._resync_event.clear()
                    self._resyncs += 1
                try:
                    raw = self._input_queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                if self.quiet_processor is None:
                    quiet_stream = raw.copy()
                else:
                    quiet_stream = np.asarray(
                        self.quiet_processor.process(raw.copy()), dtype=np.float32
                    ).reshape(-1)
                    if quiet_stream.size != self.block_size:
                        raise RuntimeError("quiet processor changed the block size")
                    if not np.isfinite(quiet_stream).all():
                        raise RuntimeError("quiet processor produced non-finite samples")
                    quiet_stream = np.ascontiguousarray(
                        quiet_stream, dtype=np.float32
                    )
                raw_pending.append((raw, quiet_stream))
                started = time.perf_counter()
                enhanced = np.asarray(
                    self.denoiser.process(raw, self.sample_rate), dtype=np.float32
                ).reshape(-1)
                elapsed = time.perf_counter() - started
                with self._timings_lock:
                    self._inference_seconds.append(elapsed)
                if enhanced.size:
                    if not np.isfinite(enhanced).all():
                        raise RuntimeError("GTCRN produced non-finite samples")
                    model_buffer = np.concatenate((model_buffer, enhanced))
                while model_buffer.size >= self.model_block_size:
                    model_block = np.ascontiguousarray(
                        model_buffer[: self.model_block_size], dtype=np.float32
                    )
                    model_buffer = model_buffer[self.model_block_size :]
                    if self.enhancement_processor is not None:
                        model_block = np.asarray(
                            self.enhancement_processor.process(model_block),
                            dtype=np.float32,
                        ).reshape(-1)
                        if model_block.size != self.model_block_size:
                            raise RuntimeError("enhancement processor changed the block size")
                    if not np.isfinite(model_block).all():
                        raise RuntimeError("enhancement processor produced non-finite samples")
                    enhanced_model_pending.append(model_block)
                    converted = self._output_resampler.process(model_block)
                    output_buffer = np.concatenate((output_buffer, converted))
                while (
                    output_buffer.size >= self.block_size
                    and raw_pending
                    and enhanced_model_pending
                ):
                    block = np.ascontiguousarray(
                        output_buffer[: self.block_size], dtype=np.float32
                    )
                    output_buffer = output_buffer[self.block_size :]
                    original, quiet_stream = raw_pending.popleft()
                    enhanced_model = enhanced_model_pending.popleft()
                    self._offer_sink(enhanced_model)
                    self._put_output_pair(
                        _OutputPair(original, block, quiet=quiet_stream)
                    )
                    self._enhanced_blocks += 1
        except BaseException as exc:  # callback must stay alive and output silence
            self._worker_error = exc
            self._stop_event.set()

    def _run_sink(self) -> None:
        assert self.enhanced_frame_sink is not None
        while True:
            item = self._sink_queue.get()
            if item is _SINK_SENTINEL:
                return
            try:
                self.enhanced_frame_sink(
                    np.asarray(item, dtype=np.float32), self.model_sample_rate
                )
            except Exception:
                self._sink_errors += 1

    def _start_workers(self) -> None:
        if self._running:
            raise RuntimeError("实时降噪已经在运行。")
        self._drain(self._input_queue)
        self._drain(self._output_queue)
        self._drain(self._sink_queue)
        self._stop_event.clear()
        self._resync_event.clear()
        self._worker_error = None
        self._active_mode = self._mode
        self._reset_stats()
        self.denoiser.reset()
        self._output_resampler.reset()
        if self.enhancement_processor is not None:
            self.enhancement_processor.reset()
        if self.quiet_processor is not None:
            self.quiet_processor.reset()
        self._prefill_output()
        if self.enhanced_frame_sink is not None:
            self._sink_worker = threading.Thread(
                target=self._run_sink, name="audiorescue-live-sink", daemon=True
            )
            self._sink_worker.start()
        self._worker = threading.Thread(
            target=self._run_worker, name="audiorescue-gtcrn", daemon=True
        )
        self._worker.start()
        self._running = True

    def start(
        self,
        *,
        input_device=None,
        output_device=None,
        latency: str | float = "low",
    ) -> None:
        """Start the worker and a mono-in/stereo-out PortAudio stream."""

        self._start_workers()
        try:
            if self._stream_factory is None:
                try:
                    import sounddevice as sd
                except ImportError as exc:  # pragma: no cover - environment-specific
                    raise RuntimeError(
                        "实时监听需要 sounddevice；请先安装 requirements.txt。"
                    ) from exc
                stream_factory = sd.Stream
            else:
                stream_factory = self._stream_factory
            self._stream = stream_factory(
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                device=(input_device, output_device),
                channels=(1, 2),
                dtype="float32",
                latency=latency,
                callback=self.audio_callback,
            )
            self._stream.start()
        except BaseException:
            self.stop()
            raise

    def _stop_sink(self) -> None:
        if self._sink_worker is None:
            return
        try:
            self._sink_queue.put_nowait(_SINK_SENTINEL)
        except queue.Full:
            self._drain(self._sink_queue)
            self._sink_queue.put_nowait(_SINK_SENTINEL)
        self._sink_worker.join(timeout=2.0)
        if self._sink_worker.is_alive():
            raise RuntimeError("下游音频消费者未能在 2 秒内停止，已禁止重启。")
        self._sink_worker = None

    def _stop_workers(self) -> None:
        self._stop_event.set()
        worker_timed_out = False
        if self._worker is not None:
            self._worker.join(timeout=2.0)
            if self._worker.is_alive():
                worker_timed_out = True
            else:
                self._worker = None
        sink_error: RuntimeError | None = None
        try:
            self._stop_sink()
        except RuntimeError as exc:
            sink_error = exc
        if worker_timed_out:
            raise RuntimeError("GTCRN 工作线程未能在 2 秒内停止，已禁止重启。")
        if sink_error is not None:
            raise sink_error
        self.denoiser.reset()
        self._output_resampler.reset()
        if self.enhancement_processor is not None:
            self.enhancement_processor.reset()
        if self.quiet_processor is not None:
            self.quiet_processor.reset()
        self._running = False

    def stop(self) -> None:
        """Stop immediately; this method is safe to call more than once."""

        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                abort = getattr(stream, "abort", None)
                if abort is not None:
                    abort()
                else:
                    stream.stop()
            finally:
                stream.close()
        self._stop_workers()

    def raise_if_failed(self) -> None:
        if self._worker_error is not None:
            raise RuntimeError("实时降噪工作线程失败。") from self._worker_error

    def snapshot_stats(self) -> LiveDenoiseStats:
        with self._timings_lock:
            timings = np.asarray(tuple(self._inference_seconds), dtype=np.float64)
        mean = float(np.mean(timings)) if timings.size else 0.0
        p95 = float(np.percentile(timings, 95)) if timings.size else 0.0
        maximum = float(np.max(timings)) if timings.size else 0.0
        block_seconds = self.block_size / self.sample_rate
        return LiveDenoiseStats(
            input_blocks=self._input_blocks,
            enhanced_blocks=self._enhanced_blocks,
            output_blocks=self._output_blocks,
            input_drops=self._input_drops,
            output_drops=self._output_drops,
            output_underruns=self._output_underruns,
            resyncs=self._resyncs,
            callback_statuses=self._callback_statuses,
            sink_drops=self._sink_drops,
            sink_errors=self._sink_errors,
            max_input_queue_depth=self._max_input_queue_depth,
            max_output_queue_depth=self._max_output_queue_depth,
            inference_mean_ms=mean * 1000.0,
            inference_p95_ms=p95 * 1000.0,
            inference_max_ms=maximum * 1000.0,
            realtime_factor=mean / block_seconds if block_seconds else 0.0,
            mode=self._mode,
            running=self._running,
        )

    def __enter__(self) -> "LiveDenoiseEngine":
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback
        self.stop()
