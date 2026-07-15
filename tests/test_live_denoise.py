import time
import unittest
from pathlib import Path

import numpy as np

from core.live_denoise import (
    DEFAULT_GTCRN_MODEL_PATH,
    GtcrnDenoiser,
    LiveDenoiseEngine,
    QuietVoiceLeveler,
    StreamingOutputResampler,
    VoiceLeveler,
    _OutputPair,
)


class _DelayedGainDenoiser:
    sample_rate = 16_000
    frame_shift_in_samples = 256

    def __init__(self, gain: float = 2.0) -> None:
        self.gain = gain
        self.pending = None

    def process(
        self, samples: np.ndarray, input_sample_rate: int | None = None
    ) -> np.ndarray:
        del input_sample_rate
        previous = self.pending
        self.pending = np.asarray(samples, dtype=np.float32).copy()
        if previous is None:
            return np.empty(0, dtype=np.float32)
        return previous * self.gain

    def flush(self) -> np.ndarray:
        if self.pending is None:
            return np.empty(0, dtype=np.float32)
        result = self.pending * self.gain
        self.pending = None
        return result

    def reset(self) -> None:
        self.pending = None


class _FailingDenoiser(_DelayedGainDenoiser):
    def process(
        self, samples: np.ndarray, input_sample_rate: int | None = None
    ) -> np.ndarray:
        del samples, input_sample_rate
        raise ValueError("synthetic worker failure")


class _DelayedDownsamplingDenoiser(_DelayedGainDenoiser):
    def process(
        self, samples: np.ndarray, input_sample_rate: int | None = None
    ) -> np.ndarray:
        self.asserted_input_rate = input_sample_rate
        previous = self.pending
        self.pending = np.asarray(samples, dtype=np.float32).copy()
        if previous is None:
            return np.empty(0, dtype=np.float32)
        return previous.reshape(256, 3).mean(axis=1) * self.gain


class _GainProcessor:
    def __init__(self, gain: float) -> None:
        self.gain = gain

    def process(self, samples: np.ndarray) -> np.ndarray:
        return np.asarray(samples, dtype=np.float32) * self.gain

    def reset(self) -> None:
        pass


class _RecordingGainProcessor(_GainProcessor):
    def __init__(self, gain: float) -> None:
        super().__init__(gain)
        self.block_sizes = []

    def process(self, samples: np.ndarray) -> np.ndarray:
        self.block_sizes.append(np.asarray(samples).size)
        return super().process(samples)


class _FakeStream:
    latest = None

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.callback = kwargs["callback"]
        self.started = False
        self.aborted = False
        self.closed = False
        type(self).latest = self

    def start(self) -> None:
        self.started = True

    def abort(self) -> None:
        self.aborted = True

    def close(self) -> None:
        self.closed = True


class _FailingStream(_FakeStream):
    def start(self) -> None:
        raise RuntimeError("synthetic stream failure")


def _wait_for(predicate, timeout: float = 1.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition was not met before timeout")


class LiveDenoiseEngineTest(unittest.TestCase):
    def make_engine(self, **kwargs) -> LiveDenoiseEngine:
        return LiveDenoiseEngine(
            _DelayedGainDenoiser(),
            startup_silence_blocks=0,
            crossfade_samples=0,
            stream_factory=_FakeStream,
            **kwargs,
        )

    def test_worker_pairs_delayed_enhancement_with_oldest_raw_block(self) -> None:
        engine = self.make_engine()
        first = np.full(256, 0.1, dtype=np.float32)
        second = np.full(256, 0.2, dtype=np.float32)
        try:
            engine.start()
            self.assertTrue(engine.submit_input(first))
            self.assertTrue(engine.submit_input(second))
            _wait_for(lambda: engine.snapshot_stats().enhanced_blocks >= 1)
            output = engine.render_output()
        finally:
            engine.stop()

        np.testing.assert_allclose(output, first * 2.0)
        self.assertTrue(_FakeStream.latest.aborted)
        self.assertTrue(_FakeStream.latest.closed)

    def test_callback_copies_mono_output_to_both_channels(self) -> None:
        engine = self.make_engine()
        pair = _OutputPair(
            raw=np.full(256, 0.2, dtype=np.float32),
            enhanced=np.full(256, 0.7, dtype=np.float32),
        )
        engine._output_queue.put_nowait(pair)
        indata = np.zeros((256, 1), dtype=np.float32)
        outdata = np.empty((256, 2), dtype=np.float32)

        engine.audio_callback(indata, outdata, 256, None, None)

        np.testing.assert_allclose(outdata[:, 0], 0.7)
        np.testing.assert_allclose(outdata[:, 1], 0.7)

    def test_callback_rejects_wrong_frame_size_with_silence(self) -> None:
        engine = self.make_engine()
        indata = np.ones((128, 1), dtype=np.float32)
        outdata = np.ones((128, 2), dtype=np.float32)

        engine.audio_callback(indata, outdata, 128, None, "overflow")

        np.testing.assert_array_equal(outdata, 0.0)
        stats = engine.snapshot_stats()
        self.assertEqual(stats.input_drops, 1)
        self.assertEqual(stats.callback_statuses, 1)

    def test_input_queue_is_bounded_and_owns_its_copy(self) -> None:
        engine = self.make_engine(input_queue_blocks=1)
        original = np.full(256, 0.3, dtype=np.float32)

        self.assertTrue(engine.submit_input(original))
        original.fill(0.9)
        self.assertFalse(engine.submit_input(original))
        queued = engine._input_queue.get_nowait()

        np.testing.assert_allclose(queued, 0.3)
        self.assertEqual(engine.snapshot_stats().input_drops, 1)

    def test_output_queue_drops_oldest_complete_pair(self) -> None:
        engine = self.make_engine(output_queue_blocks=1)
        old = _OutputPair(
            raw=np.full(256, 0.1, dtype=np.float32),
            enhanced=np.full(256, 0.2, dtype=np.float32),
        )
        new = _OutputPair(
            raw=np.full(256, 0.3, dtype=np.float32),
            enhanced=np.full(256, 0.4, dtype=np.float32),
        )

        engine._put_output_pair(old)
        engine._put_output_pair(new)

        np.testing.assert_allclose(engine.render_output(), 0.4)
        self.assertEqual(engine.snapshot_stats().output_drops, 1)

    def test_raw_enhanced_switch_uses_the_same_aligned_pair(self) -> None:
        engine = self.make_engine()
        engine._output_queue.put_nowait(
            _OutputPair(
                raw=np.full(256, 0.15, dtype=np.float32),
                enhanced=np.full(256, 0.65, dtype=np.float32),
            )
        )
        np.testing.assert_allclose(engine.render_output(), 0.65)

        engine._output_queue.put_nowait(
            _OutputPair(
                raw=np.full(256, 0.25, dtype=np.float32),
                enhanced=np.full(256, 0.75, dtype=np.float32),
            )
        )
        self.assertTrue(engine.toggle_bypass())
        np.testing.assert_allclose(engine.render_output(), 0.25)
        self.assertEqual(engine.mode, "raw")

    def test_quiet_mode_uses_raw_speech_instead_of_suppressed_denoiser(self) -> None:
        engine = LiveDenoiseEngine(
            _DelayedGainDenoiser(gain=0.05),
            quiet_processor=_GainProcessor(4.0),
            startup_silence_blocks=0,
            crossfade_samples=0,
            stream_factory=_FakeStream,
        )
        quiet = np.full(256, 0.01, dtype=np.float32)
        engine.set_mode("quiet")
        try:
            engine.start()
            self.assertTrue(engine.submit_input(quiet))
            self.assertTrue(engine.submit_input(quiet))
            _wait_for(lambda: engine.snapshot_stats().enhanced_blocks >= 1)
            output = engine.render_output()
        finally:
            engine.stop()

        np.testing.assert_allclose(output, 0.04, atol=1e-6)

    def test_quiet_mode_processes_48k_raw_blocks_without_resampling(self) -> None:
        quiet_processor = _RecordingGainProcessor(4.0)
        denoiser = _DelayedDownsamplingDenoiser(gain=0.05)
        engine = LiveDenoiseEngine(
            denoiser,
            stream_sample_rate=48_000,
            quiet_processor=quiet_processor,
            startup_silence_blocks=0,
            crossfade_samples=0,
            stream_factory=_FakeStream,
        )
        first = np.linspace(0.001, 0.01, 768, dtype=np.float32)
        second = np.full(768, 0.02, dtype=np.float32)
        engine.set_mode("quiet")
        try:
            engine.start()
            self.assertTrue(engine.submit_input(first))
            self.assertTrue(engine.submit_input(second))
            _wait_for(lambda: engine.snapshot_stats().enhanced_blocks >= 1)
            output = engine.render_output()
        finally:
            engine.stop()

        self.assertEqual(denoiser.asserted_input_rate, 48_000)
        self.assertEqual(quiet_processor.block_sizes[:2], [768, 768])
        self.assertEqual(output.size, 768)
        np.testing.assert_allclose(output, first * 4.0, atol=1e-6)

    def test_bypass_returns_to_the_selected_quiet_mode(self) -> None:
        engine = self.make_engine()
        engine.set_mode("quiet")

        self.assertTrue(engine.toggle_bypass())
        self.assertEqual(engine.mode, "raw")
        self.assertFalse(engine.toggle_bypass())
        self.assertEqual(engine.mode, "quiet")

    def test_mode_switch_crossfades_without_changing_alignment(self) -> None:
        engine = LiveDenoiseEngine(
            _DelayedGainDenoiser(),
            startup_silence_blocks=0,
            crossfade_samples=80,
            stream_factory=_FakeStream,
        )
        engine._output_queue.put_nowait(
            _OutputPair(
                raw=np.full(256, 0.2, dtype=np.float32),
                enhanced=np.full(256, 0.8, dtype=np.float32),
            )
        )
        engine.set_mode("raw")

        output = engine.render_output()

        self.assertAlmostEqual(float(output[0]), 0.8, places=6)
        self.assertAlmostEqual(float(output[79]), 0.2, places=6)
        np.testing.assert_allclose(output[80:], 0.2)

    def test_worker_failure_is_reported_and_callback_falls_back_to_silence(self) -> None:
        engine = LiveDenoiseEngine(
            _FailingDenoiser(),
            startup_silence_blocks=0,
            stream_factory=_FakeStream,
        )
        try:
            engine.start()
            engine.submit_input(np.zeros(256, dtype=np.float32))
            _wait_for(lambda: engine._worker_error is not None)
            np.testing.assert_array_equal(engine.render_output(), 0.0)
            with self.assertRaisesRegex(RuntimeError, "工作线程失败"):
                engine.raise_if_failed()
        finally:
            engine.stop()

    def test_start_failure_cleans_up_workers_and_stream(self) -> None:
        engine = LiveDenoiseEngine(
            _DelayedGainDenoiser(),
            startup_silence_blocks=0,
            stream_factory=_FailingStream,
        )

        with self.assertRaisesRegex(RuntimeError, "stream failure"):
            engine.start()

        self.assertFalse(engine.running)
        self.assertTrue(_FailingStream.latest.closed)
        engine.stop()
        engine.stop()


class VoiceLevelerTest(unittest.TestCase):
    def test_quiet_speech_is_raised_without_amplifying_silence(self) -> None:
        leveler = VoiceLeveler()
        silence = np.zeros(256, dtype=np.float32)
        np.testing.assert_array_equal(leveler.process(silence), silence)

        quiet = np.full(256, 0.01, dtype=np.float32)
        output = quiet
        for _ in range(100):
            output = leveler.process(quiet)

        self.assertGreater(float(np.sqrt(np.mean(output**2))), 0.02)
        self.assertGreater(leveler.gain, 2.0)

    def test_peak_limiter_prevents_clipping(self) -> None:
        leveler = VoiceLeveler()
        block = np.full(256, 0.01, dtype=np.float32)
        block[32] = 1.0

        output = leveler.process(block)

        self.assertLessEqual(float(np.max(np.abs(output))), 0.950001)
        self.assertTrue(np.isfinite(output).all())

    def test_quiet_voice_profile_boosts_soft_speech_more_aggressively(self) -> None:
        leveler = QuietVoiceLeveler(sample_rate=48_000, block_size=768)
        quiet = np.full(768, 10.0 ** (-50.0 / 20.0), dtype=np.float32)
        output = quiet
        for _ in range(20):
            output = leveler.process(quiet)

        self.assertGreater(leveler.gain, 4.0)
        self.assertGreater(float(np.sqrt(np.mean(output**2))), float(quiet[0]) * 4.0)

    def test_quiet_voice_profile_keeps_gain_through_a_short_pause(self) -> None:
        leveler = QuietVoiceLeveler(sample_rate=48_000, block_size=768)
        quiet = np.full(768, 10.0 ** (-50.0 / 20.0), dtype=np.float32)
        for _ in range(20):
            leveler.process(quiet)
        gain_before_pause = leveler.gain

        silence = np.zeros(768, dtype=np.float32)
        for _ in range(10):
            output = leveler.process(silence)

        np.testing.assert_array_equal(output, silence)
        self.assertGreater(leveler.gain, gain_before_pause * 0.8)

    def test_quiet_voice_profile_never_reduces_unclipped_input(self) -> None:
        leveler = QuietVoiceLeveler(sample_rate=48_000, block_size=768)
        source = np.full(768, 0.03, dtype=np.float32)
        original = source.copy()

        output = leveler.process(source)

        np.testing.assert_array_equal(source, original)
        self.assertGreaterEqual(float(np.sqrt(np.mean(output**2))), 0.03)


class StreamingOutputResamplerTest(unittest.TestCase):
    def test_16k_blocks_become_exact_48k_blocks_without_drift(self) -> None:
        resampler = StreamingOutputResampler(16_000, 48_000)
        source = np.full(256, 0.1, dtype=np.float32)

        outputs = [resampler.process(source) for _ in range(20)]

        self.assertTrue(all(block.size == 768 for block in outputs))
        merged = np.concatenate(outputs)
        self.assertEqual(merged.size, 20 * 768)
        self.assertTrue(np.isfinite(merged).all())


class RealGtcrnSmokeTest(unittest.TestCase):
    @unittest.skipUnless(
        Path(DEFAULT_GTCRN_MODEL_PATH).is_file(), "GTCRN model is not cached"
    )
    def test_cached_gtcrn_stream_runs_faster_than_realtime(self) -> None:
        try:
            denoiser = GtcrnDenoiser(DEFAULT_GTCRN_MODEL_PATH)
        except RuntimeError as exc:
            if "sherpa-onnx" in str(exc):
                self.skipTest(str(exc))
            raise
        rng = np.random.default_rng(7)
        blocks = [rng.normal(0.0, 0.03, 256).astype(np.float32) for _ in range(64)]

        started = time.perf_counter()
        output = [denoiser.process(block) for block in blocks]
        output.append(denoiser.flush())
        elapsed = time.perf_counter() - started
        merged = np.concatenate(output)

        self.assertEqual(denoiser.sample_rate, 16_000)
        self.assertEqual(denoiser.frame_shift_in_samples, 256)
        self.assertEqual(merged.size, 64 * 256)
        self.assertTrue(np.isfinite(merged).all())
        self.assertLess(elapsed, 64 * 256 / 16_000)

    @unittest.skipUnless(
        Path(DEFAULT_GTCRN_MODEL_PATH).is_file(), "GTCRN model is not cached"
    )
    def test_gtcrn_engine_accepts_48k_callback_blocks(self) -> None:
        try:
            denoiser = GtcrnDenoiser(DEFAULT_GTCRN_MODEL_PATH)
        except RuntimeError as exc:
            if "sherpa-onnx" in str(exc):
                self.skipTest(str(exc))
            raise
        engine = LiveDenoiseEngine(
            denoiser,
            stream_sample_rate=48_000,
            output_gain=0.0,
            stream_factory=_FakeStream,
        )
        rng = np.random.default_rng(19)
        try:
            engine.start()
            for _ in range(16):
                block = rng.normal(0.0, 0.02, 768).astype(np.float32)
                while not engine.submit_input(block):
                    time.sleep(0.005)
                time.sleep(0.003)
            _wait_for(lambda: engine.snapshot_stats().enhanced_blocks >= 4)
            output = engine.render_output()
        finally:
            engine.stop()

        self.assertEqual(output.size, 768)
        self.assertTrue(np.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
