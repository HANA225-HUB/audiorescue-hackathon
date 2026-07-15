import unittest
import threading

import numpy as np

from core.live_denoise import LiveDenoiseStats
from core.meeting_assistant import MeetingSnapshot
from ui.live_controller import (
    DEFAULT_INPUT_CHOICE,
    DEFAULT_OUTPUT_CHOICE,
    VIRTUAL_OUTPUT_CHOICE,
    LiveUiController,
    snapshot_to_dict,
)


class _FakeSoundDevice:
    def __init__(self, devices, default=(0, 1)):
        self._devices = list(devices)
        self.default = type("_Default", (), {"device": default})()

    def query_devices(self, device=None, kind=None):
        del kind
        if device is not None:
            return dict(self._devices[int(device)])
        return list(self._devices)


class _FakeDenoiser:
    sample_rate = 16_000
    frame_shift_in_samples = 160


class _FakeEngine:
    def __init__(self, *args, **kwargs) -> None:
        del args
        self.sink = kwargs.get("enhanced_frame_sink")
        self.running = False
        self.mode = "enhanced"
        self.started_with = None
        self.stop_calls = 0

    def start(self, *, input_device=None, output_device=None, latency="low") -> None:
        self.running = True
        self.started_with = (input_device, output_device, latency)

    def stop(self) -> None:
        self.stop_calls += 1
        self.running = False

    def set_mode(self, mode: str) -> None:
        self.mode = mode

    def raise_if_failed(self) -> None:
        return None

    def snapshot_stats(self) -> LiveDenoiseStats:
        return LiveDenoiseStats(
            input_blocks=3,
            enhanced_blocks=2,
            output_blocks=1,
            input_drops=0,
            output_drops=0,
            output_underruns=0,
            resyncs=0,
            callback_statuses=0,
            sink_drops=0,
            sink_errors=0,
            max_input_queue_depth=1,
            max_output_queue_depth=1,
            inference_mean_ms=1.0,
            inference_p95_ms=2.0,
            inference_max_ms=3.0,
            realtime_factor=0.1,
            mode=self.mode,
            running=self.running,
        )


class _SlowStartEngine(_FakeEngine):
    start_entered = threading.Event()
    release_start = threading.Event()

    def start(self, *, input_device=None, output_device=None, latency="low") -> None:
        self.started_with = (input_device, output_device, latency)
        type(self).start_entered.set()
        type(self).release_start.wait(timeout=2.0)
        self.running = True


class _FailingStopEngine(_FakeEngine):
    def stop(self) -> None:
        self.stop_calls += 1
        raise RuntimeError("stop failed /tmp/audiorescue-private.wav")


class _FakeMeeting:
    def __init__(self, *, preset="", **kwargs) -> None:
        del kwargs
        self.preset = preset
        self.running = False
        self.accepted = []
        self.request_count = 0
        self.suggestion = ""

    def start(self) -> None:
        self.running = True

    def stop(self) -> None:
        self.running = False

    def accept_audio(self, samples, sample_rate: int) -> bool:
        self.accepted.append((np.asarray(samples), sample_rate))
        return True

    def request_next_line(self) -> bool:
        self.request_count += 1
        self.suggestion = "建议先确认目标。"
        return True

    def snapshot(self) -> MeetingSnapshot:
        return MeetingSnapshot(
            status="listening" if self.running else "stopped",
            partial_text="实时字幕",
            transcript=("正式字幕",),
            suggestion=self.suggestion,
            error=None,
            asr_dropped_packets=0,
        )


def _devices(*, include_virtual=True):
    result = [
        {
            "name": "Built-in Microphone",
            "max_input_channels": 1,
            "max_output_channels": 0,
            "default_samplerate": 48000,
        },
        {
            "name": "Built-in Output",
            "max_input_channels": 0,
            "max_output_channels": 2,
            "default_samplerate": 48000,
        },
    ]
    if include_virtual:
        result.append(
            {
                "name": "BlackHole 2ch",
                "max_input_channels": 0,
                "max_output_channels": 2,
                "default_samplerate": 48000,
            }
        )
    return result


def _controller(devices=None, *, engine_instances=None, meeting_instances=None, meeting_factory=None):
    engines = engine_instances if engine_instances is not None else []
    meetings = meeting_instances if meeting_instances is not None else []

    def engine_factory(*args, **kwargs):
        engine = _FakeEngine(*args, **kwargs)
        engines.append(engine)
        return engine

    def default_meeting_factory(*args, **kwargs):
        meeting = _FakeMeeting(*args, **kwargs)
        meetings.append(meeting)
        return meeting

    return LiveUiController(
        sounddevice_loader=lambda: _FakeSoundDevice(devices or _devices()),
        ensure_model=lambda: "gtcrn.onnx",
        denoiser_factory=lambda *args, **kwargs: _FakeDenoiser(),
        engine_factory=engine_factory,
        meeting_factory=meeting_factory or default_meeting_factory,
    )


class LiveUiControllerTest(unittest.TestCase):
    def test_device_choices_include_defaults_and_virtual_output(self) -> None:
        controller = _controller()

        input_choices, output_choices, note = controller.list_devices()

        self.assertEqual(input_choices[0], DEFAULT_INPUT_CHOICE)
        self.assertEqual(output_choices[:2], [DEFAULT_OUTPUT_CHOICE, VIRTUAL_OUTPUT_CHOICE])
        self.assertTrue(any("Built-in Microphone" in choice for choice in input_choices))
        self.assertTrue(any("Built-in Output" in choice for choice in output_choices))
        self.assertIn("已刷新设备", note)

    def test_start_audio_uses_selected_devices_and_virtual_output(self) -> None:
        engines = []
        controller = _controller(engine_instances=engines)
        input_choices, _, _ = controller.list_devices()

        snapshot = controller.start_audio(input_choices[1], VIRTUAL_OUTPUT_CHOICE, "quiet")

        self.assertEqual(len(engines), 1)
        self.assertTrue(snapshot.audio_running)
        self.assertEqual(snapshot.audio_mode, "quiet")
        self.assertEqual(engines[0].started_with, (0, 2, "low"))
        self.assertIsNotNone(engines[0].sink)

    def test_repeated_start_does_not_create_second_engine(self) -> None:
        engines = []
        controller = _controller(engine_instances=engines)

        controller.start_audio(DEFAULT_INPUT_CHOICE, VIRTUAL_OUTPUT_CHOICE, "enhanced")
        snapshot = controller.start_audio(DEFAULT_INPUT_CHOICE, VIRTUAL_OUTPUT_CHOICE, "raw")

        self.assertEqual(len(engines), 1)
        self.assertEqual(snapshot.audio_mode, "raw")
        self.assertIn("未重复启动", snapshot.last_action)

    def test_missing_blackhole_returns_clear_error_without_starting_engine(self) -> None:
        engines = []
        controller = _controller(_devices(include_virtual=False), engine_instances=engines)

        snapshot = controller.start_audio(DEFAULT_INPUT_CHOICE, VIRTUAL_OUTPUT_CHOICE, "enhanced")

        self.assertEqual(engines, [])
        self.assertFalse(snapshot.audio_running)
        self.assertIn("没有找到", snapshot.audio_error)

    def test_meeting_audio_sink_suggestion_and_cleanup_share_one_instance(self) -> None:
        meetings = []
        engines = []
        controller = _controller(engine_instances=engines, meeting_instances=meetings)

        meeting_snapshot = controller.start_meeting("项目例会")
        self.assertEqual(meeting_snapshot.meeting_status, "listening")
        self.assertEqual(meetings[0].preset, "项目例会")

        self.assertTrue(controller._accept_enhanced_audio(np.zeros(160, dtype=np.float32), 16000))
        self.assertEqual(len(meetings[0].accepted), 1)

        suggestion_snapshot = controller.request_next_line()
        self.assertEqual(suggestion_snapshot.suggestion, "建议先确认目标。")
        self.assertEqual(meetings[0].request_count, 1)

        controller.start_audio(DEFAULT_INPUT_CHOICE, VIRTUAL_OUTPUT_CHOICE, "enhanced")
        controller.cleanup()
        self.assertFalse(meetings[0].running)
        self.assertFalse(engines[0].running)
        self.assertEqual(engines[0].stop_calls, 1)

    def test_meeting_start_error_is_sanitized_in_snapshot(self) -> None:
        def raising_meeting_factory(*args, **kwargs):
            del args, kwargs

            class _RaisingMeeting:
                def start(self) -> None:
                    raise RuntimeError(
                        "DASHSCOPE_API_KEY=fake_secret_value /tmp/audiorescue-test/input.wav"
                    )

            return _RaisingMeeting()

        controller = _controller(meeting_factory=raising_meeting_factory)

        snapshot = controller.start_meeting("secret preset")

        self.assertIn("[密钥]", snapshot.meeting_error)
        self.assertIn("[路径]", snapshot.meeting_error)
        self.assertNotIn("fake_secret_value", snapshot.meeting_error)
        self.assertNotIn("/tmp/audiorescue-test", snapshot.meeting_error)

    def test_snapshot_to_dict_does_not_require_runtime_instances(self) -> None:
        controller = _controller()
        snapshot = controller.snapshot()

        payload = snapshot_to_dict(snapshot)

        self.assertFalse(payload["audio_running"])
        self.assertEqual(payload["meeting_status"], "stopped")
        self.assertIn("last_action", payload)

    def test_default_speaker_output_is_blocked_before_engine_creation(self) -> None:
        engines = []
        controller = _controller(engine_instances=engines)

        snapshot = controller.start_audio(DEFAULT_INPUT_CHOICE, DEFAULT_OUTPUT_CHOICE, "enhanced")

        self.assertFalse(snapshot.audio_running)
        self.assertEqual(engines, [])
        self.assertIn("未被识别为耳机或虚拟麦", snapshot.audio_error)

    def test_virtual_input_to_virtual_output_loop_is_blocked(self) -> None:
        engines = []
        devices = [
            {
                "name": "BlackHole 2ch",
                "max_input_channels": 2,
                "max_output_channels": 2,
                "default_samplerate": 48000,
            },
            {
                "name": "AirPods",
                "max_input_channels": 0,
                "max_output_channels": 2,
                "default_samplerate": 48000,
            },
        ]
        controller = _controller(devices, engine_instances=engines)
        input_choices, _, _ = controller.list_devices()

        snapshot = controller.start_audio(input_choices[1], VIRTUAL_OUTPUT_CHOICE, "enhanced")

        self.assertFalse(snapshot.audio_running)
        self.assertEqual(engines, [])
        self.assertIn("自循环", snapshot.audio_error)

    def test_concurrent_start_does_not_create_second_engine(self) -> None:
        engines = []

        def engine_factory(*args, **kwargs):
            engine = _SlowStartEngine(*args, **kwargs)
            engines.append(engine)
            return engine

        _SlowStartEngine.start_entered.clear()
        _SlowStartEngine.release_start.clear()
        controller = LiveUiController(
            sounddevice_loader=lambda: _FakeSoundDevice(_devices()),
            ensure_model=lambda: "gtcrn.onnx",
            denoiser_factory=lambda *args, **kwargs: _FakeDenoiser(),
            engine_factory=engine_factory,
            meeting_factory=lambda *args, **kwargs: _FakeMeeting(*args, **kwargs),
        )
        result = []
        thread = threading.Thread(
            target=lambda: result.append(
                controller.start_audio(DEFAULT_INPUT_CHOICE, VIRTUAL_OUTPUT_CHOICE, "enhanced")
            )
        )

        thread.start()
        self.assertTrue(_SlowStartEngine.start_entered.wait(timeout=1.0))
        second = controller.start_audio(DEFAULT_INPUT_CHOICE, VIRTUAL_OUTPUT_CHOICE, "raw")
        _SlowStartEngine.release_start.set()
        thread.join(timeout=2.0)

        self.assertEqual(len(engines), 1)
        self.assertIn("正在启动", second.last_action)
        self.assertTrue(result[0].audio_running)

    def test_stop_during_start_stops_engine_after_start_finishes(self) -> None:
        engines = []

        def engine_factory(*args, **kwargs):
            engine = _SlowStartEngine(*args, **kwargs)
            engines.append(engine)
            return engine

        _SlowStartEngine.start_entered.clear()
        _SlowStartEngine.release_start.clear()
        controller = LiveUiController(
            sounddevice_loader=lambda: _FakeSoundDevice(_devices()),
            ensure_model=lambda: "gtcrn.onnx",
            denoiser_factory=lambda *args, **kwargs: _FakeDenoiser(),
            engine_factory=engine_factory,
            meeting_factory=lambda *args, **kwargs: _FakeMeeting(*args, **kwargs),
        )
        result = []
        thread = threading.Thread(
            target=lambda: result.append(
                controller.start_audio(DEFAULT_INPUT_CHOICE, VIRTUAL_OUTPUT_CHOICE, "enhanced")
            )
        )

        thread.start()
        self.assertTrue(_SlowStartEngine.start_entered.wait(timeout=1.0))
        stopping = controller.stop_audio()
        _SlowStartEngine.release_start.set()
        thread.join(timeout=2.0)

        self.assertIn("已排队停止", stopping.last_action)
        self.assertEqual(len(engines), 1)
        self.assertEqual(engines[0].stop_calls, 1)
        self.assertFalse(controller.snapshot().audio_running)
        self.assertIn("已停止", result[0].last_action)

    def test_stop_failure_keeps_engine_reference_for_retry(self) -> None:
        engines = []

        def engine_factory(*args, **kwargs):
            engine = _FailingStopEngine(*args, **kwargs)
            engines.append(engine)
            return engine

        controller = LiveUiController(
            sounddevice_loader=lambda: _FakeSoundDevice(_devices()),
            ensure_model=lambda: "gtcrn.onnx",
            denoiser_factory=lambda *args, **kwargs: _FakeDenoiser(),
            engine_factory=engine_factory,
            meeting_factory=lambda *args, **kwargs: _FakeMeeting(*args, **kwargs),
        )
        controller.start_audio(DEFAULT_INPUT_CHOICE, VIRTUAL_OUTPUT_CHOICE, "enhanced")

        snapshot = controller.stop_audio()

        self.assertIn("停止异常", snapshot.last_action)
        self.assertIn("[路径]", snapshot.audio_error)
        self.assertIs(controller._engine, engines[0])
        self.assertEqual(engines[0].stop_calls, 1)


if __name__ == "__main__":
    unittest.main()
