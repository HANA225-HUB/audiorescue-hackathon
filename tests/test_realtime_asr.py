import json
import unittest

import numpy as np

from core.realtime_asr import (
    DashscopeRealtimeAsr,
    build_run_task,
    float32_to_pcm16,
)


class _FakeWebSocket:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class RealtimeAsrProtocolTest(unittest.TestCase):
    def test_float32_to_pcm16_clips_and_encodes_little_endian(self) -> None:
        samples = np.array(
            [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0], dtype=np.float32
        )

        encoded = float32_to_pcm16(samples)

        decoded = np.frombuffer(encoded, dtype="<i2")
        np.testing.assert_array_equal(
            decoded,
            np.array([-32767, -32767, -16384, 0, 16384, 32767, 32767]),
        )
        with self.assertRaises(ValueError):
            float32_to_pcm16(np.array([np.nan], dtype=np.float32))

    def test_run_task_contains_realtime_pcm_parameters(self) -> None:
        task = build_run_task(
            "task-123",
            sample_rate=16_000,
            language_hint="zh",
            final_silence_ms=500,
        )

        self.assertEqual(task["header"]["action"], "run-task")
        self.assertEqual(task["header"]["streaming"], "duplex")
        self.assertEqual(task["payload"]["model"], "fun-asr-realtime")
        parameters = task["payload"]["parameters"]
        self.assertEqual(parameters["format"], "pcm")
        self.assertEqual(parameters["sample_rate"], 16_000)
        self.assertEqual(parameters["language_hints"], ["zh"])
        self.assertEqual(parameters["max_sentence_silence"], 500)
        self.assertFalse(parameters["semantic_punctuation_enabled"])
        self.assertTrue(parameters["heartbeat"])

    def test_result_messages_emit_partial_and_one_deduplicated_final(self) -> None:
        events = []
        client = DashscopeRealtimeAsr(api_key="test-key", on_event=events.append)
        client._task_id = "task-123"
        ws = _FakeWebSocket()

        partial = {
            "header": {"event": "result-generated"},
            "payload": {
                "output": {
                    "sentence": {
                        "text": "你好",
                        "sentence_end": False,
                        "sentence_id": 7,
                        "begin_time": 10,
                    }
                }
            },
        }
        final = {
            "header": {"event": "result-generated"},
            "payload": {
                "output": {
                    "sentence": {
                        "text": "你好世界",
                        "sentence_end": True,
                        "sentence_id": 7,
                        "begin_time": 10,
                        "end_time": 800,
                    }
                }
            },
        }
        heartbeat = {
            "header": {"event": "result-generated"},
            "payload": {"output": {"sentence": {"heartbeat": True}}},
        }

        client._on_message(ws, json.dumps(partial))
        client._on_message(ws, json.dumps(heartbeat))
        client._on_message(ws, json.dumps(final))
        client._on_message(ws, json.dumps(final))

        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].text, "你好")
        self.assertFalse(events[0].is_final)
        self.assertEqual(events[1].text, "你好世界")
        self.assertTrue(events[1].is_final)
        self.assertEqual(events[1].sentence_id, 7)
        self.assertEqual(events[1].end_time_ms, 800)


if __name__ == "__main__":
    unittest.main()
