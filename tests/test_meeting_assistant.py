import json
import unittest

from core.meeting_assistant import (
    LiveMeetingAssistant,
    QwenStreamingAdvisor,
    is_question,
    redact_text,
)
from core.realtime_asr import RealtimeAsrEvent


class _FakeSseResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback

    def __iter__(self):
        return iter(
            [
                b'data: {"choices":[{"delta":{"content":"\xe5\xbb\xba\xe8\xae\xae"}}]}\n',
                b'data: {"choices":[{"delta":{"content":"\xe5\x85\x88\xe7\xa1\xae\xe8\xae\xa4\xe9\x9c\x80\xe6\xb1\x82"}}]}\n',
                b"data: [DONE]\n",
            ]
        )


class MeetingAssistantTest(unittest.TestCase):
    def test_question_detection_and_sensitive_text_redaction(self) -> None:
        self.assertTrue(is_question("我们该怎么部署？"))
        self.assertTrue(is_question("这个方案可以吗"))
        self.assertFalse(is_question("我们下周完成部署。"))

        redacted = redact_text(
            "DASHSCOPE_API_KEY=secret-value "
            "13812345678 someone@example.com 11010519491231002X"
        )
        self.assertNotIn("secret-value", redacted)
        self.assertNotIn("13812345678", redacted)
        self.assertNotIn("someone@example.com", redacted)
        self.assertNotIn("11010519491231002X", redacted)
        self.assertIn("[密钥]", redacted)
        self.assertIn("[手机号]", redacted)

    def test_partial_only_updates_state_and_final_queues_advice(self) -> None:
        received = []
        clock_values = iter([1.0, 2.0])
        assistant = LiveMeetingAssistant(
            api_key="test-key",
            suggestion_interval=0,
            on_transcript=received.append,
            clock=lambda: next(clock_values),
        )
        assistant._running = True

        assistant._on_asr_event(
            RealtimeAsrEvent(text="正在说", is_final=False, sentence_id=1)
        )
        self.assertEqual(assistant.snapshot().partial_text, "正在说")
        self.assertTrue(assistant._advice_queue.empty())

        assistant._on_asr_event(
            RealtimeAsrEvent(text="这是完整发言", is_final=True, sentence_id=1)
        )
        snapshot = assistant.snapshot()
        self.assertEqual(snapshot.partial_text, "")
        self.assertEqual(snapshot.transcript, ("这是完整发言",))
        job = assistant._advice_queue.get_nowait()
        self.assertEqual(job.trigger, "next_line")
        self.assertIn("这是完整发言", job.context)
        self.assertEqual([event.is_final for event in received], [False, True])

    def test_qwen_sse_is_combined_and_disables_thinking(self) -> None:
        captured = {}

        def fake_urlopen(request, timeout):
            captured["request"] = request
            captured["timeout"] = timeout
            return _FakeSseResponse()

        deltas = []
        advisor = QwenStreamingAdvisor(
            api_key="test-key",
            timeout=3.0,
            urlopen=fake_urlopen,
        )

        result = advisor.generate(
            "项目汇报",
            "对方在问交付时间",
            "question",
            on_delta=deltas.append,
        )

        self.assertEqual(result, "建议先确认需求")
        self.assertEqual(deltas, ["建议", "先确认需求"])
        self.assertEqual(captured["timeout"], 3.0)
        payload = json.loads(captured["request"].data.decode("utf-8"))
        self.assertFalse(payload["enable_thinking"])
        self.assertTrue(payload["stream"])
        self.assertEqual(payload["model"], "qwen3.6-flash")


if __name__ == "__main__":
    unittest.main()
