import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from core.meeting_context import (
    AdviceResult,
    MeetingKnowledgeBase,
    MeetingPreset,
    MeetingScenario,
    MeetingSession,
)
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


class _FakeStructuredSseResponse(_FakeSseResponse):
    def __iter__(self):
        content = json.dumps(
            {
                "action": "SHOW",
                "kind": "ANSWER",
                "say_now": "根据资料，端到端延迟是三十二毫秒。",
                "needs_verification": False,
                "confidence": 0.91,
                "evidence_refs": [],
            },
            ensure_ascii=False,
        )
        event = json.dumps(
            {"choices": [{"delta": {"content": content}}]}, ensure_ascii=False
        ).encode("utf-8")
        return iter([b"data: " + event + b"\n", b"data: [DONE]\n"])


def _wait_for(predicate, timeout: float = 1.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition was not met before timeout")


class _StructuredAdvisor:
    def __init__(self) -> None:
        self.requests = []
        self.called = threading.Event()

    def generate_advice(self, request, *, on_delta=None) -> AdviceResult:
        del on_delta
        self.requests.append(request)
        self.called.set()
        return AdviceResult(
            action="show",
            kind="answer",
            say_now="根据本机测试，端到端延迟是三十二毫秒。",
        )


class _LegacyAdvisor:
    def __init__(self) -> None:
        self.calls = []
        self.called = threading.Event()

    def generate(self, preset, transcript, trigger, *, on_delta=None) -> str:
        del on_delta
        self.calls.append((preset, transcript, trigger))
        self.called.set()
        return "旧接口仍然可以给出建议。"


class _BlockingStructuredAdvisor:
    def __init__(self) -> None:
        self.requests = []
        self.entered = threading.Event()
        self.release = threading.Event()

    def generate_advice(self, request, *, on_delta=None) -> AdviceResult:
        del on_delta
        self.requests.append(request)
        self.entered.set()
        if not self.release.wait(1.0):
            raise TimeoutError("test did not release structured advisor")
        return AdviceResult(
            action="show",
            kind="answer",
            say_now="这条已经过期的回答不应显示。",
        )


class _BlockingStreamingLegacyAdvisor:
    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()

    def generate(self, preset, transcript, trigger, *, on_delta=None) -> str:
        del preset, transcript, trigger
        if on_delta is not None:
            on_delta("旧建议")
        self.entered.set()
        if not self.release.wait(1.0):
            raise TimeoutError("test did not release legacy advisor")
        if on_delta is not None:
            on_delta("不应继续显示")
        return "这条旧建议不应作为最终结果。"


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

    def test_qwen_endpoint_defaults_to_official_dashscope_hosts(self) -> None:
        advisor = QwenStreamingAdvisor(api_key="test-key", urlopen=lambda *_: None)
        self.assertEqual(
            advisor.endpoint,
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        )

        with self.assertRaisesRegex(ValueError, "DashScope 官方"):
            QwenStreamingAdvisor(
                api_key="test-key",
                endpoint="https://attacker-bucket.oss-cn-hangzhou.aliyuncs.com/collect",
                urlopen=lambda *_: None,
            )

    def test_qwen_structured_advice_is_buffered_and_validated(self) -> None:
        captured = {}

        def fake_urlopen(request, timeout):
            captured["request"] = request
            captured["timeout"] = timeout
            return _FakeStructuredSseResponse()

        session = MeetingSession(
            config=MeetingPreset(
                title="项目答辩",
                scenario=MeetingScenario.DEFENSE,
                objective="准确回答问题",
            ),
            session_id="structured-qwen-test",
        )
        session.append_turn("导师问延迟是多少？", source="remote_audio")
        request = session.build_request("question", "延迟是多少？")
        advisor = QwenStreamingAdvisor(
            api_key="test-key", timeout=3.0, urlopen=fake_urlopen
        )

        result = advisor.generate_advice(request)

        self.assertEqual(result.action, "show")
        self.assertEqual(result.kind, "answer")
        self.assertIn("三十二毫秒", result.say_now)
        payload = json.loads(captured["request"].data.decode("utf-8"))
        self.assertFalse(payload["enable_thinking"])
        self.assertEqual(
            [message["role"] for message in payload["messages"]],
            ["system", "user"],
        )
        self.assertNotIn("structured-qwen-test", payload["messages"][1]["content"])


class MeetingAssistantContextCompatibilityTest(unittest.TestCase):
    def _preset(self) -> MeetingPreset:
        return MeetingPreset(
            title="项目答辩",
            scenario=MeetingScenario.DEFENSE,
            user_role="学生答辩人",
            audience="导师和评委",
            objective="准确回答评委问题",
            agenda=("背景", "方案", "实验", "总结"),
        )

    def _start_advisor_thread(self, assistant: LiveMeetingAssistant) -> None:
        assistant._advisor_thread = threading.Thread(
            target=assistant._run_advisor,
            name="test-meeting-advisor",
            daemon=True,
        )
        assistant._advisor_thread.start()

    def test_structured_advisor_receives_session_request_and_material_evidence(
        self,
    ) -> None:
        advisor = _StructuredAdvisor()
        knowledge_base = MeetingKnowledgeBase()
        with tempfile.TemporaryDirectory() as tempdir:
            material = Path(tempdir) / "性能指标.md"
            material.write_text(
                "本机回环测试中，实时降噪端到端延迟是三十二毫秒。",
                encoding="utf-8",
            )
            knowledge_base.ingest_files([material])

            assistant = LiveMeetingAssistant(
                api_key="test-key",
                session_config=self._preset(),
                knowledge_base=knowledge_base,
                advisor=advisor,
                suggestion_interval=0,
                clock=lambda: 1.0,
            )
            assistant._session = MeetingSession(
                config=self._preset(),
                knowledge_base=knowledge_base,
                session_id="structured-test-session",
            )
            assistant._running = True
            self._start_advisor_thread(assistant)
            try:
                assistant._on_asr_event(
                    RealtimeAsrEvent(
                        text="你们的端到端延迟是多少？",
                        is_final=True,
                        sentence_id=1,
                    )
                )

                self.assertTrue(advisor.called.wait(1.0))
                _wait_for(
                    lambda: assistant.snapshot().suggestion
                    == "根据本机测试，端到端延迟是三十二毫秒。"
                )
            finally:
                assistant.stop()

        self.assertEqual(len(advisor.requests), 1)
        request = advisor.requests[0]
        self.assertEqual(request.session_id, "structured-test-session")
        self.assertIn(
            "你们的端到端延迟是多少？",
            json.dumps(request.payload, ensure_ascii=False),
        )
        self.assertTrue(
            any("三十二毫秒" in evidence.text for evidence in request.evidence)
        )

    def test_old_generate_only_advisor_remains_supported(self) -> None:
        advisor = _LegacyAdvisor()
        preset = self._preset()
        assistant = LiveMeetingAssistant(
            api_key="test-key",
            session_config=preset,
            advisor=advisor,
            suggestion_interval=0,
            clock=lambda: 1.0,
        )
        assistant._session = MeetingSession(
            config=preset, session_id="legacy-test-session"
        )
        assistant._running = True
        self._start_advisor_thread(assistant)
        try:
            assistant._on_asr_event(
                RealtimeAsrEvent(
                    text="旧版顾问还能回答吗？",
                    is_final=True,
                    sentence_id=2,
                )
            )

            self.assertTrue(advisor.called.wait(1.0))
            _wait_for(
                lambda: assistant.snapshot().suggestion
                == "旧接口仍然可以给出建议。"
            )
        finally:
            assistant.stop()

        self.assertEqual(len(advisor.calls), 1)
        legacy_preset, transcript, trigger = advisor.calls[0]
        self.assertIn("项目答辩", legacy_preset)
        self.assertIn("旧版顾问还能回答吗？", transcript)
        self.assertEqual(trigger, "question")

    def test_structured_result_is_discarded_when_context_changes_during_generation(
        self,
    ) -> None:
        advisor = _BlockingStructuredAdvisor()
        callbacks = []
        preset = self._preset()
        assistant = LiveMeetingAssistant(
            api_key="test-key",
            session_config=preset,
            advisor=advisor,
            on_suggestion=lambda text, is_final: callbacks.append((text, is_final)),
            clock=lambda: 1.0,
        )
        assistant._session = MeetingSession(
            config=preset, session_id="stale-result-session"
        )
        assistant._running = True
        self._start_advisor_thread(assistant)
        try:
            assistant._on_asr_event(
                RealtimeAsrEvent(
                    text="你们的延迟是多少？",
                    is_final=True,
                    sentence_id=3,
                )
            )
            self.assertTrue(advisor.entered.wait(1.0))
            self.assertEqual(assistant.snapshot().status, "thinking")

            assistant.session.append_turn(
                "模型生成期间又收到了一条新发言。", source="remote_audio"
            )
            advisor.release.set()

            _wait_for(lambda: assistant.snapshot().status == "listening")
            snapshot = assistant.snapshot()
            self.assertEqual(snapshot.suggestion, "")
            self.assertEqual(snapshot.suggestion_kind, "")
            self.assertEqual(callbacks, [])
        finally:
            advisor.release.set()
            assistant.stop()

        self.assertEqual(len(advisor.requests), 1)

    def test_legacy_stream_is_cleared_when_context_changes_during_generation(
        self,
    ) -> None:
        advisor = _BlockingStreamingLegacyAdvisor()
        callbacks = []
        preset = self._preset()
        assistant = LiveMeetingAssistant(
            api_key="test-key",
            session_config=preset,
            advisor=advisor,
            on_suggestion=lambda text, is_final: callbacks.append((text, is_final)),
            clock=lambda: 1.0,
        )
        assistant._session = MeetingSession(
            config=preset, session_id="stale-legacy-session"
        )
        assistant._running = True
        self._start_advisor_thread(assistant)
        try:
            assistant._on_asr_event(
                RealtimeAsrEvent(
                    text="请继续介绍实验。",
                    is_final=True,
                    sentence_id=4,
                )
            )
            self.assertTrue(advisor.entered.wait(1.0))
            self.assertEqual(assistant.snapshot().suggestion, "旧建议")

            assistant.session.append_turn(
                "生成期间会议内容已更新。", source="remote_audio"
            )
            advisor.release.set()

            _wait_for(lambda: assistant.snapshot().status == "listening")
            self.assertEqual(assistant.snapshot().suggestion, "")
            self.assertIn(("旧建议", False), callbacks)
            self.assertNotIn(("旧建议不应作为最终结果。", True), callbacks)
            self.assertFalse(any("不应继续显示" in text for text, _ in callbacks))
        finally:
            advisor.release.set()
            assistant.stop()


if __name__ == "__main__":
    unittest.main()
