import tempfile
import json
import os
import sys
import types
import unittest
import wave
from asyncio import run
from pathlib import Path
from unittest import mock

from app import LiveStateMiddleware, OfflineHtmlResourceMiddleware, offline_launch_app_kwargs
from ui.file_delivery import FileDeliveryMiddleware, clear_delivery_registry, register_file_for_delivery
from ui.file_staging import stage_files_for_gradio
from ui.layout import strip_remote_html_resources


def _write_pcm_wav(
    path: Path,
    frames: bytes = b"\x00\x00\x01\x00",
    *,
    sample_width: int = 2,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(48000)
        wav_file.writeframes(frames)


async def _receive():
    return {"type": "http.request", "body": b"", "more_body": False}


def _receive_body(body: bytes):
    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    return receive


def _receive_chunks(*chunks: bytes):
    remaining = list(chunks)

    async def receive():
        chunk = remaining.pop(0)
        return {
            "type": "http.request",
            "body": chunk,
            "more_body": bool(remaining),
        }

    return receive


async def _collect(app, *, scope=None):
    sent = []

    async def send(message):
        sent.append(message)

    await OfflineHtmlResourceMiddleware(app)(scope or {"type": "http", "method": "GET"}, _receive, send)
    return sent


def _response_app(*, status=200, headers=(), chunks=()):
    async def app(scope, receive, send):
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": list(headers),
            }
        )
        for index, chunk in enumerate(chunks):
            await send(
                {
                    "type": "http.response.body",
                    "body": chunk,
                    "more_body": index < len(chunks) - 1,
                }
            )

    return app


class OfflineMiddlewareTest(unittest.TestCase):
    def setUp(self) -> None:
        clear_delivery_registry()

    def tearDown(self) -> None:
        clear_delivery_registry()

    def test_launch_uses_offline_middleware(self) -> None:
        class FakeMiddleware:
            def __init__(self, cls):
                self.cls = cls

        fake_starlette = types.ModuleType("starlette")
        fake_middleware = types.ModuleType("starlette.middleware")
        fake_middleware.Middleware = FakeMiddleware
        with mock.patch.dict(
            sys.modules,
            {"starlette": fake_starlette, "starlette.middleware": fake_middleware},
        ):
            middleware = offline_launch_app_kwargs()["middleware"]
        self.assertEqual(len(middleware), 3)
        self.assertIs(middleware[0].cls, LiveStateMiddleware)
        self.assertIs(middleware[1].cls, FileDeliveryMiddleware)
        self.assertIs(middleware[2].cls, OfflineHtmlResourceMiddleware)

    def test_live_state_route_returns_same_process_snapshot_without_secret(self) -> None:
        async def passthrough_app(scope, receive, send):
            await send({"type": "http.response.start", "status": 404, "headers": []})
            await send({"type": "http.response.body", "body": b"missing"})

        sent = []

        async def send(message):
            sent.append(message)

        with mock.patch.dict(
            os.environ,
            {"DASHSCOPE_API_KEY": "SECRET_MARKER_SHOULD_NOT_RENDER"},
        ):
            run(
                LiveStateMiddleware(passthrough_app)(
                    {
                        "type": "http",
                        "method": "GET",
                        "path": "/audiorescue/live/state",
                    },
                    _receive,
                    send,
                )
            )

        body = sent[-1]["body"].decode("utf-8")
        self.assertEqual(sent[0]["status"], 200)
        self.assertIn('"audio_running"', body)
        self.assertIn('"meeting_material_names"', body)
        self.assertIn('"suggestion_sources"', body)
        self.assertIn('"needs_verification"', body)
        self.assertNotIn("SECRET_MARKER_SHOULD_NOT_RENDER", body)

    def test_live_floating_route_serves_local_helper_page(self) -> None:
        async def passthrough_app(scope, receive, send):
            await send({"type": "http.response.start", "status": 404, "headers": []})
            await send({"type": "http.response.body", "body": b"missing"})

        sent = []

        async def send(message):
            sent.append(message)

        run(
            LiveStateMiddleware(passthrough_app)(
                {
                    "type": "http",
                    "method": "GET",
                    "path": "/audiorescue/live/floating",
                },
                _receive,
                send,
            )
        )

        body = sent[-1]["body"].decode("utf-8")
        self.assertEqual(sent[0]["status"], 200)
        self.assertIn("/audiorescue/live/state", body)
        self.assertIn("AudioRescue 悬浮提示窗", body)
        self.assertIn('href="/"', body)
        self.assertIn("返回主界面", body)
        self.assertIn("window.opener.focus", body)
        self.assertIn("window.close()", body)
        self.assertIn("needs_verification", body)
        self.assertIn("suggestion_sources", body)
        self.assertIn("suggestion-meta", body)
        self.assertIn("state.suggestion_kind", body)
        self.assertIn("state.confidence", body)
        self.assertIn("对方刚刚说了什么？", body)
        self.assertIn("立即生成下一段建议", body)
        self.assertIn("根据会议资料生成回答", body)
        self.assertIn("/audiorescue/live/action", body)
        self.assertIn('runAction("next_line")', body)
        self.assertIn('runAction("answer")', body)
        self.assertIn("actionInFlight", body)
        self.assertIn("response.ok", body)
        self.assertIn('aria-live="polite"', body)
        self.assertNotIn('questionInput.value = ""', body)

    def test_live_action_route_requests_next_section(self) -> None:
        async def passthrough_app(scope, receive, send):
            raise AssertionError("action route must not reach wrapped app")

        sent = []

        async def send(message):
            sent.append(message)

        snapshot = types.SimpleNamespace(last_action="已请求下一句建议。")
        with mock.patch(
            "app.LIVE_CONTROLLER.request_next_line", return_value=snapshot
        ) as request_next, mock.patch(
            "app.snapshot_to_dict", return_value={"meeting_status": "listening"}
        ):
            run(
                LiveStateMiddleware(passthrough_app)(
                    {
                        "type": "http",
                        "method": "POST",
                        "path": "/audiorescue/live/action",
                        "headers": [(b"content-type", b"application/json")],
                    },
                    _receive_body(
                        json.dumps({"action": "next_line"}).encode("utf-8")
                    ),
                    send,
                )
            )

        payload = json.loads(sent[-1]["body"].decode("utf-8"))
        self.assertEqual(sent[0]["status"], 200)
        self.assertEqual(payload["last_action"], "已请求下一句建议。")
        self.assertEqual(payload["state"]["meeting_status"], "listening")
        request_next.assert_called_once_with()

    def test_live_action_route_submits_remote_question(self) -> None:
        async def passthrough_app(scope, receive, send):
            raise AssertionError("action route must not reach wrapped app")

        sent = []

        async def send(message):
            sent.append(message)

        snapshot = types.SimpleNamespace(last_action="已提交对方问题。")
        with mock.patch(
            "app.LIVE_CONTROLLER.request_answer", return_value=snapshot
        ) as request_answer, mock.patch(
            "app.snapshot_to_dict", return_value={"suggestion": ""}
        ):
            run(
                LiveStateMiddleware(passthrough_app)(
                    {
                        "type": "http",
                        "method": "POST",
                        "path": "/audiorescue/live/action",
                        "headers": [
                            (b"content-type", b"application/json; charset=utf-8")
                        ],
                    },
                    _receive_chunks(
                        json.dumps(
                            {
                                "action": "answer",
                                "question": "你们的项目限制是什么？",
                            },
                            ensure_ascii=False,
                        ).encode("utf-8")[:20],
                        json.dumps(
                            {
                                "action": "answer",
                                "question": "你们的项目限制是什么？",
                            },
                            ensure_ascii=False,
                        ).encode("utf-8")[20:],
                    ),
                    send,
                )
            )

        self.assertEqual(sent[0]["status"], 200)
        request_answer.assert_called_once_with("你们的项目限制是什么？")

    def test_live_action_route_rejects_invalid_requests(self) -> None:
        async def passthrough_app(scope, receive, send):
            raise AssertionError("action route must not reach wrapped app")

        cases = (
            (
                "GET",
                [(b"content-type", b"application/json")],
                b"{}",
                405,
            ),
            ("POST", [(b"content-type", b"text/plain")], b"{}", 415),
            (
                "POST",
                [(b"content-type", b"application/json")],
                b"not-json",
                400,
            ),
            (
                "POST",
                [(b"content-type", b"application/json")],
                json.dumps({"action": "answer", "question": ""}).encode("utf-8"),
                400,
            ),
            (
                "POST",
                [(b"content-type", b"application/json")],
                json.dumps({"action": "answer", "question": ["不是文本"]}).encode(
                    "utf-8"
                ),
                400,
            ),
            (
                "POST",
                [(b"content-type", b"application/json")],
                b"x" * 16_385,
                413,
            ),
        )
        for method, headers, body, expected_status in cases:
            with self.subTest(method=method, body=body):
                sent = []

                async def send(message):
                    sent.append(message)

                run(
                    LiveStateMiddleware(passthrough_app)(
                        {
                            "type": "http",
                            "method": method,
                            "path": "/audiorescue/live/action",
                            "headers": headers,
                        },
                        _receive_body(body),
                        send,
                    )
                )
                self.assertEqual(sent[0]["status"], expected_status)
                self.assertIn("error", json.loads(sent[-1]["body"].decode("utf-8")))
                if expected_status == 405:
                    self.assertEqual(dict(sent[0]["headers"])[b"allow"], b"POST")

    def test_live_action_route_hides_unexpected_controller_errors(self) -> None:
        async def passthrough_app(scope, receive, send):
            raise AssertionError("action route must not reach wrapped app")

        sent = []

        async def send(message):
            sent.append(message)

        with mock.patch(
            "app.LIVE_CONTROLLER.request_next_line",
            side_effect=RuntimeError(
                "SECRET_MARKER /private/audio/meeting.wav"
            ),
        ), mock.patch("app.LIVE_CONTROLLER.record_ui_error") as record_error:
            run(
                LiveStateMiddleware(passthrough_app)(
                    {
                        "type": "http",
                        "method": "POST",
                        "path": "/audiorescue/live/action",
                        "headers": [(b"content-type", b"application/json")],
                    },
                    _receive_body(
                        json.dumps({"action": "next_line"}).encode("utf-8")
                    ),
                    send,
                )
            )

        body = sent[-1]["body"].decode("utf-8")
        self.assertEqual(sent[0]["status"], 500)
        self.assertNotIn("SECRET_MARKER", body)
        self.assertNotIn("/private/audio", body)
        record_error.assert_called_once()

    def test_non_html_payloads_are_byte_preserved(self) -> None:
        cases = [
            (b"application/json", b'{"ok": true}'),
            (b"text/javascript", b"console.log('ok')"),
            (b"text/css", b"body{color:red}"),
            (b"audio/wav", b"RIFF\x00\x00WAVE"),
            (b"image/png", b"\x89PNG\r\n\x1a\n"),
            (b"application/octet-stream", b"\x00\x01download-bytes"),
        ]
        for content_type, body in cases:
            with self.subTest(content_type=content_type):
                sent = run(
                    _collect(
                        _response_app(headers=[(b"content-type", content_type)], chunks=[body])
                    )
                )
                self.assertEqual(sent[-1]["body"], body)
                self.assertEqual(sent[0]["headers"], [(b"content-type", content_type)])

    def test_multichunk_html_filters_remote_tags_and_preserves_metadata(self) -> None:
        chunks = [
            b"<html><head><script src=\"https://cdnjs.cloudflare.com/ajax/libs/",
            b"iframe-resizer/4.3.1/iframeResizer.contentWindow.min.js\"></script>",
            b"<script src=\"/assets/local.js\"></script><link href=\"/theme.css\" rel=\"stylesheet\">",
            b"<link href=\"https://fonts.googleapis.com/css2?family=Source+Sans+Pro\" rel=\"stylesheet\">",
            "fixture visible 文本</head><body>ok</body></html>".encode("utf-8"),
        ]
        sent = run(
            _collect(
                _response_app(
                    status=201,
                    headers=[
                        (b"content-type", b"text/html; charset=utf-8"),
                        (b"x-test", b"kept"),
                        (b"content-length", b"9999"),
                    ],
                    chunks=chunks,
                )
            )
        )
        body = sent[-1]["body"].decode("utf-8")
        headers = dict(sent[0]["headers"])
        self.assertEqual(sent[0]["status"], 201)
        self.assertEqual(headers[b"content-type"], b"text/html; charset=utf-8")
        self.assertEqual(headers[b"x-test"], b"kept")
        self.assertEqual(headers[b"content-length"], str(len(sent[-1]["body"])).encode("ascii"))
        self.assertNotIn("cdnjs.cloudflare.com", body)
        self.assertNotIn("fonts.googleapis.com", body)
        self.assertIn('src="/assets/local.js"', body)
        self.assertIn('href="/theme.css"', body)
        self.assertIn("fixture visible 文本", body)

    def test_head_bodyless_error_empty_and_streaming_responses_are_safe(self) -> None:
        html_headers = [(b"content-type", b"text/html; charset=utf-8")]
        head_sent = run(
            _collect(
                _response_app(headers=html_headers, chunks=[b"<html>head</html>"]),
                scope={"type": "http", "method": "HEAD"},
            )
        )
        self.assertEqual(head_sent[-1]["body"], b"<html>head</html>")

        for status in (204, 304):
            with self.subTest(status=status):
                sent = run(
                    _collect(
                        _response_app(status=status, headers=html_headers, chunks=[b""])
                    )
                )
                self.assertEqual(sent[0]["headers"], html_headers)
                self.assertEqual(sent[-1]["body"], b"")

        error_sent = run(
            _collect(
                _response_app(
                    status=500,
                    headers=html_headers,
                    chunks=[b'<script src="https://cdnjs.cloudflare.com/ajax/libs/iframe-resizer/x.js"></script>error'],
                )
            )
        )
        self.assertEqual(error_sent[0]["status"], 500)
        self.assertNotIn(b"cdnjs.cloudflare.com", error_sent[-1]["body"])

        empty_sent = run(_collect(_response_app(headers=html_headers, chunks=[b""])))
        self.assertEqual(empty_sent[-1]["body"], b"")

        sse_body = b"data: <script src=\"https://cdnjs.cloudflare.com/ajax/libs/x.js\"></script>\n\n"
        sse_sent = run(
            _collect(
                _response_app(headers=[(b"content-type", b"text/event-stream")], chunks=[sse_body])
            )
        )
        self.assertEqual(sse_sent[-1]["body"], sse_body)

    def test_websocket_and_api_paths_pass_through(self) -> None:
        messages = []

        async def websocket_app(scope, receive, send):
            messages.append(scope["type"])
            await send({"type": "websocket.accept"})

        async def send(message):
            messages.append(message)

        run(
            OfflineHtmlResourceMiddleware(websocket_app)(
                {"type": "websocket", "path": "/queue/join"}, _receive, send
            )
        )
        self.assertEqual(messages, ["websocket", {"type": "websocket.accept"}])

        api_body = b'{"data": ["https://fonts.googleapis.com/not-html"]}'
        api_sent = run(
            _collect(
                _response_app(headers=[(b"content-type", b"application/json")], chunks=[api_body]),
                scope={"type": "http", "method": "POST", "path": "/gradio_api/queue/join"},
            )
        )
        self.assertEqual(api_sent[-1]["body"], api_body)

    def test_filter_is_idempotent(self) -> None:
        html = (
            '<link href="https://fonts.gstatic.com" rel="preconnect">'
            '<script src="https://cdnjs.cloudflare.com/ajax/libs/iframe-resizer/x.js"></script>'
            '<script src="/local.js"></script><main>ok</main>'
        )
        once = strip_remote_html_resources(html)
        twice = strip_remote_html_resources(once)
        self.assertEqual(once, twice)
        self.assertIn('<script src="/local.js"></script>', twice)

    def test_staged_download_contents_match_sources(self) -> None:
        with tempfile.TemporaryDirectory() as root_dir, tempfile.TemporaryDirectory() as staging_dir:
            root = Path(root_dir)
            staging = Path(staging_dir)
            original = root / "original.wav"
            mixed = root / "mixed.wav"
            full = root / "full.wav"
            _write_pcm_wav(original, b"\x10\x00\x11\x00")
            _write_pcm_wav(mixed, b"\x20\x00\x21\x00")
            _write_pcm_wav(full, b"\x30\x00\x31\x00")
            original_bytes = original.read_bytes()
            mixed_bytes = mixed.read_bytes()
            full_bytes = full.read_bytes()

            staged = stage_files_for_gradio(
                {
                    "original_audio": str(original),
                    "mixed_audio": str(mixed),
                    "full_audio": str(full),
                },
                allowed_roots=(root,),
                base_dir=root,
                staging_root=staging,
            )

            self.assertEqual(Path(staged["original_audio"]).read_bytes(), original_bytes)
            self.assertEqual(Path(staged["mixed_audio"]).read_bytes(), mixed_bytes)
            self.assertEqual(Path(staged["full_audio"]).read_bytes(), full_bytes)

    def test_file_delivery_route_bypasses_html_filter_and_preserves_audio_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "original.wav"
            _write_pcm_wav(
                source,
                b'<script src="https://cdnjs.cloudflare.com/ajax/libs/x.js"></script>RIFF\x00',
            )
            url = register_file_for_delivery(
                source,
                filename="original.wav",
                allowed_roots=(root,),
            )
            self.assertIsNotNone(url)

            async def app(scope, receive, send):
                await OfflineHtmlResourceMiddleware(
                    lambda scope, receive, send: None
                )(scope, receive, send)

            sent = []

            async def send(message):
                sent.append(message)

            scope = {
                "type": "http",
                "method": "GET",
                "path": str(url),
                "raw_path": str(url).encode("ascii"),
                "headers": [],
                "query_string": b"",
            }
            run(FileDeliveryMiddleware(app)(scope, _receive, send))

            self.assertEqual(sent[0]["status"], 200)
            self.assertEqual(sent[-1]["body"], source.read_bytes())


if __name__ == "__main__":
    unittest.main()
