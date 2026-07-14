import os
import re
import sys
import tempfile
import types
import unittest
from asyncio import run
from pathlib import Path
from unittest import mock
from urllib.parse import quote, unquote

from ui.presenters import UI_TUPLE_KEYS
from ui.file_delivery import FileDeliveryMiddleware
from ui.layout import (
    CSS_PATH,
    OfflineHtmlResourceMiddleware,
    _fixture_mode_default,
    _run,
    _run_fixture,
    _run_real_pipeline,
    build_demo,
    offline_launch_app_kwargs,
    strip_remote_html_resources,
)

SENSITIVE_MARKERS = (
    "SECRET_MARKER",
    "/private/audio/input.wav",
    "C:\\Users\\secret\\input.wav",
    "https://example.test/model.pt?q=SECRET_MARKER",
)


def _assert_no_sensitive_markers(testcase: unittest.TestCase, rendered: object) -> None:
    text = str(rendered)
    decoded_once = unquote(text)
    decoded_twice = unquote(decoded_once)
    variants: set[str] = set()
    for marker in SENSITIVE_MARKERS:
        variants.update(
            {
                marker,
                quote(marker, safe=""),
                quote(quote(marker, safe=""), safe=""),
            }
        )
    for variant in variants:
        testcase.assertNotIn(variant, text)
        testcase.assertNotIn(variant, decoded_once)
        testcase.assertNotIn(variant, decoded_twice)


def _relative_luminance(color: str) -> float:
    red, green, blue = (int(color[index : index + 2], 16) / 255 for index in (1, 3, 5))
    channels = []
    for channel in (red, green, blue):
        if channel <= 0.03928:
            channels.append(channel / 12.92)
        else:
            channels.append(((channel + 0.055) / 1.055) ** 2.4)
    return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]


def _contrast_ratio(foreground: str, background: str) -> float:
    fg_luminance = _relative_luminance(foreground)
    bg_luminance = _relative_luminance(background)
    lighter = max(fg_luminance, bg_luminance)
    darker = min(fg_luminance, bg_luminance)
    return (lighter + 0.05) / (darker + 0.05)


def _css_variables(*, dark: bool = False) -> dict[str, str]:
    css = CSS_PATH.read_text(encoding="utf-8")
    root_blocks = re.findall(r":root\s*\{(?P<body>.*?)\}", css, flags=re.S)
    index = 1 if dark else 0
    body = root_blocks[index]
    return dict(re.findall(r"(--ar-[\w-]+):\s*(#[0-9a-fA-F]{6})", body))


class LayoutSafetyTest(unittest.TestCase):
    def test_competition_startup_uses_real_pipeline_by_default(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(_fixture_mode_default())

    def test_fixture_mode_requires_explicit_environment_opt_in(self) -> None:
        for value in ("1", "true", "yes"):
            with self.subTest(value=value), mock.patch.dict(
                os.environ, {"AUDIORESCUE_UI_FIXTURE": value}, clear=True
            ):
                self.assertTrue(_fixture_mode_default())

    def test_production_server_rejects_fixture_even_if_callback_is_forged(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch(
            "ui.layout._run_fixture"
        ) as run_fixture:
            rendered = _run(True, "process_result_ok", None, "标准", "", False)
        run_fixture.assert_not_called()
        self.assertTrue(any("UI_FIXTURE_DISABLED" in str(item) for item in rendered))
        self.assertTrue(any("正式模式未启用开发 fixture" in str(item) for item in rendered))

    def test_fixture_loader_exception_is_sanitized(self) -> None:
        with mock.patch(
            "ui.layout.load_fixture",
            side_effect=RuntimeError("SECRET_MARKER /private/audio/input.wav"),
        ):
            rendered = _run_fixture("process_result_ok")

        combined = "\n".join(str(item) for item in rendered)
        self.assertIn("UI_FIXTURE_LOAD_FAILED", combined)
        self.assertIn("开发 fixture 读取失败", combined)
        _assert_no_sensitive_markers(self, combined)

    def test_pipeline_import_and_call_exceptions_are_sanitized(self) -> None:
        original_import = __import__

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "core.pipeline":
                raise ImportError("SECRET_MARKER https://example.test/model.pt?q=SECRET_MARKER")
            return original_import(name, globals, locals, fromlist, level)

        input_file = types.SimpleNamespace(name="/tmp/dev_input.wav")
        with mock.patch("builtins.__import__", side_effect=guarded_import):
            import_rendered = _run_real_pipeline(input_file, "标准", "", False)
        import_combined = "\n".join(str(item) for item in import_rendered)
        self.assertIn("UI_PIPELINE_UNAVAILABLE", import_combined)
        _assert_no_sensitive_markers(self, import_combined)

        fake_pipeline = types.ModuleType("core.pipeline")

        def fake_process_audio(**kwargs):
            raise RuntimeError("SECRET_MARKER C:\\Users\\secret\\input.wav")

        fake_pipeline.process_audio = fake_process_audio
        with mock.patch.dict(sys.modules, {"core.pipeline": fake_pipeline}):
            call_rendered = _run_real_pipeline(input_file, "标准", "", False)
        call_combined = "\n".join(str(item) for item in call_rendered)
        self.assertIn("UI_PIPELINE_FAILED", call_combined)
        _assert_no_sensitive_markers(self, call_combined)

    def test_real_pipeline_smoke_uses_process_audio_contract(self) -> None:
        calls = []
        fake_pipeline = types.ModuleType("core.pipeline")

        with tempfile.TemporaryDirectory() as root_tmp, tempfile.TemporaryDirectory() as stage_tmp:
            root = Path(root_tmp)
            job_root = root / "outputs" / "ui_smoke"
            job_root.mkdir(parents=True)
            original = job_root / "original.wav"
            mixed = job_root / "mixed.wav"
            original.write_bytes(b"original")
            mixed.write_bytes(b"mixed")

            def fake_process_audio(**kwargs):
                calls.append(kwargs)
                return {
                    "job_id": "ui_smoke",
                    "status": "success",
                    "runtime": {"total_seconds": 0.1},
                    "original_audio_path": str(original),
                    "mixed_output_path": str(mixed),
                    "enhanced_audio_path": str(mixed),
                    "warnings": [],
                    "events": [],
                    "config_snapshot": {},
                }

            fake_pipeline.process_audio = fake_process_audio
            input_file = types.SimpleNamespace(name="/tmp/dev_input.wav")
            with mock.patch.dict(sys.modules, {"core.pipeline": fake_pipeline}), mock.patch(
                "ui.presenters.ROOT_DIR", root
            ), mock.patch.dict(os.environ, {"AUDIORESCUE_UI_STAGING_DIR": str(Path(stage_tmp) / "ui-stage")}):
                rendered = _run_real_pipeline(input_file, "轻度", "", True)

        self.assertEqual(len(rendered), len(UI_TUPLE_KEYS))
        self.assertEqual(
            calls,
            [
                {
                    "input_path": "/tmp/dev_input.wav",
                    "strength": 0.5,
                    "enable_events": False,
                    "reference_text": None,
                    "force_recompute": True,
                }
            ],
        )
        self.assertTrue(any("急救完成" in str(item) for item in rendered))

    def test_build_demo_smoke_does_not_expose_fixture_by_default(self) -> None:
        calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []
        events: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

        class FakeComponent:
            def __init__(self, kind: str, *args: object, **kwargs: object) -> None:
                self.kind = kind
                calls.append((kind, args, kwargs))

            def __enter__(self):
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def click(self, *args: object, **kwargs: object):
                events.append(("click", args, kwargs))
                return self

            def load(self, *args: object, **kwargs: object):
                events.append(("load", args, kwargs))
                return self

        fake_gradio = types.SimpleNamespace(
            themes=types.SimpleNamespace(Base=lambda *args, **kwargs: ("BaseTheme", args, kwargs)),
            Blocks=lambda *args, **kwargs: FakeComponent("Blocks", *args, **kwargs),
            Markdown=lambda *args, **kwargs: FakeComponent("Markdown", *args, **kwargs),
            Row=lambda *args, **kwargs: FakeComponent("Row", *args, **kwargs),
            Column=lambda *args, **kwargs: FakeComponent("Column", *args, **kwargs),
            Checkbox=lambda *args, **kwargs: FakeComponent("Checkbox", *args, **kwargs),
            Dropdown=lambda *args, **kwargs: FakeComponent("Dropdown", *args, **kwargs),
            State=lambda *args, **kwargs: FakeComponent("State", *args, **kwargs),
            Audio=lambda *args, **kwargs: FakeComponent("Audio", *args, **kwargs),
            Radio=lambda *args, **kwargs: FakeComponent("Radio", *args, **kwargs),
            Textbox=lambda *args, **kwargs: FakeComponent("Textbox", *args, **kwargs),
            Accordion=lambda *args, **kwargs: FakeComponent("Accordion", *args, **kwargs),
            Button=lambda *args, **kwargs: FakeComponent("Button", *args, **kwargs),
            HTML=lambda *args, **kwargs: FakeComponent("HTML", *args, **kwargs),
            Image=lambda *args, **kwargs: FakeComponent("Image", *args, **kwargs),
            File=lambda *args, **kwargs: FakeComponent("File", *args, **kwargs),
        )

        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.dict(
            sys.modules, {"gradio": fake_gradio}
        ):
            demo = build_demo()

        self.assertEqual(demo.kind, "Blocks")
        blocks_kwargs = calls[0][2]
        self.assertFalse(blocks_kwargs["analytics_enabled"])
        self.assertEqual(blocks_kwargs["theme"][0], "BaseTheme")
        self.assertEqual(blocks_kwargs["theme"][2]["font"][0], "system-ui")
        labels = [str(kwargs.get("label", "")) for _, _, kwargs in calls]
        self.assertNotIn("开发专用：使用前端 fixture 假数据", labels)
        self.assertEqual([kwargs.get("label") for kind, _, kwargs in calls if kind == "Audio"], ["上传音频"])
        self.assertFalse(any(kind == "File" for kind, _, _ in calls))
        self.assertFalse(any(kind == "Image" for kind, _, _ in calls))
        self.assertTrue(any(kind == "State" for kind, _, _ in calls))
        self.assertTrue(any(event == "click" for event, _, _ in events))
        self.assertFalse(any(event == "load" for event, _, _ in events))

    def test_status_component_uses_html_for_semantic_state_markup(self) -> None:
        calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

        class FakeComponent:
            def __init__(self, kind: str, *args: object, **kwargs: object) -> None:
                self.kind = kind
                calls.append((kind, args, kwargs))

            def __enter__(self):
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def click(self, *args: object, **kwargs: object):
                return self

        fake_gradio = types.SimpleNamespace(
            themes=types.SimpleNamespace(Base=lambda *args, **kwargs: ("BaseTheme", args, kwargs)),
            Blocks=lambda *args, **kwargs: FakeComponent("Blocks", *args, **kwargs),
            Markdown=lambda *args, **kwargs: FakeComponent("Markdown", *args, **kwargs),
            Row=lambda *args, **kwargs: FakeComponent("Row", *args, **kwargs),
            Column=lambda *args, **kwargs: FakeComponent("Column", *args, **kwargs),
            Checkbox=lambda *args, **kwargs: FakeComponent("Checkbox", *args, **kwargs),
            Dropdown=lambda *args, **kwargs: FakeComponent("Dropdown", *args, **kwargs),
            State=lambda *args, **kwargs: FakeComponent("State", *args, **kwargs),
            Audio=lambda *args, **kwargs: FakeComponent("Audio", *args, **kwargs),
            Radio=lambda *args, **kwargs: FakeComponent("Radio", *args, **kwargs),
            Textbox=lambda *args, **kwargs: FakeComponent("Textbox", *args, **kwargs),
            Accordion=lambda *args, **kwargs: FakeComponent("Accordion", *args, **kwargs),
            Button=lambda *args, **kwargs: FakeComponent("Button", *args, **kwargs),
            HTML=lambda *args, **kwargs: FakeComponent("HTML", *args, **kwargs),
            Image=lambda *args, **kwargs: FakeComponent("Image", *args, **kwargs),
            File=lambda *args, **kwargs: FakeComponent("File", *args, **kwargs),
        )

        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.dict(
            sys.modules, {"gradio": fake_gradio}
        ):
            build_demo()

        html_classes = [kwargs.get("elem_classes", []) for kind, _, kwargs in calls if kind == "HTML"]
        self.assertTrue(any("ar-status-panel" in classes for classes in html_classes))
        self.assertTrue(any("ar-audio" in classes for classes in html_classes))
        self.assertTrue(any("ar-download" in classes for classes in html_classes))

    def test_direct_layout_launch_kwargs_keep_file_delivery_route(self) -> None:
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

        self.assertEqual(len(middleware), 2)
        self.assertIs(middleware[0].cls, FileDeliveryMiddleware)
        self.assertIs(middleware[1].cls, OfflineHtmlResourceMiddleware)

    def test_offline_html_filter_removes_gradio_remote_defaults(self) -> None:
        html = """
        <html><head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link href="https://fonts.gstatic.com" rel="preconnect" />
        <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Sans+Pro" />
        <script src="https://cdnjs.cloudflare.com/ajax/libs/iframe-resizer/4.3.1/iframeResizer.contentWindow.min.js" async></script>
        </head><body>ok</body></html>
        """
        filtered = strip_remote_html_resources(html)
        self.assertNotIn("fonts.googleapis.com", filtered)
        self.assertNotIn("fonts.gstatic.com", filtered)
        self.assertNotIn("cdnjs.cloudflare.com", filtered)
        self.assertIn("<body>ok</body>", filtered)

    def test_offline_html_middleware_filters_html_response(self) -> None:
        async def app(scope, receive, send):
            body = (
                b'<script src="https://cdnjs.cloudflare.com/ajax/libs/iframe-resizer/4.3.1/'
                b'iframeResizer.contentWindow.min.js"></script><main>ok</main>'
            )
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [
                        (b"content-type", b"text/html; charset=utf-8"),
                        (b"content-length", str(len(body)).encode("ascii")),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": body, "more_body": False})

        sent = []

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            sent.append(message)

        run(OfflineHtmlResourceMiddleware(app)({"type": "http"}, receive, send))
        body = sent[-1]["body"].decode("utf-8")
        self.assertNotIn("cdnjs.cloudflare.com", body)
        self.assertIn("<main>ok</main>", body)
        length_header = dict(sent[0]["headers"])[b"content-length"]
        self.assertEqual(length_header, str(len(sent[-1]["body"])).encode("ascii"))

    def test_dark_theme_readability_contrast(self) -> None:
        css = CSS_PATH.read_text(encoding="utf-8")
        self.assertIn("@media (prefers-color-scheme: dark)", css)
        self.assertIn(".ar-engineering", css)
        self.assertIn(".ar-process-note", css)

        for theme, variables in (("light", _css_variables()), ("dark", _css_variables(dark=True))):
            pairs = [
                ("--ar-ink", "--ar-panel"),
                ("--ar-muted", "--ar-panel"),
                ("--ar-ink", "--ar-process-bg"),
                ("--ar-status-success-text", "--ar-status-success-bg"),
                ("--ar-status-partial-text", "--ar-status-partial-bg"),
                ("--ar-status-failed-text", "--ar-status-failed-bg"),
            ]
            for foreground, background in pairs:
                with self.subTest(theme=theme, foreground=foreground, background=background):
                    self.assertGreaterEqual(
                        _contrast_ratio(variables[foreground], variables[background]),
                        4.5,
                    )


if __name__ == "__main__":
    unittest.main()
