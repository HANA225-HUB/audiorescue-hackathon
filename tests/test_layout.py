import os
import sys
import types
import unittest
from unittest import mock

from ui.presenters import UI_TUPLE_KEYS
from ui.layout import _fixture_mode_default, _run, _run_real_pipeline, build_demo


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
        self.assertTrue(any("正式模式禁止" in str(item) for item in rendered))

    def test_real_pipeline_smoke_uses_process_audio_contract(self) -> None:
        calls = []
        fake_pipeline = types.ModuleType("core.pipeline")

        def fake_process_audio(**kwargs):
            calls.append(kwargs)
            return {
                "job_id": "ui_smoke",
                "status": "success",
                "runtime": {"total_seconds": 0.1},
                "warnings": [],
                "events": [],
                "config_snapshot": {},
            }

        fake_pipeline.process_audio = fake_process_audio
        input_file = types.SimpleNamespace(name="/tmp/dev_input.wav")
        with mock.patch.dict(sys.modules, {"core.pipeline": fake_pipeline}):
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
        labels = [str(kwargs.get("label", "")) for _, _, kwargs in calls]
        self.assertNotIn("开发专用：使用前端 fixture 假数据", labels)
        self.assertTrue(any(kind == "State" for kind, _, _ in calls))
        self.assertTrue(any(event == "click" for event, _, _ in events))
        self.assertFalse(any(event == "load" for event, _, _ in events))


if __name__ == "__main__":
    unittest.main()
