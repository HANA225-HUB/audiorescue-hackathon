import ast
import unittest
from pathlib import Path

import core.events as events_module
from core.events import EventDetectionError, detect_events
from core.schemas import EventResult


class EventIsolationTest(unittest.TestCase):
    def test_disabled_returns_empty_without_touching_detector_or_inputs(self) -> None:
        def detector(**_kwargs):
            self.fail("disabled P1 must not invoke its detector")

        result = detect_events(
            "missing-file-is-irrelevant-while-disabled.wav",
            labels=None,
            window_seconds=-1,
            hop_seconds=-1,
            detector=detector,
        )

        self.assertEqual(result, [])

    def test_enabled_uses_injected_detector(self) -> None:
        calls = []
        expected = [
            EventResult(
                label="an emergency siren",
                score=0.91,
                start_seconds=2.0,
                end_seconds=4.0,
            )
        ]

        def detector(**kwargs):
            calls.append(kwargs)
            return expected

        result = detect_events(
            "outputs/job/original.wav",
            [" an emergency siren ", "a car horn honking"],
            enabled=True,
            detector=detector,
        )

        self.assertEqual(result, expected)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["original_wav"], "outputs/job/original.wav")
        self.assertEqual(
            calls[0]["labels"],
            ["an emergency siren", "a car horn honking"],
        )
        self.assertEqual(calls[0]["window_seconds"], 2.0)
        self.assertEqual(calls[0]["hop_seconds"], 1.0)

    def test_enabled_requires_an_injected_detector(self) -> None:
        with self.assertRaisesRegex(EventDetectionError, "detector"):
            detect_events(
                "outputs/job/original.wav",
                ["an emergency siren"],
                enabled=True,
            )

    def test_detector_failure_is_wrapped_for_pipeline(self) -> None:
        original_error = OSError("model weights unavailable")

        def detector(**_kwargs):
            raise original_error

        with self.assertRaisesRegex(EventDetectionError, "EVENTS_SKIPPED") as caught:
            detect_events(
                "outputs/job/original.wav",
                ["an emergency siren"],
                enabled=True,
                detector=detector,
            )

        self.assertIs(caught.exception.__cause__, original_error)

    def test_invalid_detector_output_is_rejected(self) -> None:
        def detector(**_kwargs):
            return [{"label": "not-the-frozen-schema"}]

        with self.assertRaisesRegex(EventDetectionError, "EventResult"):
            detect_events(
                "outputs/job/original.wav",
                ["an emergency siren"],
                enabled=True,
                detector=detector,
            )

    def test_events_module_has_no_third_party_imports(self) -> None:
        source_path = Path(events_module.__file__)
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        imported_roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".", 1)[0])

        self.assertLessEqual(
            imported_roots,
            {"__future__", "collections", "core"},
        )


class EventLabelConfigTest(unittest.TestCase):
    def test_default_config_has_eight_to_twelve_labels(self) -> None:
        config_path = Path(__file__).parents[1] / "configs" / "labels.yaml"
        label_lines = [
            line
            for line in config_path.read_text(encoding="utf-8").splitlines()
            if line.startswith("  - ")
        ]

        self.assertGreaterEqual(len(label_lines), 8)
        self.assertLessEqual(len(label_lines), 12)
        self.assertEqual(len(label_lines), len(set(label_lines)))


if __name__ == "__main__":
    unittest.main()
