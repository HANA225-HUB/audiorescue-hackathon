import os
import unittest
from unittest import mock

from ui.layout import _fixture_mode_default, _run


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


if __name__ == "__main__":
    unittest.main()
