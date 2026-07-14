from __future__ import annotations

import unittest

from core.schemas import AudioMeta, ProcessResult, ProcessStatus, RuntimeStats
from ui.presenters import (
    UI_TUPLE_KEYS,
    diff_html,
    list_fixture_names,
    load_fixture,
    result_to_ui_tuple,
    result_to_view,
)


class PresenterTests(unittest.TestCase):
    def test_all_process_result_fixtures_render(self) -> None:
        names = list_fixture_names()
        self.assertGreaterEqual(len(names), 5)
        for name in names:
            with self.subTest(name=name):
                result = load_fixture(name)
                view = result_to_view(result)
                self.assertEqual(set(UI_TUPLE_KEYS), set(view.keys()))
                self.assertEqual(len(result_to_ui_tuple(result)), len(UI_TUPLE_KEYS))

    def test_cache_is_badge_not_status(self) -> None:
        result = load_fixture("process_result_cache")
        view = result_to_view(result)
        self.assertEqual(result["status"], "success")
        self.assertIn("本地缓存", view["status_md"])
        self.assertIn("缓存命中：`是`", view["runtime_md"])

    def test_partial_asr_keeps_available_results(self) -> None:
        result = load_fixture("process_result_partial_asr")
        view = result_to_view(result)
        self.assertIn("部分结果不可用", view["status_md"])
        self.assertIn("今天下午三点", view["transcript_before_md"])
        self.assertIn("未生成", view["transcript_after_md"])
        self.assertIn("ASR_AFTER_FAILED", view["warnings_html"])

    def test_failed_enhance_keeps_original_but_no_after_audio(self) -> None:
        result = load_fixture("process_result_failed_enhance")
        view = result_to_view(result)
        self.assertIn("急救未完成", view["status_md"])
        self.assertIsNotNone(view["original_audio"])
        self.assertIsNone(view["enhanced_audio"])
        self.assertIn("ENHANCE_FAILED", view["warnings_html"])

    def test_no_reference_means_no_cer_number(self) -> None:
        result = load_fixture("process_result_cache")
        view = result_to_view(result)
        self.assertIn("不显示 CER", view["cer_md"])
        self.assertNotIn("原始轨 CER：", view["cer_md"])

    def test_ui_does_not_recompute_missing_diff(self) -> None:
        result = load_fixture("process_result_partial_asr")
        rendered = diff_html(result)
        self.assertIn("UI 不重复计算 diff", rendered)

    def test_unsafe_supplied_diff_html_is_escaped(self) -> None:
        result = load_fixture("process_result_ok")
        result["text_diff_html"] = "<script>alert('x')</script>"
        rendered = diff_html(result)
        self.assertIn("已转义显示", rendered)
        self.assertIn("&lt;script&gt;", rendered)
        self.assertNotIn("<script>alert", rendered)

    def test_presenter_accepts_schema_dataclass_result(self) -> None:
        result = ProcessResult(
            job_id="dataclass_job",
            status=ProcessStatus.SUCCESS,
            runtime=RuntimeStats(total_seconds=1.0, cache_hit=True),
            input_meta=AudioMeta(
                source_name="x.wav",
                sample_rate=48000,
                channels=1,
                duration_seconds=1.0,
                peak_abs=0.5,
                rms_dbfs=None,
                clipped_ratio=0.0,
                silent_ratio=None,
                normalized_path="outputs/dataclass_job/original.wav",
            ),
            mixed_output_path="outputs/dataclass_job/enhanced_mix.wav",
        )
        view = result_to_view(result)
        self.assertIn("本地缓存", view["status_md"])
        self.assertIn("未记录", view["input_md"])

    def test_runtime_uses_frozen_contract_fields(self) -> None:
        result = load_fixture("process_result_ok")
        view = result_to_view(result)
        self.assertIn("增强模型加载", view["runtime_md"])
        self.assertIn("ASR 模型加载", view["runtime_md"])
        self.assertIn("结果持久化", view["runtime_md"])
        self.assertIn("冷启动", view["runtime_md"])

    def test_status_reads_pipeline_config_snapshot_shape(self) -> None:
        result = load_fixture("process_result_ok")
        result["config_snapshot"] = {
            "contract_version": "v0.1-contract",
            "enhancement": {"default_strength": 0.75},
            "asr": {"model": "base"},
            "request": {"strength": 0.5},
            "pipeline_version": "pipeline-v0.1.0",
        }
        view = result_to_view(result)
        self.assertIn("增强强度：`0.5`", view["status_md"])
        self.assertIn("Whisper：`base`", view["status_md"])

    def test_txt_and_json_downloads_are_placeholders_until_pipeline_contracts_paths(self) -> None:
        result = load_fixture("process_result_ok")
        view = result_to_view(result)
        self.assertIsNone(view["transcript_download"])
        self.assertIsNone(view["result_download"])


if __name__ == "__main__":
    unittest.main()
