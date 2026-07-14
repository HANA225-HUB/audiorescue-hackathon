from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock
from urllib.parse import unquote

from core.schemas import AudioMeta, ProcessResult, ProcessStatus, RuntimeStats
from ui.presenters import (
    UI_TUPLE_KEYS,
    cer_markdown,
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
        self.assertIn("data-status='success'", view["status_md"])
        self.assertIn("缓存命中：`是`", view["runtime_md"])

    def test_status_markup_has_semantic_state_classes(self) -> None:
        cases = {
            "process_result_ok": ("success", "ar-status-success", "急救完成"),
            "process_result_partial_asr": (
                "partial",
                "ar-status-partial",
                "部分结果不可用",
            ),
            "process_result_failed_enhance": ("failed", "ar-status-failed", "急救未完成"),
        }
        for fixture_name, (status, class_name, title) in cases.items():
            with self.subTest(status=status):
                rendered = result_to_view(load_fixture(fixture_name))["status_md"]
                self.assertIn(f"data-status='{status}'", rendered)
                self.assertIn(class_name, rendered)
                self.assertIn(f"状态：{status}", rendered)
                self.assertIn(title, rendered)

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
        self.assertIn("<dt>增强强度</dt><dd>0.5</dd>", view["status_md"])
        self.assertIn("<dt>Whisper</dt><dd>base</dd>", view["status_md"])

    def test_txt_and_json_downloads_are_placeholders_until_pipeline_contracts_paths(self) -> None:
        result = load_fixture("process_result_ok")
        view = result_to_view(result)
        self.assertIsNone(view["transcript_download"])
        self.assertIsNone(view["result_download"])

    def test_cer_direction_labels_improvement_tie_and_regression(self) -> None:
        result = load_fixture("process_result_ok")
        self.assertIn("（改善）", cer_markdown(result))
        result["cer_after"]["cer"] = result["cer_before"]["cer"]
        self.assertIn("持平", cer_markdown(result))
        result["cer_after"]["cer"] = 0.5
        rendered = cer_markdown(result)
        self.assertIn("（变差）", rendered)
        self.assertNotIn("下降 -", rendered)

    def test_presenter_never_serves_arbitrary_existing_server_files(self) -> None:
        result = load_fixture("process_result_ok")
        result["original_audio_path"] = "/etc/hosts"
        result["mixed_output_path"] = "/etc/hosts"
        view = result_to_view(result)
        self.assertIsNone(view["original_audio"])
        self.assertIsNone(view["enhanced_audio"])
        self.assertNotIn("/etc/hosts", view["playback_note_md"])

    def test_invalid_job_id_fails_closed_without_any_allowed_root(self) -> None:
        result = {
            "status": "success",
            "job_id": "../bad",
            "original_audio_path": "/etc/hosts",
            "mixed_output_path": "/etc/hosts",
        }
        view = result_to_view(result)
        self.assertIsNone(view["original_audio"])
        self.assertIsNone(view["enhanced_audio"])
        self.assertIsNone(view["mixed_download"])

    def test_warning_details_recursively_hide_path_and_trace_fields(self) -> None:
        result = {
            "status": "partial",
            "job_id": "safe_job",
            "warnings": [
                {
                    "code": "ASR_FAILED",
                    "message": "转写不可用",
                    "details": {
                        "path": "/private/input.wav",
                        "nested": {
                            "model_path": "/secret/model.pt",
                            "traceback": "private stack",
                            "attempt": 1,
                        },
                    },
                }
            ],
        }
        rendered = result_to_view(result)["warnings_html"]
        self.assertNotIn("/private/input.wav", rendered)
        self.assertNotIn("/secret/model.pt", rendered)
        self.assertNotIn("private stack", rendered)
        self.assertIn("attempt", rendered)

    def test_playback_note_exposes_both_loudness_tracks_when_available(self) -> None:
        result = load_fixture("process_result_ok")
        result["original_levels"] = {"peak_abs": 0.5, "rms_dbfs": -20.0}
        result["mixed_levels"] = {"peak_abs": 0.48, "rms_dbfs": -20.2}
        rendered = result_to_view(result)["playback_note_md"]
        self.assertIn("原轨", rendered)
        self.assertIn("混合增强轨", rendered)
        self.assertIn("-20.2 dBFS", rendered)

    def test_staged_gradio_paths_hide_source_sentinels_and_preserve_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as root_tmp, tempfile.TemporaryDirectory() as stage_tmp:
            root = Path(root_tmp)
            staging_root = Path(stage_tmp) / "ui-stage"
            job_root = root / "outputs" / "safe_job" / "SECRET_WORKSPACE"
            job_root.mkdir(parents=True)
            original = job_root / "SECRET_USER_original.wav"
            mixed = job_root / "SECRET_USER_mixed.wav"
            full = job_root / "SECRET_USER_full.wav"
            spectrogram = job_root / "SECRET_USER_spectrogram.png"
            waveform = job_root / "SECRET_USER_waveform.png"
            original.write_bytes(b"original-content")
            mixed.write_bytes(b"mixed-content")
            full.write_bytes(b"full-content")
            spectrogram.write_bytes(b"spectrogram-content")
            waveform.write_bytes(b"waveform-content")
            result = {
                "job_id": "safe_job",
                "status": "success",
                "runtime": {"total_seconds": 1.0},
                "input_meta": {
                    "source_name": "SECRET_USER_private_take.wav",
                    "source_format": "wav",
                    "sample_rate": 48000,
                    "channels": 1,
                    "duration_seconds": 1.0,
                    "peak_abs": 0.1,
                    "rms_dbfs": -20.0,
                    "clipped_ratio": 0.0,
                    "silent_ratio": 0.0,
                },
                "original_audio_path": str(original),
                "mixed_output_path": str(mixed),
                "enhanced_audio_path": str(mixed),
                "full_output_path": str(full),
                "spectrogram_path": str(spectrogram),
                "waveform_path": str(waveform),
                "warnings": [],
                "events": [],
                "config_snapshot": {},
            }

            with mock.patch("ui.presenters.ROOT_DIR", root), mock.patch.dict(
                "os.environ", {"AUDIORESCUE_UI_STAGING_DIR": str(staging_root)}
            ):
                view = result_to_view(result)

            self.assertEqual(Path(view["original_audio"]).read_bytes(), b"original-content")
            self.assertEqual(Path(view["enhanced_audio"]).read_bytes(), b"mixed-content")
            self.assertEqual(Path(view["mixed_download"]).read_bytes(), b"mixed-content")
            self.assertEqual(Path(view["full_download"]).read_bytes(), b"full-content")
            self.assertEqual(Path(view["spectrogram_image"]).read_bytes(), b"spectrogram-content")
            self.assertEqual(Path(view["waveform_image"]).read_bytes(), b"waveform-content")

            visible_text = "\n".join(
                str(view[key])
                for key in (
                    "status_md",
                    "input_md",
                    "playback_note_md",
                    "runtime_md",
                    "warnings_html",
                )
            )
            exposed_paths = unquote(
                " ".join(
                    str(view[key])
                    for key in (
                        "original_audio",
                        "enhanced_audio",
                        "mixed_download",
                        "full_download",
                        "spectrogram_image",
                        "waveform_image",
                    )
                )
            )
            for rendered in (visible_text, exposed_paths):
                self.assertNotIn("SECRET_WORKSPACE", rendered)
                self.assertNotIn("SECRET_USER", rendered)
                self.assertNotIn("outputs/safe_job", rendered)
                self.assertNotIn("SECRET_USER_private_take.wav", rendered)
            self.assertIn("已隐藏文件名", visible_text)
            self.assertTrue(Path(view["original_audio"]).is_relative_to(staging_root))

    def test_status_markup_covers_input_error_fixture(self) -> None:
        rendered = result_to_view(load_fixture("process_result_input_error"))["status_md"]
        self.assertIn("data-status='failed'", rendered)
        self.assertIn("ar-status-failed", rendered)
        self.assertIn("急救未完成", rendered)


if __name__ == "__main__":
    unittest.main()
