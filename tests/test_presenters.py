from __future__ import annotations

import tempfile
import binascii
import os
import re
import struct
import unittest
import wave
import zlib
from pathlib import Path
from unittest import mock
from urllib.parse import quote, unquote

from core.schemas import AudioMeta, ProcessResult, ProcessStatus, RuntimeStats
from ui.file_delivery import (
    clear_delivery_registry,
    lookup_delivery_entry,
    register_files_for_delivery as _register_files_for_delivery,
)
from ui.file_delivery import unregister_delivery_url as _unregister_delivery_url
from ui.presenters import (
    UI_TUPLE_KEYS,
    cer_markdown,
    diff_html,
    list_fixture_names,
    load_fixture,
    result_to_ui_tuple,
    result_to_view,
)

DELIVERY_URL_RE = re.compile(r"/audiorescue-files/[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+")
SENSITIVE_MARKERS = (
    "SECRET_MARKER",
    "/private/audio/input.wav",
    "C:\\Users\\secret\\input.wav",
    "https://example.test/model.pt?q=SECRET_MARKER",
    "今天下午三点我们讨论保密项目",
)

_DEFAULT = object()


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


def _write_zero_frame_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(48000)


def _valid_png_bytes() -> bytes:
    def chunk(kind: bytes, payload: bytes) -> bytes:
        crc = binascii.crc32(kind + payload) & 0xFFFFFFFF
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0)
    idat = zlib.compress(b"\x00\x00\x00\x00\x00")
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


def _valid_transcript(text: str = "ok") -> dict[str, object]:
    return {
        "text": text,
        "language": "zh",
        "segments": [],
        "runtime_seconds": 0.1,
        "model_name": "whisper-base",
        "error": None,
    }


def _write_visual_pair(root: Path) -> tuple[str, str]:
    spectrogram = root / "spectrogram.png"
    waveform = root / "waveform.png"
    spectrogram.write_bytes(_valid_png_bytes())
    waveform.write_bytes(_valid_png_bytes())
    return str(spectrogram), str(waveform)


def _delivery_urls(fragment: object) -> list[str]:
    return DELIVERY_URL_RE.findall(str(fragment))


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


def _base_delivery_result(
    root: Path,
    *,
    status: str = "success",
    original_path: str | None = None,
    mixed_path: str | None = None,
    full_path: str | None = None,
    spectrogram_path: str | None = None,
    waveform_path: str | None = None,
    transcript_before: object = _DEFAULT,
    transcript_after: object = _DEFAULT,
) -> dict[str, object]:
    return {
        "job_id": "safe_job",
        "status": status,
        "runtime": {"total_seconds": 1.0},
        "input_meta": {
            "source_name": "SECRET_MARKER_source.wav",
            "source_format": "wav",
            "sample_rate": 48000,
            "channels": 1,
            "duration_seconds": 1.0,
            "peak_abs": 0.1,
            "rms_dbfs": -20.0,
            "clipped_ratio": 0.0,
            "silent_ratio": 0.0,
        },
        "original_audio_path": original_path,
        "mixed_output_path": mixed_path,
        "enhanced_audio_path": mixed_path,
        "full_output_path": full_path,
        "transcript_before": _valid_transcript("before")
        if transcript_before is _DEFAULT
        else transcript_before,
        "transcript_after": _valid_transcript("after")
        if transcript_after is _DEFAULT
        else transcript_after,
        "spectrogram_path": spectrogram_path,
        "waveform_path": waveform_path,
        "warnings": [],
        "events": [],
        "config_snapshot": {},
    }


class PresenterTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_delivery_registry()

    def tearDown(self) -> None:
        clear_delivery_registry()

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
        self.assertIn("data-status='partial'", view["status_md"])
        self.assertIn("缓存命中：`是`", view["runtime_md"])

    def test_status_markup_has_semantic_state_classes(self) -> None:
        cases = {
            "process_result_ok": (
                "partial",
                "ar-status-partial",
                "部分结果不可用",
            ),
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

    def test_complete_success_keeps_success_markup(self) -> None:
        with tempfile.TemporaryDirectory() as root_tmp, tempfile.TemporaryDirectory() as stage_tmp:
            root = Path(root_tmp)
            staging_root = Path(stage_tmp) / "ui-stage"
            job_root = root / "outputs" / "safe_job"
            job_root.mkdir(parents=True)
            original = job_root / "original.wav"
            mixed = job_root / "mixed.wav"
            _write_pcm_wav(original)
            _write_pcm_wav(mixed, b"\x02\x00\x03\x00")
            spectrogram_path, waveform_path = _write_visual_pair(job_root)
            result = _base_delivery_result(
                root,
                original_path=str(original),
                mixed_path=str(mixed),
                spectrogram_path=spectrogram_path,
                waveform_path=waveform_path,
            )
            with mock.patch("ui.presenters.ROOT_DIR", root), mock.patch.dict(
                "os.environ", {"AUDIORESCUE_UI_STAGING_DIR": str(staging_root)}
            ):
                view = result_to_view(result)

        self.assertIn("data-status='success'", view["status_md"])
        self.assertIn("ar-status-success", view["status_md"])
        self.assertIn("急救完成", view["status_md"])
        self.assertNotIn("UI_RESULT_INCOMPLETE", view["warnings_html"])

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
        self.assertIn("<audio", view["original_audio"])
        self.assertIn("未生成", view["enhanced_audio"])
        self.assertNotIn("<audio", view["enhanced_audio"])
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
        self.assertIn("未生成", view["transcript_download"])
        self.assertIn("未生成", view["result_download"])
        self.assertEqual(_delivery_urls(view["transcript_download"]), [])
        self.assertEqual(_delivery_urls(view["result_download"]), [])

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
        self.assertIn("未生成", view["original_audio"])
        self.assertIn("未生成", view["enhanced_audio"])
        self.assertNotIn("/etc/hosts", view["original_audio"])
        self.assertNotIn("/etc/hosts", view["enhanced_audio"])
        self.assertNotIn("/etc/hosts", view["playback_note_md"])

    def test_invalid_job_id_fails_closed_without_any_allowed_root(self) -> None:
        result = {
            "status": "success",
            "job_id": "../bad",
            "original_audio_path": "/etc/hosts",
            "mixed_output_path": "/etc/hosts",
        }
        view = result_to_view(result)
        self.assertIn("未生成", view["original_audio"])
        self.assertIn("未生成", view["enhanced_audio"])
        self.assertIn("未生成", view["mixed_download"])

    def test_warning_rendering_uses_fixed_messages_and_allowlisted_details(self) -> None:
        result = {
            "status": "partial",
            "job_id": "safe_job",
            "warnings": [
                {
                    "code": "ASR_AFTER_FAILED",
                    "message": "转写不可用 SECRET_MARKER /private/audio/input.wav",
                    "module": "transcribe",
                    "recoverable": True,
                    "details": {
                        "attempt": 2,
                        "track": "after",
                        "note": "SECRET_MARKER",
                        "url": "https://example.test/model.pt?q=SECRET_MARKER",
                        "nested": {
                            "traceback": "private stack SECRET_MARKER",
                            "path": "/private/audio/input.wav",
                            "encoded": quote("/private/audio/input.wav", safe=""),
                            "double_encoded": quote(
                                quote("C:\\Users\\secret\\input.wav", safe=""),
                                safe="",
                            ),
                        },
                        "list": [
                            {"path": quote("/private/audio/input.wav", safe="")},
                            quote(quote("/private/audio/input.wav", safe=""), safe=""),
                        ],
                    },
                },
                {
                    "code": "SECRET_MARKER_UNKNOWN",
                    "message": "C:\\Users\\secret\\input.wav",
                    "module": "C:\\Users\\secret\\module.py",
                    "recoverable": False,
                    "details": {"value": "今天下午三点我们讨论保密项目"},
                },
            ],
        }
        rendered = result_to_view(result)["warnings_html"]
        self.assertIn("ASR_AFTER_FAILED", rendered)
        self.assertIn("UNKNOWN", rendered)
        self.assertIn("增强后转写不可用", rendered)
        self.assertIn("attempt", rendered)
        self.assertIn("after", rendered)
        self.assertNotIn("SECRET_MARKER_UNKNOWN", rendered)
        self.assertNotIn("C:\\Users\\secret\\module.py", rendered)
        self.assertIn("attempt", rendered)
        _assert_no_sensitive_markers(self, rendered)

    def test_transcript_error_text_is_not_displayed_raw(self) -> None:
        result = load_fixture("process_result_ok")
        result["transcript_after"] = {
            "text": "",
            "language": "zh",
            "segments": [],
            "runtime_seconds": 0.1,
            "model_name": "base",
            "error": "SECRET_MARKER /private/audio/input.wav",
        }
        rendered = result_to_view(result)["transcript_after_md"]
        self.assertIn("转写不可用：请查看警告状态。", rendered)
        _assert_no_sensitive_markers(self, rendered)

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
            _write_pcm_wav(original, b"\x10\x00\x11\x00")
            _write_pcm_wav(mixed, b"\x20\x00\x21\x00")
            _write_pcm_wav(full, b"\x30\x00\x31\x00")
            spectrogram.write_bytes(_valid_png_bytes())
            waveform.write_bytes(_valid_png_bytes())
            original_bytes = original.read_bytes()
            mixed_bytes = mixed.read_bytes()
            full_bytes = full.read_bytes()
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
                "transcript_before": _valid_transcript("before"),
                "transcript_after": _valid_transcript("after"),
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

            expected_bytes = {
                "original_audio": original_bytes,
                "enhanced_audio": mixed_bytes,
                "mixed_download": mixed_bytes,
                "full_download": full_bytes,
                "spectrogram_image": _valid_png_bytes(),
                "waveform_image": _valid_png_bytes(),
            }
            for key, expected in expected_bytes.items():
                with self.subTest(key=key):
                    urls = _delivery_urls(view[key])
                    self.assertEqual(len(set(urls)), 1)
                    entry = lookup_delivery_entry(urls[0])
                    self.assertIsNotNone(entry)
                    self.assertEqual(entry.path.read_bytes(), expected)

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
            client_html = unquote(
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
            for rendered in (visible_text, client_html):
                self.assertNotIn("SECRET_WORKSPACE", rendered)
                self.assertNotIn("SECRET_USER", rendered)
                self.assertNotIn("outputs/safe_job", rendered)
                self.assertNotIn("SECRET_USER_private_take.wav", rendered)
                self.assertNotIn(str(root), rendered)
                self.assertNotIn(str(staging_root), rendered)
                self.assertNotIn("/gradio_api/file=", rendered)
                self.assertNotIn("%2F", rendered)
                self.assertNotIn("\\", rendered)
            self.assertIn("已隐藏文件名", visible_text)
            self.assertTrue(all(url.startswith("/audiorescue-files/") for url in _delivery_urls(client_html)))

    def test_required_delivery_failures_downgrade_presentation_status_only(self) -> None:
        with tempfile.TemporaryDirectory() as root_tmp, tempfile.TemporaryDirectory() as stage_tmp:
            root = Path(root_tmp)
            staging_root = Path(stage_tmp) / "ui-stage"
            job_root = root / "outputs" / "safe_job" / "SECRET_MARKER"
            job_root.mkdir(parents=True)
            original = job_root / "SECRET_MARKER_original.wav"
            mixed = job_root / "SECRET_MARKER_mixed.wav"
            full = job_root / "SECRET_MARKER_full.wav"
            _write_pcm_wav(original)
            _write_pcm_wav(mixed, b"\x02\x00\x03\x00")
            _write_pcm_wav(full, b"\x04\x00\x05\x00")
            spectrogram_path, waveform_path = _write_visual_pair(job_root)

            cases = [
                ("original_missing", None, str(mixed), str(full), "partial"),
                ("mixed_missing", str(original), None, str(full), "partial"),
                ("both_missing", None, None, str(full), "failed"),
                (
                    "full_missing",
                    str(original),
                    str(mixed),
                    None,
                    "success",
                ),
            ]
            for name, original_path, mixed_path, full_path, expected_status in cases:
                with self.subTest(name=name), mock.patch("ui.presenters.ROOT_DIR", root), mock.patch.dict(
                    "os.environ", {"AUDIORESCUE_UI_STAGING_DIR": str(staging_root)}
                ):
                    result = _base_delivery_result(
                        root,
                        original_path=original_path,
                        mixed_path=mixed_path,
                        full_path=full_path,
                        spectrogram_path=spectrogram_path,
                        waveform_path=waveform_path,
                    )
                    original_warnings = result["warnings"]
                    view = result_to_view(result)

                self.assertEqual(result["status"], "success")
                self.assertIs(result["warnings"], original_warnings)
                self.assertEqual(result["warnings"], [])
                self.assertIn(f"data-status='{expected_status}'", view["status_md"])
                if expected_status == "success":
                    self.assertIn("UI_FILE_DELIVERY_OPTIONAL", view["warnings_html"])
                    self.assertNotIn("UI_FILE_DELIVERY_FAILED", view["warnings_html"])
                else:
                    self.assertIn("UI_FILE_DELIVERY_FAILED", view["warnings_html"])
                if original_path is None:
                    self.assertNotIn("<audio", view["original_audio"])
                else:
                    self.assertIn("<audio", view["original_audio"])
                if mixed_path is None:
                    self.assertNotIn("<audio", view["enhanced_audio"])
                    self.assertNotIn("下载 mixed.wav", view["mixed_download"])
                else:
                    self.assertIn("<audio", view["enhanced_audio"])
                    self.assertIn("下载 mixed.wav", view["mixed_download"])
                if name == "full_missing":
                    self.assertNotIn("<audio", view["full_download"])
                _assert_no_sensitive_markers(self, "\n".join(str(value) for value in view.values()))

    def test_invalid_wav_tracks_have_no_url_and_follow_required_status_rules(self) -> None:
        with tempfile.TemporaryDirectory() as root_tmp, tempfile.TemporaryDirectory() as stage_tmp:
            root = Path(root_tmp)
            staging_root = Path(stage_tmp) / "ui-stage"
            job_root = root / "outputs" / "safe_job"
            job_root.mkdir(parents=True)
            original = job_root / "original.wav"
            mixed = job_root / "mixed.wav"
            full = job_root / "full.wav"
            invalid = job_root / "invalid.wav"
            _write_pcm_wav(original)
            _write_pcm_wav(mixed, b"\x02\x00\x03\x00")
            _write_pcm_wav(full, b"\x04\x00\x05\x00")
            invalid.write_bytes(b"not a wav")
            spectrogram_path, waveform_path = _write_visual_pair(job_root)

            cases = [
                ("original_invalid", str(invalid), str(mixed), str(full), "partial"),
                ("mixed_invalid", str(original), str(invalid), str(full), "partial"),
                ("both_invalid", str(invalid), str(invalid), str(full), "failed"),
                ("full_invalid", str(original), str(mixed), str(invalid), "success"),
            ]
            for name, original_path, mixed_path, full_path, expected_status in cases:
                with self.subTest(name=name), mock.patch("ui.presenters.ROOT_DIR", root), mock.patch.dict(
                    "os.environ", {"AUDIORESCUE_UI_STAGING_DIR": str(staging_root)}
                ):
                    result = _base_delivery_result(
                        root,
                        original_path=original_path,
                        mixed_path=mixed_path,
                        full_path=full_path,
                        spectrogram_path=spectrogram_path,
                        waveform_path=waveform_path,
                    )
                    original_warnings = result["warnings"]
                    view = result_to_view(result)

                self.assertEqual(result["status"], "success")
                self.assertIs(result["warnings"], original_warnings)
                self.assertEqual(result["warnings"], [])
                self.assertIn(f"data-status='{expected_status}'", view["status_md"])
                self.assertIn("UI_WAV_INVALID", view["warnings_html"])
                if "original" in name or name == "both_invalid":
                    self.assertNotIn("<audio", view["original_audio"])
                else:
                    self.assertIn("<audio", view["original_audio"])
                if "mixed" in name or name == "both_invalid":
                    self.assertNotIn("<audio", view["enhanced_audio"])
                    self.assertNotIn("下载 mixed.wav", view["mixed_download"])
                else:
                    self.assertIn("<audio", view["enhanced_audio"])
                    self.assertIn("下载 mixed.wav", view["mixed_download"])
                if name == "full_invalid":
                    self.assertNotIn("<audio", view["full_download"])
                _assert_no_sensitive_markers(self, "\n".join(str(value) for value in view.values()))

    def test_success_result_incomplete_downgrades_to_partial_without_mutating_raw_result(self) -> None:
        with tempfile.TemporaryDirectory() as root_tmp, tempfile.TemporaryDirectory() as stage_tmp:
            root = Path(root_tmp)
            staging_root = Path(stage_tmp) / "ui-stage"
            job_root = root / "outputs" / "safe_job"
            job_root.mkdir(parents=True)
            original = job_root / "original.wav"
            mixed = job_root / "mixed.wav"
            _write_pcm_wav(original)
            _write_pcm_wav(mixed, b"\x02\x00\x03\x00")
            spectrogram_path, waveform_path = _write_visual_pair(job_root)
            malformed_transcript = {
                "text": "bad",
                "language": "zh",
                "segments": "not-list",
                "runtime_seconds": 0.1,
                "model_name": "whisper-base",
                "error": None,
            }
            cases = [
                {"name": "missing_before", "transcript_before": None},
                {"name": "missing_after", "transcript_after": None},
                {"name": "missing_both", "transcript_before": None, "transcript_after": None},
                {"name": "malformed_after", "transcript_after": malformed_transcript},
                {"name": "missing_waveform", "waveform_path": None},
                {"name": "missing_spectrogram", "spectrogram_path": None},
            ]
            for case in cases:
                with self.subTest(name=case["name"]), mock.patch("ui.presenters.ROOT_DIR", root), mock.patch.dict(
                    "os.environ", {"AUDIORESCUE_UI_STAGING_DIR": str(staging_root)}
                ):
                    result = _base_delivery_result(
                        root,
                        original_path=str(original),
                        mixed_path=str(mixed),
                        spectrogram_path=case.get("spectrogram_path", spectrogram_path),
                        waveform_path=case.get("waveform_path", waveform_path),
                        transcript_before=case.get("transcript_before", _DEFAULT),
                        transcript_after=case.get("transcript_after", _DEFAULT),
                    )
                    original_warnings = result["warnings"]
                    view = result_to_view(result)

                self.assertEqual(result["status"], "success")
                self.assertIs(result["warnings"], original_warnings)
                self.assertEqual(result["warnings"], [])
                self.assertIn("data-status='partial'", view["status_md"])
                self.assertIn("UI_RESULT_INCOMPLETE", view["warnings_html"])
                self.assertNotIn("UI_FILE_DELIVERY_FAILED", view["warnings_html"])
                _assert_no_sensitive_markers(self, "\n".join(str(value) for value in view.values()))

    def test_malformed_transcript_contract_downgrades_success_to_partial(self) -> None:
        with tempfile.TemporaryDirectory() as root_tmp, tempfile.TemporaryDirectory() as stage_tmp:
            root = Path(root_tmp)
            staging_root = Path(stage_tmp) / "ui-stage"
            job_root = root / "outputs" / "safe_job"
            job_root.mkdir(parents=True)
            original = job_root / "original.wav"
            mixed = job_root / "mixed.wav"
            _write_pcm_wav(original)
            _write_pcm_wav(mixed, b"\x02\x00\x03\x00")
            spectrogram_path, waveform_path = _write_visual_pair(job_root)
            segment = {"start": 0.0, "end": 0.1, "text": "ok"}
            cases = {
                "runtime_string": {**_valid_transcript("after"), "runtime_seconds": "0.1"},
                "runtime_negative": {**_valid_transcript("after"), "runtime_seconds": -0.1},
                "runtime_nan": {**_valid_transcript("after"), "runtime_seconds": float("nan")},
                "runtime_inf": {**_valid_transcript("after"), "runtime_seconds": float("inf")},
                "runtime_bool": {**_valid_transcript("after"), "runtime_seconds": True},
                "runtime_huge_int": {
                    **_valid_transcript("after"),
                    "runtime_seconds": 10**10000,
                },
                "top_level_extra": {**_valid_transcript("after"), "confidence": 0.9},
                "top_level_missing_key": {
                    key: value
                    for key, value in _valid_transcript("after").items()
                    if key != "language"
                },
                "segment_string": {**_valid_transcript("after"), "segments": [{**segment, "start": "0"}]},
                "segment_negative": {**_valid_transcript("after"), "segments": [{**segment, "start": -0.1}]},
                "segment_huge_start": {
                    **_valid_transcript("after"),
                    "segments": [{**segment, "start": 10**10000}],
                },
                "segment_huge_end": {
                    **_valid_transcript("after"),
                    "segments": [{**segment, "end": 10**10000}],
                },
                "segment_end_before_start": {
                    **_valid_transcript("after"),
                    "segments": [{**segment, "start": 0.2, "end": 0.1}],
                },
                "segment_extra_field": {
                    **_valid_transcript("after"),
                    "segments": [{**segment, "confidence": 0.9}],
                },
                "empty_error_string": {**_valid_transcript("after"), "error": ""},
            }
            for name, transcript_after in cases.items():
                with self.subTest(name=name), mock.patch("ui.presenters.ROOT_DIR", root), mock.patch.dict(
                    "os.environ", {"AUDIORESCUE_UI_STAGING_DIR": str(staging_root)}
                ):
                    result = _base_delivery_result(
                        root,
                        original_path=str(original),
                        mixed_path=str(mixed),
                        spectrogram_path=spectrogram_path,
                        waveform_path=waveform_path,
                        transcript_after=transcript_after,
                    )
                    original_warnings = result["warnings"]
                    view = result_to_view(result)

                self.assertEqual(result["status"], "success")
                self.assertIs(result["transcript_after"], transcript_after)
                self.assertIs(result["warnings"], original_warnings)
                self.assertEqual(result["warnings"], [])
                self.assertIn("data-status='partial'", view["status_md"])
                self.assertIn("UI_RESULT_INCOMPLETE", view["warnings_html"])
                self.assertIn("transcript_missing", view["warnings_html"])

    def test_visual_registration_failure_downgrades_success_to_partial(self) -> None:
        with tempfile.TemporaryDirectory() as root_tmp, tempfile.TemporaryDirectory() as stage_tmp:
            root = Path(root_tmp)
            staging_root = Path(stage_tmp) / "ui-stage"
            job_root = root / "outputs" / "safe_job"
            job_root.mkdir(parents=True)
            original = job_root / "original.wav"
            mixed = job_root / "mixed.wav"
            _write_pcm_wav(original)
            _write_pcm_wav(mixed, b"\x02\x00\x03\x00")
            spectrogram_path, waveform_path = _write_visual_pair(job_root)
            result = _base_delivery_result(
                root,
                original_path=str(original),
                mixed_path=str(mixed),
                spectrogram_path=spectrogram_path,
                waveform_path=waveform_path,
            )

            def fake_register_files(*args, **kwargs):
                urls = _register_files_for_delivery(*args, **kwargs)
                urls["spectrogram_image"] = None
                return urls

            with mock.patch("ui.presenters.ROOT_DIR", root), mock.patch.dict(
                "os.environ", {"AUDIORESCUE_UI_STAGING_DIR": str(staging_root)}
            ), mock.patch("ui.presenters.register_files_for_delivery", side_effect=fake_register_files):
                view = result_to_view(result)

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["warnings"], [])
        self.assertIn("data-status='partial'", view["status_md"])
        self.assertIn("UI_RESULT_INCOMPLETE", view["warnings_html"])
        self.assertIn("visual_delivery_failed", view["warnings_html"])

    def test_invalid_visual_effective_url_is_cleared_and_token_unregistered(self) -> None:
        with tempfile.TemporaryDirectory() as root_tmp, tempfile.TemporaryDirectory() as stage_tmp:
            root = Path(root_tmp)
            staging_root = Path(stage_tmp) / "ui-stage"
            job_root = root / "outputs" / "safe_job"
            job_root.mkdir(parents=True)
            original = job_root / "original.wav"
            mixed = job_root / "mixed.wav"
            spectrogram = job_root / "spectrogram.png"
            waveform = job_root / "waveform.png"
            _write_pcm_wav(original)
            _write_pcm_wav(mixed, b"\x02\x00\x03\x00")
            spectrogram.write_bytes(_valid_png_bytes())
            waveform.write_bytes(_valid_png_bytes())
            result = _base_delivery_result(
                root,
                original_path=str(original),
                mixed_path=str(mixed),
                spectrogram_path=str(spectrogram),
                waveform_path=str(waveform),
            )

            def validate_with_invalid_waveform(path: Path, *, kind: str | None) -> bool:
                if path.name == "waveform.png":
                    return False
                from ui.media_validation import validate_media_path

                return validate_media_path(path, kind=kind)

            with mock.patch("ui.presenters.ROOT_DIR", root), mock.patch.dict(
                "os.environ", {"AUDIORESCUE_UI_STAGING_DIR": str(staging_root)}
            ), mock.patch(
                "ui.presenters.validate_media_path",
                side_effect=validate_with_invalid_waveform,
            ), mock.patch(
                "ui.presenters.unregister_delivery_url",
                wraps=_unregister_delivery_url,
            ) as unregister:
                view = result_to_view(result)

        self.assertIn("data-status='partial'", view["status_md"])
        self.assertIn("UI_RESULT_INCOMPLETE", view["warnings_html"])
        self.assertIn("UI_MEDIA_INVALID", view["warnings_html"])
        self.assertIn("未生成", view["waveform_image"])
        self.assertEqual(_delivery_urls(view["waveform_image"]), [])
        unregister.assert_called()
        removed_url = unregister.call_args.args[0]
        self.assertTrue(str(removed_url).endswith("/waveform.png"))
        self.assertIsNone(lookup_delivery_entry(str(removed_url)))

    def test_registered_audio_race_before_effective_url_downgrades_and_unregisters(self) -> None:
        for mode in ("rewrite_same_metadata", "replace_legal_media"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as root_tmp, tempfile.TemporaryDirectory() as stage_tmp:
                root = Path(root_tmp)
                staging_root = Path(stage_tmp) / "ui-stage"
                job_root = root / "outputs" / "safe_job"
                job_root.mkdir(parents=True)
                original = job_root / "original.wav"
                mixed = job_root / "mixed.wav"
                _write_pcm_wav(original)
                _write_pcm_wav(mixed, b"\x02\x00\x03\x00")
                spectrogram_path, waveform_path = _write_visual_pair(job_root)
                result = _base_delivery_result(
                    root,
                    original_path=str(original),
                    mixed_path=str(mixed),
                    spectrogram_path=spectrogram_path,
                    waveform_path=waveform_path,
                )
                tampered_urls: list[str] = []

                def register_then_tamper(*args, **kwargs):
                    urls = _register_files_for_delivery(*args, **kwargs)
                    for role in ("original_audio", "mixed_audio"):
                        url = urls.get(role)
                        entry = lookup_delivery_entry(str(url)) if url else None
                        if entry is None:
                            continue
                        tampered_urls.append(str(url))
                        if mode == "rewrite_same_metadata":
                            _write_pcm_wav(entry.path, b"\x10\x00\x11\x00")
                            os.utime(entry.path, ns=(entry.mtime_ns, entry.mtime_ns))
                        else:
                            replacement = entry.path.with_name(f"{entry.path.stem}-replacement.wav")
                            _write_pcm_wav(replacement, b"\x12\x00\x13\x00")
                            os.replace(replacement, entry.path)
                    return urls

                with mock.patch("ui.presenters.ROOT_DIR", root), mock.patch.dict(
                    "os.environ", {"AUDIORESCUE_UI_STAGING_DIR": str(staging_root)}
                ), mock.patch(
                    "ui.presenters.register_files_for_delivery",
                    side_effect=register_then_tamper,
                ):
                    view = result_to_view(result)

                self.assertEqual(result["status"], "success")
                self.assertEqual(result["warnings"], [])
                self.assertIn("data-status='failed'", view["status_md"])
                self.assertIn("UI_WAV_INVALID", view["warnings_html"])
                self.assertNotIn("<audio", view["original_audio"])
                self.assertNotIn("<audio", view["enhanced_audio"])
                self.assertNotIn("下载 mixed.wav", view["mixed_download"])
                self.assertEqual(_delivery_urls(view["original_audio"]), [])
                self.assertEqual(_delivery_urls(view["enhanced_audio"]), [])
                self.assertEqual(_delivery_urls(view["mixed_download"]), [])
                self.assertGreaterEqual(len(tampered_urls), 2)
                for url in tampered_urls:
                    self.assertIsNone(lookup_delivery_entry(url))

    def test_failed_core_status_is_never_upgraded_by_available_files(self) -> None:
        with tempfile.TemporaryDirectory() as root_tmp, tempfile.TemporaryDirectory() as stage_tmp:
            root = Path(root_tmp)
            staging_root = Path(stage_tmp) / "ui-stage"
            job_root = root / "outputs" / "safe_job"
            job_root.mkdir(parents=True)
            original = job_root / "original.wav"
            mixed = job_root / "mixed.wav"
            _write_pcm_wav(original)
            _write_pcm_wav(mixed, b"\x02\x00\x03\x00")
            result = _base_delivery_result(
                root,
                status="failed",
                original_path=str(original),
                mixed_path=str(mixed),
            )
            with mock.patch("ui.presenters.ROOT_DIR", root), mock.patch.dict(
                "os.environ", {"AUDIORESCUE_UI_STAGING_DIR": str(staging_root)}
            ):
                view = result_to_view(result)

        self.assertIn("data-status='failed'", view["status_md"])
        self.assertIn("<audio", view["original_audio"])
        self.assertIn("<audio", view["enhanced_audio"])

    def test_partial_core_status_is_never_upgraded_by_complete_files(self) -> None:
        with tempfile.TemporaryDirectory() as root_tmp, tempfile.TemporaryDirectory() as stage_tmp:
            root = Path(root_tmp)
            staging_root = Path(stage_tmp) / "ui-stage"
            job_root = root / "outputs" / "safe_job"
            job_root.mkdir(parents=True)
            original = job_root / "original.wav"
            mixed = job_root / "mixed.wav"
            _write_pcm_wav(original)
            _write_pcm_wav(mixed, b"\x02\x00\x03\x00")
            spectrogram_path, waveform_path = _write_visual_pair(job_root)
            result = _base_delivery_result(
                root,
                status="partial",
                original_path=str(original),
                mixed_path=str(mixed),
                spectrogram_path=spectrogram_path,
                waveform_path=waveform_path,
            )
            with mock.patch("ui.presenters.ROOT_DIR", root), mock.patch.dict(
                "os.environ", {"AUDIORESCUE_UI_STAGING_DIR": str(staging_root)}
            ):
                view = result_to_view(result)

        self.assertEqual(result["status"], "partial")
        self.assertIn("data-status='partial'", view["status_md"])
        self.assertNotIn("UI_RESULT_INCOMPLETE", view["warnings_html"])

    def test_staging_or_registration_exceptions_fail_closed_without_leaking_text(self) -> None:
        with tempfile.TemporaryDirectory() as root_tmp:
            root = Path(root_tmp)
            job_root = root / "outputs" / "safe_job"
            job_root.mkdir(parents=True)
            original = job_root / "original.wav"
            mixed = job_root / "mixed.wav"
            _write_pcm_wav(original)
            _write_pcm_wav(mixed, b"\x02\x00\x03\x00")
            result = _base_delivery_result(
                root,
                original_path=str(original),
                mixed_path=str(mixed),
            )

            patches = [
                mock.patch(
                    "ui.presenters.stage_files_for_gradio",
                    side_effect=OSError("SECRET_MARKER /private/audio/input.wav"),
                ),
                mock.patch(
                    "ui.presenters.register_files_for_delivery",
                    side_effect=RuntimeError("SECRET_MARKER C:\\Users\\secret\\input.wav"),
                ),
            ]
            for patcher in patches:
                with self.subTest(patcher=str(patcher)), mock.patch(
                    "ui.presenters.ROOT_DIR", root
                ), patcher:
                    view = result_to_view(result)
                self.assertIn("data-status='failed'", view["status_md"])
                self.assertIn("UI_FILE_DELIVERY_FAILED", view["warnings_html"])
                self.assertNotIn("<audio", view["original_audio"])
                self.assertNotIn("<audio", view["enhanced_audio"])
                _assert_no_sensitive_markers(self, "\n".join(str(value) for value in view.values()))

    def test_status_markup_covers_input_error_fixture(self) -> None:
        rendered = result_to_view(load_fixture("process_result_input_error"))["status_md"]
        self.assertIn("data-status='failed'", rendered)
        self.assertIn("ar-status-failed", rendered)
        self.assertIn("急救未完成", rendered)

    def test_many_sensitive_names_do_not_reach_visible_text_or_staged_paths(self) -> None:
        with tempfile.TemporaryDirectory() as root_tmp, tempfile.TemporaryDirectory() as stage_tmp:
            root = Path(root_tmp)
            staging_root = Path(stage_tmp) / "ui-stage"
            rendered_parts = []
            for index in range(50):
                source_name = f"SECRET_USER_take_{index:02d}.wav"
                job_root = root / "outputs" / "safe_job" / f"SECRET_WORKSPACE_{index:02d}"
                job_root.mkdir(parents=True, exist_ok=True)
                original = job_root / source_name
                mixed = job_root / f"SECRET_USER_mix_{index:02d}.wav"
                full = job_root / f"SECRET_USER_full_{index:02d}.wav"
                _write_pcm_wav(original, index.to_bytes(2, "little") * 2)
                _write_pcm_wav(mixed, (index + 1).to_bytes(2, "little") * 2)
                _write_pcm_wav(full, (index + 2).to_bytes(2, "little") * 2)
                result = {
                    "job_id": "safe_job",
                    "status": "success",
                    "runtime": {"total_seconds": 1.0},
                    "input_meta": {"source_name": source_name, "duration_seconds": 1.0},
                    "original_audio_path": str(original),
                    "mixed_output_path": str(mixed),
                    "enhanced_audio_path": str(mixed),
                    "full_output_path": str(full),
                    "warnings": [],
                    "events": [],
                    "config_snapshot": {},
                }
                with mock.patch("ui.presenters.ROOT_DIR", root), mock.patch.dict(
                    "os.environ", {"AUDIORESCUE_UI_STAGING_DIR": str(staging_root)}
                ):
                    view = result_to_view(result)
                rendered_parts.extend(
                    str(view[key])
                    for key in (
                        "status_md",
                        "input_md",
                        "playback_note_md",
                        "original_audio",
                        "enhanced_audio",
                        "mixed_download",
                        "full_download",
                    )
                )

            rendered = unquote("\n".join(rendered_parts))
            self.assertNotIn("SECRET_USER", rendered)
            self.assertNotIn("SECRET_WORKSPACE", rendered)
            self.assertNotIn("outputs/safe_job", rendered)
            self.assertNotIn(str(root), rendered)
            self.assertNotIn(str(staging_root), rendered)
            self.assertNotIn("/gradio_api/file=", rendered)
            self.assertNotIn("%2F", rendered)
            self.assertTrue(all(url.startswith("/audiorescue-files/") for url in _delivery_urls(rendered)))
            self.assertGreaterEqual(rendered.count("original.wav"), 50)
            self.assertGreaterEqual(rendered.count("mixed.wav"), 50)
            self.assertGreaterEqual(rendered.count("full.wav"), 50)


if __name__ == "__main__":
    unittest.main()
