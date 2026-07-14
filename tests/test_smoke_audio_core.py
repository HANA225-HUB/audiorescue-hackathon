import contextlib
import io
import json
import logging
import sys
import tempfile
import unittest
import warnings
import wave
from urllib.parse import quote
from pathlib import Path
from unittest import mock

import numpy as np

from core.schemas import AudioMeta, TranscriptResult
from core.schemas import EnhancementError
from scripts import smoke_audio_core as smoke

_POSIX_CACHE_MARKER = "/private/cache/alpha.wav"
_WINDOWS_CACHE_MARKER = "C:/private/cache/beta.wav"
_URL_MARKER = "https://example.test/audio.wav?token=secret"
_CHINESE_TRANSCRIPT_MARKER = "\u4e2d\u6587\u8f6c\u5199\u6807\u8bb0"
_CLI_SECRET_MARKER = "secret-cli-marker-\u4e2d\u6587-token"
_LONG_FREE_TEXT = "free-text-" + ("x" * 2048)


def _write_pcm16_wav(path: Path, amplitude: float = 0.1) -> None:
    samples = np.full(48_000, amplitude, dtype=np.float32)
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(48_000)
        handle.writeframes(pcm.tobytes())


def _fake_normalize(input_path: str, output_path: str) -> AudioMeta:
    output = Path(output_path)
    _write_pcm16_wav(output, amplitude=0.1)
    return AudioMeta(
        source_name=Path(input_path).name,
        sample_rate=48_000,
        channels=1,
        duration_seconds=1.0,
        peak_abs=0.1,
        rms_dbfs=-20.0,
        clipped_ratio=0.0,
        silent_ratio=0.0,
        normalized_path=str(output.resolve()),
        original_sample_rate=48_000,
        original_channels=1,
        source_format="wav",
    )


def _noisy_fake_normalize(input_path: str, output_path: str) -> AudioMeta:
    print(f"stdout leak {_POSIX_CACHE_MARKER} {_CHINESE_TRANSCRIPT_MARKER} {_LONG_FREE_TEXT}")
    print(f"stderr leak {_WINDOWS_CACHE_MARKER} transcript marker", file=sys.stderr)
    warnings.warn(
        f"warning leak /private/model.pt {_CHINESE_TRANSCRIPT_MARKER}",
        UserWarning,
    )
    logging.getLogger("audiorescue.noisy").warning("logging leak %s", _URL_MARKER)
    return _fake_normalize(input_path, output_path)


def _fake_enhance(input_wav: str, output_full_wav: str, output_mix_wav: str, strength: float):
    _write_pcm16_wav(Path(output_full_wav), amplitude=0.2)
    _write_pcm16_wav(Path(output_mix_wav), amplitude=0.15)
    return {
        "full_output_path": str(Path(output_full_wav).resolve()),
        "mixed_output_path": str(Path(output_mix_wav).resolve()),
        "strength": strength,
        "runtime_seconds": 0.25,
        "model_name": "fake-deepfilternet3",
        "warnings": [],
    }


def _noisy_enhance_error(*_args, **_kwargs):
    print(f"enhance stdout leak {_POSIX_CACHE_MARKER} {_CHINESE_TRANSCRIPT_MARKER}")
    print(f"enhance stderr leak {_WINDOWS_CACHE_MARKER} {_LONG_FREE_TEXT}", file=sys.stderr)
    warnings.warn(f"enhance warning leak {_URL_MARKER}", RuntimeWarning)
    logging.getLogger("audiorescue.noisy").error("enhance logging leak %s", _URL_MARKER)
    raise EnhancementError(
        "backend unavailable: private transcript must not be printed",
        details={
            "stage": "enhance",
            "exception_type": "RuntimeError",
            "path": "C:/private/cache/source.wav",
            "message": _CHINESE_TRANSCRIPT_MARKER,
            "url": _URL_MARKER,
        },
    )


def _fake_transcribe(audio_path: str, language: str = "zh", *, model_name: str, device: str):
    return TranscriptResult(
        text="private transcript must not be printed",
        language=language,
        segments=[{"start": 0.0, "end": 1.0, "text": "private segment"}],
        runtime_seconds=0.5,
        model_name=f"whisper-{model_name}",
        error=None,
    )


def _assert_marker_absent(testcase: unittest.TestCase, rendered: str, marker: str) -> None:
    testcase.assertNotIn(marker, rendered)
    encoded = quote(marker, safe="")
    testcase.assertNotIn(encoded, rendered)
    testcase.assertNotIn(quote(encoded, safe=""), rendered)


class SmokeAudioCoreTest(unittest.TestCase):
    def _invoke_smoke(self, args: list[str], *, patch_normalize: bool = False):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(sys, "argv", ["smoke_audio_core.py", *args]))
            if patch_normalize:
                stack.enter_context(
                    mock.patch.object(smoke, "normalize_audio", side_effect=_fake_normalize)
                )
            stack.enter_context(contextlib.redirect_stdout(stdout))
            stack.enter_context(contextlib.redirect_stderr(stderr))
            code = smoke.main()
        return code, stdout.getvalue(), stderr.getvalue()

    def _run_normalize_only_with_cli(self, args: list[str]):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            input_wav = temp / "input.wav"
            output_dir = temp / "outputs"
            _write_pcm16_wav(input_wav)
            code, stdout, stderr = self._invoke_smoke(
                [
                    "--input",
                    str(input_wav),
                    "--output-dir",
                    str(output_dir),
                    "--normalize-only",
                    *args,
                ],
                patch_normalize=True,
            )
        return code, stdout, stderr, json.loads(stdout)

    def test_asr_model_summary_uses_safe_label_without_cli_value_leak(self) -> None:
        cases = [
            ("base", "base", None),
            (_POSIX_CACHE_MARKER, "custom", _POSIX_CACHE_MARKER),
            (_WINDOWS_CACHE_MARKER, "custom", _WINDOWS_CACHE_MARKER),
            (_URL_MARKER, "custom", _URL_MARKER),
            (_CLI_SECRET_MARKER, "custom", _CLI_SECRET_MARKER),
            (_LONG_FREE_TEXT, "custom", _LONG_FREE_TEXT[:80]),
        ]
        for value, expected_label, marker in cases:
            with self.subTest(asr_model=value[:16]):
                code, stdout, stderr, payload = self._run_normalize_only_with_cli(
                    ["--asr-model", value]
                )

                self.assertEqual(code, 0)
                self.assertEqual(stderr, "")
                self.assertEqual(payload["asr_request"]["model"], expected_label)
                self.assertEqual(payload["asr_request"]["device"], "auto")
                if marker is not None:
                    _assert_marker_absent(self, stdout, marker)

    def test_device_summary_uses_safe_family_without_cli_value_leak(self) -> None:
        cases = [
            ("auto", "auto", None),
            ("cpu", "cpu", None),
            ("cuda", "cuda", None),
            ("cuda:0", "cuda", None),
            ("mps", "mps", None),
            (_POSIX_CACHE_MARKER, "custom", _POSIX_CACHE_MARKER),
            (_WINDOWS_CACHE_MARKER, "custom", _WINDOWS_CACHE_MARKER),
            (_URL_MARKER, "custom", _URL_MARKER),
            (_CLI_SECRET_MARKER, "custom", _CLI_SECRET_MARKER),
            (_LONG_FREE_TEXT, "custom", _LONG_FREE_TEXT[:80]),
        ]
        for value, expected_label, marker in cases:
            with self.subTest(device=value[:16]):
                code, stdout, stderr, payload = self._run_normalize_only_with_cli(
                    ["--device", value]
                )

                self.assertEqual(code, 0)
                self.assertEqual(stderr, "")
                self.assertEqual(payload["asr_request"]["model"], "base")
                self.assertEqual(payload["asr_request"]["device"], expected_label)
                if marker is not None:
                    _assert_marker_absent(self, stdout, marker)

    def test_custom_asr_values_do_not_leak_through_success_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            input_wav = temp / "input.wav"
            output_dir = temp / "outputs"
            model_value = str(temp / f"{_CLI_SECRET_MARKER}.pt")
            device_value = f"private-device-{_CLI_SECRET_MARKER}"
            _write_pcm16_wav(input_wav)
            stdout = io.StringIO()
            stderr = io.StringIO()
            argv = [
                "smoke_audio_core.py",
                "--input",
                str(input_wav),
                "--output-dir",
                str(output_dir),
                "--asr-model",
                model_value,
                "--device",
                device_value,
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(smoke, "normalize_audio", side_effect=_fake_normalize),
                mock.patch.object(smoke, "enhance_audio", side_effect=_fake_enhance),
                mock.patch.object(smoke, "transcribe_audio", side_effect=_fake_transcribe),
                mock.patch.object(
                    smoke,
                    "get_enhancer_model_fingerprint",
                    return_value={
                        "model_name": "DeepFilterNet3",
                        "checkpoint_path": "checkpoints/model_120.ckpt.best",
                        "checkpoint_sha256": "a" * 64,
                    },
                ),
                mock.patch.object(
                    smoke,
                    "get_asr_model_fingerprint",
                    return_value={
                        "model_name": f"whisper-{_CLI_SECRET_MARKER}",
                        "device": device_value,
                        "checkpoint_path": f"{_CLI_SECRET_MARKER}.pt",
                        "checkpoint_sha256": "D" * 64,
                    },
                ),
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ):
                code = smoke.main()

        stdout_value = stdout.getvalue()
        stderr_value = stderr.getvalue()
        payload = json.loads(stdout_value)
        rendered = json.dumps(payload, ensure_ascii=False)

        self.assertEqual(code, 0)
        self.assertEqual(stderr_value, "")
        self.assertEqual(payload["asr_request"], {"model": "custom", "device": "custom"})
        self.assertEqual(
            payload["runs"][0]["model_fingerprints"]["asr"]["model_name"],
            "whisper-custom",
        )
        self.assertEqual(
            payload["runs"][0]["model_fingerprints"]["asr"]["checkpoint_path"],
            "custom",
        )
        self.assertEqual(
            payload["runs"][0]["transcript_before"]["model_name"],
            "whisper-custom",
        )
        self.assertEqual(
            payload["runs"][0]["transcript_after"]["model_name"],
            "whisper-custom",
        )
        _assert_marker_absent(self, rendered, _CLI_SECRET_MARKER)

    def test_parse_failures_return_single_safe_input_invalid_json(self) -> None:
        cases = [
            (["--unknown-arg", _CLI_SECRET_MARKER], "unknown_argument", _CLI_SECRET_MARKER),
            (["--runs", _CLI_SECRET_MARKER], "invalid_value", _CLI_SECRET_MARKER),
            (["--runs"], "missing_value", None),
            (["--runs", "0"], "invalid_range", None),
            (["--runs", "-1"], "invalid_range", None),
        ]
        for args, expected_reason, marker in cases:
            with self.subTest(args=args[:1]):
                code, stdout, stderr = self._invoke_smoke(args)
                payload = json.loads(stdout)
                rendered = json.dumps(payload, ensure_ascii=False)

                self.assertEqual(code, 2)
                self.assertEqual(stderr, "")
                self.assertEqual(payload["runs"][0]["error"]["code"], "INPUT_INVALID")
                self.assertEqual(
                    payload["runs"][0]["error"]["message"],
                    "input audio is invalid",
                )
                self.assertEqual(
                    payload["runs"][0]["error"]["details"],
                    {"category": "argument_parse", "reason": expected_reason},
                )
                self.assertNotIn("usage:", stdout.lower())
                self.assertNotIn("Traceback", stdout)
                if marker is not None:
                    _assert_marker_absent(self, rendered, marker)

    def test_cli_help_remains_static_argparse_output(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            mock.patch.object(
                sys,
                "argv",
                ["smoke_audio_core.py", "--help", _CLI_SECRET_MARKER],
            ),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            with self.assertRaises(SystemExit) as raised:
                smoke.main()

        self.assertEqual(raised.exception.code, 0)
        self.assertIn("AudioRescue A-layer smoke test", stdout.getvalue())
        self.assertNotIn(_CLI_SECRET_MARKER, stdout.getvalue())
        self.assertEqual(stderr.getvalue(), "")

    def test_cli_pre_run_boundary_does_not_swallow_system_exit(self) -> None:
        argv = [
            "smoke_audio_core.py",
            "--input",
            "unused.wav",
            "--output-dir",
            "unused_outputs",
        ]
        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch.object(smoke, "_run_smoke", side_effect=SystemExit(7)),
        ):
            with self.assertRaises(SystemExit) as raised:
                smoke.main()

        self.assertEqual(raised.exception.code, 7)

    def test_smoke_runs_twice_and_prints_sanitized_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            input_wav = temp / "private_input.wav"
            output_dir = temp / "private_outputs"
            _write_pcm16_wav(input_wav)
            stdout = io.StringIO()
            argv = [
                "smoke_audio_core.py",
                "--input",
                str(input_wav),
                "--output-dir",
                str(output_dir),
                "--runs",
                "2",
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(smoke, "normalize_audio", side_effect=_fake_normalize),
                mock.patch.object(smoke, "enhance_audio", side_effect=_fake_enhance),
                mock.patch.object(smoke, "transcribe_audio", side_effect=_fake_transcribe),
                mock.patch.object(
                    smoke,
                    "get_enhancer_model_fingerprint",
                    return_value={
                        "model_name": "DeepFilterNet3",
                        "checkpoint_path": "checkpoints/model_120.ckpt.best",
                        "checkpoint_sha256": "a" * 64,
                        "config_path": "config.ini",
                        "config_sha256": "b" * 64,
                    },
                ),
                mock.patch.object(
                    smoke,
                    "get_asr_model_fingerprint",
                    return_value={
                        "model_name": "whisper-base",
                        "checkpoint_path": "base.pt",
                        "checkpoint_sha256": "c" * 64,
                    },
                ),
                mock.patch.object(
                    smoke,
                    "_probe_environment",
                    return_value={
                        "python": "3.12.0",
                        "torch_cuda_available": False,
                        "gpu": None,
                    },
                ),
                mock.patch.object(
                    smoke,
                    "_cuda_after_run",
                    return_value={
                        "torch_importable": True,
                        "torch_cuda_available": False,
                        "max_memory_allocated_bytes": 0,
                    },
                ),
                contextlib.redirect_stdout(stdout),
            ):
                code = smoke.main()

            self.assertEqual(code, 0)
            rendered = stdout.getvalue()
            payload = json.loads(rendered)

        self.assertEqual([item["run_kind"] for item in payload["runs"]], ["cold", "warm"])
        self.assertIn("model_fingerprints", payload["runs"][0])
        for run in payload["runs"]:
            self.assertEqual(run["output_files"]["original"]["filename"], "original.wav")
            self.assertEqual(run["output_files"]["enhanced_full"]["sample_rate"], 48_000)
            self.assertEqual(run["output_files"]["enhanced_mix"]["channels"], 1)
            self.assertEqual(run["output_files"]["enhanced_mix"]["sample_width_bits"], 16)
        self.assertNotIn(temp_dir, rendered)
        self.assertNotIn("private transcript", rendered)
        self.assertNotIn("private segment", rendered)
        self.assertNotIn("private_outputs", rendered)

    def test_normalize_only_runs_multiple_times_without_model_calls(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            input_wav = temp / "input.wav"
            output_dir = temp / "outputs"
            _write_pcm16_wav(input_wav)
            stdout = io.StringIO()
            argv = [
                "smoke_audio_core.py",
                "--input",
                str(input_wav),
                "--output-dir",
                str(output_dir),
                "--runs",
                "2",
                "--normalize-only",
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(smoke, "normalize_audio", side_effect=_fake_normalize),
                mock.patch.object(smoke, "enhance_audio") as enhance_mock,
                mock.patch.object(smoke, "transcribe_audio") as transcribe_mock,
                contextlib.redirect_stdout(stdout),
            ):
                code = smoke.main()

            self.assertEqual(code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(len(payload["runs"]), 2)
            self.assertNotIn("model_fingerprints", payload["runs"][0])
            enhance_mock.assert_not_called()
            transcribe_mock.assert_not_called()

    def test_error_evidence_sanitizes_path_like_details(self) -> None:
        error = EnhancementError(
            "failed",
            details={
                "cache_path": "C:/private/cache/model.bin",
                "checkpoint_dir": "C:/private/checkpoints",
                "exception_type": "ModuleNotFoundError",
            },
        )

        evidence = smoke._audio_rescue_error_evidence(error)

        self.assertEqual(evidence["details"]["cache_path"], "<redacted>")
        self.assertEqual(evidence["details"]["checkpoint_dir"], "<redacted>")
        self.assertEqual(evidence["details"]["exception_type"], "ModuleNotFoundError")

    def test_error_evidence_drops_unknown_free_text_recursively(self) -> None:
        error = EnhancementError(
            "failed",
            details={
                "stage": "enhance",
                "component": "deepfilternet",
                "exception_type": "RuntimeError",
                "retryable": False,
                "message": "今天下午三点，我们在实验室讨论语音处理项目的最终方案。",
                "english_transcript": "please send help from the old lab recording",
                "nested": {
                    "notes": [
                        "private transcript must not be printed",
                        {"url": "https://example.test/audio.wav?token=secret"},
                        {"path": "C:/private/cache/original_patient_file.wav"},
                    ],
                },
                "very_long": "x" * 512,
            },
        )

        evidence = smoke._audio_rescue_error_evidence(error)
        rendered = json.dumps(evidence, ensure_ascii=False)

        self.assertEqual(evidence["details"]["stage"], "enhance")
        self.assertEqual(evidence["details"]["component"], "deepfilternet")
        self.assertEqual(evidence["details"]["exception_type"], "RuntimeError")
        self.assertFalse(evidence["details"]["retryable"])
        self.assertNotIn("今天下午三点", rendered)
        self.assertNotIn("please send help", rendered)
        self.assertNotIn("private transcript", rendered)
        self.assertNotIn("original_patient_file", rendered)
        self.assertNotIn("example.test", rendered)
        self.assertNotIn("x" * 64, rendered)

    def test_main_returns_structured_error_without_free_text_leak(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            input_wav = temp / "private_input.wav"
            output_dir = temp / "private_outputs"
            _write_pcm16_wav(input_wav)
            stdout = io.StringIO()
            stderr = io.StringIO()
            argv = [
                "smoke_audio_core.py",
                "--input",
                str(input_wav),
                "--output-dir",
                str(output_dir),
            ]
            error = EnhancementError(
                "backend unavailable: private transcript must not be printed",
                details={
                    "stage": "enhance",
                    "exception_type": "RuntimeError",
                    "path": "C:/private/cache/source.wav",
                    "message": "今天下午三点，我们在实验室讨论语音处理项目的最终方案。",
                },
            )
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(smoke, "normalize_audio", side_effect=_fake_normalize),
                mock.patch.object(smoke, "enhance_audio", side_effect=_noisy_enhance_error),
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ):
                code = smoke.main()

            payload = json.loads(stdout.getvalue())
            rendered = json.dumps(payload, ensure_ascii=False)

        self.assertEqual(code, 2)
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(payload["runs"][0]["error"]["code"], "ENHANCE_FAILED")
        self.assertEqual(payload["runs"][0]["error"]["message"], "enhancement failed")
        self.assertEqual(payload["runs"][0]["error"]["details"]["path"], "<redacted>")
        self.assertIn("cli_log_capture", payload)
        self.assertGreater(payload["cli_log_capture"]["stdout_lines"], 0)
        self.assertGreater(payload["cli_log_capture"]["stderr_lines"], 0)
        self.assertGreater(payload["cli_log_capture"]["warning_count"], 0)
        self.assertGreater(payload["cli_log_capture"]["logging_count"], 0)
        self.assertNotIn(_CHINESE_TRANSCRIPT_MARKER, rendered)
        self.assertNotIn(_LONG_FREE_TEXT[:80], rendered)
        self.assertNotIn("example.test", rendered)
        self.assertNotIn("private transcript", rendered)
        self.assertNotIn("今天下午三点", rendered)
        self.assertNotIn("source.wav", rendered)


    def test_cli_captures_third_party_output_without_breaking_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            input_wav = temp / "private_input.wav"
            output_dir = temp / "private_outputs"
            _write_pcm16_wav(input_wav)
            stdout = io.StringIO()
            stderr = io.StringIO()
            noisy_logger = logging.getLogger("audiorescue.noisy")
            previous_propagate = noisy_logger.propagate
            leaky_handler = logging.StreamHandler(stderr)
            noisy_logger.addHandler(leaky_handler)
            noisy_logger.propagate = False
            argv = [
                "smoke_audio_core.py",
                "--input",
                str(input_wav),
                "--output-dir",
                str(output_dir),
            ]
            try:
                with (
                    mock.patch.object(sys, "argv", argv),
                    mock.patch.object(smoke, "normalize_audio", side_effect=_noisy_fake_normalize),
                    mock.patch.object(smoke, "enhance_audio", side_effect=_fake_enhance),
                    mock.patch.object(smoke, "transcribe_audio", side_effect=_fake_transcribe),
                    mock.patch.object(
                        smoke,
                        "get_enhancer_model_fingerprint",
                        return_value={
                            "model_name": "DeepFilterNet3",
                            "checkpoint_path": "checkpoints/model_120.ckpt.best",
                            "checkpoint_sha256": "a" * 64,
                            "config_path": "config.ini",
                            "config_sha256": "b" * 64,
                        },
                    ),
                    mock.patch.object(
                        smoke,
                        "get_asr_model_fingerprint",
                        return_value={
                            "model_name": "whisper-base",
                            "checkpoint_path": "base.pt",
                            "checkpoint_sha256": "c" * 64,
                        },
                    ),
                    contextlib.redirect_stdout(stdout),
                    contextlib.redirect_stderr(stderr),
                ):
                    code = smoke.main()
            finally:
                noisy_logger.removeHandler(leaky_handler)
                noisy_logger.propagate = previous_propagate

            payload = json.loads(stdout.getvalue())
            rendered = json.dumps(payload, ensure_ascii=False)

        self.assertEqual(code, 0)
        self.assertEqual(stderr.getvalue(), "")
        self.assertIn("cli_log_capture", payload)
        self.assertGreater(payload["cli_log_capture"]["stdout_lines"], 0)
        self.assertGreater(payload["cli_log_capture"]["stderr_lines"], 0)
        self.assertGreater(payload["cli_log_capture"]["warning_count"], 0)
        self.assertGreater(payload["cli_log_capture"]["logging_count"], 0)
        self.assertGreaterEqual(payload["cli_log_capture"]["warning_categories"]["UserWarning"], 1)
        self.assertEqual(payload["cli_log_capture"]["logging_levels"]["WARNING"], 1)
        self.assertNotIn("stdout leak", rendered)
        self.assertNotIn("stderr leak", rendered)
        self.assertNotIn("warning leak", rendered)
        self.assertNotIn("logging leak", rendered)
        self.assertNotIn("transcript marker", rendered)
        self.assertNotIn(_CHINESE_TRANSCRIPT_MARKER, rendered)
        self.assertNotIn(_LONG_FREE_TEXT[:80], rendered)
        self.assertNotIn("example.test", rendered)

    def test_cli_pre_run_output_dir_file_error_returns_json_without_leak(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            input_wav = temp / "private_input.wav"
            marker = "cache_private_transcript_token_secret_marker"
            output_path = temp / marker
            _write_pcm16_wav(input_wav)
            output_path.write_text("not a directory", encoding="utf-8")
            stdout = io.StringIO()
            stderr = io.StringIO()
            argv = [
                "smoke_audio_core.py",
                "--input",
                str(input_wav),
                "--output-dir",
                str(output_path),
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ):
                code = smoke.main()

            payload = json.loads(stdout.getvalue())
            rendered = json.dumps(payload, ensure_ascii=False)

        self.assertEqual(code, 2)
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(payload["runs"][0]["error"]["code"], "INTERNAL_ERROR")
        self.assertEqual(payload["runs"][0]["error"]["message"], "internal error")
        self.assertEqual(payload["runs"][0]["error"]["details"]["exception_type"], "FileExistsError")
        self.assertIn("cli_log_capture", payload)
        self.assertNotIn(marker, rendered)
        self.assertNotIn("File exists", rendered)
        self.assertNotIn(str(output_path), rendered)

    def test_cli_pre_run_boundary_does_not_swallow_keyboard_interrupt(self) -> None:
        argv = [
            "smoke_audio_core.py",
            "--input",
            "unused.wav",
            "--output-dir",
            "unused_outputs",
        ]
        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch.object(smoke, "_run_smoke", side_effect=KeyboardInterrupt),
        ):
            with self.assertRaises(KeyboardInterrupt):
                smoke.main()

    def test_cli_unexpected_exception_returns_json_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            input_wav = temp / "private_input.wav"
            output_dir = temp / "private_outputs"
            _write_pcm16_wav(input_wav)
            stdout = io.StringIO()
            stderr = io.StringIO()
            argv = [
                "smoke_audio_core.py",
                "--input",
                str(input_wav),
                "--output-dir",
                str(output_dir),
            ]

            def explode(*_args, **_kwargs):
                print("unexpected stdout marker /private/cache/gamma.wav")
                print("unexpected stderr marker C:/private/cache/delta.wav", file=sys.stderr)
                raise RuntimeError("traceback marker private transcript must not be printed")

            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(smoke, "normalize_audio", side_effect=_fake_normalize),
                mock.patch.object(smoke, "enhance_audio", side_effect=explode),
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ):
                code = smoke.main()

            payload = json.loads(stdout.getvalue())
            rendered = json.dumps(payload, ensure_ascii=False)

        self.assertEqual(code, 2)
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(payload["runs"][0]["error"]["code"], "INTERNAL_ERROR")
        self.assertEqual(payload["runs"][0]["error"]["message"], "internal error")
        self.assertEqual(payload["runs"][0]["error"]["details"]["exception_type"], "RuntimeError")
        self.assertIn("cli_log_capture", payload)
        self.assertNotIn("traceback marker", rendered)
        self.assertNotIn("unexpected stdout", rendered)
        self.assertNotIn("unexpected stderr", rendered)
        self.assertNotIn("private transcript", rendered)


if __name__ == "__main__":
    unittest.main()
