import contextlib
import io
import json
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

import numpy as np

from core.schemas import AudioMeta, TranscriptResult
from core.schemas import EnhancementError
from scripts import smoke_audio_core as smoke


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


def _fake_transcribe(audio_path: str, language: str = "zh", *, model_name: str, device: str):
    return TranscriptResult(
        text="private transcript must not be printed",
        language=language,
        segments=[{"start": 0.0, "end": 1.0, "text": "private segment"}],
        runtime_seconds=0.5,
        model_name=f"whisper-{model_name}",
        error=None,
    )


class SmokeAudioCoreTest(unittest.TestCase):
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
                mock.patch.object(smoke, "enhance_audio", side_effect=error),
                contextlib.redirect_stdout(stdout),
            ):
                code = smoke.main()

            payload = json.loads(stdout.getvalue())
            rendered = json.dumps(payload, ensure_ascii=False)

        self.assertEqual(code, 2)
        self.assertEqual(payload["runs"][0]["error"]["code"], "ENHANCE_FAILED")
        self.assertEqual(payload["runs"][0]["error"]["message"], "enhancement failed")
        self.assertEqual(payload["runs"][0]["error"]["details"]["path"], "<redacted>")
        self.assertNotIn("private transcript", rendered)
        self.assertNotIn("今天下午三点", rendered)
        self.assertNotIn("source.wav", rendered)


if __name__ == "__main__":
    unittest.main()
