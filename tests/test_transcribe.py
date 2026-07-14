import sys
import tempfile
import time
import types
import unittest
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

import numpy as np

from core.schemas import ASRInferenceError


def _write_pcm16_wav(path: Path) -> None:
    samples = np.zeros(48_000, dtype=np.float32)
    pcm = (samples * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(48_000)
        handle.writeframes(pcm.tobytes())


class _FakeWhisperModel:
    def __init__(self, payload: dict | Exception) -> None:
        self.payload = payload
        self.kwargs = None

    def transcribe(self, audio_path: str, **kwargs):
        self.kwargs = kwargs
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


class TranscribeContractTest(unittest.TestCase):
    def test_transcribe_audio_returns_serializable_transcript_result(self) -> None:
        from core import transcribe

        payload = {
            "text": "今天下午三点",
            "language": "zh",
            "segments": [
                {"start": 0, "end": 1.25, "text": "今天下午三点", "tokens": [1, 2]},
            ],
        }
        fake = _FakeWhisperModel(payload)
        with tempfile.TemporaryDirectory() as temp_dir:
            wav_path = Path(temp_dir) / "original.wav"
            _write_pcm16_wav(wav_path)
            with mock.patch.object(transcribe, "load_asr", return_value=fake):
                result = transcribe.transcribe_audio(str(wav_path), language="zh")

        self.assertEqual(result.text, "今天下午三点")
        self.assertEqual(result.language, "zh")
        self.assertEqual(
            result.segments,
            [{"start": 0.0, "end": 1.25, "text": "今天下午三点"}],
        )
        self.assertIsNone(result.error)
        self.assertGreaterEqual(result.runtime_seconds, 0.0)
        self.assertIn("whisper", result.model_name)
        self.assertEqual(fake.kwargs["language"], "zh")
        self.assertEqual(fake.kwargs["task"], "transcribe")
        self.assertEqual(fake.kwargs["temperature"], 0.0)
        self.assertFalse(fake.kwargs["condition_on_previous_text"])
        self.assertIsNone(fake.kwargs["initial_prompt"])

    def test_transcribe_audio_allows_empty_text_without_error(self) -> None:
        from core import transcribe

        fake = _FakeWhisperModel({"text": "", "language": "zh", "segments": []})
        with tempfile.TemporaryDirectory() as temp_dir:
            wav_path = Path(temp_dir) / "quiet.wav"
            _write_pcm16_wav(wav_path)
            with mock.patch.object(transcribe, "load_asr", return_value=fake):
                result = transcribe.transcribe_audio(str(wav_path), language="zh")

        self.assertEqual(result.text, "")
        self.assertIsNone(result.error)
        self.assertEqual(result.segments, [])

    def test_transcribe_audio_raises_asr_inference_error_on_real_failure(self) -> None:
        from core import transcribe

        fake = _FakeWhisperModel(RuntimeError("decoder failed"))
        with tempfile.TemporaryDirectory() as temp_dir:
            wav_path = Path(temp_dir) / "original.wav"
            _write_pcm16_wav(wav_path)
            with mock.patch.object(transcribe, "load_asr", return_value=fake):
                with self.assertRaises(ASRInferenceError):
                    transcribe.transcribe_audio(str(wav_path), language="zh")

    def test_transcribe_audio_uses_explicit_non_default_model_and_device(self) -> None:
        from core import transcribe

        fake = _FakeWhisperModel({"text": "ok", "language": "zh", "segments": []})
        with tempfile.TemporaryDirectory() as temp_dir:
            wav_path = Path(temp_dir) / "original.wav"
            _write_pcm16_wav(wav_path)
            with mock.patch.object(
                transcribe, "load_asr", return_value=fake
            ) as loader:
                result = transcribe.transcribe_audio(
                    str(wav_path),
                    language="zh",
                    model_name="tiny",
                    device="cpu",
                )

        loader.assert_called_once_with("tiny", "cpu")
        self.assertEqual(result.model_name, "whisper-tiny")
        self.assertFalse(fake.kwargs["fp16"])

    def test_load_asr_singleton_is_thread_safe(self) -> None:
        from core import transcribe

        loaded_model = object()
        load_calls: list[tuple[str, str]] = []

        def fake_load_model(model_name: str, *, device: str):
            load_calls.append((model_name, device))
            time.sleep(0.02)
            return loaded_model

        fake_whisper = types.SimpleNamespace(load_model=fake_load_model)
        with (
            mock.patch.object(transcribe, "_ASR_MODEL", None),
            mock.patch.object(transcribe, "_ASR_KEY", None),
            mock.patch.dict(sys.modules, {"whisper": fake_whisper}),
        ):
            with ThreadPoolExecutor(max_workers=8) as executor:
                models = list(
                    executor.map(
                        lambda _: transcribe.load_asr("tiny", "cpu"), range(8)
                    )
                )

        self.assertEqual(load_calls, [("tiny", "cpu")])
        self.assertTrue(all(model is loaded_model for model in models))


if __name__ == "__main__":
    unittest.main()
