import sys
import tempfile
import time
import types
import unittest
import wave
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
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


class _OverlappingWhisperModel(_FakeWhisperModel):
    def __init__(self, payload: dict) -> None:
        super().__init__(payload)
        self.active_calls = 0
        self.max_active_calls = 0

    def transcribe(self, audio_path: str, **kwargs):
        self.active_calls += 1
        self.max_active_calls = max(self.max_active_calls, self.active_calls)
        try:
            time.sleep(0.02)
            return super().transcribe(audio_path, **kwargs)
        finally:
            self.active_calls -= 1


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

    def test_asr_fingerprint_reports_sanitized_checkpoint_hash(self) -> None:
        from core import transcribe

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_root = Path(temp_dir) / "cache"
            checkpoint = cache_root / "whisper" / "base.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"whisper-base")
            fake_whisper = types.SimpleNamespace(
                _MODELS={"base": "https://example.invalid/base.pt"}
            )

            with (
                mock.patch.dict("os.environ", {"XDG_CACHE_HOME": str(cache_root)}),
                mock.patch.dict(sys.modules, {"whisper": fake_whisper}),
            ):
                fingerprint = transcribe.get_asr_model_fingerprint("base", "cpu")

        self.assertEqual(fingerprint["model_name"], "whisper-base")
        self.assertEqual(fingerprint["device"], "cpu")
        self.assertEqual(fingerprint["checkpoint_path"], "base.pt")
        self.assertEqual(
            fingerprint["checkpoint_sha256"],
            sha256(b"whisper-base").hexdigest(),
        )
        for value in fingerprint.values():
            if isinstance(value, str):
                self.assertNotIn(temp_dir, value)

    def test_asr_fingerprint_for_custom_checkpoint_does_not_expose_parent_path(self) -> None:
        from core import transcribe

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "private" / "custom.pt"
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b"custom-checkpoint")

            fingerprint = transcribe.get_asr_model_fingerprint(str(checkpoint), "cpu")

        self.assertEqual(fingerprint["model_name"], "whisper-custom")
        self.assertEqual(fingerprint["checkpoint_path"], "custom.pt")
        self.assertEqual(
            fingerprint["checkpoint_sha256"],
            sha256(b"custom-checkpoint").hexdigest(),
        )
        for value in fingerprint.values():
            if isinstance(value, str):
                self.assertNotIn(temp_dir, value)

    def test_asr_fingerprint_mismatch_fails_before_caching_model(self) -> None:
        from core import transcribe

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_root = Path(temp_dir) / "cache"
            checkpoint = cache_root / "whisper" / "tiny.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"tiny-checkpoint")
            loaded_model = object()

            def fake_load_model(model_name: str, *, device: str):
                return loaded_model

            fake_whisper = types.SimpleNamespace(
                load_model=fake_load_model,
                _MODELS={"tiny": "https://example.invalid/tiny.pt"},
            )
            with (
                mock.patch.object(transcribe, "_ASR_MODEL", None),
                mock.patch.object(transcribe, "_ASR_KEY", None),
                mock.patch.object(transcribe, "_ASR_FINGERPRINT_KEY", None),
                mock.patch.dict(sys.modules, {"whisper": fake_whisper}),
                mock.patch.dict("os.environ", {"XDG_CACHE_HOME": str(cache_root)}),
            ):
                with self.assertRaises(ASRInferenceError):
                    transcribe.load_asr(
                        "tiny",
                        "cpu",
                        expected_fingerprint={"checkpoint_sha256": "0" * 64},
                    )

        self.assertIsNone(transcribe._ASR_MODEL)

    def test_asr_empty_expected_hash_fails_closed(self) -> None:
        from core import transcribe

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_root = Path(temp_dir) / "cache"
            checkpoint = cache_root / "whisper" / "tiny.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"tiny-checkpoint")
            fake_whisper = types.SimpleNamespace(
                _MODELS={"tiny": "https://example.invalid/tiny.pt"},
            )

            with (
                mock.patch.dict(sys.modules, {"whisper": fake_whisper}),
                mock.patch.dict("os.environ", {"XDG_CACHE_HOME": str(cache_root)}),
            ):
                with self.assertRaises(ASRInferenceError):
                    transcribe.verify_asr_model_fingerprint(
                        {"checkpoint_sha256": ""},
                        "tiny",
                        "cpu",
                    )

    def test_asr_correct_fingerprint_does_not_force_duplicate_load(self) -> None:
        from core import transcribe

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_root = Path(temp_dir) / "cache"
            checkpoint = cache_root / "whisper" / "tiny.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"tiny-checkpoint")
            loaded_model = object()
            load_calls: list[tuple[str, str]] = []

            def fake_load_model(model_name: str, *, device: str):
                load_calls.append((model_name, device))
                return loaded_model

            fake_whisper = types.SimpleNamespace(
                load_model=fake_load_model,
                _MODELS={"tiny": "https://example.invalid/tiny.pt"},
            )
            expected = {
                "checkpoint_sha256": sha256(b"tiny-checkpoint").hexdigest(),
            }
            with (
                mock.patch.object(transcribe, "_ASR_MODEL", None),
                mock.patch.object(transcribe, "_ASR_CACHE_IDENTITY", None),
                mock.patch.object(transcribe, "_ASR_FINGERPRINT_KEY", None),
                mock.patch.dict(sys.modules, {"whisper": fake_whisper}),
                mock.patch.dict("os.environ", {"XDG_CACHE_HOME": str(cache_root)}),
            ):
                first = transcribe.load_asr(
                    "tiny", "cpu", expected_fingerprint=expected
                )
                second = transcribe.load_asr(
                    "tiny", "cpu", expected_fingerprint=expected
                )

        self.assertIs(first, loaded_model)
        self.assertIs(second, loaded_model)
        self.assertEqual(load_calls, [("tiny", "cpu")])

    def test_asr_reloads_when_checkpoint_changes_on_disk(self) -> None:
        from core import transcribe

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_root = Path(temp_dir) / "cache"
            checkpoint = cache_root / "whisper" / "tiny.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"tiny-v1")
            loaded_models = [object(), object()]
            load_calls: list[tuple[str, str]] = []

            def fake_load_model(model_name: str, *, device: str):
                load_calls.append((model_name, device))
                return loaded_models[len(load_calls) - 1]

            fake_whisper = types.SimpleNamespace(
                load_model=fake_load_model,
                _MODELS={"tiny": "https://example.invalid/tiny.pt"},
            )
            with (
                mock.patch.object(transcribe, "_ASR_MODEL", None),
                mock.patch.object(transcribe, "_ASR_CACHE_IDENTITY", None),
                mock.patch.dict(sys.modules, {"whisper": fake_whisper}),
                mock.patch.dict("os.environ", {"XDG_CACHE_HOME": str(cache_root)}),
            ):
                first = transcribe.load_asr("tiny", "cpu")
                checkpoint.write_bytes(b"tiny-v2")
                second = transcribe.load_asr("tiny", "cpu")

        self.assertIs(first, loaded_models[0])
        self.assertIs(second, loaded_models[1])
        self.assertEqual(load_calls, [("tiny", "cpu"), ("tiny", "cpu")])

    def test_transcribe_audio_serializes_model_inference(self) -> None:
        from core import transcribe

        fake = _OverlappingWhisperModel({"text": "ok", "language": "zh", "segments": []})
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            wav_paths = []
            for index in range(2):
                wav_path = temp / f"audio_{index}.wav"
                _write_pcm16_wav(wav_path)
                wav_paths.append(str(wav_path))

            with mock.patch.object(transcribe, "load_asr", return_value=fake):
                with ThreadPoolExecutor(max_workers=2) as executor:
                    list(executor.map(transcribe.transcribe_audio, wav_paths))

        self.assertEqual(fake.max_active_calls, 1)


if __name__ == "__main__":
    unittest.main()
