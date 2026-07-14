import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

import numpy as np

from core.schemas import EnhancementError


def _write_pcm16_wav(path: Path, samples: np.ndarray, sample_rate: int = 48_000) -> None:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def _read_pcm16_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        raw = handle.readframes(handle.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


class _FakeEnhancer:
    model_name = "fake-deepfilternet3"

    def __init__(self, wet_samples: np.ndarray) -> None:
        self._wet_samples = wet_samples

    def enhance_file(self, input_wav: str, output_full_wav: str) -> None:
        _write_pcm16_wav(Path(output_full_wav), self._wet_samples)


class EnhancementContractTest(unittest.TestCase):
    def test_enhance_audio_writes_full_and_mixed_outputs_with_strength(self) -> None:
        from core import enhance

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            dry = np.full(48_000, 0.2, dtype=np.float32)
            wet = np.full(48_000, 0.8, dtype=np.float32)
            input_wav = temp / "original.wav"
            full_wav = temp / "enhanced_full.wav"
            mix_wav = temp / "enhanced_mix.wav"
            _write_pcm16_wav(input_wav, dry)

            with mock.patch.object(enhance, "load_enhancer", return_value=_FakeEnhancer(wet)):
                output = enhance.enhance_audio(
                    str(input_wav),
                    str(full_wav),
                    str(mix_wav),
                    strength=0.75,
                )

            self.assertEqual(Path(output["full_output_path"]), full_wav.resolve())
            self.assertEqual(Path(output["mixed_output_path"]), mix_wav.resolve())
            self.assertEqual(output["strength"], 0.75)
            self.assertEqual(output["model_name"], "fake-deepfilternet3")
            self.assertGreaterEqual(output["runtime_seconds"], 0.0)
            self.assertEqual(output["warnings"], [])
            self.assertTrue(full_wav.exists())
            self.assertTrue(mix_wav.exists())
            mixed = _read_pcm16_wav(mix_wav)
            self.assertAlmostEqual(float(np.mean(mixed)), 0.65, places=3)

    def test_enhance_audio_rejects_invalid_strength(self) -> None:
        from core.enhance import enhance_audio

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            input_wav = temp / "original.wav"
            _write_pcm16_wav(input_wav, np.zeros(48_000, dtype=np.float32))
            with self.assertRaises(EnhancementError):
                enhance_audio(
                    str(input_wav),
                    str(temp / "full.wav"),
                    str(temp / "mix.wav"),
                    strength=1.5,
                )

    def test_enhance_audio_allows_playable_silence(self) -> None:
        from core import enhance

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            silence = np.zeros(48_000, dtype=np.float32)
            input_wav = temp / "original.wav"
            full_wav = temp / "enhanced_full.wav"
            mix_wav = temp / "enhanced_mix.wav"
            _write_pcm16_wav(input_wav, silence)

            with mock.patch.object(
                enhance, "load_enhancer", return_value=_FakeEnhancer(silence)
            ):
                output = enhance.enhance_audio(
                    str(input_wav),
                    str(full_wav),
                    str(mix_wav),
                )

            self.assertEqual(Path(output["mixed_output_path"]), mix_wav.resolve())
            self.assertTrue(mix_wav.exists())

    def test_enhance_audio_wraps_backend_failures(self) -> None:
        from core import enhance

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            input_wav = temp / "original.wav"
            _write_pcm16_wav(input_wav, np.zeros(48_000, dtype=np.float32))
            with mock.patch.object(enhance, "load_enhancer", side_effect=RuntimeError("boom")):
                with self.assertRaises(EnhancementError):
                    enhance.enhance_audio(
                        str(input_wav),
                        str(temp / "full.wav"),
                        str(temp / "mix.wav"),
                    )


if __name__ == "__main__":
    unittest.main()
