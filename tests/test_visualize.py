from __future__ import annotations

import math
import tempfile
import unittest
import wave
from pathlib import Path

from core.visualize import create_spectrogram_comparison, create_waveform_comparison


def _write_tone(path: Path, duration_seconds: float = 0.25, frequency: float = 440.0, sample_rate: int = 8000) -> None:
    frames = bytearray()
    frame_count = int(sample_rate * duration_seconds)
    for index in range(frame_count):
        value = int(16000 * math.sin(2 * math.pi * frequency * index / sample_rate))
        frames.extend(value.to_bytes(2, byteorder="little", signed=True))
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(bytes(frames))


class VisualizeTests(unittest.TestCase):
    def test_visualize_functions_exist(self) -> None:
        self.assertTrue(callable(create_waveform_comparison))
        self.assertTrue(callable(create_spectrogram_comparison))

    def test_generate_comparison_images_when_dependencies_exist(self) -> None:
        try:
            import matplotlib  # noqa: F401
            import numpy  # noqa: F401
            import soundfile  # noqa: F401
        except Exception:
            self.skipTest("visualization dependencies are not installed")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            original = tmp_path / "original.wav"
            enhanced = tmp_path / "enhanced.wav"
            waveform = tmp_path / "waveform.png"
            spectrogram = tmp_path / "spectrogram.png"
            _write_tone(original, duration_seconds=0.35, frequency=440.0)
            _write_tone(enhanced, duration_seconds=0.25, frequency=550.0)

            self.assertEqual(create_waveform_comparison(str(original), str(enhanced), str(waveform)), str(waveform))
            self.assertTrue(waveform.exists())
            self.assertGreater(waveform.stat().st_size, 0)

            self.assertEqual(
                create_spectrogram_comparison(str(original), str(enhanced), str(spectrogram)),
                str(spectrogram),
            )
            self.assertTrue(spectrogram.exists())
            self.assertGreater(spectrogram.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
