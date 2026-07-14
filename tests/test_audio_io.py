import math
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from core.schemas import ErrorCode, InputAudioError, InputTooLongError


def _write_pcm16_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim == 1:
        channels = 1
        interleaved = samples
    else:
        channels = samples.shape[1]
        interleaved = samples.reshape(-1)
    pcm = np.clip(interleaved, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


class AudioIOContractTest(unittest.TestCase):
    def test_normalize_audio_writes_48khz_mono_pcm16_and_preserves_source_meta(self) -> None:
        from core.audio_io import normalize_audio

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            input_path = temp / "stereo_44100.wav"
            output_path = temp / "original.wav"
            t = np.linspace(0.0, 1.2, int(44_100 * 1.2), endpoint=False)
            left = 0.25 * np.sin(2.0 * math.pi * 440.0 * t)
            right = 0.15 * np.sin(2.0 * math.pi * 660.0 * t)
            _write_pcm16_wav(input_path, np.column_stack([left, right]), 44_100)

            meta = normalize_audio(str(input_path), str(output_path))

            self.assertTrue(output_path.exists())
            with wave.open(str(output_path), "rb") as handle:
                self.assertEqual(handle.getframerate(), 48_000)
                self.assertEqual(handle.getnchannels(), 1)
                self.assertEqual(handle.getsampwidth(), 2)
                self.assertAlmostEqual(handle.getnframes() / 48_000, 1.2, places=2)

            self.assertEqual(meta.sample_rate, 48_000)
            self.assertEqual(meta.channels, 1)
            self.assertEqual(meta.original_sample_rate, 44_100)
            self.assertEqual(meta.original_channels, 2)
            self.assertEqual(meta.source_format, "wav")
            self.assertEqual(Path(meta.normalized_path), output_path.resolve())
            self.assertGreater(meta.peak_abs, 0.0)
            self.assertIsNotNone(meta.rms_dbfs)

    def test_normalize_audio_rejects_missing_empty_and_too_long_inputs(self) -> None:
        from core.audio_io import normalize_audio

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            with self.assertRaises(InputAudioError):
                normalize_audio(str(temp / "missing.wav"), str(temp / "out.wav"))

            empty = temp / "empty.wav"
            empty.write_bytes(b"")
            with self.assertRaises(InputAudioError):
                normalize_audio(str(empty), str(temp / "out.wav"))

            too_long = temp / "too_long.wav"
            samples = np.zeros(61 * 8_000, dtype=np.float32)
            _write_pcm16_wav(too_long, samples, 8_000)
            with self.assertRaises(InputTooLongError):
                normalize_audio(str(too_long), str(temp / "out.wav"))

    def test_inspect_audio_reports_warning_items_for_clipping_and_near_silence(self) -> None:
        from core.audio_io import inspect_audio

        clipped = np.array([0.0, 1.0, -1.0, 0.1], dtype=np.float32)
        clipped_report = inspect_audio(clipped, 48_000)
        clipped_codes = {item.code for item in clipped_report["warnings"]}
        self.assertIn(ErrorCode.INPUT_CLIPPED, clipped_codes)

        quiet = np.zeros(48_000, dtype=np.float32)
        quiet_report = inspect_audio(quiet, 48_000)
        quiet_codes = {item.code for item in quiet_report["warnings"]}
        self.assertIn(ErrorCode.INPUT_NEAR_SILENT, quiet_codes)
        self.assertEqual(quiet_report["peak_abs"], 0.0)


if __name__ == "__main__":
    unittest.main()
