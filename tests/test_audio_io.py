import json
import math
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

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
        from core import audio_io

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            with self.assertRaises(InputAudioError):
                audio_io.normalize_audio(
                    str(temp / "missing.wav"), str(temp / "out.wav")
                )

            empty = temp / "empty.wav"
            empty.write_bytes(b"")
            with self.assertRaises(InputAudioError):
                audio_io.normalize_audio(str(empty), str(temp / "out.wav"))

            too_long = temp / "too_long.wav"
            samples = np.zeros(61 * 8_000, dtype=np.float32)
            _write_pcm16_wav(too_long, samples, 8_000)
            with mock.patch.object(audio_io, "_read_wav") as read_wav:
                with self.assertRaises(InputTooLongError):
                    audio_io.normalize_audio(
                        str(too_long), str(temp / "out.wav")
                    )
                read_wav.assert_not_called()

    def test_oversized_non_wav_is_rejected_before_external_tools(self) -> None:
        from core import audio_io

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            input_path = temp / "oversized.mp3"
            with input_path.open("wb") as handle:
                handle.seek(audio_io.MAX_INPUT_BYTES)
                handle.write(b"\0")

            with mock.patch.object(audio_io.subprocess, "run") as runner:
                with self.assertRaises(InputAudioError) as raised:
                    audio_io.normalize_audio(
                        str(input_path), str(temp / "out.wav")
                    )

            self.assertEqual(raised.exception.public_message, "输入音频文件过大")
            runner.assert_not_called()

    def test_ffprobe_rejects_overlong_non_wav_before_decode(self) -> None:
        from core import audio_io

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            input_path = temp / "overlong.mp3"
            input_path.write_bytes(b"fake mp3")
            probe_result = mock.Mock(
                returncode=0,
                stdout=json.dumps(
                    {
                        "format": {"duration": "61.0", "size": "8"},
                        "streams": [{"duration": "61.0"}],
                    }
                ),
                stderr="",
            )
            with (
                mock.patch.object(
                    audio_io.shutil, "which", return_value="/tools/ffprobe"
                ),
                mock.patch.object(
                    audio_io.subprocess, "run", return_value=probe_result
                ) as runner,
            ):
                with self.assertRaises(InputTooLongError):
                    audio_io.normalize_audio(
                        str(input_path), str(temp / "out.wav")
                    )

            self.assertEqual(runner.call_count, 1)
            command = runner.call_args.args[0]
            self.assertIn("-nostdin", command)
            self.assertEqual(
                runner.call_args.kwargs["timeout"],
                audio_io.FFPROBE_TIMEOUT_SECONDS,
            )

    def test_non_wav_probe_failure_has_stable_input_error(self) -> None:
        from core import audio_io

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            input_path = temp / "corrupt.m4a"
            input_path.write_bytes(b"not media")
            probe_result = mock.Mock(
                returncode=1, stdout="", stderr="invalid container"
            )
            with (
                mock.patch.object(
                    audio_io.shutil, "which", return_value="/tools/ffprobe"
                ),
                mock.patch.object(
                    audio_io.subprocess, "run", return_value=probe_result
                ),
            ):
                with self.assertRaises(InputAudioError) as raised:
                    audio_io.normalize_audio(
                        str(input_path), str(temp / "out.wav")
                    )

            self.assertEqual(
                raised.exception.public_message,
                "输入音频无法读取或不包含音频流",
            )

    def test_ffmpeg_decode_has_no_stdin_timeout_or_truncation(self) -> None:
        from core import audio_io

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            input_path = temp / "valid-duration.mp3"
            input_path.write_bytes(b"fake mp3")
            probe_result = mock.Mock(
                returncode=0,
                stdout=json.dumps(
                    {
                        "format": {"duration": "1.0", "size": "8"},
                        "streams": [{"duration": "1.0"}],
                    }
                ),
                stderr="",
            )
            timeout = subprocess.TimeoutExpired(cmd="ffmpeg", timeout=120)
            with (
                mock.patch.object(
                    audio_io.shutil,
                    "which",
                    side_effect=lambda name: f"/tools/{name}",
                ),
                mock.patch.object(
                    audio_io.subprocess,
                    "run",
                    side_effect=[probe_result, timeout],
                ) as runner,
            ):
                with self.assertRaises(InputAudioError) as raised:
                    audio_io.normalize_audio(
                        str(input_path), str(temp / "out.wav")
                    )

            self.assertEqual(raised.exception.public_message, "输入音频解码超时")
            self.assertEqual(runner.call_count, 2)
            decode_command = runner.call_args_list[1].args[0]
            decode_kwargs = runner.call_args_list[1].kwargs
            self.assertIn("-nostdin", decode_command)
            self.assertNotIn("-t", decode_command)
            self.assertEqual(decode_kwargs["stdin"], subprocess.DEVNULL)
            self.assertEqual(
                decode_kwargs["timeout"], audio_io.FFMPEG_TIMEOUT_SECONDS
            )

    def test_corrupt_wav_header_has_stable_input_error(self) -> None:
        from core import audio_io

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            input_path = temp / "corrupt.wav"
            input_path.write_bytes(b"not a wav")

            with self.assertRaises(InputAudioError) as raised:
                audio_io.normalize_audio(str(input_path), str(temp / "out.wav"))

            self.assertEqual(raised.exception.public_message, "输入 WAV 头无法读取")

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
