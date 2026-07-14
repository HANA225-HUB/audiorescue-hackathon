import tempfile
import time
import unittest
import wave
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
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


class _OverlappingEnhancer(_FakeEnhancer):
    def __init__(self, wet_samples: np.ndarray) -> None:
        super().__init__(wet_samples)
        self.active_calls = 0
        self.max_active_calls = 0

    def enhance_file(self, input_wav: str, output_full_wav: str) -> None:
        self.active_calls += 1
        self.max_active_calls = max(self.max_active_calls, self.active_calls)
        try:
            time.sleep(0.02)
            super().enhance_file(input_wav, output_full_wav)
        finally:
            self.active_calls -= 1


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
        from core import enhance

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            input_wav = temp / "original.wav"
            _write_pcm16_wav(input_wav, np.zeros(48_000, dtype=np.float32))
            with mock.patch.object(enhance, "load_enhancer") as loader:
                for invalid_strength in (
                    1.5,
                    True,
                    "0.75",
                    b"0.75",
                    None,
                    float("nan"),
                    float("inf"),
                ):
                    with self.subTest(strength=invalid_strength):
                        with self.assertRaises(EnhancementError):
                            enhance.enhance_audio(
                                str(input_wav),
                                str(temp / "full.wav"),
                                str(temp / "mix.wav"),
                                strength=invalid_strength,
                            )
                loader.assert_not_called()

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

    def test_runtime_excludes_model_loading(self) -> None:
        from core import enhance

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            samples = np.full(48_000, 0.2, dtype=np.float32)
            input_wav = temp / "original.wav"
            _write_pcm16_wav(input_wav, samples)
            backend = _FakeEnhancer(samples)
            timer = mock.Mock(side_effect=[10.0, 12.5])

            def load_backend():
                self.assertEqual(timer.call_count, 0)
                return backend

            with (
                mock.patch.object(enhance, "load_enhancer", side_effect=load_backend),
                mock.patch.object(enhance.time, "perf_counter", timer),
            ):
                output = enhance.enhance_audio(
                    str(input_wav),
                    str(temp / "full.wav"),
                    str(temp / "mix.wav"),
                )

        self.assertEqual(output["runtime_seconds"], 2.5)

    def test_load_enhancer_singleton_is_thread_safe(self) -> None:
        from core import enhance

        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "DeepFilterNet3"
            checkpoint_dir = model_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True)
            (model_dir / "config.ini").write_bytes(b"config")
            (checkpoint_dir / "model_120.ckpt.best").write_bytes(b"checkpoint")

            loaded_backend = object()
            load_calls: list[str | None] = []

            def fake_backend(model_dir_arg: str | None, *, fingerprint=None):
                load_calls.append(model_dir_arg)
                time.sleep(0.02)
                return loaded_backend

            with (
                mock.patch.object(enhance, "_ENHANCER_BACKEND", None),
                mock.patch.object(enhance, "_ENHANCER_MODEL_DIR", None),
                mock.patch.object(enhance, "_ENHANCER_CACHE_IDENTITY", None),
                mock.patch.object(
                    enhance, "_DeepFilterNetBackend", side_effect=fake_backend
                ),
            ):
                with ThreadPoolExecutor(max_workers=8) as executor:
                    backends = list(
                        executor.map(
                            lambda _: enhance.load_enhancer(str(model_dir)), range(8)
                        )
                    )

        self.assertEqual(load_calls, [str(model_dir)])
        self.assertTrue(all(backend is loaded_backend for backend in backends))

    def test_enhancer_fingerprint_reports_sanitized_config_and_checkpoint_hashes(self) -> None:
        from core import enhance

        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "DeepFilterNet3"
            checkpoint_dir = model_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True)
            config = model_dir / "config.ini"
            checkpoint = checkpoint_dir / "model_120.ckpt.best"
            older_checkpoint = checkpoint_dir / "model_7.ckpt.best"
            config.write_bytes(b"config")
            checkpoint.write_bytes(b"checkpoint-120")
            older_checkpoint.write_bytes(b"checkpoint-7")

            fingerprint = enhance.get_enhancer_model_fingerprint(str(model_dir))

        self.assertEqual(fingerprint["model_name"], "DeepFilterNet3")
        self.assertEqual(fingerprint["model_dir_name"], "DeepFilterNet3")
        self.assertEqual(fingerprint["config_path"], "config.ini")
        self.assertEqual(fingerprint["checkpoint_path"], "checkpoints/model_120.ckpt.best")
        self.assertEqual(fingerprint["config_sha256"], sha256(b"config").hexdigest())
        self.assertEqual(
            fingerprint["checkpoint_sha256"],
            sha256(b"checkpoint-120").hexdigest(),
        )
        for value in fingerprint.values():
            if isinstance(value, str):
                self.assertNotIn(temp_dir, value)

    def test_enhancer_fingerprint_mismatch_fails_before_caching_backend(self) -> None:
        from core import enhance

        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "DeepFilterNet3"
            checkpoint_dir = model_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True)
            (model_dir / "config.ini").write_bytes(b"config")
            (checkpoint_dir / "model_120.ckpt.best").write_bytes(b"checkpoint")

            with (
                mock.patch.object(enhance, "_ENHANCER_BACKEND", None),
                mock.patch.object(enhance, "_ENHANCER_MODEL_DIR", None),
                mock.patch.object(enhance, "_ENHANCER_FINGERPRINT_KEY", None),
                mock.patch.object(enhance, "_DeepFilterNetBackend") as backend_loader,
            ):
                with self.assertRaises(EnhancementError):
                    enhance.load_enhancer(
                        str(model_dir),
                        expected_fingerprint={
                            "config_sha256": "0" * 64,
                            "checkpoint_sha256": sha256(b"checkpoint").hexdigest(),
                        },
                    )

        backend_loader.assert_not_called()
        self.assertIsNone(enhance._ENHANCER_BACKEND)

    def test_enhancer_empty_expected_hash_fails_closed(self) -> None:
        from core import enhance

        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "DeepFilterNet3"
            checkpoint_dir = model_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True)
            (model_dir / "config.ini").write_bytes(b"config")
            (checkpoint_dir / "model_120.ckpt.best").write_bytes(b"checkpoint")

            with self.assertRaises(EnhancementError):
                enhance.verify_enhancer_model_fingerprint(
                    {
                        "config_sha256": "",
                        "checkpoint_sha256": sha256(b"checkpoint").hexdigest(),
                    },
                    str(model_dir),
                )

    def test_enhancer_correct_fingerprint_does_not_force_duplicate_load(self) -> None:
        from core import enhance

        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "DeepFilterNet3"
            checkpoint_dir = model_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True)
            (model_dir / "config.ini").write_bytes(b"config")
            (checkpoint_dir / "model_120.ckpt.best").write_bytes(b"checkpoint")
            loaded_backend = object()
            load_calls: list[str | None] = []

            def fake_backend(model_dir_arg: str | None, fingerprint=None):
                load_calls.append(model_dir_arg)
                return loaded_backend

            expected = {
                "config_sha256": sha256(b"config").hexdigest(),
                "checkpoint_sha256": sha256(b"checkpoint").hexdigest(),
            }
            with (
                mock.patch.object(enhance, "_ENHANCER_BACKEND", None),
                mock.patch.object(enhance, "_ENHANCER_MODEL_DIR", None),
                mock.patch.object(enhance, "_ENHANCER_FINGERPRINT_KEY", None),
                mock.patch.object(enhance, "_DeepFilterNetBackend", side_effect=fake_backend),
            ):
                first = enhance.load_enhancer(
                    str(model_dir), expected_fingerprint=expected
                )
                second = enhance.load_enhancer(
                    str(model_dir), expected_fingerprint=expected
                )

        self.assertIs(first, loaded_backend)
        self.assertIs(second, loaded_backend)
        self.assertEqual(load_calls, [str(model_dir)])

    def test_enhancer_reloads_when_checkpoint_changes_on_disk(self) -> None:
        from core import enhance

        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "DeepFilterNet3"
            checkpoint_dir = model_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True)
            (model_dir / "config.ini").write_bytes(b"config")
            checkpoint = checkpoint_dir / "model_120.ckpt.best"
            checkpoint.write_bytes(b"checkpoint-v1")
            loaded_backends = [object(), object()]
            load_calls: list[dict | None] = []

            def fake_backend(model_dir_arg: str | None, fingerprint=None):
                load_calls.append(fingerprint)
                return loaded_backends[len(load_calls) - 1]

            with (
                mock.patch.object(enhance, "_ENHANCER_BACKEND", None),
                mock.patch.object(enhance, "_ENHANCER_CACHE_IDENTITY", None),
                mock.patch.object(
                    enhance, "_DeepFilterNetBackend", side_effect=fake_backend
                ),
            ):
                first = enhance.load_enhancer(str(model_dir))
                checkpoint.write_bytes(b"checkpoint-v2")
                second = enhance.load_enhancer(str(model_dir))

        self.assertIs(first, loaded_backends[0])
        self.assertIs(second, loaded_backends[1])
        self.assertEqual(len(load_calls), 2)

    def test_enhance_audio_serializes_backend_inference(self) -> None:
        from core import enhance

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            samples = np.full(48_000, 0.2, dtype=np.float32)
            backend = _OverlappingEnhancer(samples)
            inputs = []
            for index in range(2):
                input_wav = temp / f"original_{index}.wav"
                _write_pcm16_wav(input_wav, samples)
                inputs.append(
                    (
                        str(input_wav),
                        str(temp / f"full_{index}.wav"),
                        str(temp / f"mix_{index}.wav"),
                    )
                )

            with mock.patch.object(enhance, "load_enhancer", return_value=backend):
                with ThreadPoolExecutor(max_workers=2) as executor:
                    list(executor.map(lambda args: enhance.enhance_audio(*args), inputs))

        self.assertEqual(backend.max_active_calls, 1)


if __name__ == "__main__":
    unittest.main()
