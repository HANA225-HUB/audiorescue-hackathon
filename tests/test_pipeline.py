import json
import shutil
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import yaml

from core.pipeline import PipelineDependencies, process_audio
from core.schemas import (
    ASRInferenceError,
    AudioMeta,
    EnhancementError,
    ErrorCode,
    ProcessStatus,
    TranscriptResult,
    WarningItem,
)


def write_pcm16_wav(path: Path, *, frames: int = 48_000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(48_000)
        audio.writeframes(b"\x00\x00" * frames)


class FakeBackend:
    def __init__(self) -> None:
        self.normalize_calls = 0
        self.enhance_calls = 0
        self.asr_calls: list[str] = []
        self.asr_inference_identities: list[tuple[str, str]] = []
        self.fail_enhancement = False
        self.fail_after_asr = False
        self.fail_events = False
        self.meta_duration_seconds = 1.0
        self.reported_strength: float | None = None
        self.enhancement_model = "DeepFilterNet3"
        self.asr_model = "base"
        self.after_asr_error: str | None = None
        self.after_segments_override = None
        self.enhancement_warnings: list[WarningItem] = []
        self.enhancer_load_calls = 0
        self.asr_load_calls: list[tuple[str, str]] = []

    def normalize_audio(
        self,
        input_path: str,
        output_path: str,
        target_sr: int = 48_000,
        mono: bool = True,
    ) -> AudioMeta:
        self.normalize_calls += 1
        shutil.copyfile(input_path, output_path)
        return AudioMeta(
            source_name=Path(input_path).name,
            sample_rate=target_sr,
            channels=1 if mono else 2,
            duration_seconds=self.meta_duration_seconds,
            peak_abs=0.2,
            rms_dbfs=-24.0,
            clipped_ratio=0.0,
            silent_ratio=0.0,
            normalized_path=output_path,
            original_sample_rate=48_000,
            original_channels=1,
            source_format="wav",
        )

    def enhance_audio(
        self,
        input_wav: str,
        output_full_wav: str,
        output_mix_wav: str,
        strength: float = 0.75,
    ) -> dict:
        self.enhance_calls += 1
        if self.fail_enhancement:
            raise EnhancementError("测试增强失败")
        shutil.copyfile(input_wav, output_full_wav)
        shutil.copyfile(input_wav, output_mix_wav)
        return {
            "full_output_path": output_full_wav,
            "mixed_output_path": output_mix_wav,
            "strength": strength if self.reported_strength is None else self.reported_strength,
            "runtime_seconds": 0.01,
            "model_name": self.enhancement_model,
            "warnings": self.enhancement_warnings,
        }

    def transcribe_audio(
        self,
        audio_path: str,
        language: str = "zh",
        *,
        model_name: str = "base",
        device: str = "auto",
    ) -> TranscriptResult:
        self.asr_calls.append(audio_path)
        self.asr_inference_identities.append((model_name, device))
        is_after = Path(audio_path).name == "enhanced_mix.wav"
        if is_after and self.fail_after_asr:
            raise ASRInferenceError("测试增强轨转写失败")
        text = "今天下午三点" if is_after else "今天下午两点"
        return TranscriptResult(
            text=text,
            language=language,
            segments=(
                self.after_segments_override
                if is_after and self.after_segments_override is not None
                else [{"start": 0.0, "end": 1.0, "text": text}]
            ),
            runtime_seconds=0.01,
            model_name=f"whisper-{self.asr_model}",
            error=self.after_asr_error if is_after else None,
        )

    @staticmethod
    def create_waveform_comparison(
        original_wav: str, enhanced_wav: str, output_path: str
    ) -> str:
        Path(output_path).write_bytes(b"fake waveform png")
        return output_path

    @staticmethod
    def create_spectrogram_comparison(
        original_wav: str, enhanced_wav: str, output_path: str
    ) -> str:
        Path(output_path).write_bytes(b"fake spectrogram png")
        return output_path

    @staticmethod
    def build_text_diff(before: str, after: str) -> str:
        return f"<span>{before}</span><span>{after}</span>"

    def detect_events(self, *args, **kwargs):
        if self.fail_events:
            raise RuntimeError("optional detector unavailable")
        return []

    def load_enhancer(self):
        self.enhancer_load_calls += 1
        return object()

    def load_asr(self, model_name: str, device: str):
        self.asr_load_calls.append((model_name, device))
        return object()

    def dependencies(self, *, include_loaders: bool = False) -> PipelineDependencies:
        return PipelineDependencies(
            normalize_audio=self.normalize_audio,
            enhance_audio=self.enhance_audio,
            transcribe_audio=self.transcribe_audio,
            create_waveform_comparison=self.create_waveform_comparison,
            create_spectrogram_comparison=self.create_spectrogram_comparison,
            build_text_diff=self.build_text_diff,
            detect_events=self.detect_events,
            load_enhancer=self.load_enhancer if include_loaders else None,
            load_asr=self.load_asr if include_loaders else None,
        )


class PipelineIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.input_path = self.root / "input.wav"
        write_pcm16_wav(self.input_path)
        self.config_path = self.root / "app.yaml"
        self.output_root = self.root / "outputs"
        self.cache_enabled = False
        self._write_config()
        self.backend = FakeBackend()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _write_config(self) -> None:
        config = {
            "contract_version": "v0.1-contract",
            "app": {"output_root": str(self.output_root), "concurrency": 1},
            "audio": {
                "target_sample_rate": 48_000,
                "channels": 1,
                "pcm_subtype": "PCM_16",
                "min_duration_seconds": 1,
                "max_duration_seconds": 60,
            },
            "enhancement": {
                "model": "DeepFilterNet3",
                "default_strength": 0.75,
            },
            "asr": {
                "model": "base",
                "language": "zh",
                "device": "cpu",
                "temperature": 0.0,
            },
            "comparison": {"loudness_dsp_matching": False},
            "cache": {
                "enabled": self.cache_enabled,
                "demo_only": False,
                "allow_user_audio": True,
            },
        }
        self.config_path.write_text(
            yaml.safe_dump(config, allow_unicode=True), encoding="utf-8"
        )

    def run_pipeline(self, **overrides):
        arguments = {
            "input_path": str(self.input_path),
            "strength": 0.75,
            "enable_events": False,
            "reference_text": "今天下午三点",
            "force_recompute": False,
        }
        arguments.update(overrides)
        with (
            patch.dict("os.environ", {"AUDIORESCUE_CONFIG": str(self.config_path)}),
            patch(
                "core.pipeline._load_default_dependencies",
                return_value=self.backend.dependencies(),
            ),
            patch("core.pipeline._PROJECT_ROOT", self.root),
        ):
            return process_audio(**arguments)

    def test_success_persists_complete_result(self) -> None:
        result = self.run_pipeline()

        self.assertEqual(result.status, ProcessStatus.SUCCESS)
        self.assertEqual(result.enhanced_audio_path, result.mixed_output_path)
        self.assertIsNotNone(result.transcript_before)
        self.assertIsNotNone(result.transcript_after)
        self.assertIsNotNone(result.cer_before)
        self.assertIsNotNone(result.cer_after)
        self.assertEqual(result.original_levels.peak_abs, 0.0)
        self.assertIsNone(result.original_levels.rms_dbfs)
        self.assertEqual(result.mixed_levels.peak_abs, 0.0)
        self.assertIsNone(result.mixed_levels.rms_dbfs)
        self.assertTrue(Path(result.waveform_path).is_file())
        self.assertTrue(Path(result.spectrogram_path).is_file())

        job_root = Path(result.original_audio_path).parent
        persisted = json.loads((job_root / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(persisted["status"], "success")
        self.assertGreaterEqual(persisted["runtime"]["total_seconds"], 0.0)
        self.assertTrue((job_root / "run.log").is_file())

    def test_one_asr_failure_is_partial_and_preserves_other_results(self) -> None:
        self.backend.fail_after_asr = True
        result = self.run_pipeline()

        self.assertEqual(result.status, ProcessStatus.PARTIAL)
        self.assertIsNotNone(result.transcript_before)
        self.assertIsNone(result.transcript_after)
        self.assertTrue(Path(result.mixed_output_path).is_file())
        self.assertIn(ErrorCode.ASR_AFTER_FAILED, [item.code for item in result.warnings])

    def test_enhancement_failure_is_failed_but_keeps_original_asr(self) -> None:
        self.backend.fail_enhancement = True
        result = self.run_pipeline()

        self.assertEqual(result.status, ProcessStatus.FAILED)
        self.assertIsNotNone(result.original_audio_path)
        self.assertIsNotNone(result.transcript_before)
        self.assertIsNone(result.mixed_output_path)
        self.assertIn(ErrorCode.ENHANCE_FAILED, [item.code for item in result.warnings])

    def test_optional_event_failure_does_not_downgrade_p0(self) -> None:
        self.backend.fail_events = True
        result = self.run_pipeline(enable_events=True)

        self.assertEqual(result.status, ProcessStatus.SUCCESS)
        self.assertIn(ErrorCode.EVENTS_SKIPPED, [item.code for item in result.warnings])

    def test_invalid_request_does_not_load_dependencies(self) -> None:
        missing = self.root / "missing.wav"
        with (
            patch.dict("os.environ", {"AUDIORESCUE_CONFIG": str(self.config_path)}),
            patch("core.pipeline._load_default_dependencies") as dependency_loader,
            patch("core.pipeline._PROJECT_ROOT", self.root),
        ):
            result = process_audio(str(missing))

        self.assertEqual(result.status, ProcessStatus.FAILED)
        dependency_loader.assert_not_called()
        self.assertIn(ErrorCode.INPUT_INVALID, [item.code for item in result.warnings])

    def test_second_identical_request_uses_cache(self) -> None:
        self.cache_enabled = True
        self._write_config()
        first = self.run_pipeline()
        normalize_calls = self.backend.normalize_calls
        job_directories = set(self.output_root.iterdir())
        second = self.run_pipeline()

        self.assertEqual(first.status, ProcessStatus.SUCCESS)
        self.assertEqual(second.status, ProcessStatus.SUCCESS)
        self.assertTrue(second.runtime.cache_hit)
        self.assertEqual(self.backend.normalize_calls, normalize_calls)
        self.assertEqual(set(self.output_root.iterdir()), job_directories)
        self.assertIn(ErrorCode.CACHE_USED, [item.code for item in second.warnings])

    def test_incomplete_success_cache_is_a_miss_and_recomputes(self) -> None:
        self.cache_enabled = True
        self._write_config()
        first = self.run_pipeline()
        Path(first.waveform_path).unlink()
        normalize_calls = self.backend.normalize_calls

        second = self.run_pipeline()

        self.assertEqual(second.status, ProcessStatus.SUCCESS)
        self.assertFalse(second.runtime.cache_hit)
        self.assertEqual(self.backend.normalize_calls, normalize_calls + 1)
        self.assertTrue(Path(second.waveform_path).is_file())

    def test_corrupt_nonempty_cached_wav_is_a_miss_and_recomputes(self) -> None:
        self.cache_enabled = True
        self._write_config()
        first = self.run_pipeline()
        Path(first.mixed_output_path).write_bytes(b"not a wav" * 20)
        normalize_calls = self.backend.normalize_calls

        second = self.run_pipeline()

        self.assertEqual(second.status, ProcessStatus.SUCCESS)
        self.assertFalse(second.runtime.cache_hit)
        self.assertEqual(self.backend.normalize_calls, normalize_calls + 1)

    def test_missing_optional_partial_artifact_is_a_cache_miss(self) -> None:
        self.cache_enabled = True
        self._write_config()
        self.backend.fail_after_asr = True
        first = self.run_pipeline()
        Path(first.waveform_path).unlink()
        normalize_calls = self.backend.normalize_calls

        second = self.run_pipeline()

        self.assertEqual(second.status, ProcessStatus.PARTIAL)
        self.assertFalse(second.runtime.cache_hit)
        self.assertEqual(self.backend.normalize_calls, normalize_calls + 1)

    def test_cache_failure_never_fails_p0(self) -> None:
        with patch(
            "core.pipeline._prepare_cache",
            side_effect=PermissionError("read-only cache"),
        ):
            result = self.run_pipeline()

        self.assertEqual(result.status, ProcessStatus.SUCCESS)
        self.assertFalse(result.runtime.cache_hit)

    def test_force_recompute_bypasses_existing_cache(self) -> None:
        self.cache_enabled = True
        self._write_config()
        first = self.run_pipeline()
        normalize_calls = self.backend.normalize_calls
        second = self.run_pipeline(force_recompute=True)

        self.assertEqual(first.status, ProcessStatus.SUCCESS)
        self.assertEqual(second.status, ProcessStatus.SUCCESS)
        self.assertFalse(second.runtime.cache_hit)
        self.assertEqual(self.backend.normalize_calls, normalize_calls + 1)

    def test_real_wav_duration_rejects_lying_short_metadata(self) -> None:
        write_pcm16_wav(self.input_path, frames=4_800)
        self.backend.meta_duration_seconds = 1.0
        result = self.run_pipeline()

        self.assertEqual(result.status, ProcessStatus.FAILED)
        self.assertIn(ErrorCode.INPUT_INVALID, [item.code for item in result.warnings])

    def test_real_wav_duration_rejects_lying_overlong_metadata(self) -> None:
        write_pcm16_wav(self.input_path, frames=48_000 * 61)
        self.backend.meta_duration_seconds = 1.0
        result = self.run_pipeline()

        self.assertEqual(result.status, ProcessStatus.FAILED)
        self.assertIn(ErrorCode.INPUT_TOO_LONG, [item.code for item in result.warnings])

    def test_wrong_enhancement_strength_is_failed(self) -> None:
        self.backend.reported_strength = 0.5
        result = self.run_pipeline(strength=0.75)

        self.assertEqual(result.status, ProcessStatus.FAILED)
        self.assertIn(ErrorCode.OUTPUT_INVALID, [item.code for item in result.warnings])

    def test_asr_error_field_is_partial_not_success(self) -> None:
        self.backend.after_asr_error = "decoder failed"
        result = self.run_pipeline()

        self.assertEqual(result.status, ProcessStatus.PARTIAL)
        self.assertIn(ErrorCode.ASR_AFTER_FAILED, [item.code for item in result.warnings])

    def test_nan_audio_meta_is_failed_but_result_json_is_valid(self) -> None:
        self.backend.meta_duration_seconds = float("nan")
        result = self.run_pipeline()

        self.assertEqual(result.status, ProcessStatus.FAILED)
        self.assertIn(ErrorCode.INPUT_INVALID, [item.code for item in result.warnings])
        persisted = json.loads(
            (self.output_root / result.job_id / "result.json").read_text(encoding="utf-8")
        )
        self.assertEqual(persisted["status"], "failed")

    def test_non_json_asr_segment_is_partial_and_result_is_persisted(self) -> None:
        self.backend.after_segments_override = [object()]
        result = self.run_pipeline()

        self.assertEqual(result.status, ProcessStatus.PARTIAL)
        self.assertIsNone(result.transcript_after)
        self.assertIn(ErrorCode.ASR_AFTER_FAILED, [item.code for item in result.warnings])
        job_root = self.output_root / result.job_id
        persisted = json.loads((job_root / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(persisted["status"], "partial")
        self.assertFalse((job_root / "transcript_after.json").exists())

    def test_nonfinite_and_bool_strength_are_rejected_before_backend(self) -> None:
        for strength in (float("nan"), float("inf"), True, "0.75"):
            with self.subTest(strength=strength):
                result = self.run_pipeline(strength=strength)
                self.assertEqual(result.status, ProcessStatus.FAILED)
                self.assertIn(
                    ErrorCode.INPUT_INVALID, [item.code for item in result.warnings]
                )
                json.loads(
                    (self.output_root / result.job_id / "result.json").read_text(
                        encoding="utf-8"
                    )
                )
        self.assertEqual(self.backend.normalize_calls, 0)

    def test_non_json_enhancement_warning_fails_contract_safely(self) -> None:
        self.backend.enhancement_warnings = [
            WarningItem(
                code=ErrorCode.OUTPUT_PEAK_PROTECTED,
                message="test",
                module="enhance",
                recoverable=True,
                details={"unsafe": object()},
            )
        ]
        result = self.run_pipeline()

        self.assertEqual(result.status, ProcessStatus.FAILED)
        self.assertIn(ErrorCode.OUTPUT_INVALID, [item.code for item in result.warnings])
        json.loads(
            (self.output_root / result.job_id / "result.json").read_text(encoding="utf-8")
        )

    def test_warmup_key_changes_with_model(self) -> None:
        dependencies = self.backend.dependencies(include_loaders=True)
        with (
            patch.dict("os.environ", {"AUDIORESCUE_CONFIG": str(self.config_path)}),
            patch("core.pipeline._load_default_dependencies", return_value=dependencies),
            patch("core.pipeline._PROJECT_ROOT", self.root),
            patch("core.pipeline._ENHANCER_WARMED_KEY", None),
            patch("core.pipeline._ASR_WARMED_KEY", None),
        ):
            first = process_audio(str(self.input_path), reference_text="今天下午三点")
            config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
            config["asr"]["model"] = "tiny"
            self.config_path.write_text(
                yaml.safe_dump(config, allow_unicode=True), encoding="utf-8"
            )
            self.backend.asr_model = "tiny"
            second = process_audio(str(self.input_path), reference_text="今天下午三点")

        self.assertEqual(first.status, ProcessStatus.SUCCESS)
        self.assertEqual(second.status, ProcessStatus.SUCCESS)
        self.assertEqual(self.backend.enhancer_load_calls, 1)
        self.assertEqual(self.backend.asr_load_calls, [("base", "cpu"), ("tiny", "cpu")])
        self.assertEqual(
            self.backend.asr_inference_identities,
            [
                ("base", "cpu"),
                ("base", "cpu"),
                ("tiny", "cpu"),
                ("tiny", "cpu"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
