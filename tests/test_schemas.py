import json
import unittest

from core.schemas import (
    ASRInferenceError,
    AudioLevelMetrics,
    AudioMeta,
    ErrorCode,
    ProcessResult,
    ProcessStatus,
    TranscriptResult,
)


class SchemaContractTest(unittest.TestCase):
    def test_success_result_is_json_serializable(self) -> None:
        meta = AudioMeta(
            source_name="fixture.wav",
            sample_rate=48_000,
            channels=1,
            duration_seconds=12.0,
            peak_abs=0.8,
            rms_dbfs=-20.0,
            clipped_ratio=0.0,
            silent_ratio=0.02,
            normalized_path="outputs/job/original.wav",
            original_sample_rate=44_100,
            original_channels=2,
            source_format="wav",
        )
        transcript = TranscriptResult(
            text="今天下午三点",
            language="zh",
            segments=[{"start": 0.0, "end": 2.0, "text": "今天下午三点"}],
            runtime_seconds=0.2,
            model_name="whisper-base",
        )
        result = ProcessResult(
            job_id="job",
            status=ProcessStatus.SUCCESS,
            input_meta=meta,
            original_audio_path="outputs/job/original.wav",
            enhanced_audio_path="outputs/job/enhanced_mix.wav",
            full_output_path="outputs/job/enhanced_full.wav",
            mixed_output_path="outputs/job/enhanced_mix.wav",
            original_levels=AudioLevelMetrics(peak_abs=0.8, rms_dbfs=-20.0),
            mixed_levels=AudioLevelMetrics(peak_abs=0.74, rms_dbfs=-21.2),
            transcript_before=transcript,
            transcript_after=transcript,
        )

        payload = result.to_dict()
        self.assertEqual(payload["status"], "success")
        self.assertEqual(
            payload["enhanced_audio_path"], payload["mixed_output_path"]
        )
        self.assertEqual(
            payload["original_levels"],
            {"peak_abs": 0.8, "rms_dbfs": -20.0},
        )
        self.assertEqual(
            payload["mixed_levels"],
            {"peak_abs": 0.74, "rms_dbfs": -21.2},
        )
        json.dumps(payload, ensure_ascii=False)

    def test_asr_error_maps_to_warning(self) -> None:
        error = ASRInferenceError("转写失败", detail="private traceback")
        warning = error.to_warning(code=ErrorCode.ASR_AFTER_FAILED)
        self.assertEqual(warning.code, ErrorCode.ASR_AFTER_FAILED)
        self.assertTrue(warning.recoverable)
        self.assertNotIn("traceback", warning.message)

    def test_mixed_output_populates_compatibility_alias(self) -> None:
        result = ProcessResult(
            job_id="job",
            status=ProcessStatus.PARTIAL,
            mixed_output_path="outputs/job/enhanced_mix.wav",
        )
        self.assertEqual(result.enhanced_audio_path, result.mixed_output_path)

    def test_digital_silence_levels_are_json_serializable(self) -> None:
        silence = AudioLevelMetrics(peak_abs=0.0, rms_dbfs=None)
        result = ProcessResult(
            job_id="silent-job",
            status=ProcessStatus.PARTIAL,
            original_levels=silence,
            mixed_levels=silence,
        )

        payload = result.to_dict()
        self.assertEqual(
            payload["original_levels"],
            {"peak_abs": 0.0, "rms_dbfs": None},
        )
        json.dumps(payload, ensure_ascii=False, allow_nan=False)

    def test_level_metrics_reject_invalid_peak(self) -> None:
        for peak in (-0.01, 1.01, float("nan"), float("inf")):
            with self.subTest(peak=peak), self.assertRaises(ValueError):
                AudioLevelMetrics(peak_abs=peak, rms_dbfs=-20.0)

    def test_level_metrics_reject_non_finite_rms(self) -> None:
        for rms in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(rms=rms), self.assertRaises(ValueError):
                AudioLevelMetrics(peak_abs=0.5, rms_dbfs=rms)

    def test_to_dict_rejects_non_json_and_non_finite_values(self) -> None:
        with self.assertRaises(TypeError):
            ProcessResult(
                job_id="bad-object",
                status=ProcessStatus.FAILED,
                config_snapshot={"unsafe": object()},
            ).to_dict()
        with self.assertRaises(ValueError):
            ProcessResult(
                job_id="bad-number",
                status=ProcessStatus.FAILED,
                config_snapshot={"unsafe": float("nan")},
            ).to_dict()


if __name__ == "__main__":
    unittest.main()
