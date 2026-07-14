import json
import unittest

from core.schemas import (
    ASRInferenceError,
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
            transcript_before=transcript,
            transcript_after=transcript,
        )

        payload = result.to_dict()
        self.assertEqual(payload["status"], "success")
        self.assertEqual(
            payload["enhanced_audio_path"], payload["mixed_output_path"]
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


if __name__ == "__main__":
    unittest.main()
