# A Backend Contract V1

> Owner: C integration layer
> Applies to: `core/audio_io.py`, `core/enhance.py`, `core/transcribe.py`, `core/pipeline.py`

This document records the public A-layer runtime contract. It intentionally uses neutral dataset examples only. Runtime schemas remain defined by `core/schemas.py`; schema changes must be reviewed through the integration owner before A/B code depends on them.

## 1. Public Schemas

A-facing code may import these public schema and exception types from `core.schemas`:

```text
AudioLevelMetrics
AudioMeta
TranscriptResult
TranscriptSegment
EnhancementOutput
WarningItem
ErrorCode
InputAudioError
InputTooLongError
EnhancementError
OutputValidationError
ASRInferenceError
```

Nullable fields:

- `AudioMeta.rms_dbfs` and `silent_ratio` may be `None` when they cannot be computed reliably.
- `AudioMeta.original_sample_rate`, `original_channels`, and `source_format` may be `None` when source metadata is unavailable.
- `TranscriptResult.language` may be `None` when the ASR backend does not report it.
- `TranscriptResult.error` must be `None` for successful or valid empty transcripts; real ASR failures raise `ASRInferenceError`.
- `ProcessResult` fields for paths, transcripts, CER, figures, and level metrics may be `None` when that stage did not run or did not complete.
- `events`, `warnings`, and `config_snapshot` are never `None`; they are empty collections when there is nothing to report.

## 2. A-Layer Function Signatures

```python
def normalize_audio(
    input_path: str,
    output_path: str,
    target_sr: int = 48_000,
    mono: bool = True,
) -> AudioMeta:
    ...


def inspect_audio(samples, sample_rate: int) -> dict:
    ...


def load_enhancer(model_dir: str | None = None):
    ...


def enhance_audio(
    input_wav: str,
    output_full_wav: str,
    output_mix_wav: str,
    strength: float = 0.75,
) -> EnhancementOutput:
    ...


def load_asr(model_name: str, device: str):
    ...


def transcribe_audio(
    audio_path: str,
    language: str = "zh",
    *,
    model_name: str = "base",
    device: str = "auto",
) -> TranscriptResult:
    ...
```

`EnhancementOutput` contains:

```text
full_output_path
mixed_output_path
strength
runtime_seconds
model_name
warnings: list[WarningItem]
```

`TranscriptResult.segments` must be strict JSON data:

```json
[{"start": 0.0, "end": 2.4, "text": "neutral transcript text"}]
```

Do not return backend objects, tensors, model instances, or any object that cannot be serialized with strict JSON rules.

## 3. Output Directory Contract

The pipeline creates the job directory and owns the final result files:

```text
outputs/{job_id}/
├── original.wav
├── enhanced_full.wav
├── enhanced_mix.wav
├── transcript_before.json
├── transcript_after.json
├── waveform_compare.png
├── spectrogram_compare.png
├── result.json
└── run.log
```

A writes only the paths passed by the pipeline:

- `original.wav`
- `enhanced_full.wav`
- `enhanced_mix.wav`

A does not choose the `job_id`, rename output files, or write `result.json`. It may create parent directories for the exact target paths, but it must not alter the directory layout. A write failure must not overwrite source audio or create an empty success artifact.

## 4. Errors, Warnings, And Status

| Situation | A behavior | Pipeline warning code | Final status |
|---|---|---|---|
| Missing, empty, undecodable, or zero-duration input | Raise `InputAudioError` | `INPUT_INVALID` | `failed` |
| Input longer than the configured maximum | Raise `InputTooLongError`; do not truncate silently | `INPUT_TOO_LONG` | `failed` |
| Clipped input | Return metadata with `INPUT_CLIPPED` warning | `INPUT_CLIPPED` | not downgraded by itself |
| Near-silent input | Return metadata with `INPUT_NEAR_SILENT` warning | `INPUT_NEAR_SILENT` | not downgraded by itself; empty ASR may later make it `partial` |
| Resample or downmix occurred | Return source metadata and warning | `INPUT_RESAMPLED` / `INPUT_STEREO_DOWNMIXED` | not downgraded |
| Enhancement backend failure | Raise `EnhancementError` | `ENHANCE_FAILED` | `failed` |
| Enhanced output empty, invalid, NaN, or unreadable | Raise `OutputValidationError` | `OUTPUT_INVALID` | `failed` |
| ASR inference failure | Raise `ASRInferenceError` | pipeline maps before/after failure | `partial` if enhancement succeeded |
| ASR returns valid empty text | Return `text=""`, `error=None` | `ASR_EMPTY` | `partial` |
| Visualization failure | A does not handle it | `VIS_FAILED` | `partial` |
| Event tagging disabled | A does not handle it | none | no effect |
| Event tagging enabled but failed | A does not handle it | `EVENTS_SKIPPED` | no effect on P0 |
| Cache hit | A does not handle it | `CACHE_USED` | preserve cached status |

Additional rules:

- Before and after ASR are independent. Failure on one side must not cancel the other side.
- If enhancement fails, the pipeline may preserve original-track outputs, but it must not fabricate enhanced-track success.
- User-facing messages must not include tracebacks. Full exception details belong only in local logs.
- `ASRInferenceError` is stage-agnostic; the pipeline maps it to the before or after warning code according to call site.

## 5. Frozen Runtime Configuration

The public runtime configuration is defined by `configs/app.yaml`.

| Item | V1 contract |
|---|---|
| Normalized audio | WAV, 48 kHz, mono, PCM16 |
| Input duration | 1-60 seconds by default |
| Enhancement backend | DeepFilterNet3-compatible enhancer |
| Default dry/wet | `mixed=(1-strength)*original+strength*enhanced` |
| Default strength | `0.75` |
| ASR model class | multilingual Whisper-compatible model |
| ASR language/task | `zh` / `transcribe` |
| Device selection | `auto`; runtime resolves to an available supported device |
| Decode policy | `temperature=0`, `condition_on_previous_text=false`, no `initial_prompt` |
| Concurrency | 1 |

Original and enhanced tracks must use the same ASR model, language, device policy, and decoding settings. Reference text is only allowed in the CER computation layer after ASR output exists; it must never become an ASR prompt or model-facing hint.

## 6. A/B Loudness And Playback Fairness

P0 does not add loudness DSP to make one side sound better. The frozen rules are:

1. Keep `original.wav`, `enhanced_full.wav`, and `enhanced_mix.wav` audit-ready.
2. Use the same player volume and system volume for both A/B sides.
3. Report both tracks' `rms_dbfs` and `peak_abs`.
4. Blind listening must not treat "louder" as "clearer" without evidence.
5. The page's default after-audio playback and after-ASR use `enhanced_mix.wav`.
6. `enhanced_full.wav` is for full-strength debugging and download.
7. `ProcessResult.enhanced_audio_path` is a compatibility alias for `mixed_output_path`.

Public loudness evidence:

```python
AudioLevelMetrics(
    peak_abs: float,        # finite, 0.0..1.0
    rms_dbfs: float | None, # finite when non-silent; None for digital silence
)
```

`ProcessResult.original_levels` corresponds to `original_audio_path`; `ProcessResult.mixed_levels` corresponds to `mixed_output_path`. Do not use `NaN`, `Infinity`, or `-Infinity` to represent silence.

If LUFS is added later, it must create a new derived playback copy rather than overwrite the three standard tracks, and the contract version must be updated.

## 7. Cache Responsibility

- `core/cache.py` owns job-result caching, cache keys, cache-hit display, and invalidation.
- A owns only process-local enhancer/ASR model singletons and official backend weight caches.
- A does not cache `ProcessResult` and must not bypass processing based on filenames.
- A changed input hash, strength, model, language, code version, or config version must invalidate old results.

## 8. Runtime Timing Fields

`RuntimeStats` reports:

```text
decode_seconds
enhancer_load_seconds
enhancement_seconds
asr_load_seconds
asr_before_seconds
asr_after_seconds
visualization_seconds
event_seconds
persistence_seconds
total_seconds
cache_hit
cold_start
device
```

Timing definitions:

- `enhance_audio.runtime_seconds` measures only the enhancement call, not model loading.
- `TranscriptResult.runtime_seconds` measures that track's audio preparation and decode, not model loading.
- If models are preloaded before a job, job-level load fields are zero and `cold_start=false`.
- If the first request lazy-loads models, load time is recorded and `cold_start=true`.
- `total_seconds` covers `process_audio()` entry through durable `result.json` persistence, including model load that happens inside the request.
- Reports should distinguish cold-start and warm-run timings; user-facing headline numbers should identify the condition used.

## 9. Public Fixture And Local Dataset Roles

The repository fixture under `tests/fixtures/` is synthetic and contains no private speech or reference transcript. It is suitable for decode, enhancement, output-layout, structured-error, and cache checks. It is not CER evidence.

For local dataset work, use an ignored root and an explicit local spec:

```text
data_local/controlled/dev/sample_mix_1.wav
data_local/raw/clean/speaker_1/sample_clean_1.wav
data_local/raw/noise/noise_class_1.wav
```

Protected or holdout-style splits remain unavailable for tuning until code, config, manifest, authorization, dataset spec, and commit state are frozen. Public documentation must describe roles and fields, not private member mappings, exact private counts, candidate IDs, server paths, transcripts, or local absolute paths.

Expected core artifacts remain:

```text
outputs/{job_id}/original.wav
outputs/{job_id}/enhanced_full.wav
outputs/{job_id}/enhanced_mix.wav
```

Acceptance checks require readable WAV files, 48 kHz mono PCM16, finite metadata, structured errors, and no leakage of local private paths, reference text, credentials, model cache paths, or transcript contents beyond intended transcript fields.

## 10. Event Tagging Boundary

A provides the normalized `original.wav` path and `AudioMeta` to the pipeline. Event-tagging code reads the original track and does not modify normalization, enhancement, ASR, or CER behavior. When event tagging is disabled, it produces no warning; when explicitly enabled and unavailable, the pipeline appends `EVENTS_SKIPPED`.
