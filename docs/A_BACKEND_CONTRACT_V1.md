# A Backend Contract V1

This document records the public A-layer contract without private dataset values. Runtime schemas remain defined in `core/schemas.py`.

## Audio I/O

- Inputs are local audio files selected by the caller.
- Normalization outputs 48 kHz, mono, PCM16 WAV.
- Evidence may include duration, sample rate, channels, peak, RMS, clipping ratio, silence ratio, and SHA-256.
- Paths in public reports must be relative, opaque, or otherwise sanitized.

## Enhancement

`enhance_audio(input_wav, output_full_wav, output_mix_wav, strength)` writes:

- `enhanced_full.wav`: full enhanced signal.
- `enhanced_mix.wav`: dry/wet mix used for default playback.

The function returns structured runtime metadata and warnings. Backend exceptions are wrapped as structured errors.

## ASR

`transcribe_audio(audio_path, language="zh", model_name=<value>, device=<value>)` returns a `TranscriptResult`.

- Raw `model_name` and `device` reach the ASR backend.
- Public JSON summarizes custom model/device values with safe labels.
- Reference text must never be passed as an ASR prompt.

## Smoke Evidence

The public smoke fixture under `tests/fixtures/` is synthetic and has no reference transcript. It is suitable for checking decode, enhancement, output layout, and structured errors; it is not CER evidence.

For real local data, callers must use an ignored dataset root and a local spec. Public docs should use neutral examples such as:

```text
data_local/controlled/dev/sample_mix_1.wav
data_local/raw/clean/speaker_1/sample_clean_1.wav
```

Holdout-style rows remain unavailable for tuning until code, config, manifest, authorization, and commit state are frozen.

## Output Layout

Expected job artifacts:

```text
outputs/{job_id}/original.wav
outputs/{job_id}/enhanced_full.wav
outputs/{job_id}/enhanced_mix.wav
outputs/{job_id}/result.json
```

Acceptance checks verify readable WAVs, finite numeric metadata, structured errors, and no leakage of local private paths, reference text, credentials, model cache paths, or transcript contents beyond the intended transcript fields.
