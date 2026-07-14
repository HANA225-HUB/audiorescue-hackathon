# AudioRescue Offline Demo Runbook

This runbook is the shortest public path for preparing and running the competition demo without network access. It avoids machine-specific paths, server details, private audio names, transcript text, and model cache locations.

## Scope

- Production demo: fixture mode off; the UI calls `process_audio`.
- Fixture demo: UI state and offline page delivery only; it is not evidence that the real audio pipeline improves speech.
- Real smoke: only with approved short samples and already-prepared model assets.
- No server, tunnel, public URL, private recording, frozen test split, or remote model operation is part of the live demo path.

## Environment Rules

- Use Python 3.10 or 3.11.
- Install PyTorch and torchaudio as a matched pair: same version family and same CPU or CUDA wheel source.
- Do not install `latest` packages during the competition window.
- `requirements.txt` covers project-level packages; torch and torchaudio remain environment-specific prerequisites.
- FFmpeg must be on `PATH`.
- DeepFilterNet3, Whisper `base`, Gradio, and soundfile must import before rehearsal.
- GPU is optional. If CUDA is used, record it as an environment-specific acceleration choice, not a generic requirement.

## First Online Preparation

Run these while network access is available:

```bash
python -m pip check
ffmpeg -version
python -c "import torch, torchaudio, soundfile, whisper, gradio; import df"
```

Then prepare the model assets:

1. Load DeepFilterNet3 once.
2. Load Whisper `base` once with language `zh`.
3. Save only safe version and fingerprint summaries.
4. Do not publish raw logs, cache locations, model paths, or transcript text.

Run a no-model smoke first:

```bash
python scripts/smoke_audio_core.py --normalize-only
```

Run a real smoke only after model assets are prepared and the sample is approved:

```bash
python scripts/smoke_audio_core.py --runs 2 --asr-model base --device auto
```

The smoke CLI is expected to write one JSON document to stdout. Normal successful runs should keep stderr empty. Ordinary fatal errors are reported as a fixed `INTERNAL_ERROR` object with a non-zero exit code. Share only the safe JSON summary fields, not raw logs.

## Offline Rehearsal

Before disconnecting:

1. Confirm the production UI starts with fixture mode off.
2. Confirm approved demo inputs and precomputed fallback assets are available.
3. Confirm no model download is pending.
4. Keep the verified Python environment and model cache intact.

After disconnecting:

```bash
AUDIORESCUE_UI_FIXTURE=0 python app.py
```

Open the local page shown by Gradio. Process one approved short sample and check:

- original, mixed, and full enhanced audio are playable;
- output WAV files are 48 kHz, mono, PCM16;
- the page shows success, partial, or failed state clearly;
- file links use the controlled relative delivery route with opaque IDs and neutral filenames;
- HEAD, GET, and Range requests work for delivered files;
- external network requests are zero.

For UI-only fixture rehearsal:

```bash
AUDIORESCUE_UI_FIXTURE=1 python app.py
```

Fixture output may be used to verify page state, layout, and offline resource delivery only.

## Operator Handling

- Missing model: stop real processing, use the precomputed same-version fallback, and state that live model assets are unavailable.
- Empty transcription: report it as an ASR outcome; do not fill or edit text manually.
- Partial result: show playable artifacts and warnings; do not claim full success.
- Failed result: show the structured error state; do not publish raw logs.
- Slow run: switch to a shorter approved sample or precomputed fallback after the rehearsed timeout.

## Lightweight Checks

These checks do not run real ASR or enhancement:

```bash
python -m py_compile app.py core/*.py ui/*.py scripts/*.py
python -m unittest discover -s tests -p "test_*.py" -q
git diff --check
```

## Full Smoke Checks

These require prepared model assets and approved samples:

```bash
python scripts/smoke_audio_core.py --normalize-only
python scripts/smoke_audio_core.py --runs 2 --asr-model base --device auto
```

Record Python version, torch and torchaudio versions, FFmpeg availability, model fingerprint prefix, elapsed time, test count, and pass/fail status. Do not record private paths, raw logs, transcript text, or cache locations.

## Cleanup

- Stop the Gradio process.
- Remove only temporary outputs created during rehearsal.
- Remove only temporary staging files created for the current run.
- Do not delete the verified Python environment or prepared model cache.
- Do not clean private data by broad recursive commands during the competition window.
