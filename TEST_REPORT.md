# AudioRescue Public Test Report

> Rolling public report. Do not publish private paths, transcript text, raw logs, server details, model cache locations, or private sample identifiers. Items without current evidence stay pending.

## Candidate Component State

| Component | Candidate SHA | Status |
|---|---|---|
| C integration / PR #1 | `5f5addfa504164c3463a512a98e8b6878c091de7` | Frozen integration baseline |
| A backend / PR #8 | `31b35aaa1bfd523c29b6d53312b883b6710a8caf` | Frozen for integration; PR remains draft and unmerged |
| B UI / PR #5 | `2650d5bd5d3c968012be28a70b2799bfe87da467` | Frozen for integration; PR remains unmerged |
| Collaboration docs / PR #7 | `934c44ba7a542c26af4af12c4567e59c5c237c04` | Frozen for integration; separate docs PR remains unmerged |
| Public demo docs / PR #9 | this branch | Draft public documentation sync; not a product merge |
| Final four-way combination | AR-R3-B-021 accepted | Fixed heads combined without conflicts; targeted 185 OK / 1 skipped; full 285 OK / 1 skipped; `py_compile`, diff, repo-safety, fixture browser, privacy, and UI-core contract gates passed |

## Fixed Configuration

| Item | Public value |
|---|---|
| Contract | `v0.1-contract` |
| Audio normalization | 48 kHz, mono, PCM16 WAV |
| Enhancer | DeepFilterNet3 |
| Enhancement strength | default `0.75`; candidate options `0.50`, `0.75`, `1.00` |
| ASR | OpenAI Whisper multilingual `base` |
| Language/task | `zh` / `transcribe` |
| Device | `auto`: CUDA when available, otherwise CPU |
| Demo fixture | public synthetic fixture; useful for I/O and UI state only |

## Evidence Summary

| Evidence | Status | Public notes |
|---|---|---|
| A004 CPU real smoke | Accepted as CPU chain evidence | Cold run about 8.82 s and warm run about 6.85 s were reported on the prepared local environment; DeepFilterNet3 and Whisper base fingerprint prefixes matched the expected values; output tracks were 48 kHz, mono, PCM16, and about 12 s. The public synthetic fixture produced empty enhanced transcription, so it proves the chain runs but not effect quality. |
| A007/A009 smoke CLI safety | Implemented in PR #8 | CLI stdout is one safe JSON document; normal stderr should be empty; ordinary fatal errors return fixed `INTERNAL_ERROR` evidence with a non-zero exit code. Do not publish raw logs. |
| B019/B021 file delivery and UI gate | Accepted for integration | Client file delivery uses a controlled relative route with opaque IDs and neutral filenames; fixture browser gate reported external requests at zero and normal media HEAD/GET/Range as 200/200/206. Do not publish token URLs. |
| GPU supplement | Provisional | Server/GPU evidence is not formal speed evidence and is not part of the live demo path. |
| Final combined tree | Accepted for integration evidence | AR-R3-B-021 reported the frozen candidate heads combined without conflicts and without P0/P1 findings. This is not merge authorization and does not prove user-facing enhancement quality. |

## Public Automatic Test Log

| Time | Scope | Command | Result |
|---|---|---|---|
| 2026-07-14 | Historical C baseline | `python -m unittest discover -s tests -q` | Historical counts only; superseded by later PR checks |
| 2026-07-15 | A backend PR #8 | PR checks | Frozen head reported CLEAN with 6/6 success |
| 2026-07-15 | B UI PR #5 | PR checks | Frozen head reported CLEAN with 6/6 success |
| 2026-07-15 | Collaboration docs PR #7 | PR checks | Frozen head reported CLEAN with 6/6 success |
| 2026-07-15 | Final four-way combination | AR-R3-B-021 | No conflicts; targeted 185 OK / 1 skipped; full 285 OK / 1 skipped; `py_compile`, diff, repo-safety, browser, privacy, and UI-core contract gates passed |

## Required Local Checks

Lightweight checks that should not run real ASR or enhancement:

```bash
python -m py_compile app.py core/*.py ui/*.py scripts/*.py
python -m unittest discover -s tests -p "test_*.py" -q
git diff --check
```

Smoke checks that require prepared model assets and approved short samples:

```bash
python scripts/smoke_audio_core.py --normalize-only
python scripts/smoke_audio_core.py --runs 2 --asr-model base --device auto
```

## Still Required Before Final Freeze

- C final review and explicit merge decision.
- Offline rehearsal with fixture mode off.
- Manual A/B listening on approved samples.
- User-facing real-sample enhancement and transcription quality validation.
- Safe public evidence export: versions, elapsed time, fingerprint prefixes, test counts, pass/fail status only.

## Public Data Policy

- Public docs may say "approved short sample" or "private ledger".
- Public docs must not include private reference text, member-to-sample mappings, private recording filenames, raw transcripts, server details, model cache paths, or raw logs.
- Frozen evaluation data remains private and is not used for tuning or public debugging.
