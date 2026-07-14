# AudioRescue Public Test Report

> Rolling public report. Do not publish private paths, transcript text, raw logs, server details, model cache locations, or private sample identifiers. Items without current evidence stay pending.

## Candidate Component State

| Component | Candidate SHA | Status |
|---|---|---|
| C integration | `5f5addfa504164c3463a512a98e8b6878c091de7` | Baseline for this docs branch |
| A backend / PR #8 | `9b5bea652c90d2067115501fdfb3708bdd8add3b` | Checks green at last A report |
| B UI / PR #5 | `015d25bd149867ed558938adc81fb19c64260c61` | B012 reported checks green |
| Collaboration docs / PR #7 | `934c44ba7a542c26af4af12c4567e59c5c237c04` | Separate docs PR; not modified here |
| Final four-way combination | Pending | B013 is in progress; not yet accepted as final freeze |

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
| B012 file delivery | Reported complete by B | Client file delivery should use a controlled relative route with opaque IDs and neutral filenames; HEAD, GET, and Range are part of final validation. Do not publish token URLs. |
| GPU supplement | Provisional | Server/GPU evidence is not counted as final accepted evidence until the router records explicit acceptance. |
| Final combined tree | Pending | B013 must confirm the four candidate heads together before this report can say final combination passed. |

## Public Automatic Test Log

| Time | Scope | Command | Result |
|---|---|---|---|
| 2026-07-14 | Historical C baseline | `python -m unittest discover -s tests -q` | Historical counts only; superseded by later PR checks |
| 2026-07-14 | A backend PR #8 | PR checks | Reported 6/6 success at A009 handoff |
| 2026-07-14 | B UI PR #5 | PR checks | Reported 6/6 success at B012 handoff |
| 2026-07-14 | Final combination | B013 dispatch | In progress; no final pass claim yet |

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

- B013 final four-way combination handoff.
- C review of the combined tree and public report.
- Offline rehearsal with fixture mode off.
- Browser validation that external requests are zero.
- Verification that delivered file URLs use only the controlled relative route.
- Manual A/B listening on approved samples.
- Safe public evidence export: versions, elapsed time, fingerprint prefixes, test counts, pass/fail status only.

## Public Data Policy

- Public docs may say "approved short sample" or "private ledger".
- Public docs must not include private reference text, member-to-sample mappings, private recording filenames, raw transcripts, server details, model cache paths, or raw logs.
- Frozen evaluation data remains private and is not used for tuning or public debugging.
