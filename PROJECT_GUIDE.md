# PROJECT_GUIDE

## Scope

P0 covers offline short-audio normalization, enhancement, two-pass ASR, A/B comparison, visualization, text diffing, and structured local evidence. CLAP, streaming, training, account systems, and mobile clients are outside P0 unless explicitly re-scoped.

## Collaboration Rules

1. `main` stays runnable; feature work happens on branches.
2. One person or agent owns a file at a time.
3. Shared contracts change through the schema owner before A/B consume them.
4. Each commit should solve one reviewable problem.
5. Do not commit models, caches, full datasets, private audio, run outputs, logs, or local specs.
6. Reference text is for metrics and review only; it must not be passed into model prompts.
7. Do not edit ASR output manually before computing metrics.
8. A/B work starts from the current integration branch announced by C; do not force-push shared history.
9. Holdout-style evaluation splits run only after code, config, manifest, authorization, and commit state are frozen. If a guarded run records a receipt, do not delete it to rerun.

## File Ownership

- A: backend audio I/O, enhancement, ASR, smoke evidence, and related tests.
- B: app/UI, presenters, file staging/delivery, visualization, text diffing, and related tests.
- C: schemas, pipeline, metrics, cache/events, evaluation orchestration, integration docs, and release decisions.

## Local Data Boundary

- Source masters and standardized WAVs stay under ignored local data roots.
- Public Git may contain only synthetic fixtures or reviewed redistributable demo assets.
- Full manifests, private specs, reference text, holdout material, model weights, and run outputs are not committed.
- Dataset tooling reads a local spec when real collection details are needed. The repository tracks only the neutral example spec.
- Authorization or license evidence is recorded in the local private ledger; unknown rights means the asset is not used for demo or evaluation.

## Current Stable References

- Contract version: `v0.1-contract`
- Runtime config: `configs/app.yaml`
- A contract: `docs/A_BACKEND_CONTRACT_V1.md`
- Data-tool example spec: `configs/dataset_spec.example.json`
