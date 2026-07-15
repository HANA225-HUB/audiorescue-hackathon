"""Initialize an ignored local AudioRescue dataset workspace from a spec.

The generated tree is intentionally private. This script only creates
directories and text templates; it never creates, moves, renames, or overwrites
audio recordings.
"""

from __future__ import annotations

import argparse
import csv
import io
from dataclasses import dataclass
from pathlib import Path

from scripts.dataset_spec import (
    DatasetSpec,
    DatasetSpecError,
    ensure_private_repo_root,
    load_dataset_spec,
    safe_join,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class InitReport:
    root: Path
    created_directories: tuple[Path, ...]
    created_files: tuple[Path, ...]
    preserved_files: tuple[Path, ...]


def _render_tsv(rows: list[dict[str, str]], fieldnames: list[str]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames, delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _render_csv(rows: list[dict[str, str]], fieldnames: list[str]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _directories(spec: DatasetSpec) -> tuple[str, ...]:
    paths = {Path("outputs"), Path("source_original/clean"), Path("source_original/noise"), Path("source_original/real")}
    for path in spec.required_audio_paths():
        paths.add(path.parent)
    return tuple(path.as_posix() for path in sorted(paths, key=lambda item: item.as_posix()))


def _transcript_rows(spec: DatasetSpec) -> list[dict[str, str]]:
    return [
        {
            "clean_id": item.id,
            "speaker_id": item.speaker_id,
            "sentence_id": item.sentence_id,
            "split": item.split,
            "clean_path": item.path.as_posix(),
            "reference_text": item.reference_text,
        }
        for item in spec.clean_recordings
    ]


def _metadata_rows(spec: DatasetSpec) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for item in spec.clean_recordings:
        rows.append(
            {
                "asset_id": item.id,
                "source_type": "clean",
                "source_original_path": "",
                "relative_path": item.path.as_posix(),
                "speaker_id": item.speaker_id,
                "sentence_id": item.sentence_id,
                "noise_type": "",
                "recording_device": "TODO",
                "recording_location": "TODO",
                "recording_date": "TODO",
                "recorder": "TODO",
                "distance_cm": "TODO",
                "consent_status": spec.consent_tokens.pending,
                "source_original_sha256": "",
                "standardized_sha256": "",
                "notes": "",
            }
        )
    for item in spec.noise_recordings:
        rows.append(
            {
                "asset_id": item.id,
                "source_type": "noise",
                "source_original_path": "",
                "relative_path": item.path.as_posix(),
                "speaker_id": "",
                "sentence_id": "",
                "noise_type": item.noise_type,
                "recording_device": "TODO",
                "recording_location": "TODO",
                "recording_date": "TODO",
                "recorder": "TODO",
                "distance_cm": "",
                "consent_status": spec.consent_tokens.pending,
                "source_original_sha256": "",
                "standardized_sha256": "",
                "notes": "Avoid identifiable private speech.",
            }
        )
    for item in spec.real_recordings:
        rows.append(
            {
                "asset_id": item.sample_id,
                "source_type": "real",
                "source_original_path": "",
                "relative_path": item.path.as_posix(),
                "speaker_id": item.speaker_id,
                "sentence_id": item.sentence_id,
                "noise_type": item.noise_type,
                "recording_device": "TODO",
                "recording_location": "TODO",
                "recording_date": "TODO",
                "recorder": "TODO",
                "distance_cm": "TODO",
                "consent_status": spec.consent_tokens.pending,
                "source_original_sha256": "",
                "standardized_sha256": "",
                "notes": "",
            }
        )
    return rows


def _readme_text(spec: DatasetSpec) -> str:
    return f"""# Local AudioRescue Dataset Workspace

This directory is private and must stay ignored by Git. It is initialized from a local dataset spec and contains placeholders only.

Dataset version: `{spec.dataset_version}`

Workflow:
- Put immutable source recordings under `source_original/`.
- Put standardized 48 kHz mono PCM16 WAV files under the relative paths declared by the spec.
- Fill `recording_metadata.csv` with source and standardized SHA-256 values before using the formal approval token.
- Generate `manifest.csv` with `scripts/build_dataset.py --spec <local-spec>`.
- Keep holdout-style splits isolated until the agreed one-shot evaluation point.

No audio files were generated by this initializer.
"""


def _licenses_text(spec: DatasetSpec) -> str:
    return f"""# Dataset Authorization Record

Nothing in this workspace may be published, submitted, or used for evaluation until the local owner records an explicit decision.

Allowed manifest tokens for this spec:
- development: `{spec.consent_tokens.pending}`
- formal evaluation: `{spec.consent_tokens.approved}`

Record consent per source in `recording_metadata.csv`; do not infer public release rights from local playback permission.
"""


def _require_private_root(dataset_root: Path) -> None:
    """Refuse a repository-local dataset directory that Git could track."""

    try:
        ensure_private_repo_root(dataset_root, project_root=PROJECT_ROOT)
    except DatasetSpecError as exc:
        raise ValueError(
            "dataset root inside the repository must be ignored by Git; "
            "use data_local/ or add an explicit ignore rule before initialization"
        ) from exc


def initialize_dataset(
    root: str | Path,
    *,
    spec_path: str | Path | None = None,
    example_mode: bool = False,
) -> InitReport:
    """Create an idempotent private workspace without overwriting text or audio."""

    dataset_root = Path(root).expanduser().resolve()
    _require_private_root(dataset_root)
    if dataset_root.exists() and not dataset_root.is_dir():
        raise NotADirectoryError("dataset root is not a directory")
    spec = load_dataset_spec(spec_path, allow_example=example_mode)
    created_directories: list[Path] = []
    created_files: list[Path] = []
    preserved_files: list[Path] = []

    if not dataset_root.exists():
        dataset_root.mkdir(parents=True)
        created_directories.append(dataset_root)

    for relative in _directories(spec):
        try:
            directory = safe_join(dataset_root, relative)
        except DatasetSpecError as exc:
            raise ValueError("unsafe path declared by dataset spec") from exc
        if not directory.exists():
            directory.mkdir(parents=True)
            created_directories.append(directory)
        elif not directory.is_dir():
            raise NotADirectoryError(f"expected directory but found file: {directory}")

    templates = {
        dataset_root / "README.md": _readme_text(spec),
        dataset_root / "transcripts.tsv": _render_tsv(
            _transcript_rows(spec),
            [
                "clean_id",
                "speaker_id",
                "sentence_id",
                "split",
                "clean_path",
                "reference_text",
            ],
        ),
        dataset_root / "recording_metadata.csv": _render_csv(
            _metadata_rows(spec),
            [
                "asset_id",
                "source_type",
                "source_original_path",
                "relative_path",
                "speaker_id",
                "sentence_id",
                "noise_type",
                "recording_device",
                "recording_location",
                "recording_date",
                "recorder",
                "distance_cm",
                "consent_status",
                "source_original_sha256",
                "standardized_sha256",
                "notes",
            ],
        ),
        dataset_root / "LICENSES.md": _licenses_text(spec),
    }
    for path, content in templates.items():
        try:
            path = safe_join(dataset_root, path.relative_to(dataset_root))
        except (DatasetSpecError, ValueError) as exc:
            raise ValueError("unsafe path declared by dataset spec") from exc
        if path.exists():
            preserved_files.append(path)
            continue
        path.write_text(content, encoding="utf-8", newline="")
        created_files.append(path)

    return InitReport(
        root=dataset_root,
        created_directories=tuple(created_directories),
        created_files=tuple(created_files),
        preserved_files=tuple(preserved_files),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Initialize an ignored AudioRescue dataset workspace."
    )
    parser.add_argument("--root", default="data_local")
    parser.add_argument("--spec", default=None, help="Local dataset spec JSON")
    parser.add_argument(
        "--example-spec",
        action="store_true",
        help="explicitly use the tracked synthetic example spec",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        report = initialize_dataset(
            args.root,
            spec_path=args.spec,
            example_mode=args.example_spec,
        )
    except (DatasetSpecError, ValueError, NotADirectoryError):
        print("Dataset initialization refused: DATASET_INPUT_INVALID")
        return 2
    print("Dataset workspace: <dataset_root>")
    print(f"Created directories: {len(report.created_directories)}")
    print(f"Created templates: {len(report.created_files)}")
    print(f"Preserved existing templates: {len(report.preserved_files)}")
    print("No audio file was created, moved, renamed, or overwritten.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
