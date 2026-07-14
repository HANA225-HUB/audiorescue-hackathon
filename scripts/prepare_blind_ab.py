"""Prepare byte-identical, anonymized A/B listening ballots.

The script does not normalize loudness or touch audio samples.  It only copies
the two supplied tracks under neutral names and stores the answer key in a
separate administrator directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import secrets
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class BlindPair:
    sample_id: str
    original_path: Path
    enhanced_path: Path


@dataclass(frozen=True, slots=True)
class BlindTestReport:
    output_root: Path
    pair_count: int
    rater_count: int
    answer_key: Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_pairs_csv(path: str | Path) -> list[BlindPair]:
    csv_path = Path(path).expanduser().resolve()
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "original_path", "enhanced_path"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                "pairs CSV must contain sample_id, original_path, enhanced_path"
            )
        pairs = []
        for row_number, row in enumerate(reader, start=2):
            sample_id = (row.get("sample_id") or "").strip()
            if not sample_id:
                raise ValueError(f"row {row_number}: sample_id is empty")
            pairs.append(
                BlindPair(
                    sample_id=sample_id,
                    original_path=_resolve_csv_path(csv_path, row["original_path"]),
                    enhanced_path=_resolve_csv_path(csv_path, row["enhanced_path"]),
                )
            )
    return pairs


def _resolve_csv_path(csv_path: Path, raw_value: str) -> Path:
    value = Path(raw_value).expanduser()
    if not value.is_absolute():
        value = csv_path.parent / value
    return value.resolve()


def _validate_pairs(pairs: list[BlindPair]) -> None:
    if not pairs:
        raise ValueError("at least one A/B pair is required")
    identifiers: set[str] = set()
    for pair in pairs:
        if not pair.sample_id or any(character in pair.sample_id for character in "/\\"):
            raise ValueError("sample_id must be non-empty and cannot contain path separators")
        if pair.sample_id in identifiers:
            raise ValueError(f"duplicate sample_id: {pair.sample_id}")
        identifiers.add(pair.sample_id)
        for label, path in (
            ("original", pair.original_path),
            ("enhanced", pair.enhanced_path),
        ):
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(
                    f"{pair.sample_id}: {label} track is missing or empty: {path}"
                )
        if pair.original_path.samefile(pair.enhanced_path):
            raise ValueError(f"{pair.sample_id}: original and enhanced are the same file")


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _ensure_private_destination(destination: Path) -> None:
    """Refuse blind answers/paths in a repository location Git can track."""

    project_root = Path(__file__).resolve().parents[1]
    try:
        relative = destination.relative_to(project_root)
    except ValueError:
        return
    check = subprocess.run(
        ["git", "check-ignore", "-q", "--", relative.as_posix()],
        cwd=project_root,
        check=False,
        capture_output=True,
    )
    if check.returncode != 0:
        raise ValueError(
            "blind-test output inside the repository must be ignored by Git; "
            "use data_local/blind_ab/... or a directory outside the repository"
        )


def prepare_blind_test(
    pairs: list[BlindPair],
    output_root: str | Path,
    *,
    raters: tuple[str, ...] = ("A", "B", "C"),
    seed: int | None = None,
) -> BlindTestReport:
    """Create balanced private ballots without altering either audio track."""

    _validate_pairs(pairs)
    if not raters or len(set(raters)) != len(raters):
        raise ValueError("raters must be a non-empty unique list")
    for rater in raters:
        if not rater or any(character in rater for character in "/\\"):
            raise ValueError("rater names cannot be empty or contain path separators")

    destination = Path(output_root).expanduser().resolve()
    _ensure_private_destination(destination)
    if destination.exists():
        raise FileExistsError(
            f"blind-test output already exists; choose a new session path: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        admin = temporary / "admin_keep_private"
        admin.mkdir()
        for rater in raters:
            (temporary / f"rater_{rater}" / "audio").mkdir(parents=True)

        answer_rows: list[dict[str, str]] = []
        ballot_rows: dict[str, list[dict[str, str]]] = {rater: [] for rater in raters}
        # Production callers get an unpredictable assignment.  A deterministic
        # seed remains available only when tests or an administrator explicitly
        # request one; it is never copied into any rater package.
        session_seed = secrets.randbits(128) if seed is None else seed
        rng = random.Random(session_seed)
        rater_cycle = list(raters)
        rng.shuffle(rater_cycle)
        cycle_index = {rater: index for index, rater in enumerate(rater_cycle)}
        starting_phase = rng.randrange(2)
        for pair_index, pair in enumerate(pairs, start=1):
            blind_id = f"T{pair_index:03d}"
            original_digest = sha256_file(pair.original_path)
            enhanced_digest = sha256_file(pair.enhanced_path)

            for rater in raters:
                # Latin-style alternating schedule: each rater sees original as
                # A/B equally often (difference <= 1), while every odd-rater
                # pair remains balanced 2:1 rather than exposing one fixed side.
                original_is_a = (
                    (pair_index - 1 + cycle_index[rater] + starting_phase) % 2 == 0
                )
                rater_root = temporary / f"rater_{rater}"
                path_a = rater_root / "audio" / f"{blind_id}_A.wav"
                path_b = rater_root / "audio" / f"{blind_id}_B.wav"
                source_a = pair.original_path if original_is_a else pair.enhanced_path
                source_b = pair.enhanced_path if original_is_a else pair.original_path
                shutil.copyfile(source_a, path_a)
                shutil.copyfile(source_b, path_b)

                answer_rows.append(
                    {
                        "rater": rater,
                        "blind_id": blind_id,
                        "sample_id": pair.sample_id,
                        "A_role": "original" if original_is_a else "enhanced",
                        "B_role": "enhanced" if original_is_a else "original",
                        "original_sha256": original_digest,
                        "enhanced_sha256": enhanced_digest,
                    }
                )
                ballot_rows[rater].append(
                    {
                        "blind_id": blind_id,
                        "clip_A": f"audio/{blind_id}_A.wav",
                        "clip_B": f"audio/{blind_id}_B.wav",
                        "preference_A_B_tie": "",
                        "clarity_A_1to5": "",
                        "clarity_B_1to5": "",
                        "artifacts_A": "",
                        "artifacts_B": "",
                        "notes": "",
                    }
                )

        answer_key = admin / "answer_key.csv"
        _write_csv(
            answer_key,
            [
                "rater",
                "blind_id",
                "sample_id",
                "A_role",
                "B_role",
                "original_sha256",
                "enhanced_sha256",
            ],
            answer_rows,
        )
        (admin / "session.json").write_text(
            json.dumps(
                {
                    "assignment_seed": session_seed,
                    "pair_count": len(pairs),
                    "raters": list(raters),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        for rater, rows in ballot_rows.items():
            _write_csv(
                temporary / f"rater_{rater}" / "ballot.csv",
                [
                    "blind_id",
                    "clip_A",
                    "clip_B",
                    "preference_A_B_tie",
                    "clarity_A_1to5",
                    "clarity_B_1to5",
                    "artifacts_A",
                    "artifacts_B",
                    "notes",
                ],
                rows,
            )
        (admin / "README.txt").write_text(
            "Do not show answer_key.csv to raters before all ballots are returned.\n"
            "Audio files are byte-for-byte copies; no loudness matching was applied.\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return BlindTestReport(
        output_root=destination,
        pair_count=len(pairs),
        rater_count=len(raters),
        answer_key=destination / "admin_keep_private" / "answer_key.csv",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare anonymized A/B listening ballots without changing audio."
    )
    parser.add_argument("--pairs", required=True, help="CSV with sample_id and two paths.")
    parser.add_argument("--output", required=True, help="New private session directory.")
    parser.add_argument(
        "--raters",
        default="A,B,C",
        help="Comma-separated unique rater IDs (default: A,B,C).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Explicit deterministic assignment seed for tests/reproduction only; "
            "the production default is cryptographically random."
        ),
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    raters = tuple(item.strip() for item in args.raters.split(",") if item.strip())
    report = prepare_blind_test(
        read_pairs_csv(args.pairs),
        args.output,
        raters=raters,
        seed=args.seed,
    )
    print(f"Blind session: {report.output_root}")
    print(f"Pairs: {report.pair_count}; raters: {report.rater_count}")
    print(f"Keep private until ballots return: {report.answer_key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
