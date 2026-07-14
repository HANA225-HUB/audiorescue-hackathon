import hashlib
import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRIVATE_AUDIO_SUFFIXES = {
    ".wav",
    ".m4a",
    ".aac",
    ".mp3",
    ".flac",
    ".aiff",
    ".aif",
    ".caf",
    ".ogg",
    ".opus",
    ".webm",
    ".mp4",
    ".mov",
    ".3gp",
    ".amr",
    ".wma",
    ".mkv",
}
LICENSE_COLUMNS = (
    "文件",
    "来源",
    "说话人",
    "允许现场播放",
    "允许评委包",
    "允许公开 GitHub",
    "SHA-256",
    "确认日期",
)


def tracked_paths() -> list[Path]:
    output = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
    ).stdout
    return [Path(item.decode("utf-8")) for item in output.split(b"\0") if item]


def audio_license_problems(
    project_root: Path,
    audio_paths: list[Path],
    *,
    ledger_relative: Path,
) -> list[str]:
    """Return publication-gate failures for one tracked audio collection."""

    ledger_path = project_root / ledger_relative
    if not ledger_path.is_file():
        return [f"{ledger_relative.as_posix()} is missing"]

    problems: list[str] = []
    records: dict[str, tuple[str, str, int]] = {}
    header_seen = False
    for line_number, line in enumerate(
        ledger_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = tuple(cell.strip() for cell in stripped.strip("|").split("|"))
        if cells == LICENSE_COLUMNS:
            header_seen = True
            continue
        if not header_seen:
            continue
        if len(cells) == len(LICENSE_COLUMNS) and all(
            cell and set(cell) <= {"-", ":", " "} for cell in cells
        ):
            continue
        if len(cells) != len(LICENSE_COLUMNS):
            problems.append(
                f"LICENSES.md:{line_number} must contain exactly "
                f"{len(LICENSE_COLUMNS)} columns"
            )
            continue

        repository_path = cells[0]
        if repository_path in records:
            problems.append(
                f"LICENSES.md:{line_number} duplicates {repository_path!r}"
            )
            continue
        records[repository_path] = (cells[5], cells[6], line_number)

    if not header_seen:
        problems.append("LICENSES.md is missing the required authorization table header")

    for relative_path in audio_paths:
        repository_path = relative_path.as_posix()
        record = records.get(repository_path)
        if record is None:
            problems.append(
                f"{repository_path} has no exact LICENSES.md record"
            )
            continue

        public_github, recorded_sha256, line_number = record
        if public_github != "yes":
            problems.append(
                f"LICENSES.md:{line_number} must set 允许公开 GitHub=yes "
                f"for {repository_path}"
            )

        audio_path = project_root / relative_path
        if not audio_path.is_file():
            problems.append(f"{repository_path} is tracked but missing from the worktree")
            continue
        actual_sha256 = hashlib.sha256(audio_path.read_bytes()).hexdigest()
        if recorded_sha256 != actual_sha256:
            problems.append(
                f"LICENSES.md:{line_number} SHA-256 does not match "
                f"{repository_path}"
            )

    return problems


def demo_license_problems(
    project_root: Path, demo_audio_paths: list[Path]
) -> list[str]:
    """Backward-compatible helper for the public demo gate tests."""

    return audio_license_problems(
        project_root,
        demo_audio_paths,
        ledger_relative=Path("demo_assets/LICENSES.md"),
    )


class RepositorySafetyTest(unittest.TestCase):
    def test_private_dataset_is_not_tracked(self) -> None:
        leaked = [path for path in tracked_paths() if path.parts[:1] == ("data_local",)]
        self.assertEqual(leaked, [], f"private data_local files are tracked: {leaked}")

    def test_tracked_audio_is_only_fixture_or_reviewed_demo(self) -> None:
        tracked = tracked_paths()
        tracked_audio = [
            path for path in tracked if path.suffix.lower() in PRIVATE_AUDIO_SUFFIXES
        ]
        invalid_locations = [
            path
            for path in tracked_audio
            if path.parts[:2] != ("tests", "fixtures")
            and path.parts[:1] != ("demo_assets",)
        ]
        self.assertEqual(
            invalid_locations,
            [],
            f"audio outside tests/fixtures or demo_assets is tracked: {invalid_locations}",
        )

        demo_manifest = yaml.safe_load(
            (PROJECT_ROOT / "configs/demo.yaml").read_text(encoding="utf-8")
        )
        declared = {Path(sample["file"]) for sample in demo_manifest["samples"]}
        undeclared_demo = [
            path
            for path in tracked_audio
            if path.parts[:1] == ("demo_assets",) and path not in declared
        ]
        self.assertEqual(
            undeclared_demo,
            [],
            f"demo audio is tracked without configs/demo.yaml metadata: {undeclared_demo}",
        )
        tracked_demo = [
            path for path in tracked_audio if path.parts[:1] == ("demo_assets",)
        ]
        tracked_fixtures = [
            path for path in tracked_audio if path.parts[:2] == ("tests", "fixtures")
        ]
        if tracked_fixtures:
            self.assertIn(
                Path("tests/fixtures/LICENSES.md"),
                tracked,
                "fixture audio is tracked but its publication ledger is not tracked",
            )
        self.assertEqual(
            audio_license_problems(
                PROJECT_ROOT,
                tracked_fixtures,
                ledger_relative=Path("tests/fixtures/LICENSES.md"),
            ),
            [],
            "tracked fixture audio failed the LICENSES.md publication gate",
        )
        if tracked_demo:
            self.assertIn(
                Path("demo_assets/LICENSES.md"),
                tracked,
                "demo audio is tracked but its authorization ledger is not tracked",
            )
        self.assertEqual(
            demo_license_problems(PROJECT_ROOT, tracked_demo),
            [],
            "tracked demo audio failed the LICENSES.md publication gate",
        )

    def test_demo_license_gate_requires_exact_path_public_yes_and_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            relative = Path("demo_assets/example.wav")
            audio_path = root / relative
            audio_path.parent.mkdir(parents=True)
            audio_path.write_bytes(b"authorized demo bytes")
            digest = hashlib.sha256(audio_path.read_bytes()).hexdigest()
            ledger = root / "demo_assets" / "LICENSES.md"
            prefix = (
                "| 文件 | 来源 | 说话人 | 允许现场播放 | 允许评委包 | "
                "允许公开 GitHub | SHA-256 | 确认日期 |\n"
                "|---|---|---|---|---|---|---|---|\n"
            )

            ledger.write_text(prefix, encoding="utf-8")
            self.assertTrue(demo_license_problems(root, [relative]))

            ledger.write_text(
                prefix
                + f"| {relative.as_posix()} | team | A | yes | yes | no | "
                f"{digest} | 2026-07-14 |\n",
                encoding="utf-8",
            )
            self.assertTrue(
                any(
                    "GitHub=yes" in problem
                    for problem in demo_license_problems(root, [relative])
                )
            )

            ledger.write_text(
                prefix
                + f"| {relative.as_posix()} | team | A | yes | yes | yes | "
                f"{'0' * 64} | 2026-07-14 |\n",
                encoding="utf-8",
            )
            self.assertTrue(
                any(
                    "SHA-256" in problem
                    for problem in demo_license_problems(root, [relative])
                )
            )

            ledger.write_text(
                prefix
                + f"| {relative.as_posix()} | team | A | yes | yes | yes | "
                f"{digest} | 2026-07-14 |\n",
                encoding="utf-8",
            )
            self.assertEqual(demo_license_problems(root, [relative]), [])

    def test_ignore_rules_cover_private_original_formats(self) -> None:
        ignore_lines = {
            line.strip()
            for line in (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        self.assertIn("data_local/", ignore_lines)
        for suffix in PRIVATE_AUDIO_SUFFIXES - {".wav"}:
            self.assertIn(f"*{suffix}", ignore_lines)


if __name__ == "__main__":
    unittest.main()
