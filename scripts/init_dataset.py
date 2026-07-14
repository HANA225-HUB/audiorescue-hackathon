"""Initialize the private AudioRescue-CN-Mini-v1 recording workspace.

The generated ``data_local`` tree is intentionally ignored by Git.  This
script only creates directories and text templates; it never creates, moves,
renames, or overwrites audio recordings.
"""

from __future__ import annotations

import argparse
import csv
import io
import subprocess
from dataclasses import dataclass
from pathlib import Path


DATASET_VERSION = "AudioRescue-CN-Mini-v1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]

SENTENCES = {
    "s01": "今天下午三点，我们在实验室讨论语音处理项目的最终方案。",
    "s02": "请记录会议中的三个重点：数据来源、模型效果和系统稳定性。",
    "s03": "如果现场网络中断，系统仍然可以在本地完成音频增强和文字转写。",
}

LATIN_SQUARE = {
    "A": {"s01": "fan", "s02": "keyboard", "s03": "traffic"},
    "B": {"s01": "keyboard", "s02": "traffic", "s03": "fan"},
    "C": {"s01": "traffic", "s02": "fan", "s03": "keyboard"},
}

REAL_RECORDINGS = (
    ("A", "s03", "traffic"),
    ("B", "s01", "fan"),
    ("C", "s02", "keyboard"),
)

DIRECTORIES = (
    "source_original/clean/spkA",
    "source_original/clean/spkB",
    "source_original/clean/spkC",
    "source_original/noise",
    "source_original/real",
    "raw/clean/spkA",
    "raw/clean/spkB",
    "raw/clean/spkC",
    "raw/noise",
    "raw/real",
    "controlled/dev",
    "controlled/locked_test",
    "outputs",
)


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


def _transcript_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for speaker in ("A", "B", "C"):
        for sentence_id, reference_text in SENTENCES.items():
            noise_type = LATIN_SQUARE[speaker][sentence_id]
            rows.append(
                {
                    "speaker_id": speaker,
                    "sentence_id": sentence_id.upper(),
                    "split": "dev" if sentence_id in {"s01", "s02"} else "locked_test",
                    "noise_type": noise_type,
                    "clean_path": (
                        f"raw/clean/spk{speaker}/"
                        f"clean_spk{speaker}_{sentence_id}.wav"
                    ),
                    "reference_text": reference_text,
                }
            )
    return rows


def _metadata_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for item in _transcript_rows():
        speaker = item["speaker_id"]
        sentence = item["sentence_id"].lower()
        rows.append(
            {
                "asset_id": f"clean_spk{speaker}_{sentence}",
                "source_type": "clean",
                "source_original_path": "",
                "relative_path": item["clean_path"],
                "speaker_id": speaker,
                "sentence_id": item["sentence_id"],
                "noise_type": "",
                "recording_device": "TODO",
                "recording_location": "TODO",
                "recording_date": "TODO",
                "recorder": "TODO",
                "distance_cm": "TODO",
                "consent_status": "pending",
                "source_original_sha256": "",
                "standardized_sha256": "",
                "notes": "",
            }
        )

    for noise in ("fan", "keyboard", "traffic"):
        rows.append(
            {
                "asset_id": f"noise_{noise}_take01",
                "source_type": "noise",
                "source_original_path": "",
                "relative_path": f"raw/noise/noise_{noise}_take01.wav",
                "speaker_id": "",
                "sentence_id": "",
                "noise_type": noise,
                "recording_device": "TODO",
                "recording_location": "TODO",
                "recording_date": "TODO",
                "recorder": "TODO",
                "distance_cm": "",
                "consent_status": "pending",
                "source_original_sha256": "",
                "standardized_sha256": "",
                "notes": "Avoid identifiable private speech.",
            }
        )

    for speaker, sentence, noise in REAL_RECORDINGS:
        rows.append(
            {
                "asset_id": f"real_spk{speaker}_{noise}_r01",
                "source_type": "real",
                "source_original_path": "",
                "relative_path": f"raw/real/real_spk{speaker}_{noise}_r01.wav",
                "speaker_id": speaker,
                "sentence_id": sentence.upper(),
                "noise_type": noise,
                "recording_device": "TODO",
                "recording_location": "TODO",
                "recording_date": "TODO",
                "recorder": "TODO",
                "distance_cm": "TODO",
                "consent_status": "pending",
                "source_original_sha256": "",
                "standardized_sha256": "",
                "notes": "",
            }
        )
    return rows


def _readme_text() -> str:
    return f"""# {DATASET_VERSION} 本地工作区

该目录不会提交到 Git。录音设备产生的 M4A/AAC/WAV 母带先放入 `source_original/`，从此只读，不裁剪、不改名、不覆盖。只将从母带导出的最终标准化 WAV 放入 `raw/` 的固定路径。

## 录音落盘清单

- 不可变母带：`source_original/clean/spkA..C/`、`source_original/noise/`、`source_original/real/`
- 9 条 clean：`raw/clean/spkA..C/clean_spkX_s01..s03.wav`
- 3 条 noise：`raw/noise/noise_fan|keyboard|traffic_take01.wav`
- 3 条 real：`raw/real/real_spkA_traffic_r01.wav`、`real_spkB_fan_r01.wav`、`real_spkC_keyboard_r01.wav`

进入混音前，最终 WAV 必须是 48kHz、单声道、PCM16。clean 开头和结尾各保留约 0.5 秒；noise 为 45–60 秒且不得包含可辨认的未授权谈话。

## 文本台账

- `transcripts.tsv`：固定参考文本、切分和拉丁方噪声分配；不得用转写结果反向修改。
- `recording_metadata.csv`：录音设备、地点、日期、录制人、授权状态，以及原始母带/标准化 WAV 的两个 SHA-256。
- `LICENSES.md`：三位说话人必须确认展示/提交范围。
- `manifest.csv`：由固定混音工具生成，不要手工编写混音参数。

`locked_test` 在配置冻结前不用于调整模型、强度或参数。
"""


def _licenses_text() -> str:
    return f"""# {DATASET_VERSION} 数据授权记录

> 未将下表的 `pending` 改为明确选项前，任何对应音频都不得放入公开仓库、演示包或比赛提交包。

## 说话人同意记录

| 说话人 | 比赛现场播放 | 评委提交包 | 公开 GitHub | 确认日期 | 确认方式 |
|---|---|---|---|---|---|
| A | pending | pending | pending | TODO | TODO |
| B | pending | pending | pending | TODO | TODO |
| C | pending | pending | pending | TODO | TODO |

允许值：`yes` / `no`。三个用途分别确认，不得从“可以现场播放”推定“可以公开发布”。

## 自录素材

- clean speech：团队成员自录，用途范围以上表为准。
- noise stems：团队自录；确认不含可辨认的未授权谈话或商业音乐。
- real noisy recordings：说话人自录，用途范围以上表为准。

## 外部数据

当前正式主数据不包含外部素材。如后续引入 VoiceBank-DEMAND 或 ESC-50，必须在此增加来源链接、数据版本、具体文件和原许可条款，不能只写“公开数据集”。
"""


def _require_private_root(dataset_root: Path) -> None:
    """Refuse a repository-local dataset directory that Git could track."""

    try:
        relative = dataset_root.relative_to(PROJECT_ROOT)
    except ValueError:
        return
    check = subprocess.run(
        ["git", "check-ignore", "-q", "--", relative.as_posix()],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
    )
    if check.returncode != 0:
        raise ValueError(
            "dataset root inside the repository must be ignored by Git; "
            "use data_local/ or add an explicit ignore rule before initialization"
        )


def initialize_dataset(root: str | Path) -> InitReport:
    """Create an idempotent private workspace without overwriting text or audio."""

    dataset_root = Path(root).expanduser().resolve()
    _require_private_root(dataset_root)
    created_directories: list[Path] = []
    created_files: list[Path] = []
    preserved_files: list[Path] = []

    if not dataset_root.exists():
        dataset_root.mkdir(parents=True)
        created_directories.append(dataset_root)
    elif not dataset_root.is_dir():
        raise NotADirectoryError(f"dataset root is not a directory: {dataset_root}")

    for relative in DIRECTORIES:
        directory = dataset_root / relative
        if not directory.exists():
            directory.mkdir(parents=True)
            created_directories.append(directory)
        elif not directory.is_dir():
            raise NotADirectoryError(f"expected directory but found file: {directory}")

    templates = {
        dataset_root / "README.md": _readme_text(),
        dataset_root / "transcripts.tsv": _render_tsv(
            _transcript_rows(),
            [
                "speaker_id",
                "sentence_id",
                "split",
                "noise_type",
                "clean_path",
                "reference_text",
            ],
        ),
        dataset_root / "recording_metadata.csv": _render_csv(
            _metadata_rows(),
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
        dataset_root / "LICENSES.md": _licenses_text(),
    }
    for path, content in templates.items():
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
        description="Initialize the ignored AudioRescue-CN-Mini-v1 workspace."
    )
    parser.add_argument(
        "--root",
        default="data_local",
        help="Private dataset directory (default: data_local).",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = initialize_dataset(args.root)
    print(f"Dataset workspace: {report.root}")
    print(f"Created directories: {len(report.created_directories)}")
    print(f"Created templates: {len(report.created_files)}")
    print(f"Preserved existing templates: {len(report.preserved_files)}")
    print("No audio file was created, moved, renamed, or overwritten.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
