#!/usr/bin/env python3
"""Standardize immutable recording masters to the dataset WAV contract.

The recording ledger is the source of truth:

* ``source_original_path`` must resolve below ``data_root/source_original``;
* ``relative_path`` must resolve below ``data_root/raw``;
* source masters are opened only for hashing and as converter inputs;
* output WAVs are committed only after format and source-integrity checks;
* both SHA-256 columns are updated by atomically replacing the CSV file.

FFmpeg is preferred on every platform.  On macOS, ``/usr/bin/afconvert`` is
used when FFmpeg is unavailable.  Tests and callers may inject a converter so
the safety and bookkeeping logic does not depend on either program.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import wave
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SAMPLE_RATE = 48_000
CHANNELS = 1
SAMPLE_WIDTH_BYTES = 2
METADATA_FILENAME = "recording_metadata.csv"
AFCONVERT_PATH = Path("/usr/bin/afconvert")
REQUIRED_METADATA_FIELDS = (
    "asset_id",
    "source_original_path",
    "relative_path",
    "source_original_sha256",
    "standardized_sha256",
)

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
Converter = Callable[[Path, Path], None]


class StandardizationError(RuntimeError):
    """Base error for a safe, actionable standardization failure."""


class MetadataError(StandardizationError):
    """Raised when the ledger is incomplete or violates the path contract."""


class MissingSourceError(MetadataError):
    """Raised in final mode when one or more master paths are still blank."""


class SourceIntegrityError(StandardizationError):
    """Raised when a master hash is wrong or changes during conversion."""


class OutputExistsError(StandardizationError):
    """Raised when an output exists and overwrite was not explicitly allowed."""


class ConversionError(StandardizationError):
    """Raised when no converter is available or the converter fails."""


class WavValidationError(StandardizationError):
    """Raised when a converted file is not 48 kHz mono PCM16 WAV."""


@dataclass(frozen=True, slots=True)
class RecordingTask:
    row_index: int
    asset_id: str
    source_path: Path
    output_path: Path
    source_sha256: str


@dataclass(frozen=True, slots=True)
class StandardizationReport:
    data_root: Path
    metadata_path: Path
    processed_assets: tuple[str, ...]
    processed_outputs: tuple[Path, ...]
    skipped_missing_sources: tuple[str, ...]

    @property
    def processed_count(self) -> int:
        return len(self.processed_assets)

    @property
    def skipped_count(self) -> int:
        return len(self.skipped_missing_sources)


@dataclass(frozen=True, slots=True)
class CommandConverter:
    """Callable wrapper for one external conversion backend."""

    backend: str
    executable: str
    runner: Callable[..., Any] = field(
        default=subprocess.run,
        repr=False,
        compare=False,
    )

    def __call__(self, source_path: Path, destination_path: Path) -> None:
        if self.backend == "ffmpeg":
            command = [
                self.executable,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source_path),
                "-map_metadata",
                "-1",
                "-vn",
                "-ac",
                str(CHANNELS),
                "-ar",
                str(SAMPLE_RATE),
                "-c:a",
                "pcm_s16le",
                str(destination_path),
            ]
        elif self.backend == "afconvert":
            command = [
                self.executable,
                "-f",
                "WAVE",
                "-d",
                f"LEI16@{SAMPLE_RATE}",
                "-c",
                str(CHANNELS),
                str(source_path),
                str(destination_path),
            ]
        else:  # Defensive guard for manually constructed instances.
            raise ConversionError(f"未知转换后端：{self.backend}")

        try:
            self.runner(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except FileNotFoundError as exc:
            raise ConversionError(
                f"转换程序不存在：{self.executable}"
            ) from exc
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or str(exc)).strip()
            if len(detail) > 800:
                detail = detail[-800:]
            raise ConversionError(
                f"{self.backend} 转换失败：{detail or '未返回错误信息'}"
            ) from exc
        except OSError as exc:
            raise ConversionError(
                f"无法启动 {self.backend}（{self.executable}）：{exc}"
            ) from exc


def select_converter(
    *,
    which: Callable[[str], str | None] = shutil.which,
    platform: str = sys.platform,
    afconvert_path: str | Path = AFCONVERT_PATH,
    runner: Callable[..., Any] = subprocess.run,
) -> CommandConverter:
    """Select FFmpeg first, then the built-in macOS converter."""

    ffmpeg_path = which("ffmpeg")
    if ffmpeg_path:
        return CommandConverter("ffmpeg", ffmpeg_path, runner)

    fallback = Path(afconvert_path)
    if platform == "darwin" and fallback.is_file() and os.access(fallback, os.X_OK):
        return CommandConverter("afconvert", str(fallback), runner)

    raise ConversionError(
        "未找到 ffmpeg"
        + (
            f"，且 macOS 备用程序不可用：{fallback}"
            if platform == "darwin"
            else "；请先安装 ffmpeg 后重试"
        )
    )


def sha256_file(path: Path) -> str:
    """Return a lowercase SHA-256 digest without loading the file into RAM."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_standard_wav(path: Path) -> None:
    """Require a non-empty, complete, uncompressed 48 kHz mono PCM16 WAV."""

    if not path.is_file():
        raise WavValidationError(f"转换器未生成输出：{path}")

    try:
        with wave.open(str(path), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            compression = wav_file.getcomptype()
            frame_count = wav_file.getnframes()
            frame_bytes = wav_file.readframes(frame_count)
    except (EOFError, OSError, wave.Error) as exc:
        raise WavValidationError(f"无法读取 WAV：{path}（{exc}）") from exc

    problems: list[str] = []
    if sample_rate != SAMPLE_RATE:
        problems.append(f"采样率 {sample_rate} Hz")
    if channels != CHANNELS:
        problems.append(f"声道数 {channels}")
    if sample_width != SAMPLE_WIDTH_BYTES:
        problems.append(f"位深 {sample_width * 8} bit")
    if compression != "NONE":
        problems.append(f"压缩类型 {compression}")
    if frame_count <= 0:
        problems.append("音频为空")

    expected_bytes = frame_count * channels * sample_width
    if len(frame_bytes) != expected_bytes:
        problems.append(
            f"音频数据不完整（应为 {expected_bytes} 字节，"
            f"实际 {len(frame_bytes)} 字节）"
        )
    if problems:
        raise WavValidationError(
            f"{path} 不符合 48kHz/mono/PCM16 WAV：" + "、".join(problems)
        )


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _load_metadata(metadata_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with metadata_path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise MetadataError(f"台账缺少表头：{metadata_path}")
            fieldnames = list(reader.fieldnames)
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise MetadataError(f"无法读取台账：{metadata_path}（{exc}）") from exc

    if len(fieldnames) != len(set(fieldnames)):
        raise MetadataError("台账表头存在重复列，请先修正 CSV")
    missing_fields = [
        field for field in REQUIRED_METADATA_FIELDS if field not in fieldnames
    ]
    if missing_fields:
        raise MetadataError(
            f"台账缺少必需列：{', '.join(missing_fields)}"
        )
    if not rows:
        raise MetadataError(f"台账没有录音条目：{metadata_path}")
    for line_number, row in enumerate(rows, start=2):
        if None in row:
            raise MetadataError(
                f"台账第 {line_number} 行的列数超过表头，请先修正 CSV"
            )
        missing_cells = [field for field, value in row.items() if value is None]
        if missing_cells:
            raise MetadataError(
                f"台账第 {line_number} 行缺少列值：{', '.join(missing_cells)}"
            )
    return fieldnames, rows


def _normalise_hash(value: str, *, asset_id: str, field_name: str) -> str:
    stripped = value.strip()
    if stripped and not _SHA256_RE.fullmatch(stripped):
        raise MetadataError(
            f"{asset_id} 的 {field_name} 不是 64 位 SHA-256：{stripped!r}"
        )
    return stripped.lower()


def _resolve_source_path(
    value: str,
    *,
    data_root: Path,
    source_root: Path,
    asset_id: str,
) -> Path:
    entered = Path(value).expanduser()
    candidate = entered if entered.is_absolute() else data_root / entered
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise MetadataError(
            f"{asset_id} 的母带路径不存在或无法解析：{value}"
        ) from exc
    if not resolved.is_file():
        raise MetadataError(f"{asset_id} 的母带不是文件：{resolved}")
    if not _is_within(resolved, source_root):
        raise MetadataError(
            f"{asset_id} 的母带必须位于 {source_root} 内：{resolved}"
        )
    return resolved


def _resolve_output_path(
    value: str,
    *,
    data_root: Path,
    raw_root: Path,
    asset_id: str,
) -> Path:
    entered = Path(value)
    if entered.is_absolute():
        raise MetadataError(f"{asset_id} 的 relative_path 必须是相对路径：{value}")
    try:
        resolved = (data_root / entered).resolve(strict=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise MetadataError(
            f"{asset_id} 的输出路径无法解析：{value}"
        ) from exc
    if not _is_within(resolved, raw_root):
        raise MetadataError(
            f"{asset_id} 的标准化输出必须位于 {raw_root} 内：{resolved}"
        )
    if resolved.suffix.lower() != ".wav":
        raise MetadataError(f"{asset_id} 的输出必须是 .wav：{resolved}")
    return resolved


def _build_plan(
    rows: list[dict[str, str]],
    *,
    data_root: Path,
    source_root: Path,
    raw_root: Path,
    available_only: bool,
    overwrite: bool,
) -> tuple[list[RecordingTask], list[str]]:
    missing_assets: list[str] = []
    seen_asset_ids: set[str] = set()

    for index, row in enumerate(rows, start=2):
        asset_id = row["asset_id"].strip() or f"CSV 第 {index} 行"
        if asset_id in seen_asset_ids:
            raise MetadataError(f"台账 asset_id 重复：{asset_id}")
        seen_asset_ids.add(asset_id)
        if not row["source_original_path"].strip():
            missing_assets.append(asset_id)

    if missing_assets and not available_only:
        preview = "、".join(missing_assets[:8])
        if len(missing_assets) > 8:
            preview += f"等 {len(missing_assets)} 项"
        raise MissingSourceError(
            f"最终模式要求所有母带路径已登记；"
            f"当前有 {len(missing_assets)} 个 source_original_path 为空：{preview}。"
            "可先用 --available-only 仅处理已就绪录音。"
        )

    tasks: list[RecordingTask] = []
    seen_sources: dict[Path, str] = {}
    seen_outputs: dict[Path, str] = {}
    for row_index, row in enumerate(rows):
        asset_id = row["asset_id"].strip() or f"CSV 第 {row_index + 2} 行"
        source_value = row["source_original_path"].strip()
        if not source_value:
            continue

        output_value = row["relative_path"].strip()
        if not output_value:
            raise MetadataError(f"{asset_id} 的 relative_path 为空")
        source_path = _resolve_source_path(
            source_value,
            data_root=data_root,
            source_root=source_root,
            asset_id=asset_id,
        )
        output_path = _resolve_output_path(
            output_value,
            data_root=data_root,
            raw_root=raw_root,
            asset_id=asset_id,
        )
        if source_path == output_path:
            raise MetadataError(f"{asset_id} 的母带与输出路径不得相同")
        if source_path in seen_sources:
            raise MetadataError(
                f"{asset_id} 与 {seen_sources[source_path]} 指向同一母带：{source_path}"
            )
        if output_path in seen_outputs:
            raise MetadataError(
                f"{asset_id} 与 {seen_outputs[output_path]} 指向同一输出：{output_path}"
            )
        seen_sources[source_path] = asset_id
        seen_outputs[output_path] = asset_id

        if output_path.exists() and not overwrite:
            raise OutputExistsError(
                f"拒绝覆盖已有输出：{output_path}。"
                "确认需要重新标准化后显式传入 --overwrite。"
            )
        if output_path.exists() and not output_path.is_file():
            raise MetadataError(f"输出路径已存在且不是文件：{output_path}")

        stored_source_hash = _normalise_hash(
            row["source_original_sha256"],
            asset_id=asset_id,
            field_name="source_original_sha256",
        )
        _normalise_hash(
            row["standardized_sha256"],
            asset_id=asset_id,
            field_name="standardized_sha256",
        )
        current_source_hash = sha256_file(source_path)
        if stored_source_hash and stored_source_hash != current_source_hash:
            raise SourceIntegrityError(
                f"{asset_id} 的母带 SHA-256 与台账不一致；"
                f"台账={stored_source_hash}，当前={current_source_hash}。"
                "为保护母带已中止。"
            )

        tasks.append(
            RecordingTask(
                row_index=row_index,
                asset_id=asset_id,
                source_path=source_path,
                output_path=output_path,
                source_sha256=current_source_hash,
            )
        )
    return tasks, missing_assets


def _new_unused_path(directory: Path, *, prefix: str, suffix: str) -> Path:
    descriptor, value = tempfile.mkstemp(dir=directory, prefix=prefix, suffix=suffix)
    os.close(descriptor)
    path = Path(value)
    path.unlink()
    return path


def _atomic_write_metadata(
    metadata_path: Path,
    fieldnames: list[str],
    rows: list[dict[str, str]],
) -> None:
    descriptor, temporary_value = tempfile.mkstemp(
        dir=metadata_path.parent,
        prefix=f".{metadata_path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_value)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_path, metadata_path.stat().st_mode & 0o777)
        os.replace(temporary_path, metadata_path)
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary_path.unlink(missing_ok=True)
        raise


def _invoke_converter(converter: Converter, source: Path, destination: Path) -> None:
    try:
        converter(source, destination)
    except StandardizationError:
        raise
    except Exception as exc:
        raise ConversionError(f"转换 {source.name} 失败：{exc}") from exc


def _commit_one_task(
    task: RecordingTask,
    *,
    converter: Converter,
    overwrite: bool,
    metadata_path: Path,
    fieldnames: list[str],
    rows: list[dict[str, str]],
) -> None:
    current_hash = sha256_file(task.source_path)
    if current_hash != task.source_sha256:
        raise SourceIntegrityError(
            f"{task.asset_id} 的母带在计划生成后发生变化；已中止"
        )

    task.output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = _new_unused_path(
        task.output_path.parent,
        prefix=f".{task.output_path.stem}.",
        suffix=".wav",
    )
    backup_output: Path | None = None
    output_installed = False
    old_source_hash = rows[task.row_index]["source_original_sha256"]
    old_output_hash = rows[task.row_index]["standardized_sha256"]

    try:
        _invoke_converter(converter, task.source_path, temporary_output)
        validate_standard_wav(temporary_output)

        source_hash_after = sha256_file(task.source_path)
        if source_hash_after != task.source_sha256:
            raise SourceIntegrityError(
                f"{task.asset_id} 的母带在转换期间发生变化；"
                f"转换前={task.source_sha256}，转换后={source_hash_after}。"
                "输出和台账均未提交。"
            )
        standardized_hash = sha256_file(temporary_output)

        # Re-check at commit time so a concurrent file cannot be overwritten
        # merely because the path was empty while the plan was built.
        if task.output_path.exists():
            if not overwrite:
                raise OutputExistsError(
                    f"输出在转换期间已被创建，拒绝覆盖：{task.output_path}"
                )
            if not task.output_path.is_file():
                raise MetadataError(
                    f"输出路径已存在且不是文件：{task.output_path}"
                )
            backup_output = _new_unused_path(
                task.output_path.parent,
                prefix=f".{task.output_path.name}.",
                suffix=".backup",
            )
            os.replace(task.output_path, backup_output)

        os.replace(temporary_output, task.output_path)
        output_installed = True
        rows[task.row_index]["source_original_sha256"] = task.source_sha256
        rows[task.row_index]["standardized_sha256"] = standardized_hash
        _atomic_write_metadata(metadata_path, fieldnames, rows)
    except Exception:
        rows[task.row_index]["source_original_sha256"] = old_source_hash
        rows[task.row_index]["standardized_sha256"] = old_output_hash
        if output_installed:
            task.output_path.unlink(missing_ok=True)
        if backup_output is not None and backup_output.exists():
            os.replace(backup_output, task.output_path)
        raise
    finally:
        temporary_output.unlink(missing_ok=True)
        if backup_output is not None:
            backup_output.unlink(missing_ok=True)


def standardize_recordings(
    data_root: str | Path,
    *,
    metadata_path: str | Path | None = None,
    overwrite: bool = False,
    available_only: bool = False,
    converter: Converter | None = None,
) -> StandardizationReport:
    """Convert ledger entries and atomically record their source/output hashes.

    By default this is the final, strict mode: every row must have a populated
    ``source_original_path``.  ``available_only=True`` skips blank paths so the
    team can standardize recordings incrementally while preserving the same
    checks for every available item.
    """

    try:
        root = Path(data_root).expanduser().resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise MetadataError(f"数据根路径不存在或无法解析：{data_root}") from exc
    if not root.is_dir():
        raise MetadataError(f"数据根路径不是目录：{root}")

    metadata_candidate = (
        Path(metadata_path).expanduser()
        if metadata_path is not None
        else root / METADATA_FILENAME
    )
    if not metadata_candidate.is_absolute():
        metadata_candidate = root / metadata_candidate
    try:
        ledger = metadata_candidate.resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise MetadataError(f"台账不存在：{metadata_candidate}") from exc
    if not ledger.is_file():
        raise MetadataError(f"台账不是文件：{ledger}")
    if not _is_within(ledger, root):
        raise MetadataError(f"台账必须位于数据根目录内：{ledger}")

    try:
        source_root = (root / "source_original").resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise MetadataError(f"缺少母带目录：{root / 'source_original'}") from exc
    if not source_root.is_dir() or not _is_within(source_root, root):
        raise MetadataError(
            f"source_original 必须是数据根目录内的真实目录：{source_root}"
        )

    raw_entry = root / "raw"
    raw_entry.mkdir(parents=True, exist_ok=True)
    raw_root = raw_entry.resolve(strict=True)
    if not raw_root.is_dir() or not _is_within(raw_root, root):
        raise MetadataError(f"raw 必须是数据根目录内的真实目录：{raw_root}")

    fieldnames, rows = _load_metadata(ledger)
    tasks, missing_assets = _build_plan(
        rows,
        data_root=root,
        source_root=source_root,
        raw_root=raw_root,
        available_only=available_only,
        overwrite=overwrite,
    )
    if not tasks:
        return StandardizationReport(
            data_root=root,
            metadata_path=ledger,
            processed_assets=(),
            processed_outputs=(),
            skipped_missing_sources=tuple(missing_assets),
        )

    selected_converter = converter or select_converter()
    processed_assets: list[str] = []
    processed_outputs: list[Path] = []
    for task in tasks:
        _commit_one_task(
            task,
            converter=selected_converter,
            overwrite=overwrite,
            metadata_path=ledger,
            fieldnames=fieldnames,
            rows=rows,
        )
        processed_assets.append(task.asset_id)
        processed_outputs.append(task.output_path)

    return StandardizationReport(
        data_root=root,
        metadata_path=ledger,
        processed_assets=tuple(processed_assets),
        processed_outputs=tuple(processed_outputs),
        skipped_missing_sources=tuple(missing_assets),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "将 source_original 中的不可变母带标准化为 "
            "48kHz/mono/PCM16 WAV，并原子更新 SHA-256 台账。"
        )
    )
    parser.add_argument(
        "--data-root",
        default="data_local",
        help="数据根目录（默认：data_local）",
    )
    parser.add_argument(
        "--metadata",
        default=None,
        help="数据根目录内的 CSV 台账路径（默认：recording_metadata.csv）",
    )
    parser.add_argument(
        "--available-only",
        action="store_true",
        help="仅处理已填 source_original_path 的条目",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="显式允许原子替换已有 raw WAV",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    try:
        report = standardize_recordings(
            arguments.data_root,
            metadata_path=arguments.metadata,
            overwrite=arguments.overwrite,
            available_only=arguments.available_only,
        )
    except StandardizationError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2

    print(
        f"[PASS] 标准化 {report.processed_count} 条；"
        f"跳过空母带路径 {report.skipped_count} 条"
    )
    for asset_id, output_path in zip(
        report.processed_assets,
        report.processed_outputs,
    ):
        print(f"  {asset_id}: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
