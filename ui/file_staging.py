"""Controlled file staging for Gradio file-bearing components."""

from __future__ import annotations

import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

from .media_validation import media_kind_for_role, validate_media_path

DEFAULT_STAGING_TTL_SECONDS = 60 * 60
STAGING_ENV = "AUDIORESCUE_UI_STAGING_DIR"
STAGING_TTL_ENV = "AUDIORESCUE_UI_STAGING_TTL_SECONDS"

_ROLE_FILENAMES = {
    "original_audio": "original.wav",
    "mixed_audio": "mixed.wav",
    "full_audio": "full.wav",
    "spectrogram_image": "spectrogram.png",
    "waveform_image": "waveform.png",
}

def default_staging_root() -> Path:
    configured = os.environ.get(STAGING_ENV)
    if configured:
        return Path(configured).expanduser()
    return Path(tempfile.gettempdir()) / "audiorescue-ui-staging"


def staging_ttl_seconds() -> int:
    try:
        return max(60, int(os.environ.get(STAGING_TTL_ENV, DEFAULT_STAGING_TTL_SECONDS)))
    except ValueError:
        return DEFAULT_STAGING_TTL_SECONDS


def resolve_allowed_file(
    path_value: Any,
    *,
    allowed_roots: tuple[Path, ...],
    base_dir: Path,
) -> Path | None:
    if not path_value or not allowed_roots:
        return None

    path = Path(str(path_value)).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    try:
        resolved = path.resolve()
    except (OSError, RuntimeError):
        return None
    if not resolved.is_file():
        return None

    for root in allowed_roots:
        try:
            resolved.relative_to(root.resolve())
            return resolved
        except (OSError, RuntimeError, ValueError):
            continue
    return None


def _is_session_dir(path: Path) -> bool:
    return (
        not path.is_symlink()
        and path.is_dir()
        and len(path.name) == 32
        and all(char in "0123456789abcdef" for char in path.name)
    )


def cleanup_stale_staging(
    *,
    staging_root: Path | None = None,
    ttl_seconds: int | None = None,
    now: float | None = None,
) -> None:
    root = staging_root or default_staging_root()
    ttl = staging_ttl_seconds() if ttl_seconds is None else ttl_seconds
    timestamp = time.time() if now is None else now
    if not root.exists():
        return

    for child in root.iterdir():
        if not _is_session_dir(child):
            continue
        try:
            age = timestamp - child.stat().st_mtime
        except OSError:
            continue
        if age > ttl:
            try:
                from .file_delivery import invalidate_delivery_under

                invalidate_delivery_under(child)
            except Exception:
                pass
            shutil.rmtree(child, ignore_errors=True)


def _prepare_session_dir(staging_root: Path) -> Path:
    staging_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(staging_root, 0o700)
    for _ in range(10):
        session_dir = staging_root / uuid.uuid4().hex
        try:
            session_dir.mkdir(mode=0o700)
        except FileExistsError:
            continue
        return session_dir
    raise RuntimeError("无法创建 UI staging 目录")


def is_valid_pcm_wav(path: Path) -> bool:
    """Return whether path is a non-empty, fully readable PCM WAV file."""

    return validate_media_path(path, kind="wav")


def stage_files_for_gradio(
    paths_by_role: Mapping[str, Any],
    *,
    allowed_roots: tuple[Path, ...],
    base_dir: Path,
    staging_root: Path | None = None,
) -> dict[str, str | None]:
    """Copy allowed source files to neutral, app-owned paths for Gradio.

    Gradio encodes server-side file paths in generated URLs. Returning a copied
    staging path avoids exposing the original ProcessResult path, job output
    tree, user name, source file name, or repository location in href/src.
    """

    staged: dict[str, str | None] = {role: None for role in paths_by_role}
    resolved_by_role: dict[str, Path] = {}
    for role, path_value in paths_by_role.items():
        if role not in _ROLE_FILENAMES:
            continue
        resolved = resolve_allowed_file(
            path_value,
            allowed_roots=allowed_roots,
            base_dir=base_dir,
        )
        if resolved is not None:
            if not validate_media_path(resolved, kind=media_kind_for_role(role)):
                continue
            resolved_by_role[role] = resolved

    if not resolved_by_role:
        return staged

    root = staging_root or default_staging_root()
    try:
        cleanup_stale_staging(staging_root=root)
        session_dir = _prepare_session_dir(root)
    except (OSError, RuntimeError):
        return staged
    for role, source in resolved_by_role.items():
        destination = session_dir / _ROLE_FILENAMES[role]
        try:
            shutil.copyfile(source, destination)
            os.chmod(destination, 0o600)
            if not validate_media_path(destination, kind=media_kind_for_role(role)):
                raise OSError
        except OSError:
            destination.unlink(missing_ok=True)
            staged[role] = None
            continue
        staged[role] = str(destination)
    return staged
