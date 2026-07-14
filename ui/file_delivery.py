"""Opaque, in-process delivery URLs for UI staged files."""

from __future__ import annotations

import mimetypes
import re
import secrets
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

DELIVERY_PREFIX = "/audiorescue-files"
_DELIVERY_PREFIX_BYTES = DELIVERY_PREFIX.encode("ascii")
_OPAQUE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{16,96}$")
_FILENAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_NEUTRAL_CONTENT_TYPES = {
    ".wav": "audio/wav",
    ".wave": "audio/wav",
    ".png": "image/png",
    ".txt": "text/plain; charset=utf-8",
    ".json": "application/json",
}


@dataclass(frozen=True)
class DeliveryEntry:
    token: str
    filename: str
    path: Path
    size: int
    mtime_ns: int
    inode: int
    device: int
    expires_at: float

    @property
    def url(self) -> str:
        return f"{DELIVERY_PREFIX}/{self.token}/{self.filename}"


_REGISTRY: dict[str, DeliveryEntry] = {}
_REGISTRY_LOCK = threading.RLock()


def _default_allowed_roots() -> tuple[Path, ...]:
    from .file_staging import default_staging_root

    return (default_staging_root(),)


def _default_ttl_seconds() -> int:
    from .file_staging import staging_ttl_seconds

    return staging_ttl_seconds()


def _safe_token() -> str:
    return secrets.token_urlsafe(24)


def _resolve_under_allowed_roots(path_value: Any, allowed_roots: tuple[Path, ...]) -> Path | None:
    if not path_value:
        return None
    candidate = Path(str(path_value)).expanduser()
    if not candidate.is_absolute():
        return None
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    if candidate.is_symlink() or not resolved.is_file():
        return None

    for root in allowed_roots:
        try:
            resolved.relative_to(root.resolve())
            return resolved
        except (OSError, RuntimeError, ValueError):
            continue
    return None


def cleanup_delivery_registry(*, now: float | None = None) -> None:
    timestamp = time.time() if now is None else now
    with _REGISTRY_LOCK:
        expired = [
            token
            for token, entry in _REGISTRY.items()
            if entry.expires_at <= timestamp
        ]
        for token in expired:
            _REGISTRY.pop(token, None)


def clear_delivery_registry() -> None:
    """Clear the process-local registry. Intended for tests and process reset."""

    with _REGISTRY_LOCK:
        _REGISTRY.clear()


def invalidate_delivery_under(root: Path) -> None:
    try:
        resolved_root = root.resolve()
    except (OSError, RuntimeError):
        return
    with _REGISTRY_LOCK:
        stale = []
        for token, entry in _REGISTRY.items():
            try:
                entry.path.relative_to(resolved_root)
            except ValueError:
                continue
            stale.append(token)
        for token in stale:
            _REGISTRY.pop(token, None)


def register_file_for_delivery(
    path_value: Any,
    *,
    filename: str | None = None,
    allowed_roots: tuple[Path, ...] | None = None,
    ttl_seconds: int | None = None,
    now: float | None = None,
) -> str | None:
    """Register a staged file and return a path-only opaque URL.

    The returned URL contains no server path information. The real file path is
    retained only in this process-local registry.
    """

    roots = _default_allowed_roots() if allowed_roots is None else allowed_roots
    resolved = _resolve_under_allowed_roots(path_value, roots)
    if resolved is None:
        return None

    neutral_name = filename or resolved.name
    if (
        not _FILENAME_RE.fullmatch(neutral_name)
        or "/" in neutral_name
        or "\\" in neutral_name
        or ":" in neutral_name
        or ".." in neutral_name
    ):
        return None

    try:
        stat_result = resolved.stat()
    except OSError:
        return None
    if not resolved.is_file() or resolved.is_symlink():
        return None

    timestamp = time.time() if now is None else now
    ttl = _default_ttl_seconds() if ttl_seconds is None else max(1, ttl_seconds)
    cleanup_delivery_registry(now=timestamp)
    with _REGISTRY_LOCK:
        for _ in range(10):
            token = _safe_token()
            if token not in _REGISTRY:
                break
        else:
            return None
        _REGISTRY[token] = DeliveryEntry(
            token=token,
            filename=neutral_name,
            path=resolved,
            size=stat_result.st_size,
            mtime_ns=stat_result.st_mtime_ns,
            inode=stat_result.st_ino,
            device=stat_result.st_dev,
            expires_at=timestamp + ttl,
        )
        return _REGISTRY[token].url


def register_files_for_delivery(
    files_by_role: Mapping[str, str | None],
    *,
    filenames_by_role: Mapping[str, str] | None = None,
    allowed_roots: tuple[Path, ...] | None = None,
    ttl_seconds: int | None = None,
) -> dict[str, str | None]:
    filenames = filenames_by_role or {}
    urls: dict[str, str | None] = {}
    for role, path in files_by_role.items():
        try:
            urls[role] = register_file_for_delivery(
                path,
                filename=filenames.get(role),
                allowed_roots=allowed_roots,
                ttl_seconds=ttl_seconds,
            )
        except Exception:
            urls[role] = None
    return urls


def lookup_delivery_entry(url_or_token: str) -> DeliveryEntry | None:
    token = url_or_token
    if token.startswith(f"{DELIVERY_PREFIX}/"):
        parts = token.split("/")
        if len(parts) != 4:
            return None
        token = parts[2]
    with _REGISTRY_LOCK:
        return _REGISTRY.get(token)


def _response_headers(entry: DeliveryEntry, *, status_size: int) -> list[tuple[bytes, bytes]]:
    content_type = _NEUTRAL_CONTENT_TYPES.get(
        entry.path.suffix.lower(),
        mimetypes.guess_type(entry.filename)[0] or "application/octet-stream",
    )
    return [
        (b"content-type", content_type.encode("ascii")),
        (b"accept-ranges", b"bytes"),
        (b"cache-control", b"no-store"),
        (b"content-disposition", f'inline; filename="{entry.filename}"'.encode("ascii")),
        (b"content-length", str(status_size).encode("ascii")),
    ]


def _plain_response(status: int, body: bytes = b"") -> tuple[int, list[tuple[bytes, bytes]], bytes]:
    return (
        status,
        [
            (b"content-type", b"text/plain; charset=utf-8"),
            (b"cache-control", b"no-store"),
            (b"content-length", str(len(body)).encode("ascii")),
        ],
        body,
    )


def _raw_path_bytes(scope: Mapping[str, Any]) -> bytes:
    raw_path = scope.get("raw_path")
    if isinstance(raw_path, (bytes, bytearray)):
        return bytes(raw_path)
    if isinstance(raw_path, str):
        return raw_path.encode("utf-8", "ignore")
    return str(scope.get("path", "")).encode("utf-8", "ignore")


def _parse_request(scope: Mapping[str, Any]) -> tuple[str, str] | None:
    raw_path = _raw_path_bytes(scope)
    if scope.get("query_string"):
        return None
    if b"%" in raw_path or b"\\" in raw_path or b":" in raw_path:
        return None
    if not raw_path.startswith(_DELIVERY_PREFIX_BYTES + b"/"):
        return None
    suffix = raw_path[len(_DELIVERY_PREFIX_BYTES) + 1 :]
    parts = suffix.split(b"/")
    if len(parts) != 2:
        return None
    token_raw, filename_raw = parts
    if b".." in token_raw or b".." in filename_raw:
        return None
    try:
        token = token_raw.decode("ascii")
        filename = filename_raw.decode("ascii")
    except UnicodeDecodeError:
        return None
    if not _OPAQUE_ID_RE.fullmatch(token) or not _FILENAME_RE.fullmatch(filename):
        return None
    return token, filename


def _validated_entry(scope: Mapping[str, Any]) -> tuple[DeliveryEntry | None, int]:
    parsed = _parse_request(scope)
    if parsed is None:
        return None, 404
    token, filename = parsed
    timestamp = time.time()
    with _REGISTRY_LOCK:
        entry = _REGISTRY.get(token)
        if entry is None:
            return None, 404
        if entry.expires_at <= timestamp:
            _REGISTRY.pop(token, None)
            return None, 410
        if filename != entry.filename:
            return None, 404

    path = entry.path
    try:
        stat_result = path.stat()
    except OSError:
        with _REGISTRY_LOCK:
            _REGISTRY.pop(token, None)
        return None, 410
    if (
        path.is_symlink()
        or not path.is_file()
        or stat_result.st_size != entry.size
        or stat_result.st_mtime_ns != entry.mtime_ns
        or stat_result.st_ino != entry.inode
        or stat_result.st_dev != entry.device
    ):
        with _REGISTRY_LOCK:
            _REGISTRY.pop(token, None)
        return None, 410
    return entry, 200


def _range_from_header(range_header: bytes | None, size: int) -> tuple[int, int, int] | None:
    if not range_header:
        return 200, 0, max(0, size - 1)
    try:
        text = range_header.decode("ascii").strip()
    except UnicodeDecodeError:
        return None
    if not text.startswith("bytes=") or "," in text:
        return None
    start_text, separator, end_text = text[6:].partition("-")
    if separator != "-":
        return None
    try:
        if start_text == "":
            suffix_length = int(end_text)
            if suffix_length <= 0:
                return None
            start = max(0, size - suffix_length)
            end = max(0, size - 1)
        else:
            start = int(start_text)
            end = int(end_text) if end_text else size - 1
    except ValueError:
        return None
    if size <= 0 or start < 0 or end < start or start >= size:
        return None
    return 206, start, min(end, size - 1)


async def send_delivery_response(scope: Mapping[str, Any], send) -> bool:
    """Serve a registered file if scope targets the delivery prefix."""

    path = str(scope.get("path", ""))
    raw_path = _raw_path_bytes(scope)
    if not path.startswith(f"{DELIVERY_PREFIX}/") and not raw_path.startswith(
        _DELIVERY_PREFIX_BYTES + b"/"
    ):
        return False

    method = str(scope.get("method", "GET")).upper()
    if method not in {"GET", "HEAD"}:
        status, headers, body = _plain_response(405, b"method not allowed")
        await send({"type": "http.response.start", "status": status, "headers": headers})
        await send({"type": "http.response.body", "body": body if method != "HEAD" else b""})
        return True

    entry, miss_status = _validated_entry(scope)
    if entry is None:
        body = b"gone" if miss_status == 410 else b"not found"
        status, headers, response_body = _plain_response(miss_status, body)
        await send({"type": "http.response.start", "status": status, "headers": headers})
        await send(
            {
                "type": "http.response.body",
                "body": response_body if method != "HEAD" else b"",
                "more_body": False,
            }
        )
        return True

    headers_dict = {
        key.lower(): value for key, value in scope.get("headers", [])
    }
    parsed_range = _range_from_header(headers_dict.get(b"range"), entry.size)
    if parsed_range is None:
        headers = [
            (b"content-range", f"bytes */{entry.size}".encode("ascii")),
            (b"content-length", b"0"),
            (b"accept-ranges", b"bytes"),
            (b"cache-control", b"no-store"),
        ]
        await send({"type": "http.response.start", "status": 416, "headers": headers})
        await send({"type": "http.response.body", "body": b"", "more_body": False})
        return True

    status, start, end = parsed_range
    length = 0 if entry.size == 0 else end - start + 1
    headers = _response_headers(entry, status_size=length)
    if status == 206:
        headers.append((b"content-range", f"bytes {start}-{end}/{entry.size}".encode("ascii")))
    await send({"type": "http.response.start", "status": status, "headers": headers})
    if method == "HEAD":
        body = b""
    else:
        with entry.path.open("rb") as file:
            file.seek(start)
            body = file.read(length)
    await send({"type": "http.response.body", "body": body, "more_body": False})
    return True


class FileDeliveryMiddleware:
    """ASGI middleware serving only registered opaque file URLs."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") == "http" and await send_delivery_response(scope, send):
            return
        await self.app(scope, receive, send)
