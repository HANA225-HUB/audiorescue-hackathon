"""Deterministic, JSON-only task-result cache for the C-owned pipeline.

The cache deliberately stores only caller-provided JSON results. It never
copies input audio, model weights, or other artifacts, and it never derives a
hit from a filename. Cache identity is based on the input file's content plus
all result-affecting configuration supplied by the pipeline.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any


CACHE_FORMAT_VERSION = 1
CACHE_KEY_VERSION = 1
DEFAULT_CACHE_ROOT = Path(".cache") / "task-results"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_HASH_CHUNK_SIZE = 1024 * 1024


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file's bytes.

    The path, filename, timestamp, and other filesystem metadata are not part
    of the digest. Missing or unreadable inputs intentionally propagate an
    ``OSError`` so the pipeline can classify the input failure.
    """

    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(_HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    """Serialize JSON data deterministically for hashing or persistence.

    Callers must pass JSON-compatible data with finite numbers. In particular,
    sets, paths, model objects, tensors, and NaN/Infinity are rejected rather
    than being converted implicitly into unstable representations.
    """

    return json.dumps(
        _normalize_json_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def build_cache_key(
    input_path: str | Path,
    *,
    strength: float,
    model_name: str,
    language: str,
    contract_version: str,
    config_version: str,
    code_version: str,
    processing_config: Mapping[str, Any] | None = None,
) -> str:
    """Build a cache key from input content and frozen processing identity.

    ``processing_config`` must include every additional option that can affect
    the returned result, for example ASR decoding settings, ``enable_events``,
    or a reference text used for CER. ``force_recompute`` and ``job_id`` do not
    belong in the key because they control execution rather than result
    semantics.
    """

    if isinstance(strength, bool) or not isinstance(strength, (int, float)):
        raise TypeError("strength must be a number")
    numeric_strength = float(strength)
    if not math.isfinite(numeric_strength):
        raise ValueError("strength must be finite")

    identity_fields = {
        "model_name": model_name,
        "language": language,
        "contract_version": contract_version,
        "config_version": config_version,
        "code_version": code_version,
    }
    for field_name, field_value in identity_fields.items():
        if not isinstance(field_value, str) or not field_value.strip():
            raise ValueError(f"{field_name} must be a non-empty string")

    key_payload = {
        "cache_key_version": CACHE_KEY_VERSION,
        "input_sha256": sha256_file(input_path),
        "strength": numeric_strength,
        "model_name": model_name,
        "language": language,
        "contract_version": contract_version,
        "config_version": config_version,
        "code_version": code_version,
        "processing_config": dict(processing_config or {}),
    }
    canonical = canonical_json(key_payload).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


class LocalTaskCache:
    """Atomic local cache for JSON-ready task results.

    Typical pipeline usage::

        key = build_cache_key(input_path, ...)
        cache = LocalTaskCache()
        if (payload := cache.load(key)) is not None:
            return payload
        cache.save(key, result.to_dict())

    A successful lookup returns a fresh ``dict``. Missing, unreadable,
    malformed, truncated, or mismatched entries are all safe cache misses.
    """

    def __init__(self, root: str | Path = DEFAULT_CACHE_ROOT) -> None:
        self.root = Path(root)

    def path_for_key(self, cache_key: str) -> Path:
        """Return the sole JSON path associated with a validated key."""

        _validate_cache_key(cache_key)
        return self.root / f"{cache_key}.json"

    def load(self, cache_key: str) -> dict[str, Any] | None:
        """Load a cached JSON result, treating any corruption as a miss."""

        path = self.path_for_key(cache_key)
        try:
            raw = path.read_text(encoding="utf-8")
            envelope = json.loads(raw, parse_constant=_reject_json_constant)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
            return None

        if not isinstance(envelope, dict):
            return None
        if envelope.get("cache_format_version") != CACHE_FORMAT_VERSION:
            return None
        if envelope.get("cache_key") != cache_key:
            return None

        payload = envelope.get("payload")
        if not isinstance(payload, dict):
            return None
        return payload

    def save(
        self,
        cache_key: str,
        payload: Mapping[str, Any],
    ) -> Path:
        """Atomically save a JSON result and return its cache-file path.

        Serialization is completed before touching an existing entry. The new
        bytes are then flushed to a temporary file in the same directory and
        atomically replace the destination, so readers never observe a partial
        JSON document.
        """

        path = self.path_for_key(cache_key)
        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping")

        envelope = {
            "cache_format_version": CACHE_FORMAT_VERSION,
            "cache_key": cache_key,
            "payload": dict(payload),
        }
        encoded = (canonical_json(envelope) + "\n").encode("utf-8")

        self.root.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            file_descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{cache_key}.",
                suffix=".tmp",
                dir=self.root,
            )
            temporary_path = Path(temporary_name)
            with os.fdopen(file_descriptor, "wb") as temporary_file:
                temporary_file.write(encoded)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass

        return path


def _validate_cache_key(cache_key: str) -> None:
    if not isinstance(cache_key, str) or not _SHA256_RE.fullmatch(cache_key):
        raise ValueError("cache_key must be a lowercase SHA-256 hex digest")


def _normalize_json_value(value: Any) -> Any:
    """Return a strict JSON value without implicit or order-unstable casts."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON numbers must be finite")
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("JSON object keys must be strings")
            normalized[key] = _normalize_json_value(item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_normalize_json_value(item) for item in value]
    raise TypeError(f"value is not JSON-compatible: {type(value).__name__}")


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


__all__ = [
    "CACHE_FORMAT_VERSION",
    "CACHE_KEY_VERSION",
    "DEFAULT_CACHE_ROOT",
    "LocalTaskCache",
    "build_cache_key",
    "canonical_json",
    "sha256_file",
]
