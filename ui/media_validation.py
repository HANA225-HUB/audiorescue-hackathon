"""Bounded validation for UI-delivered media files."""

from __future__ import annotations

import binascii
import hashlib
import os
import struct
from dataclasses import dataclass
from pathlib import Path

MAX_MEDIA_BYTES = 16 * 1024 * 1024
_READ_CHUNK_BYTES = 64 * 1024
_WAV_FRAME_SIZE = 2
_MAX_WAV_HEADER_BYTES = 4096
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_MAX_PNG_DIMENSION = 16384


@dataclass(frozen=True)
class ValidatedMedia:
    size: int
    digest: str
    data: bytes


def media_kind_for_role(role: str) -> str | None:
    if role in {"original_audio", "mixed_audio", "full_audio"}:
        return "wav"
    if role in {"spectrogram_image", "waveform_image"}:
        return "png"
    return None


def media_kind_for_name(name: str | Path) -> str | None:
    suffix = Path(str(name)).suffix.lower()
    if suffix in {".wav", ".wave"}:
        return "wav"
    if suffix == ".png":
        return "png"
    return None


def validate_media_path(path: Path, *, kind: str | None) -> bool:
    if kind is None:
        return True
    try:
        stat_result = path.stat()
    except OSError:
        return False
    if not path.is_file() or path.is_symlink():
        return False
    if stat_result.st_size <= 0 or stat_result.st_size > MAX_MEDIA_BYTES:
        return False
    try:
        with path.open("rb") as file:
            data = file.read(MAX_MEDIA_BYTES + 1)
    except OSError:
        return False
    if len(data) != stat_result.st_size:
        return False
    return validate_media_bytes(data, kind=kind)


def validate_and_hash_fd(fd: int, *, kind: str | None) -> ValidatedMedia | None:
    try:
        stat_result = os.fstat(fd)
    except OSError:
        return None
    if stat_result.st_size <= 0 or stat_result.st_size > MAX_MEDIA_BYTES:
        return None
    try:
        os.lseek(fd, 0, os.SEEK_SET)
    except OSError:
        return None

    remaining = stat_result.st_size
    chunks: list[bytes] = []
    digest = hashlib.sha256()
    while remaining > 0:
        try:
            chunk = os.read(fd, min(_READ_CHUNK_BYTES, remaining))
        except OSError:
            return None
        if not chunk:
            return None
        chunks.append(chunk)
        digest.update(chunk)
        remaining -= len(chunk)

    try:
        extra = os.read(fd, 1)
    except OSError:
        return None
    if extra:
        return None

    data = b"".join(chunks)
    if kind is not None and not validate_media_bytes(data, kind=kind):
        return None
    return ValidatedMedia(size=len(data), digest=digest.hexdigest(), data=data)


def validate_media_bytes(data: bytes, *, kind: str) -> bool:
    if kind == "wav":
        return _validate_wav(data)
    if kind == "png":
        return _validate_png(data)
    return False


def _u16_le(data: bytes, offset: int) -> int:
    return struct.unpack_from("<H", data, offset)[0]


def _u32_le(data: bytes, offset: int) -> int:
    return struct.unpack_from("<I", data, offset)[0]


def _u32_be(data: bytes, offset: int) -> int:
    return struct.unpack_from(">I", data, offset)[0]


def _validate_wav(data: bytes) -> bool:
    if len(data) < 44 or len(data) > MAX_MEDIA_BYTES:
        return False
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return False
    riff_size = _u32_le(data, 4)
    if riff_size != len(data) - 8:
        return False

    position = 12
    seen_chunks: set[bytes] = set()
    fmt_payload: bytes | None = None
    data_payload_size: int | None = None
    while position < len(data):
        if position + 8 > len(data):
            return False
        chunk_id = data[position : position + 4]
        chunk_size = _u32_le(data, position + 4)
        payload_start = position + 8
        payload_end = payload_start + chunk_size
        if payload_end > len(data):
            return False
        padded_end = payload_end + (chunk_size % 2)
        if padded_end > len(data):
            return False
        if chunk_id in seen_chunks:
            return False
        seen_chunks.add(chunk_id)
        payload = data[payload_start:payload_end]
        if chunk_id == b"fmt ":
            if chunk_size != 16:
                return False
            fmt_payload = payload
        elif chunk_id == b"data":
            if payload_start > _MAX_WAV_HEADER_BYTES:
                return False
            if chunk_size == 0 or chunk_size % _WAV_FRAME_SIZE != 0:
                return False
            data_payload_size = chunk_size
        position = padded_end

    if position != len(data) or fmt_payload is None or data_payload_size is None:
        return False

    audio_format = _u16_le(fmt_payload, 0)
    channels = _u16_le(fmt_payload, 2)
    sample_rate = _u32_le(fmt_payload, 4)
    byte_rate = _u32_le(fmt_payload, 8)
    block_align = _u16_le(fmt_payload, 12)
    bits_per_sample = _u16_le(fmt_payload, 14)
    return (
        audio_format == 1
        and channels == 1
        and sample_rate == 48000
        and byte_rate == 96000
        and block_align == _WAV_FRAME_SIZE
        and bits_per_sample == 16
    )


def _validate_png(data: bytes) -> bool:
    if len(data) < 45 or len(data) > MAX_MEDIA_BYTES:
        return False
    if not data.startswith(_PNG_SIGNATURE):
        return False

    position = len(_PNG_SIGNATURE)
    seen_ihdr = False
    seen_idat = False
    seen_iend = False
    while position < len(data):
        if seen_iend or position + 12 > len(data):
            return False
        length = _u32_be(data, position)
        chunk_type = data[position + 4 : position + 8]
        payload_start = position + 8
        payload_end = payload_start + length
        crc_end = payload_end + 4
        if payload_end > len(data) or crc_end > len(data):
            return False
        payload = data[payload_start:payload_end]
        expected_crc = _u32_be(data, payload_end)
        actual_crc = binascii.crc32(chunk_type + payload) & 0xFFFFFFFF
        if expected_crc != actual_crc:
            return False

        if chunk_type == b"IHDR":
            if seen_ihdr or position != len(_PNG_SIGNATURE) or length != 13:
                return False
            if not _valid_ihdr(payload):
                return False
            seen_ihdr = True
        elif chunk_type == b"IDAT":
            if not seen_ihdr or seen_iend or length == 0:
                return False
            seen_idat = True
        elif chunk_type == b"IEND":
            if length != 0 or not seen_ihdr or not seen_idat:
                return False
            seen_iend = True
        elif not seen_ihdr:
            return False
        position = crc_end

    return seen_iend and position == len(data)


def _valid_ihdr(payload: bytes) -> bool:
    width = _u32_be(payload, 0)
    height = _u32_be(payload, 4)
    bit_depth = payload[8]
    color_type = payload[9]
    compression = payload[10]
    filter_method = payload[11]
    interlace = payload[12]
    if not (1 <= width <= _MAX_PNG_DIMENSION and 1 <= height <= _MAX_PNG_DIMENSION):
        return False
    valid_depths = {
        0: {1, 2, 4, 8, 16},
        2: {8, 16},
        3: {1, 2, 4, 8},
        4: {8, 16},
        6: {8, 16},
    }
    return (
        bit_depth in valid_depths.get(color_type, set())
        and compression == 0
        and filter_method == 0
        and interlace in {0, 1}
    )
