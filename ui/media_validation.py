"""Bounded validation for UI-delivered media files."""

from __future__ import annotations

import binascii
import hashlib
import os
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

MAX_MEDIA_BYTES = 16 * 1024 * 1024
_READ_CHUNK_BYTES = 64 * 1024
_WAV_FRAME_SIZE = 2
_MAX_WAV_HEADER_BYTES = 4096
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_MAX_PNG_DIMENSION = 16384
_MAX_PNG_PIXELS = 8 * 1024 * 1024
_MAX_PNG_DECOMPRESSED_BYTES = 32 * 1024 * 1024


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
            if fmt_payload is None or not _valid_wav_fmt(fmt_payload):
                return False
            if payload_start > _MAX_WAV_HEADER_BYTES:
                return False
            if chunk_size == 0 or chunk_size % _WAV_FRAME_SIZE != 0:
                return False
            data_payload_size = chunk_size
        position = padded_end

    if position != len(data) or fmt_payload is None or data_payload_size is None:
        return False

    return _valid_wav_fmt(fmt_payload)


def _valid_wav_fmt(fmt_payload: bytes) -> bool:
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
    ihdr: dict[str, int] | None = None
    seen_plte = False
    seen_idat = False
    idat_finished = False
    seen_iend = False
    idat_payloads: list[bytes] = []
    while position < len(data):
        if seen_iend or position + 12 > len(data):
            return False
        length = _u32_be(data, position)
        chunk_type = data[position + 4 : position + 8]
        if not _valid_png_chunk_type(chunk_type):
            return False
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
            if ihdr is not None or position != len(_PNG_SIGNATURE) or length != 13:
                return False
            ihdr = _png_ihdr(payload)
            if ihdr is None:
                return False
        elif chunk_type == b"PLTE":
            if ihdr is None or seen_plte or seen_idat or not _valid_plte(payload, ihdr):
                return False
            seen_plte = True
        elif chunk_type == b"IDAT":
            if ihdr is None or seen_iend or idat_finished or length == 0:
                return False
            if ihdr["color_type"] == 3 and not seen_plte:
                return False
            seen_idat = True
            idat_payloads.append(payload)
        elif chunk_type == b"IEND":
            if length != 0 or ihdr is None or not seen_idat:
                return False
            seen_iend = True
        elif ihdr is None:
            return False
        elif _is_critical_png_chunk(chunk_type):
            return False
        elif seen_idat:
            idat_finished = True
        position = crc_end

    return (
        seen_iend
        and position == len(data)
        and ihdr is not None
        and _valid_png_zlib_scanlines(idat_payloads, ihdr)
    )


def _valid_png_chunk_type(chunk_type: bytes) -> bool:
    return len(chunk_type) == 4 and all(
        (65 <= byte <= 90) or (97 <= byte <= 122) for byte in chunk_type
    )


def _is_critical_png_chunk(chunk_type: bytes) -> bool:
    return not bool(chunk_type[0] & 0x20)


def _png_ihdr(payload: bytes) -> dict[str, int] | None:
    width = _u32_be(payload, 0)
    height = _u32_be(payload, 4)
    bit_depth = payload[8]
    color_type = payload[9]
    compression = payload[10]
    filter_method = payload[11]
    interlace = payload[12]
    pixels = width * height
    if (
        not (1 <= width <= _MAX_PNG_DIMENSION and 1 <= height <= _MAX_PNG_DIMENSION)
        or pixels > _MAX_PNG_PIXELS
    ):
        return None
    valid_depths = {
        0: {1, 2, 4, 8, 16},
        2: {8, 16},
        3: {1, 2, 4, 8},
        4: {8, 16},
        6: {8, 16},
    }
    if not (
        bit_depth in valid_depths.get(color_type, set())
        and compression == 0
        and filter_method == 0
        and interlace == 0
    ):
        return None
    channels_by_color = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
    bits_per_pixel = channels_by_color[color_type] * bit_depth
    row_bytes = (width * bits_per_pixel + 7) // 8
    decompressed_bytes = height * (row_bytes + 1)
    if decompressed_bytes <= 0 or decompressed_bytes > _MAX_PNG_DECOMPRESSED_BYTES:
        return None
    return {
        "width": width,
        "height": height,
        "bit_depth": bit_depth,
        "color_type": color_type,
        "row_bytes": row_bytes,
        "decompressed_bytes": decompressed_bytes,
    }


def _valid_plte(payload: bytes, ihdr: dict[str, int]) -> bool:
    if len(payload) == 0 or len(payload) % 3 != 0 or len(payload) > 256 * 3:
        return False
    color_type = ihdr["color_type"]
    if color_type in {0, 4}:
        return False
    if color_type == 3:
        entries = len(payload) // 3
        return entries <= 2 ** ihdr["bit_depth"]
    return True


def _valid_png_zlib_scanlines(idat_payloads: list[bytes], ihdr: dict[str, int]) -> bool:
    compressed = b"".join(idat_payloads)
    if not compressed:
        return False
    expected = ihdr["decompressed_bytes"]
    try:
        decompressor = zlib.decompressobj()
        decoded = decompressor.decompress(compressed, expected + 1)
        if len(decoded) > expected:
            return False
        decoded += decompressor.flush(expected + 1 - len(decoded))
    except zlib.error:
        return False
    if (
        not decompressor.eof
        or decompressor.unused_data
        or decompressor.unconsumed_tail
        or len(decoded) != expected
    ):
        return False
    row_stride = ihdr["row_bytes"] + 1
    for offset in range(0, len(decoded), row_stride):
        if decoded[offset] > 4:
            return False
    return True
