from __future__ import annotations

import binascii
import struct
import unittest
import zlib

from ui.media_validation import MAX_MEDIA_BYTES, validate_media_bytes


def _chunk(chunk_id: bytes, payload: bytes) -> bytes:
    padding = b"\x00" if len(payload) % 2 else b""
    return chunk_id + struct.pack("<I", len(payload)) + payload + padding


def _wav_bytes(
    *,
    audio_format: int = 1,
    channels: int = 1,
    sample_rate: int = 48000,
    bits_per_sample: int = 16,
    block_align: int = 2,
    byte_rate: int = 96000,
    data: bytes = b"\x00\x00\x01\x00",
    chunks: list[bytes] | None = None,
    riff_size_delta: int = 0,
) -> bytes:
    fmt = struct.pack(
        "<HHIIHH",
        audio_format,
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
    )
    payload = b"".join(chunks if chunks is not None else [_chunk(b"fmt ", fmt), _chunk(b"data", data)])
    return b"RIFF" + struct.pack("<I", len(payload) + 4 + riff_size_delta) + b"WAVE" + payload


def _png_chunk(kind: bytes, payload: bytes, *, crc_delta: int = 0) -> bytes:
    crc = (binascii.crc32(kind + payload) + crc_delta) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", crc)


def _png_bytes(
    *,
    width: int = 1,
    height: int = 1,
    bit_depth: int = 8,
    color_type: int = 6,
    interlace: int = 0,
    idat_payload: bytes | None = None,
    raw_scanlines: bytes | None = None,
    include_plte: bool = False,
    plte_payload: bytes = b"\x00\x00\x00",
    extra_chunks: list[tuple[bytes, bytes]] | None = None,
    include_idat: bool = True,
    include_iend: bool = True,
    crc_delta: int = 0,
) -> bytes:
    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, bit_depth, color_type, 0, 0, interlace)
    chunks = [_png_chunk(b"IHDR", ihdr, crc_delta=crc_delta)]
    if include_plte:
        chunks.append(_png_chunk(b"PLTE", plte_payload))
    for kind, payload in extra_chunks or []:
        chunks.append(_png_chunk(kind, payload))
    if include_idat:
        if idat_payload is None:
            scanlines = b"\x00\x00\x00\x00\x00" if raw_scanlines is None else raw_scanlines
            idat_payload = zlib.compress(scanlines)
        chunks.append(_png_chunk(b"IDAT", idat_payload))
    if include_iend:
        chunks.append(_png_chunk(b"IEND", b""))
    return signature + b"".join(chunks)


class MediaValidationTests(unittest.TestCase):
    def test_accepts_strict_frozen_wav_and_png(self) -> None:
        self.assertTrue(validate_media_bytes(_wav_bytes(), kind="wav"))
        self.assertTrue(validate_media_bytes(_png_bytes(), kind="png"))

    def test_rejects_malformed_wav_shapes(self) -> None:
        cases = {
            "half_sample": _wav_bytes(data=b"\x00"),
            "wrong_riff_length": _wav_bytes(riff_size_delta=1),
            "wrong_block_align": _wav_bytes(block_align=4),
            "wrong_byte_rate": _wav_bytes(byte_rate=48000),
            "forty_bit": _wav_bytes(bits_per_sample=40, block_align=5, byte_rate=240000),
            "non_48k": _wav_bytes(sample_rate=44100, byte_rate=88200),
            "stereo": _wav_bytes(channels=2, block_align=4, byte_rate=192000),
            "eight_bit": _wav_bytes(bits_per_sample=8, block_align=1, byte_rate=48000),
            "duplicate_fmt": _wav_bytes(
                chunks=[
                    _chunk(
                        b"fmt ",
                        struct.pack("<HHIIHH", 1, 1, 48000, 96000, 2, 16),
                    ),
                    _chunk(
                        b"fmt ",
                        struct.pack("<HHIIHH", 1, 1, 48000, 96000, 2, 16),
                    ),
                    _chunk(b"data", b"\x00\x00"),
                ]
            ),
            "data_before_fmt": _wav_bytes(
                chunks=[
                    _chunk(b"data", b"\x00\x00"),
                    _chunk(
                        b"fmt ",
                        struct.pack("<HHIIHH", 1, 1, 48000, 96000, 2, 16),
                    ),
                ]
            ),
            "overrun_chunk": b"RIFF\x14\x00\x00\x00WAVEdata\xff\x00\x00\x00\x00\x00",
            "trailing_garbage": _wav_bytes() + b"junk",
            "oversized_header": _wav_bytes(
                chunks=[
                    _chunk(b"fmt ", struct.pack("<HHIIHH", 1, 1, 48000, 96000, 2, 16)),
                    _chunk(b"JUNK", b"x" * 5000),
                    _chunk(b"data", b"\x00\x00"),
                ]
            ),
        }
        for name, data in cases.items():
            with self.subTest(name=name):
                self.assertFalse(validate_media_bytes(data, kind="wav"))

    def test_rejects_oversized_declared_media(self) -> None:
        oversized = b"RIFF" + struct.pack("<I", MAX_MEDIA_BYTES + 1) + b"WAVE"
        self.assertFalse(validate_media_bytes(oversized, kind="wav"))

    def test_rejects_malformed_png_shapes(self) -> None:
        valid = _png_bytes()
        cases = {
            "empty": b"",
            "random": b"not a png",
            "truncated": valid[:-3],
            "bad_crc": _png_bytes(crc_delta=1),
            "zero_width": _png_bytes(width=0),
            "zero_height": _png_bytes(height=0),
            "no_idat": _png_bytes(include_idat=False),
            "no_iend": _png_bytes(include_iend=False),
            "trailing_garbage": valid + b"x",
            "non_zlib_idat": _png_bytes(idat_payload=b"not zlib"),
            "zlib_trailing": _png_bytes(idat_payload=zlib.compress(b"\x00\x00\x00\x00\x00") + b"x"),
            "wrong_scanline_size": _png_bytes(raw_scanlines=b"\x00\x00"),
            "bad_filter_byte": _png_bytes(raw_scanlines=b"\x05\x00\x00\x00\x00"),
            "indexed_without_plte": _png_bytes(
                color_type=3,
                bit_depth=8,
                raw_scanlines=b"\x00\x00",
            ),
            "indexed_plte_too_large": _png_bytes(
                color_type=3,
                bit_depth=1,
                include_plte=True,
                plte_payload=b"\x00\x00\x00" * 3,
                raw_scanlines=b"\x00\x00",
            ),
            "unknown_critical_chunk": _png_bytes(extra_chunks=[(b"ABCD", b"")]),
            "interlaced": _png_bytes(interlace=1),
        }
        for name, data in cases.items():
            with self.subTest(name=name):
                self.assertFalse(validate_media_bytes(data, kind="png"))


if __name__ == "__main__":
    unittest.main()
