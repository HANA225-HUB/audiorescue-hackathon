"""Create the private integration smoke fixture on macOS.

This artifact is deliberately synthetic and must never be reported as an
evaluation or demonstration result.
"""

from __future__ import annotations

import math
import random
import shutil
import struct
import subprocess
import tempfile
import wave
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "tests" / "fixtures" / "dev_smoke_s01_fan.wav"
TEXT = "今天下午三点，我们在实验室讨论语音处理项目的最终方案。"
SAMPLE_RATE = 48_000
TARGET_SECONDS = 12.0
RANDOM_SEED = 20260714


def require(command: str) -> str:
    path = shutil.which(command)
    if path is None:
        raise SystemExit(f"Required macOS command not found: {command}")
    return path


def read_pcm16_mono(path: Path) -> list[float]:
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            raise RuntimeError("afconvert did not produce mono PCM16")
        frames = wav.readframes(wav.getnframes())
    values = struct.unpack(f"<{len(frames) // 2}h", frames)
    return [value / 32768.0 for value in values]


def write_pcm16_mono(path: Path, samples: list[float]) -> None:
    encoded = bytearray()
    for sample in samples:
        value = max(-1.0, min(1.0, sample))
        encoded.extend(struct.pack("<h", round(value * 32767.0)))
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(bytes(encoded))


def main() -> None:
    say = require("say")
    afconvert = require("afconvert")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        aiff_path = temp / "speech.aiff"
        wav_path = temp / "speech.wav"
        subprocess.run(
            [say, "-v", "Tingting", "-r", "145", "-o", str(aiff_path), TEXT],
            check=True,
        )
        subprocess.run(
            [
                afconvert,
                "-f",
                "WAVE",
                "-d",
                f"LEI16@{SAMPLE_RATE}",
                "-c",
                "1",
                str(aiff_path),
                str(wav_path),
            ],
            check=True,
        )
        speech = read_pcm16_mono(wav_path)

    target_length = round(TARGET_SECONDS * SAMPLE_RATE)
    pad_total = max(0, target_length - len(speech))
    pad_left = min(round(0.5 * SAMPLE_RATE), pad_total // 2)
    padded = [0.0] * pad_left + speech
    padded.extend([0.0] * max(0, target_length - len(padded)))
    padded = padded[:target_length]

    rng = random.Random(RANDOM_SEED)
    low_pass = 0.0
    mixed: list[float] = []
    for index, sample in enumerate(padded):
        white = rng.uniform(-1.0, 1.0)
        low_pass = 0.997 * low_pass + 0.003 * white
        hum = math.sin(2.0 * math.pi * 100.0 * index / SAMPLE_RATE)
        noise = 0.055 * low_pass + 0.008 * hum
        mixed.append(sample + noise)

    peak = max(abs(sample) for sample in mixed) or 1.0
    gain = min(1.0, 0.95 / peak)
    write_pcm16_mono(OUTPUT, [sample * gain for sample in mixed])
    print(f"created {OUTPUT}")


if __name__ == "__main__":
    main()
