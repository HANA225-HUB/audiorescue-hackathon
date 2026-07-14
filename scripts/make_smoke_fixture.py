"""Create the public, deterministic integration smoke fixture.

The waveform is generated entirely from mathematical tones and seeded noise;
it contains no human/TTS recording and carries no evaluation claim.
"""

from __future__ import annotations

import math
import random
import struct
import wave
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "tests" / "fixtures" / "dev_smoke_s01_fan.wav"
SAMPLE_RATE = 48_000
TARGET_SECONDS = 12.0
RANDOM_SEED = 20260714


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
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    target_length = round(TARGET_SECONDS * SAMPLE_RATE)
    rng = random.Random(RANDOM_SEED)
    low_pass = 0.0
    mixed: list[float] = []
    for index in range(target_length):
        time_seconds = index / SAMPLE_RATE
        white = rng.uniform(-1.0, 1.0)
        low_pass = 0.997 * low_pass + 0.003 * white
        hum = math.sin(2.0 * math.pi * 100.0 * index / SAMPLE_RATE)
        fan = 0.20 * low_pass + 0.012 * hum

        # A repeated, non-speech chirp makes before/after waveform handling
        # observable without embedding any voice or copyrighted recording.
        phase = time_seconds % 1.5
        envelope = 0.0
        if 0.35 <= time_seconds <= TARGET_SECONDS - 0.35 and phase < 0.9:
            edge = min(phase / 0.06, (0.9 - phase) / 0.06, 1.0)
            envelope = max(0.0, edge)
        chirp_frequency = 260.0 + 420.0 * min(phase / 0.9, 1.0)
        chirp = envelope * (
            0.15 * math.sin(2.0 * math.pi * chirp_frequency * time_seconds)
            + 0.05 * math.sin(2.0 * math.pi * 2.0 * chirp_frequency * time_seconds)
        )
        mixed.append(chirp + fan)

    peak = max(abs(sample) for sample in mixed) or 1.0
    gain = min(1.0, 0.95 / peak)
    write_pcm16_mono(OUTPUT, [sample * gain for sample in mixed])
    print(f"created {OUTPUT}")


if __name__ == "__main__":
    main()
