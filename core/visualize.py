"""Waveform and spectrogram comparison helpers owned by B."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _load_audio(path: str) -> tuple[Any, int]:
    try:
        import numpy as np
        import soundfile as sf
    except Exception as exc:  # pragma: no cover - depends on local env.
        raise RuntimeError("生成可视化需要安装 numpy 和 soundfile。") from exc

    samples, sample_rate = sf.read(path, always_2d=False)
    if getattr(samples, "ndim", 1) > 1:
        samples = samples.mean(axis=1)
    samples = np.asarray(samples, dtype=float)
    if samples.size == 0:
        raise ValueError(f"音频为空，无法生成可视化：{path}")
    return samples, int(sample_rate)


def _ensure_parent(output_path: str) -> Path:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _shared_crop(original: Any, enhanced: Any, sample_rate: int) -> tuple[Any, Any, bool]:
    min_len = min(len(original), len(enhanced))
    length_delta = abs(len(original) - len(enhanced)) / sample_rate
    return original[:min_len], enhanced[:min_len], length_delta > 0.2


def create_waveform_comparison(original_wav: str, enhanced_wav: str, output_path: str) -> str:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:  # pragma: no cover - depends on local env.
        raise RuntimeError("生成波形图需要安装 matplotlib 和 numpy。") from exc

    original, sr_original = _load_audio(original_wav)
    enhanced, sr_enhanced = _load_audio(enhanced_wav)
    if sr_original != sr_enhanced:
        raise ValueError("前后音频采样率不同，不能生成公平波形对照。")

    original, enhanced, length_warning = _shared_crop(original, enhanced, sr_original)
    time_axis = np.arange(len(original)) / sr_original
    limit = max(float(np.max(np.abs(original))), float(np.max(np.abs(enhanced))), 1e-6)
    target = _ensure_parent(output_path)

    fig, axes = plt.subplots(2, 1, figsize=(9, 4), sharex=True, sharey=True)
    axes[0].plot(time_axis, original, color="#555555", linewidth=0.8)
    axes[0].set_title("Original")
    axes[1].plot(time_axis, enhanced, color="#1769aa", linewidth=0.8)
    axes[1].set_title("Enhanced (mixed)")
    for axis in axes:
        axis.set_ylim(-limit, limit)
        axis.set_ylabel("Amplitude")
        axis.grid(True, alpha=0.25)
    if length_warning:
        axes[0].text(0.01, 0.92, "Length differs >0.2s; common range shown", transform=axes[0].transAxes)
    axes[1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(target, dpi=140)
    plt.close(fig)
    return str(target)


def create_spectrogram_comparison(original_wav: str, enhanced_wav: str, output_path: str) -> str:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:  # pragma: no cover - depends on local env.
        raise RuntimeError("生成声谱图需要安装 matplotlib 和 numpy。") from exc

    original, sr_original = _load_audio(original_wav)
    enhanced, sr_enhanced = _load_audio(enhanced_wav)
    if sr_original != sr_enhanced:
        raise ValueError("前后音频采样率不同，不能生成公平声谱图。")

    original, enhanced, length_warning = _shared_crop(original, enhanced, sr_original)
    n_fft = 1024
    hop_length = 256
    target = _ensure_parent(output_path)

    def stft_db(samples):
        window = np.hanning(n_fft)
        frames = []
        max_start = max(len(samples) - n_fft + 1, 1)
        for start in range(0, max_start, hop_length):
            frame = samples[start : start + n_fft]
            if len(frame) < n_fft:
                frame = np.pad(frame, (0, n_fft - len(frame)))
            spectrum = np.fft.rfft(frame * window)
            frames.append(np.abs(spectrum))
        magnitude = np.stack(frames, axis=1)
        return 20 * np.log10(np.maximum(magnitude, 1e-8))

    original_db = stft_db(original)
    enhanced_db = stft_db(enhanced)
    vmin = min(float(original_db.min()), float(enhanced_db.min()))
    vmax = max(float(original_db.max()), float(enhanced_db.max()))
    extent = [0, len(original) / sr_original, 0, sr_original / 2]

    fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True, sharey=True)
    image = None
    for axis, data, title in (
        (axes[0], original_db, "Original"),
        (axes[1], enhanced_db, "Enhanced (mixed)"),
    ):
        image = axis.imshow(
            data,
            origin="lower",
            aspect="auto",
            cmap="magma",
            vmin=vmin,
            vmax=vmax,
            extent=extent,
        )
        axis.set_title(title)
        axis.set_ylabel("Frequency (Hz)")
    if length_warning:
        axes[0].text(0.01, 0.92, "Length differs >0.2s; common range shown", transform=axes[0].transAxes)
    axes[1].set_xlabel("Time (s)")
    fig.colorbar(image, ax=axes, label="dB")
    fig.savefig(target, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return str(target)
