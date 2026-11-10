"""Acoustic spectrogram and waveform analysis module for forensic examination.

Provides high-resolution visual signal analysis for audio exhibits:
- Time-domain waveform visualization.
- Short-Time Fourier Transform (STFT) linear frequency spectrogram.
- Log-scaled Mel-frequency spectrogram.
- Average Power Spectral Density (PSD) frequency distribution.
- Temporal trajectory analysis: Spectral Centroid, Bandwidth, Rolloff, ZCR, RMS.
- Mel-Frequency Cepstral Coefficients (MFCC) 2D heatmap across time.

All figures adhere to forensic publication standards with precise axes, units,
and sampling parameters.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from src.utils import project_root

logger = logging.getLogger(__name__)


def compute_stft_frames(
    waveform: np.ndarray,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Short-Time Fourier Transform power frames and STFT complex matrix.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        stft_matrix (freq_bins, n_frames), power_spectrogram, frame_times
    """
    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=-1)

    window = np.hanning(n_fft)
    pad_amount = n_fft // 2
    padded_wave = np.pad(waveform, pad_amount, mode="reflect")

    n_frames = (len(padded_wave) - n_fft) // hop_length + 1
    if n_frames <= 0:
        n_frames = 1
        padded_wave = np.pad(waveform, (0, max(0, n_fft - len(waveform))), mode="constant")

    shape = (n_frames, n_fft)
    strides = (padded_wave.strides[0] * hop_length, padded_wave.strides[0])
    frames = np.lib.stride_tricks.as_strided(padded_wave, shape=shape, strides=strides)

    windowed_frames = frames * window
    stft_matrix = np.fft.rfft(windowed_frames, axis=1).T  # (n_bins, n_frames)
    power_spec = np.abs(stft_matrix) ** 2

    return stft_matrix, power_spec, np.arange(n_frames) * hop_length


def hz_to_mel(hz: np.ndarray) -> np.ndarray:
    """Convert Hertz to Mel scale using standard Slaney formula."""
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: np.ndarray) -> np.ndarray:
    """Convert Mel scale to Hertz."""
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def create_mel_filterbank(
    n_fft: int,
    sample_rate: int,
    n_mels: int = 128,
    f_min: float = 0.0,
    f_max: Optional[float] = None,
) -> np.ndarray:
    """Generate triangular Mel filterbank matrix."""
    if f_max is None:
        f_max = sample_rate / 2.0

    n_bins = n_fft // 2 + 1
    fft_freqs = np.linspace(0.0, sample_rate / 2.0, n_bins)

    mel_min = hz_to_mel(np.array([f_min]))[0]
    mel_max = hz_to_mel(np.array([f_max]))[0]
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    filterbank = np.zeros((n_mels, n_bins), dtype=np.float64)
    for i in range(n_mels):
        left = hz_points[i]
        center = hz_points[i + 1]
        right = hz_points[i + 2]

        # Up slope
        up_mask = (fft_freqs >= left) & (fft_freqs <= center)
        if center > left:
            filterbank[i, up_mask] = (fft_freqs[up_mask] - left) / (center - left)

        # Down slope
        down_mask = (fft_freqs >= center) & (fft_freqs <= right)
        if right > center:
            filterbank[i, down_mask] = (right - fft_freqs[down_mask]) / (right - center)

        # Slaney area normalization
        enorm = 2.0 / (right - left) if right > left else 1.0
        filterbank[i] *= enorm

    return filterbank


def compute_spectral_trajectories(
    waveform: np.ndarray,
    sample_rate: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mfcc: int = 20,
) -> Dict[str, np.ndarray]:
    """Compute frame-level time series for spectral centroid, bandwidth, rolloff, ZCR, RMS, and MFCCs."""
    stft_mat, power_spec, frame_indices = compute_stft_frames(waveform, n_fft=n_fft, hop_length=hop_length)
    n_bins, n_frames = power_spec.shape
    fft_freqs = np.linspace(0.0, sample_rate / 2.0, n_bins)
    time_axis = frame_indices / sample_rate

    # Spectral centroid: sum(f * P) / sum(P)
    power_sum = np.sum(power_spec, axis=0) + 1e-12
    centroid = np.sum(fft_freqs[:, np.newaxis] * power_spec, axis=0) / power_sum

    # Spectral bandwidth: sqrt(sum((f - centroid)^2 * P) / sum(P))
    deviation = fft_freqs[:, np.newaxis] - centroid[np.newaxis, :]
    bandwidth = np.sqrt(np.sum((deviation**2) * power_spec, axis=0) / power_sum)

    # Spectral rolloff (85% energy)
    cum_power = np.cumsum(power_spec, axis=0)
    thresh = 0.85 * cum_power[-1, :]
    rolloff = np.zeros(n_frames, dtype=np.float64)
    for t in range(n_frames):
        idx = np.searchsorted(cum_power[:, t], thresh[t])
        idx = min(idx, n_bins - 1)
        rolloff[t] = fft_freqs[idx]

    # Frame-wise Zero Crossing Rate (ZCR)
    window = np.hanning(n_fft)
    pad_amount = n_fft // 2
    padded = np.pad(waveform, pad_amount, mode="reflect")
    zcr = np.zeros(n_frames, dtype=np.float64)
    rms = np.zeros(n_frames, dtype=np.float64)

    for t in range(n_frames):
        start = t * hop_length
        chunk = padded[start : start + n_fft]
        if len(chunk) < n_fft:
            continue
        zcr[t] = 0.5 * np.mean(np.abs(np.diff(np.signbit(chunk))))
        rms[t] = np.sqrt(np.mean(chunk**2) + 1e-12)

    # MFCC frames (DCT-II of log Mel energy)
    fbank = create_mel_filterbank(n_fft, sample_rate, n_mels=128)
    mel_power = np.dot(fbank, power_spec) + 1e-12
    ref_val = np.max(mel_power) if np.max(mel_power) > 0 else 1.0
    mel_db = 10.0 * np.log10(np.maximum(mel_power, 1e-12)) - 10.0 * np.log10(ref_val)
    mel_db = np.maximum(mel_db, -80.0)

    # DCT Type-II
    n_mels = mel_db.shape[0]
    dct_basis = np.empty((n_mfcc, n_mels))
    dct_basis[0, :] = 1.0 / np.sqrt(n_mels)
    samples = np.arange(1, 2 * n_mels, 2) * (np.pi / (2.0 * n_mels))
    for i in range(1, n_mfcc):
        dct_basis[i, :] = np.cos(i * samples) * np.sqrt(2.0 / n_mels)
    mfcc_matrix = np.dot(dct_basis, mel_db)  # shape: (n_mfcc, n_frames)

    return {
        "time_axis": time_axis,
        "centroid": centroid,
        "bandwidth": bandwidth,
        "rolloff": rolloff,
        "zcr": zcr,
        "rms": rms,
        "mfcc_matrix": mfcc_matrix,
        "power_spec": power_spec,
        "mel_db": mel_db,
        "fft_freqs": fft_freqs,
    }


def generate_forensic_visualizations(
    waveform: np.ndarray,
    sample_rate: int,
    output_dir: Optional[Union[str, Path]] = None,
    file_stem: str = "evidence",
    title_suffix: str = "",
) -> Dict[str, Path]:
    """Generate high-resolution forensic visualization figures for an audio exhibit.

    Parameters
    ----------
    waveform : np.ndarray
        Floating-point audio signal.
    sample_rate : int
        Audio sampling rate.
    output_dir : Union[str, Path], optional
        Destination directory. Defaults to ``results/graphs/``.
    file_stem : str, default="evidence"
        Prefix stem for output image filenames.
    title_suffix : str, default=""
        Optional context string appended to plot titles (e.g. "Genuine Exhibit").

    Returns
    -------
    Dict[str, Path]
        Mapping of visualization names to their saved image filepaths.
    """
    root = project_root()
    out_dir = Path(output_dir) if output_dir else root / "results" / "graphs"
    out_dir.mkdir(parents=True, exist_ok=True)

    duration = float(waveform.size / sample_rate) if sample_rate > 0 else 0.0
    time_wave = np.linspace(0.0, duration, len(waveform))

    n_fft = 2048
    hop_length = 512
    traj = compute_spectral_trajectories(
        waveform, sample_rate, n_fft=n_fft, hop_length=hop_length, n_mfcc=20
    )

    saved_figures: Dict[str, Path] = {}

    # =========================================================================
    # 1. Master Multi-Panel Forensic Overview Figure (7-Panel Figure)
    # =========================================================================
    fig = plt.figure(figsize=(14, 16), dpi=150)
    plt.subplots_adjust(hspace=0.45)

    title_main = f"Forensic Acoustic Analysis: {file_stem} {title_suffix}".strip()
    fig.suptitle(title_main, fontsize=16, fontweight="bold", y=0.99)

    # Panel 1: Waveform
    ax1 = fig.add_subplot(6, 1, 1)
    ax1.plot(time_wave, waveform, color="#1f77b4", linewidth=0.6, alpha=0.9)
    ax1.set_xlim([0, duration])
    ax1.set_ylim([-1.05, 1.05])
    ax1.set_ylabel("Amplitude", fontsize=10)
    ax1.set_title(f"1. Audio Waveform (Sampling Rate: {sample_rate:,} Hz, Duration: {duration:.2f}s)", fontsize=11, loc="left")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Panel 2: STFT Spectrogram (dB)
    ax2 = fig.add_subplot(6, 1, 2)
    spec_db = 10.0 * np.log10(np.maximum(traj["power_spec"], 1e-12))
    spec_db -= np.max(spec_db)
    im2 = ax2.imshow(
        spec_db,
        origin="lower",
        aspect="auto",
        cmap="magma",
        extent=[0, duration, 0, sample_rate / 2],
        vmin=-80,
        vmax=0,
    )
    ax2.set_ylabel("Frequency (Hz)", fontsize=10)
    ax2.set_title("2. Short-Time Fourier Transform (STFT) Spectrogram (dB)", fontsize=11, loc="left")
    cbar2 = fig.colorbar(im2, ax=ax2, pad=0.015)
    cbar2.set_label("dBFS", fontsize=9)

    # Panel 3: Mel Spectrogram (dB)
    ax3 = fig.add_subplot(6, 1, 3)
    im3 = ax3.imshow(
        traj["mel_db"],
        origin="lower",
        aspect="auto",
        cmap="viridis",
        extent=[0, duration, 0, 128],
        vmin=-80,
        vmax=0,
    )
    ax3.set_ylabel("Mel Band (1-128)", fontsize=10)
    ax3.set_title("3. Log Mel-Frequency Spectrogram (128 Bands, Slaney Scaling)", fontsize=11, loc="left")
    cbar3 = fig.colorbar(im3, ax=ax3, pad=0.015)
    cbar3.set_label("dB", fontsize=9)

    # Panel 4: MFCC Heatmap (20 coefficients)
    ax4 = fig.add_subplot(6, 1, 4)
    im4 = ax4.imshow(
        traj["mfcc_matrix"],
        origin="lower",
        aspect="auto",
        cmap="coolwarm",
        extent=[0, duration, 1, 20],
    )
    ax4.set_ylabel("MFCC Index", fontsize=10)
    ax4.set_yticks([1, 5, 10, 15, 20])
    ax4.set_title("4. Mel-Frequency Cepstral Coefficients (MFCC 1-20)", fontsize=11, loc="left")
    cbar4 = fig.colorbar(im4, ax=ax4, pad=0.015)
    cbar4.set_label("Magnitude", fontsize=9)

    # Panel 5: Spectral Dynamics (Centroid & Rolloff)
    ax5 = fig.add_subplot(6, 1, 5)
    ax5.plot(traj["time_axis"], traj["rolloff"], color="#d62728", label="Spectral Rolloff (85%)", linewidth=1.1)
    ax5.plot(traj["time_axis"], traj["centroid"], color="#2ca02c", label="Spectral Centroid", linewidth=1.1)
    ax5.set_xlim([0, duration])
    ax5.set_ylim([0, sample_rate / 2])
    ax5.set_ylabel("Frequency (Hz)", fontsize=10)
    ax5.set_title("5. Spectral Dynamics Trajectory (Centroid and 85% Rolloff)", fontsize=11, loc="left")
    ax5.legend(loc="upper right", fontsize=9)
    ax5.grid(True, linestyle="--", alpha=0.5)

    # Panel 6: Signal Energy & ZCR
    ax6 = fig.add_subplot(6, 1, 6)
    ax6_twin = ax6.twinx()
    p_rms = ax6.plot(traj["time_axis"], traj["rms"], color="#1f77b4", label="RMS Energy", linewidth=1.1)
    p_zcr = ax6_twin.plot(traj["time_axis"], traj["zcr"], color="#ff7f0e", label="Zero-Crossing Rate", linewidth=1.1, linestyle=":")
    ax6.set_xlim([0, duration])
    ax6.set_xlabel("Time (seconds)", fontsize=10)
    ax6.set_ylabel("RMS Amplitude", fontsize=10, color="#1f77b4")
    ax6_twin.set_ylabel("ZCR", fontsize=10, color="#ff7f0e")
    ax6.set_title("6. Signal Energy Dynamics and Zero-Crossing Rate", fontsize=11, loc="left")
    plots = p_rms + p_zcr
    labels = [p.get_label() for p in plots]
    ax6.legend(plots, labels, loc="upper right", fontsize=9)
    ax6.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    overview_path = out_dir / f"forensic_overview_{file_stem}.png"
    fig.savefig(overview_path, format="png", bbox_inches="tight")
    plt.close(fig)
    saved_figures["overview"] = overview_path

    # =========================================================================
    # 2. Individual High-Resolution Figures for Modular Reports
    # =========================================================================
    # (a) Waveform & STFT Spectrogram Pair
    fig_spec, (s_ax1, s_ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, dpi=120)
    s_ax1.plot(time_wave, waveform, color="#1f77b4", linewidth=0.7)
    s_ax1.set_ylabel("Amplitude")
    s_ax1.set_title(f"Waveform ({sample_rate} Hz, Mono)")
    s_ax1.grid(True, linestyle="--", alpha=0.5)

    im_s = s_ax2.imshow(
        spec_db,
        origin="lower",
        aspect="auto",
        cmap="magma",
        extent=[0, duration, 0, sample_rate / 2],
        vmin=-80,
        vmax=0,
    )
    s_ax2.set_xlabel("Time (seconds)")
    s_ax2.set_ylabel("Frequency (Hz)")
    s_ax2.set_title("STFT Spectrogram (dBFS)")
    fig_spec.colorbar(im_s, ax=s_ax2, pad=0.02).set_label("dB")
    fig_spec.tight_layout()
    spec_path = out_dir / f"spectrogram_{file_stem}.png"
    fig_spec.savefig(spec_path, format="png", bbox_inches="tight")
    plt.close(fig_spec)
    saved_figures["spectrogram"] = spec_path

    # (b) Mel Spectrogram & MFCC
    fig_mel, (m_ax1, m_ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, dpi=120)
    im_m1 = m_ax1.imshow(
        traj["mel_db"],
        origin="lower",
        aspect="auto",
        cmap="viridis",
        extent=[0, duration, 0, 128],
        vmin=-80,
        vmax=0,
    )
    m_ax1.set_ylabel("Mel Band (1-128)")
    m_ax1.set_title("Mel Spectrogram (dB)")
    fig_mel.colorbar(im_m1, ax=m_ax1, pad=0.02).set_label("dB")

    im_m2 = m_ax2.imshow(
        traj["mfcc_matrix"],
        origin="lower",
        aspect="auto",
        cmap="coolwarm",
        extent=[0, duration, 1, 20],
    )
    m_ax2.set_xlabel("Time (seconds)")
    m_ax2.set_ylabel("MFCC Index")
    m_ax2.set_yticks([1, 5, 10, 15, 20])
    m_ax2.set_title("MFCC 1-20 Heatmap")
    fig_mel.colorbar(im_m2, ax=m_ax2, pad=0.02).set_label("Value")
    fig_mel.tight_layout()
    mel_path = out_dir / f"mel_mfcc_{file_stem}.png"
    fig_mel.savefig(mel_path, format="png", bbox_inches="tight")
    plt.close(fig_mel)
    saved_figures["mel_mfcc"] = mel_path

    return saved_figures
