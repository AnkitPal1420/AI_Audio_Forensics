"""Audio feature extraction for synthetic and genuine voice analysis.

Extracts standard forensic acoustic and spectral representations:
1. Mel-Frequency Cepstral Coefficients (20 MFCCs, mean and std).
2. Spectral Centroid (mean and std).
3. Spectral Bandwidth (mean and std).
4. Spectral Rolloff (85% energy percentile, mean and std).
5. Zero-Crossing Rate (mean and std).
6. Root-Mean-Square Energy (RMS, mean and std).
7. Chroma Feature Representation (12 pitch classes, mean and std, plus aggregate).
8. Spectral Contrast (octave sub-bands, peak vs valley energy difference).
9. Spectral Flatness (Wiener entropy measure of tonality vs noisiness).

All features are implemented deterministically using pure NumPy and standard
signal-processing transforms, adhering to standard decibel-scaled mel power
representations to ensure full mathematical consistency between training
and inference pipelines.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from src.audio_preprocessing import load_audio

logger = logging.getLogger(__name__)

AudioPath = Union[str, Path]

# Canonical 26 feature schema used for baseline forensic classifiers
CANONICAL_FEATURE_NAMES: List[str] = [
    "chroma_stft",
    "rms",
    "spectral_centroid",
    "spectral_bandwidth",
    "rolloff",
    "zero_crossing_rate",
] + [f"mfcc{i}" for i in range(1, 21)]


def extract_basic_features(
    waveform: np.ndarray,
    sample_rate: int,
    n_mfcc: int = 20,
) -> Dict[str, float]:
    """Extract standard forensic acoustic features from a 1D mono waveform.

    Extracts:
    - 20 MFCC coefficients (or ``n_mfcc``, up to 40)
    - Spectral centroid
    - Spectral bandwidth
    - Spectral rolloff (85% threshold)
    - Zero-crossing rate
    - RMS energy
    - Chroma (12 pitch classes)
    - Spectral contrast (octave sub-bands)
    - Spectral flatness

    Parameters
    ----------
    waveform : np.ndarray
        One-dimensional mono audio signal.
    sample_rate : int
        Sample rate of the audio in Hertz.
    n_mfcc : int, default=20
        Number of MFCC coefficients to extract (1 to 40).

    Returns
    -------
    Dict[str, float]
        Dictionary mapping feature names to computed floating-point statistics.
    """
    samples = np.asarray(waveform, dtype=np.float32)
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("waveform must be a non-empty one-dimensional signal")
    if not np.all(np.isfinite(samples)):
        raise ValueError("waveform must contain only finite values")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer")
    if n_mfcc <= 0:
        raise ValueError("n_mfcc must be a positive integer")
    if n_mfcc > 40:
        raise ValueError("n_mfcc cannot exceed 40")

    # Frame configuration
    frame_length = min(2048, samples.size)
    hop_length = max(1, frame_length // 4)
    pad_len = frame_length // 2

    if samples.size < frame_length:
        pad_width = frame_length - samples.size
        padded = np.pad(samples, (pad_len, pad_len + pad_width), mode="reflect")
    else:
        padded = np.pad(samples, (pad_len, pad_len), mode="reflect")

    # Strided frame view
    num_frames = max(1, (padded.size - frame_length) // hop_length + 1)
    shape = (num_frames, frame_length)
    strides = (padded.strides[0] * hop_length, padded.strides[0])
    frames = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)

    # Windowing and short-time Fourier transform (STFT)
    window = np.hanning(frame_length).astype(np.float32)
    windowed_frames = frames * window
    spectrum = np.fft.rfft(windowed_frames, axis=1).T  # (freq_bins, num_frames)
    frequencies = np.fft.rfftfreq(frame_length, d=1.0 / sample_rate)
    magnitude = np.abs(spectrum)
    power = magnitude**2

    # 1. Root-Mean-Square (RMS) Energy
    rms_per_frame = np.sqrt(np.mean(frames**2, axis=1))
    rms_mean = float(np.mean(rms_per_frame))
    rms_std = float(np.std(rms_per_frame))

    # 2. Zero-Crossing Rate (ZCR)
    signs = np.sign(frames)
    signs[signs == 0] = 1
    zcr_per_frame = np.mean(np.abs(np.diff(signs, axis=1)) > 0, axis=1)
    zcr_mean = float(np.mean(zcr_per_frame))
    zcr_std = float(np.std(zcr_per_frame))

    # 3. Spectral Centroid
    frame_energy = np.sum(magnitude, axis=0)
    centroid = np.divide(
        frequencies @ magnitude,
        frame_energy,
        out=np.zeros_like(frame_energy),
        where=frame_energy > 0,
    )
    spectral_centroid_mean = float(np.mean(centroid))
    spectral_centroid_std = float(np.std(centroid))

    # 4. Spectral Bandwidth
    freq_diff = frequencies[:, None] - centroid[None, :]
    bandwidth = np.sqrt(
        np.divide(
            np.sum((freq_diff**2) * magnitude, axis=0),
            frame_energy,
            out=np.zeros_like(frame_energy),
            where=frame_energy > 0,
        )
    )
    spectral_bandwidth_mean = float(np.mean(bandwidth))
    spectral_bandwidth_std = float(np.std(bandwidth))

    # 5. Spectral Rolloff (85% cumulative spectral energy)
    cumulative_energy = np.cumsum(magnitude, axis=0)
    rolloff_threshold = 0.85 * frame_energy
    rolloff_indices = np.argmax(cumulative_energy >= rolloff_threshold[None, :], axis=0)
    rolloff = frequencies[rolloff_indices]
    spectral_rolloff_mean = float(np.mean(rolloff))
    spectral_rolloff_std = float(np.std(rolloff))

    # 6. Spectral Flatness (Wiener entropy)
    eps = 1e-10
    log_power = np.log(power + eps)
    geom_mean = np.exp(np.mean(log_power, axis=0))
    arith_mean = np.mean(power, axis=0) + eps
    flatness = np.divide(geom_mean, arith_mean)
    spectral_flatness_mean = float(np.mean(flatness))
    spectral_flatness_std = float(np.std(flatness))

    # 7. Chroma (12 pitch classes)
    chroma_filters = np.zeros((12, frequencies.size), dtype=np.float32)
    valid_freq_mask = frequencies > 60.0  # Above C2 (~65.4 Hz)
    if np.any(valid_freq_mask):
        midi_notes = 12.0 * np.log2(frequencies[valid_freq_mask] / 440.0) + 69.0
        pitch_classes = np.mod(np.round(midi_notes), 12).astype(int)
        for pc in range(12):
            mask = pitch_classes == pc
            indices = np.where(valid_freq_mask)[0][mask]
            chroma_filters[pc, indices] = 1.0
        row_sums = chroma_filters.sum(axis=1, keepdims=True)
        chroma_filters = np.divide(chroma_filters, row_sums, out=chroma_filters, where=row_sums > 0)
    raw_chroma = chroma_filters @ power
    chroma_norm = np.linalg.norm(raw_chroma, axis=0, keepdims=True)
    chroma = np.divide(raw_chroma, chroma_norm, out=raw_chroma, where=chroma_norm > 0)
    chroma_mean = float(np.mean(chroma))
    chroma_std = float(np.std(chroma))

    # 8. Spectral Contrast (6 octave sub-bands: 200 to Nyquist)
    band_edges = [200.0, 400.0, 800.0, 1600.0, 3200.0, 6400.0, sample_rate / 2.0]
    contrasts: List[np.ndarray] = []
    for i in range(len(band_edges) - 1):
        low, high = band_edges[i], band_edges[i + 1]
        band_mask = (frequencies >= low) & (frequencies < high)
        if np.any(band_mask):
            sub_mag = magnitude[band_mask, :]
            peak = np.quantile(sub_mag, 0.85, axis=0)
            valley = np.quantile(sub_mag, 0.15, axis=0)
            peak_db = 20.0 * np.log10(np.maximum(peak, 1e-6))
            valley_db = 20.0 * np.log10(np.maximum(valley, 1e-6))
            contrasts.append(peak_db - valley_db)
        else:
            contrasts.append(np.zeros(num_frames, dtype=np.float32))
    contrast_bands = np.array(contrasts)
    overall_contrast = np.mean(contrast_bands, axis=0)
    spectral_contrast_mean = float(np.mean(overall_contrast))
    spectral_contrast_std = float(np.std(overall_contrast))

    # 9. Mel-Frequency Cepstral Coefficients (MFCCs) with standard Slaney & dB scaling
    n_mels = 128
    mel_pts = np.linspace(
        0.0,
        2595.0 * np.log10(1.0 + (sample_rate / 2.0) / 700.0),
        n_mels + 2,
    )
    hz_pts = 700.0 * (10.0 ** (mel_pts / 2595.0) - 1.0)
    mel_filters = np.zeros((n_mels, frequencies.size), dtype=np.float32)
    for i in range(n_mels):
        low, mid, high = hz_pts[i], hz_pts[i + 1], hz_pts[i + 2]
        rising = (frequencies - low) / (mid - low)
        falling = (high - frequencies) / (high - mid)
        mel_filters[i] = np.maximum(0.0, np.minimum(rising, falling)) * (2.0 / (high - low))

    mel_spec = mel_filters @ power
    # Standard 10*log10(power) decibel representation (ref=1.0, 80 dB floor)
    db_mel = 10.0 * np.log10(np.maximum(1e-10, mel_spec))
    db_mel = np.maximum(db_mel, np.max(db_mel) - 80.0)

    # DCT type 2 orthogonal basis
    coeffs = np.arange(n_mels, dtype=np.float32)[:, None]
    bands = np.arange(n_mels, dtype=np.float32)[None, :]
    dct_basis = np.cos(np.pi / n_mels * (bands + 0.5) * coeffs)
    dct_basis[0] *= np.sqrt(1.0 / n_mels)
    dct_basis[1:] *= np.sqrt(2.0 / n_mels)
    mfcc_all = (dct_basis @ db_mel)[:n_mfcc]

    # Aggregate dictionary
    features: Dict[str, float] = {
        "duration_seconds": float(samples.size / sample_rate),
        "rms_mean": rms_mean,
        "rms_std": rms_std,
        "zero_crossing_rate_mean": zcr_mean,
        "zero_crossing_rate_std": zcr_std,
        "spectral_centroid_mean": spectral_centroid_mean,
        "spectral_centroid_std": spectral_centroid_std,
        "spectral_bandwidth_mean": spectral_bandwidth_mean,
        "spectral_bandwidth_std": spectral_bandwidth_std,
        "spectral_rolloff_mean": spectral_rolloff_mean,
        "spectral_rolloff_std": spectral_rolloff_std,
        "spectral_flatness_mean": spectral_flatness_mean,
        "spectral_flatness_std": spectral_flatness_std,
        "spectral_contrast_mean": spectral_contrast_mean,
        "spectral_contrast_std": spectral_contrast_std,
        "chroma_mean": chroma_mean,
        "chroma_std": chroma_std,
        # Direct canonical aliases
        "chroma_stft": chroma_mean,
        "rms": rms_mean,
        "spectral_centroid": spectral_centroid_mean,
        "spectral_bandwidth": spectral_bandwidth_mean,
        "rolloff": spectral_rolloff_mean,
        "zero_crossing_rate": zcr_mean,
    }

    # Per-band contrast
    for b_idx, b_arr in enumerate(contrast_bands, start=1):
        features[f"spectral_contrast_band_{b_idx}_mean"] = float(np.mean(b_arr))

    # Per-class chroma
    for pc_idx, pc_arr in enumerate(chroma, start=1):
        features[f"chroma_{pc_idx}_mean"] = float(np.mean(pc_arr))
        features[f"chroma_{pc_idx}_std"] = float(np.std(pc_arr))

    # Per-coefficient MFCC
    for m_idx, m_arr in enumerate(mfcc_all, start=1):
        features[f"mfcc_{m_idx}_mean"] = float(np.mean(m_arr))
        features[f"mfcc_{m_idx}_std"] = float(np.std(m_arr))
        features[f"mfcc{m_idx}"] = float(np.mean(m_arr))

    return features


def extract_features(
    waveform: np.ndarray,
    sample_rate: int,
    n_mfcc: int = 20,
) -> Dict[str, float]:
    """Alias for ``extract_basic_features`` providing standard feature naming."""
    return extract_basic_features(waveform=waveform, sample_rate=sample_rate, n_mfcc=n_mfcc)


def validate_feature_vector(
    features: Dict[str, float],
    expected_names: Sequence[str] = CANONICAL_FEATURE_NAMES,
) -> np.ndarray:
    """Validate feature dictionary against expected schema and return ordered vector.

    Raises
    ------
    ValueError
        If required features are missing, non-finite, or malformed.
    """
    if not isinstance(features, dict):
        raise ValueError(f"Features must be a dictionary, got {type(features)}")

    vector = []
    missing = []
    invalid = []

    for name in expected_names:
        if name not in features:
            missing.append(name)
        else:
            val = features[name]
            if not np.isfinite(val):
                invalid.append((name, val))
            vector.append(float(val))

    if missing:
        raise ValueError(f"Feature vector validation failed: Missing required features: {missing}")
    if invalid:
        raise ValueError(f"Feature vector validation failed: Non-finite feature values: {invalid}")

    return np.array([vector], dtype=np.float32)


def extract_features_from_file(
    path: AudioPath,
    sample_rate: int = 16_000,
    n_mfcc: int = 20,
    normalize: bool = True,
    max_duration: Optional[float] = None,
) -> Dict[str, Any]:
    """Load an audio file, standardise it, and extract all forensic features."""
    file_path = Path(path)
    waveform, sr = load_audio(
        file_path,
        sample_rate=sample_rate,
        mono=True,
        normalize=normalize,
        max_duration=max_duration,
    )
    features = extract_basic_features(waveform, sample_rate=sr, n_mfcc=n_mfcc)
    result: Dict[str, Any] = {
        "file_path": str(file_path),
        "file_name": file_path.name,
        "sample_rate": sr,
    }
    result.update(features)
    return result


def extract_slice_features_from_audio(
    waveform: np.ndarray,
    sample_rate: int = 16_000,
    slice_duration: float = 1.0,
    n_mfcc: int = 20,
) -> List[Dict[str, float]]:
    """Slice an audio waveform into uniform temporal windows and extract features per slice."""
    slice_samples = int(round(slice_duration * sample_rate))
    if slice_samples <= 0:
        raise ValueError(f"Invalid slice duration: {slice_duration}")

    total_samples = waveform.size
    slices: List[Dict[str, float]] = []

    for start_idx in range(0, total_samples, slice_samples):
        end_idx = min(start_idx + slice_samples, total_samples)
        chunk = waveform[start_idx:end_idx]
        if chunk.size < sample_rate // 4:  # Skip chunks shorter than 250ms
            continue
        feat = extract_basic_features(chunk, sample_rate=sample_rate, n_mfcc=n_mfcc)
        feat["slice_start_sec"] = float(start_idx / sample_rate)
        feat["slice_end_sec"] = float(end_idx / sample_rate)
        slices.append(feat)

    return slices


def extract_dataset_features(
    dataset_dir: Optional[AudioPath] = None,
    output_csv: Optional[AudioPath] = None,
    sample_rate: int = 16_000,
    n_mfcc: int = 20,
    max_duration_per_file: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Extract features across all genuine and synthetic recordings and save to CSV."""
    from src.utils import project_root

    root = project_root()
    base_dir = Path(dataset_dir) if dataset_dir else root / "dataset"
    out_file = Path(output_csv) if output_csv else root / "results" / "reports" / "extracted_features.csv"

    classes = {"real": "genuine", "fake": "synthetic"}
    all_rows: List[Dict[str, Any]] = []

    for sub_dir, label in classes.items():
        src_path = base_dir / sub_dir
        if not src_path.is_dir():
            continue

        for audio_file in sorted(src_path.iterdir()):
            if not audio_file.is_file() or audio_file.suffix.lower() not in {".wav", ".mp3", ".flac"}:
                continue

            try:
                row = extract_features_from_file(
                    audio_file,
                    sample_rate=sample_rate,
                    n_mfcc=n_mfcc,
                    max_duration=max_duration_per_file,
                )
                row["label"] = label
                row["category"] = sub_dir
                all_rows.append(row)
                logger.info("Extracted features for %s [%s]", audio_file.name, label)
            except Exception as exc:
                logger.error("Failed to extract features from %s: %s", audio_file, exc)

    if all_rows:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(all_rows[0].keys())
        with out_file.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)

    return all_rows
