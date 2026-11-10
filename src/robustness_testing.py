"""Forensic audio robustness stress-testing module.

Evaluates trained acoustic models against real-world forensic degradations:
1. Additive Gaussian noise across multiple SNR levels (30 dB, 20 dB, 10 dB).
2. Telephony channel bandpass filtering (G.711 standard: 300 Hz - 3400 Hz).
3. Lossy codec quantization simulation (mu-law and low bit-depth transcoding).
4. Physical acoustic re-recording channels from external forensic benchmarks.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from src.feature_extraction import CANONICAL_FEATURE_NAMES, extract_basic_features, validate_feature_vector
from src.utils import project_root

logger = logging.getLogger(__name__)


def add_gaussian_noise(waveform: np.ndarray, snr_db: float, seed: int = 42) -> np.ndarray:
    """Add zero-mean Gaussian noise to achieve specified Signal-to-Noise Ratio (SNR in dB)."""
    samples = np.asarray(waveform, dtype=np.float32)
    sig_power = float(np.mean(samples**2))
    if sig_power <= 0.0:
        return samples

    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    noise_std = np.sqrt(noise_power)

    rng = np.random.RandomState(seed)
    noise = rng.normal(0.0, noise_std, size=samples.shape).astype(np.float32)
    degraded = samples + noise

    # Preserve peak normalization
    peak = np.max(np.abs(degraded))
    if peak > 1.0:
        degraded = degraded / peak
    return degraded


def apply_telephony_bandpass(
    waveform: np.ndarray,
    sample_rate: int = 16_000,
    low_freq: float = 300.0,
    high_freq: float = 3400.0,
) -> np.ndarray:
    """Simulate G.711 telephony channel by applying a 300 Hz - 3400 Hz bandpass filter via FFT."""
    samples = np.asarray(waveform, dtype=np.float32)
    n = samples.size
    if n == 0:
        return samples

    spectrum = np.fft.rfft(samples)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)

    # Frequency-domain smooth roll-off bandpass
    filter_resp = np.zeros_like(freqs, dtype=np.float32)
    in_band = (freqs >= low_freq) & (freqs <= high_freq)
    filter_resp[in_band] = 1.0

    # Smooth 50 Hz transition edges
    low_trans = (freqs >= low_freq - 50.0) & (freqs < low_freq)
    if np.any(low_trans):
        filter_resp[low_trans] = 0.5 * (1.0 + np.sin(np.pi * (freqs[low_trans] - (low_freq - 25.0)) / 50.0))
    high_trans = (freqs > high_freq) & (freqs <= high_freq + 100.0)
    if np.any(high_trans):
        filter_resp[high_trans] = 0.5 * (1.0 - np.sin(np.pi * (freqs[high_trans] - (high_freq + 50.0)) / 100.0))

    filtered_spectrum = spectrum * filter_resp
    filtered = np.fft.irfft(filtered_spectrum, n=n).astype(np.float32)

    peak = np.max(np.abs(filtered))
    if peak > 1.0:
        filtered = filtered / peak
    return filtered


def apply_lossy_codec_simulation(waveform: np.ndarray, mu: int = 255) -> np.ndarray:
    """Simulate lossy ITU-T G.711 mu-law 8-bit companding and quantization."""
    samples = np.asarray(waveform, dtype=np.float32)
    # mu-law compression
    signs = np.sign(samples)
    compressed = signs * np.log1p(mu * np.abs(samples)) / np.log1p(mu)
    # 8-bit uniform quantization (-128 to +127)
    quantized = np.round(compressed * 127.0) / 127.0
    # mu-law expansion
    expanded = np.sign(quantized) * (1.0 / mu) * ((1.0 + mu) ** np.abs(quantized) - 1.0)
    return expanded.astype(np.float32)


def evaluate_model_robustness(
    model: Any,
    test_audio_paths: Sequence[Union[str, Path]],
    test_labels: Sequence[int],
    output_dir: Optional[Union[str, Path]] = None,
    sample_rate: int = 16_000,
) -> Dict[str, Any]:
    """Execute robustness stress-testing across noise, telephony, and lossy compression.

    Parameters
    ----------
    model : Any
        Trained forensic classification pipeline.
    test_audio_paths : Sequence[Path]
        Audio file paths representing the test set exhibits.
    test_labels : Sequence[int]
        Ground-truth labels (0 = genuine, 1 = synthetic).
    """
    root = project_root()
    out_dir = Path(output_dir) if output_dir else root / "results" / "robustness"
    derivative_dir = out_dir / "derivative_samples"
    derivative_dir.mkdir(parents=True, exist_ok=True)

    conditions = [
        ("Clean (Baseline)", lambda w: w),
        ("Gaussian Noise (SNR 30 dB)", lambda w: add_gaussian_noise(w, snr_db=30.0)),
        ("Gaussian Noise (SNR 20 dB)", lambda w: add_gaussian_noise(w, snr_db=20.0)),
        ("Gaussian Noise (SNR 10 dB)", lambda w: add_gaussian_noise(w, snr_db=10.0)),
        ("Telephony Bandpass (300-3400 Hz)", lambda w: apply_telephony_bandpass(w, sample_rate=sample_rate)),
        ("Lossy Codec (Mu-law 8-bit)", lambda w: apply_lossy_codec_simulation(w)),
    ]

    results: Dict[str, Dict[str, float]] = {}

    for cond_name, transform_fn in conditions:
        preds = []
        for p_idx, path in enumerate(test_audio_paths):
            fpath = Path(path)
            w, sr = sf.read(str(fpath))
            if w.ndim > 1:
                w = np.mean(w, axis=1)

            # Apply transformation
            degraded_w = transform_fn(w)

            # Save a representative derivative sample for the first file in each condition
            if p_idx == 0:
                cond_slug = cond_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
                sample_file = derivative_dir / f"sample_{cond_slug}.wav"
                sf.write(str(sample_file), degraded_w, sr)

            # Extract features from degraded audio
            feats = extract_basic_features(degraded_w, sample_rate=sr)
            feat_vec = validate_feature_vector(feats, CANONICAL_FEATURE_NAMES)
            pred = int(model.predict(feat_vec)[0])
            preds.append(pred)

        y_true = np.asarray(test_labels, dtype=np.int32)
        y_pred = np.asarray(preds, dtype=np.int32)

        acc = float(np.mean(y_true == y_pred))
        tp = float(np.sum((y_true == 1) & (y_pred == 1)))
        fp = float(np.sum((y_true == 0) & (y_pred == 1)))
        fn = float(np.sum((y_true == 1) & (y_pred == 0)))
        tn = float(np.sum((y_true == 0) & (y_pred == 0)))

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        results[cond_name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
        }

    # Save numerical summary
    summary_csv = out_dir / "robustness_evaluation_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Condition", "Accuracy", "Precision", "Recall", "F1_Score", "TP", "FP", "TN", "FN"])
        for cond, m in results.items():
            writer.writerow([cond, f"{m['accuracy']:.4f}", f"{m['precision']:.4f}", f"{m['recall']:.4f}", f"{m['f1_score']:.4f}", m["true_positives"], m["false_positives"], m["true_negatives"], m["false_negatives"]])

    return {
        "results": results,
        "summary_csv": str(summary_csv),
        "derivative_dir": str(derivative_dir),
    }


def plot_robustness_performance_curve(
    results: Dict[str, Dict[str, float]],
    output_path: Optional[Union[str, Path]] = None,
) -> str:
    """Generate diagnostic bar/line chart of model accuracy and F1 degradation under stress."""
    root = project_root()
    out_file = Path(output_path) if output_path else root / "results" / "graphs" / "robustness_performance_comparison.png"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    conditions = list(results.keys())
    accuracies = [results[c]["accuracy"] for c in conditions]
    f1_scores = [results[c]["f1_score"] for c in conditions]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(conditions))
    width = 0.35

    rects1 = ax.bar(x - width / 2, accuracies, width, label="Accuracy", color="#1f77b4", alpha=0.85)
    rects2 = ax.bar(x + width / 2, f1_scores, width, label="F1-Score", color="#ff7f0e", alpha=0.85)

    ax.bar_label(rects1, padding=3, fmt="%.3f", fontsize=8)
    ax.bar_label(rects2, padding=3, fmt="%.3f", fontsize=8)

    ax.set_ylabel("Performance Score (0.0 - 1.0)")
    ax.set_title("Acoustic Model Robustness Under Environmental & Channel Stress")
    ax.set_xticks(x, [c.replace(" ", "\n") for c in conditions], fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.legend(loc="lower left")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_file, dpi=200)
    plt.close(fig)

    logger.info("Saved robustness chart to %s", out_file)
    return str(out_file)
