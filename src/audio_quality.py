"""Audio quality assessment and out-of-distribution (OOD) detection module.

This module provides signal quality screening and domain validation before
forensic inference:
- Detects insufficient duration, severe clipping, excessive silence, and low SNR.
- Checks acoustic feature vectors against validated empirical training bounds.
- Prevents forced classifications on compromised or out-of-domain evidence.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from src.utils import project_root

logger = logging.getLogger(__name__)


def estimate_snr_db(waveform: np.ndarray, frame_length: int = 1024, hop_length: int = 512) -> float:
    """Estimate audio Signal-to-Noise Ratio (SNR) in decibels from signal power dynamics.

    Uses frame-wise energy percentiles (95th percentile for signal speech activity,
    5th percentile for stationary ambient noise floor).
    """
    if waveform.size < frame_length:
        return 0.0

    # Calculate frame-wise root-mean-square energy
    n_frames = (waveform.size - frame_length) // hop_length + 1
    if n_frames <= 0:
        return 0.0

    # Vectorized frame extraction
    shape = (n_frames, frame_length)
    strides = (waveform.strides[0] * hop_length, waveform.strides[0])
    frames = np.lib.stride_tricks.as_strided(waveform, shape=shape, strides=strides)
    frame_rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)

    p95_signal = float(np.percentile(frame_rms, 95))
    p05_noise = float(np.percentile(frame_rms, 5))

    # If the signal has steady, high energy throughout (e.g. continuous tone/speech without pauses),
    # the 5th percentile is not ambient silence, but active signal.
    if np.mean(frame_rms) > 0.02 and (p95_signal / max(p05_noise, 1e-6)) < 2.0:
        return 25.0

    if p05_noise < 1e-6:
        p05_noise = 1e-6

    snr_ratio = (p95_signal / p05_noise) ** 2
    snr_db = 10.0 * np.log10(max(snr_ratio, 1.0))
    return float(snr_db)


def assess_audio_quality(
    waveform: np.ndarray,
    sample_rate: int,
    min_duration_sec: float = 0.5,
    max_clipping_ratio: float = 0.05,
    max_silence_ratio: float = 0.80,
    min_snr_db: float = 5.0,
) -> Dict[str, Any]:
    """Perform pre-classification acoustic quality checks on normalized audio signal.

    Parameters
    ----------
    waveform : np.ndarray
        Floating-point audio signal array.
    sample_rate : int
        Audio sampling rate in Hertz.
    min_duration_sec : float, default=0.5
        Minimum duration required for statistical feature reliability.
    max_clipping_ratio : float, default=0.05
        Maximum tolerable ratio of clipped samples (|y| >= 0.999).
    max_silence_ratio : float, default=0.80
        Maximum tolerable fraction of silent audio frames.
    min_snr_db : float, default=5.0
        Minimum required SNR in decibels for reliable feature extraction.

    Returns
    -------
    Dict[str, Any]
        Dictionary with pass/fail status, quality metrics, and warning flags.
    """
    duration = float(waveform.size / sample_rate) if sample_rate > 0 else 0.0
    flags: List[str] = []

    # 1. Duration check
    if duration < min_duration_sec:
        flags.append(f"INSUFFICIENT_DURATION: Duration {duration:.2f}s is below minimum {min_duration_sec}s.")

    # 2. Clipping check
    clipped_count = int(np.sum(np.abs(waveform) >= 0.999))
    clipping_ratio = float(clipped_count / waveform.size) if waveform.size > 0 else 0.0
    if clipping_ratio > max_clipping_ratio:
        flags.append(
            f"EXCESSIVE_CLIPPING: {clipping_ratio:.2%} samples are clipped (threshold: {max_clipping_ratio:.2%})."
        )

    # 3. Silence / Energy check
    frame_len = min(1024, max(128, waveform.size // 10)) if waveform.size > 0 else 1024
    if waveform.size >= frame_len:
        hop = frame_len // 2
        n_frames = (waveform.size - frame_len) // hop + 1
        shape = (n_frames, frame_len)
        strides = (waveform.strides[0] * hop, waveform.strides[0])
        frames = np.lib.stride_tricks.as_strided(waveform, shape=shape, strides=strides)
        f_rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
        silent_frames = int(np.sum(f_rms < 0.005))
        silence_ratio = float(silent_frames / n_frames)
    else:
        silence_ratio = 1.0 if np.max(np.abs(waveform)) < 0.005 else 0.0

    if silence_ratio > max_silence_ratio:
        flags.append(
            f"EXCESSIVE_SILENCE: {silence_ratio:.2%} of frames are silent (threshold: {max_silence_ratio:.2%})."
        )

    # 4. Signal-to-noise ratio
    snr_est = estimate_snr_db(waveform, frame_length=frame_len, hop_length=frame_len // 2)
    if snr_est < min_snr_db and duration >= min_duration_sec:
        flags.append(f"LOW_SNR: Estimated SNR {snr_est:.1f} dB is below minimum {min_snr_db:.1f} dB.")

    is_valid = len(flags) == 0

    return {
        "status": "PASS" if is_valid else "FAIL",
        "duration_seconds": duration,
        "sample_rate": sample_rate,
        "clipping_ratio": clipping_ratio,
        "silence_ratio": silence_ratio,
        "estimated_snr_db": snr_est,
        "quality_flags": flags,
        "is_reliable_for_inference": is_valid,
    }


def load_training_distribution_stats(stats_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load cached empirical training distribution statistics."""
    root = project_root()
    path = Path(stats_path) if stats_path else root / "models" / "training_distribution_stats.json"

    if not path.is_file():
        logger.warning("Training distribution statistics file not found at %s. OOD detection will be skipped.", path)
        return {}

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def assess_feature_distribution(
    features: Dict[str, float],
    stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assess whether extracted acoustic features lie within validated training distribution.

    Parameters
    ----------
    features : Dict[str, float]
        Dictionary of extracted forensic features.
    stats : Dict[str, Any], optional
        Pre-loaded distribution statistics. If None, loads from `models/training_distribution_stats.json`.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing distribution classification, max z-score, and outlier features.
    """
    if stats is None:
        stats = load_training_distribution_stats()

    if not stats or "means" not in stats or "stds" not in stats:
        return {
            "status": "UNCHECKED",
            "is_in_distribution": True,
            "max_z_score": 0.0,
            "mean_z_score": 0.0,
            "outlier_features": [],
            "reason": "Training distribution statistics unavailable.",
        }

    means = stats["means"]
    stds = stats["stds"]
    thresholds = stats.get("ood_thresholds", {})
    max_z_thresh = thresholds.get("max_z_score_threshold", 6.5)
    mean_z_thresh = thresholds.get("mean_z_score_threshold", 2.2)

    z_scores: Dict[str, float] = {}
    outliers: List[Dict[str, Any]] = []

    for feat_name, mean_val in means.items():
        if feat_name in features:
            val = float(features[feat_name])
            std_val = float(stds.get(feat_name, 1.0))
            if std_val < 1e-6:
                std_val = 1.0
            z = abs((val - mean_val) / std_val)
            z_scores[feat_name] = z
            if z > max_z_thresh:
                outliers.append(
                    {
                        "feature": feat_name,
                        "value": val,
                        "training_mean": mean_val,
                        "z_score": round(z, 2),
                    }
                )

    if not z_scores:
        return {
            "status": "UNCHECKED",
            "is_in_distribution": True,
            "max_z_score": 0.0,
            "mean_z_score": 0.0,
            "outlier_features": [],
        }

    max_z = float(np.max(list(z_scores.values())))
    mean_z = float(np.mean(list(z_scores.values())))

    is_ood = (max_z > max_z_thresh) or (mean_z > mean_z_thresh)

    return {
        "status": "OUT_OF_DISTRIBUTION" if is_ood else "IN_DISTRIBUTION",
        "is_in_distribution": not is_ood,
        "max_z_score": round(max_z, 3),
        "mean_z_score": round(mean_z, 3),
        "max_z_threshold": max_z_thresh,
        "mean_z_threshold": mean_z_thresh,
        "outlier_count": len(outliers),
        "outlier_features": outliers,
    }
