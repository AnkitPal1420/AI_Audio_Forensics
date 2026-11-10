"""Acoustic Anomaly Detection Engine.

Extracts multidimensional frame-level acoustic descriptors (spectral flatness,
spectral flux, autocorrelation pitch proxy, spectral centroid, spectral rolloff,
and energy) and applies unsupervised outlier detection (Isolation Forest / Local Outlier Factor)
to flag statistical anomalies in vocal recordings.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from database.models import AnalysisFinding
from src.tampering_analysis import compute_frame_stft

logger = logging.getLogger(__name__)


@dataclass
class AnomalyDetectionResult:
    """Container for acoustic anomaly detection findings and frame scores."""

    run_id: str
    evidence_id: str
    input_sha256: str
    duration_seconds: float
    sample_rate: int
    algorithm: str  # IsolationForest or LOF
    outlier_fraction: float
    total_anomaly_frames: int
    findings: List[AnalysisFinding]
    time_points: List[float]
    anomaly_scores: List[float]
    spectral_flatness: List[float]
    spectral_centroid: List[float]
    pitch_stability: List[float]
    score_threshold: float
    disclaimer: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "evidence_id": self.evidence_id,
            "input_sha256": self.input_sha256,
            "duration_seconds": self.duration_seconds,
            "sample_rate": self.sample_rate,
            "algorithm": self.algorithm,
            "outlier_fraction": self.outlier_fraction,
            "total_anomaly_frames": self.total_anomaly_frames,
            "findings": [f.to_dict() for f in self.findings],
            "time_points": self.time_points,
            "anomaly_scores": self.anomaly_scores,
            "spectral_flatness": self.spectral_flatness,
            "spectral_centroid": self.spectral_centroid,
            "pitch_stability": self.pitch_stability,
            "score_threshold": self.score_threshold,
            "disclaimer": self.disclaimer,
        }


def compute_spectral_flatness(mag_spec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Compute spectral flatness (Wiener entropy) per frame: geometric_mean / arithmetic_mean."""
    power_spec = mag_spec ** 2 + eps
    # Geometric mean: exp(mean(log(p)))
    log_power = np.log(power_spec)
    geom_mean = np.exp(np.mean(log_power, axis=0))
    arith_mean = np.mean(power_spec, axis=0)
    flatness = geom_mean / (arith_mean + eps)
    return np.clip(flatness, 0.0, 1.0).astype(np.float32)


def compute_spectral_centroid(mag_spec: np.ndarray, sample_rate: int) -> np.ndarray:
    """Compute spectral centroid (center of mass of spectrum) in Hz per frame."""
    n_bins = mag_spec.shape[0]
    freqs = np.linspace(0, sample_rate / 2, n_bins)
    sum_mag = np.sum(mag_spec, axis=0) + 1e-9
    centroid = np.dot(freqs, mag_spec) / sum_mag
    return centroid.astype(np.float32)


def compute_pitch_autocorrelation_proxy(
    audio: np.ndarray,
    sample_rate: int,
    frame_length_sec: float = 0.030,
    hop_length_sec: float = 0.010,
    min_f0: float = 60.0,
    max_f0: float = 500.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate pitch and pitch stability proxy via normalized short-time autocorrelation."""
    frame_len = int(frame_length_sec * sample_rate)
    hop_len = int(hop_length_sec * sample_rate)
    num_frames = max(1, (len(audio) - frame_len) // hop_len + 1)

    min_lag = int(sample_rate / max_f0)
    max_lag = int(sample_rate / min_f0)

    f0_est = np.zeros(num_frames, dtype=np.float32)
    autocorr_peak = np.zeros(num_frames, dtype=np.float32)

    for i in range(num_frames):
        start = i * hop_len
        chunk = audio[start : start + frame_len]
        if np.std(chunk) < 1e-4:
            continue

        chunk_centered = chunk - np.mean(chunk)
        # Fast autocorrelation via FFT
        r = np.correlate(chunk_centered, chunk_centered, mode="full")
        r = r[len(r) // 2 :]  # Non-negative lags
        r_norm = r[0] + 1e-9

        if max_lag < len(r):
            search_window = r[min_lag : max_lag + 1] / r_norm
            best_idx = np.argmax(search_window)
            peak_val = search_window[best_idx]
            lag = min_lag + best_idx
            if peak_val > 0.3:
                f0_est[i] = sample_rate / lag
                autocorr_peak[i] = peak_val

    return f0_est, autocorr_peak


def detect_acoustic_anomalies(
    audio: np.ndarray,
    sample_rate: int,
    evidence_id: str,
    run_id: Optional[str] = None,
    input_sha256: str = "",
    contamination: float = 0.05,
    method: str = "IsolationForest",
) -> AnomalyDetectionResult:
    """Run multidimensional unsupervised acoustic anomaly detection.
    
    Args:
        audio: 1D float32 audio.
        sample_rate: sampling frequency.
        evidence_id: evidence ID.
        run_id: optional run ID.
        input_sha256: file hash.
        contamination: expected fraction of outliers (0.01 - 0.15).
        method: 'IsolationForest' or 'LOF'.
    """
    if run_id is None:
        run_id = f"RUN-ANOM-{uuid.uuid4().hex[:8].upper()}"

    duration_sec = len(audio) / sample_rate
    mag_spec, times = compute_frame_stft(audio, sample_rate, frame_length_sec=0.030, hop_length_sec=0.015)

    # 1. Compute acoustic features per frame
    flatness = compute_spectral_flatness(mag_spec)
    centroid = compute_spectral_centroid(mag_spec, sample_rate)
    rms_power = np.sqrt(np.mean(mag_spec ** 2, axis=0) + 1e-12)
    f0_est, pitch_stability = compute_pitch_autocorrelation_proxy(
        audio, sample_rate, frame_length_sec=0.030, hop_length_sec=0.015
    )

    min_len = min(len(flatness), len(centroid), len(rms_power), len(f0_est), len(times))
    times = times[:min_len]
    flatness = flatness[:min_len]
    centroid = centroid[:min_len]
    rms_power = rms_power[:min_len]
    f0_est = f0_est[:min_len]
    pitch_stability = pitch_stability[:min_len]

    # Feature matrix: shape (num_frames, 5)
    X = np.column_stack([
        flatness,
        centroid / (sample_rate / 2.0),
        np.log1p(rms_power),
        f0_est / 500.0,
        pitch_stability,
    ])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if method == "LOF":
        model = LocalOutlierFactor(n_neighbors=20, contamination=contamination, novelty=False)
        preds = model.fit_predict(X_scaled)  # -1 for outliers, 1 for inliers
        raw_scores = -model.negative_outlier_factor_  # higher means more anomalous
        score_thresh = float(np.percentile(raw_scores, 100 * (1 - contamination)))
    else:
        model = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=42,
            n_jobs=1,
        )
        preds = model.fit_predict(X_scaled)
        # IsolationForest decision_function: lower is more anomalous
        # Invert so higher score = higher anomaly
        scores = -model.decision_function(X_scaled)
        raw_scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-9)
        score_thresh = float(np.percentile(raw_scores, 100 * (1 - contamination)))

    anom_indices = np.where(preds == -1)[0]
    total_anomaly_frames = len(anom_indices)
    outlier_fraction = total_anomaly_frames / max(1, min_len)

    # Cluster consecutive anomaly frames into distinct findings
    findings: List[AnalysisFinding] = []
    if len(anom_indices) > 0:
        clusters = []
        cur = [anom_indices[0]]
        for idx in anom_indices[1:]:
            if idx == cur[-1] + 1:
                cur.append(idx)
            else:
                clusters.append(cur)
                cur = [idx]
        clusters.append(cur)

        for cl in clusters:
            start_t = float(times[cl[0]])
            end_t = float(times[cl[-1]] + 0.015)
            peak_score = float(np.max(raw_scores[cl]))
            severity = "HIGH" if peak_score > score_thresh * 1.3 else ("MEDIUM" if peak_score > score_thresh else "LOW")

            finding = AnalysisFinding(
                finding_id=f"FIND-ANOM-{uuid.uuid4().hex[:8].upper()}",
                run_id=run_id,
                evidence_id=evidence_id,
                start_sec=round(start_t, 3),
                end_sec=round(end_t, 3),
                indicator="ACOUSTIC_OUTLIER",
                value=round(peak_score, 4),
                unit="anomaly_score",
                threshold=round(score_thresh, 4),
                severity=severity,
                interpretation_flag="MEASUREMENT",
                description=(
                    f"Acoustic frame cluster at {start_t:.2f}s-{end_t:.2f}s is a statistical outlier "
                    f"under {method} (peak score {peak_score:.3f} vs threshold {score_thresh:.3f}). "
                    "Features diverge markedly from the recording's dominant vocal distribution."
                ),
            )
            findings.append(finding)

    # Decimate curves for plotting
    decimate = max(1, min_len // 400)
    sampled_times = [round(float(t), 3) for t in times[::decimate]]
    sampled_scores = [round(float(s), 4) for s in raw_scores[::decimate]]
    sampled_flatness = [round(float(f), 4) for f in flatness[::decimate]]
    sampled_centroid = [round(float(c), 1) for c in centroid[::decimate]]
    sampled_stability = [round(float(p), 4) for p in pitch_stability[::decimate]]

    disclaimer = (
        "FORENSIC NOTICE: Acoustic anomaly detection employs unsupervised statistical modeling to flag "
        "spectral outliers. An outlier may reflect unusual phonetics, sudden emotional shifts, acoustic transients, "
        "or background environment changes. Outliers are investigative leads, NOT definitive evidence of synthetic speech."
    )

    return AnomalyDetectionResult(
        run_id=run_id,
        evidence_id=evidence_id,
        input_sha256=input_sha256,
        duration_seconds=round(duration_sec, 3),
        sample_rate=sample_rate,
        algorithm=method,
        outlier_fraction=round(float(outlier_fraction), 4),
        total_anomaly_frames=total_anomaly_frames,
        findings=findings,
        time_points=sampled_times,
        anomaly_scores=sampled_scores,
        spectral_flatness=sampled_flatness,
        spectral_centroid=sampled_centroid,
        pitch_stability=sampled_stability,
        score_threshold=round(score_thresh, 4),
        disclaimer=disclaimer,
    )
