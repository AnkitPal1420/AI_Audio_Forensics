"""Signal-Level Audio Manipulation & Tampering Analysis Module.

Provides forensic signal processing algorithms for detecting potential edit points,
abrupt spectral flux jumps, RMS energy step changes, background noise floor shifts,
repeated segments (clone-and-paste splicing), and CUSUM change points.

Forensic Rule:
All detected events are flagged as objective MEASUREMENTS or INVESTIGATIVE INDICATORS,
never as definitive proof of malicious tampering.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf

from database.models import AnalysisFinding, AnalysisRun
from src.utils import project_root

logger = logging.getLogger(__name__)


@dataclass
class TamperingAnalysisResult:
    """Complete result container for signal tampering analysis."""

    run_id: str
    evidence_id: str
    input_sha256: str
    duration_seconds: float
    sample_rate: int
    findings: List[AnalysisFinding]
    time_points: List[float]
    spectral_flux: List[float]
    rms_energy: List[float]
    noise_floor: List[float]
    flux_threshold: float
    rms_step_threshold: float
    noise_shift_threshold: float
    total_anomalies: int
    status: str
    disclaimer: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "evidence_id": self.evidence_id,
            "input_sha256": self.input_sha256,
            "duration_seconds": self.duration_seconds,
            "sample_rate": self.sample_rate,
            "findings": [f.to_dict() for f in self.findings],
            "time_points": self.time_points,
            "spectral_flux": self.spectral_flux,
            "rms_energy": self.rms_energy,
            "noise_floor": self.noise_floor,
            "flux_threshold": self.flux_threshold,
            "rms_step_threshold": self.rms_step_threshold,
            "noise_shift_threshold": self.noise_shift_threshold,
            "total_anomalies": self.total_anomalies,
            "status": self.status,
            "disclaimer": self.disclaimer,
        }


def compute_frame_stft(
    audio: np.ndarray,
    sample_rate: int,
    frame_length_sec: float = 0.025,
    hop_length_sec: float = 0.010,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute magnitude STFT spectrogram and time points using pure NumPy.
    
    Args:
        audio: 1D float32 audio array.
        sample_rate: sampling frequency.
        frame_length_sec: window duration (default 25 ms).
        hop_length_sec: hop duration (default 10 ms).
        
    Returns:
        (magnitude_spec: 2D array [freq_bins, frames], times: 1D array)
    """
    n_fft = int(2 ** np.ceil(np.log2(frame_length_sec * sample_rate)))
    hop_length = int(hop_length_sec * sample_rate)
    window = np.hanning(n_fft)

    # Pad audio
    pad_len = n_fft // 2
    padded = np.pad(audio, (pad_len, pad_len), mode="reflect")
    num_frames = max(1, (len(padded) - n_fft) // hop_length + 1)

    frames = np.zeros((num_frames, n_fft), dtype=np.float32)
    for i in range(num_frames):
        start = i * hop_length
        frames[i] = padded[start : start + n_fft] * window

    # FFT
    complex_spec = np.fft.rfft(frames, axis=1)  # shape: (num_frames, n_fft // 2 + 1)
    mag_spec = np.abs(complex_spec).T  # shape: (freq_bins, num_frames)

    times = np.arange(num_frames) * hop_length_sec
    return mag_spec, times


def compute_spectral_flux(mag_spec: np.ndarray) -> np.ndarray:
    """Compute half-wave rectified spectral flux between consecutive frames.
    
    Spectral flux measures rapid changes in frequency distribution.
    """
    if mag_spec.shape[1] < 2:
        return np.zeros(mag_spec.shape[1], dtype=np.float32)

    diff = mag_spec[:, 1:] - mag_spec[:, :-1]
    # Half-wave rectification: only positive energy increases
    diff_rect = np.maximum(0, diff)
    flux_values = np.sqrt(np.mean(diff_rect ** 2, axis=0))
    # Prepend 0 to match number of frames
    flux = np.concatenate(([0.0], flux_values)).astype(np.float32)
    return flux


def compute_frame_rms(
    audio: np.ndarray,
    sample_rate: int,
    frame_length_sec: float = 0.025,
    hop_length_sec: float = 0.010,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute frame-level Root-Mean-Square (RMS) energy."""
    frame_len = int(frame_length_sec * sample_rate)
    hop_len = int(hop_length_sec * sample_rate)

    pad_len = frame_len // 2
    padded = np.pad(audio, (pad_len, pad_len), mode="reflect")
    num_frames = max(1, (len(padded) - frame_len) // hop_len + 1)

    rms = np.zeros(num_frames, dtype=np.float32)
    for i in range(num_frames):
        start = i * hop_len
        chunk = padded[start : start + frame_len]
        rms[i] = np.sqrt(np.mean(chunk ** 2) + 1e-12)

    times = np.arange(num_frames) * hop_length_sec
    return rms, times


def compute_noise_floor_profile(
    rms_energy: np.ndarray,
    window_frames: int = 50,
) -> np.ndarray:
    """Estimate background noise floor over a sliding window (10th percentile)."""
    n = len(rms_energy)
    noise_floor = np.zeros(n, dtype=np.float32)
    half_win = max(1, window_frames // 2)

    for i in range(n):
        start = max(0, i - half_win)
        end = min(n, i + half_win + 1)
        noise_floor[i] = np.percentile(rms_energy[start:end], 10)

    return noise_floor


def detect_cusum_changepoints(
    signal: np.ndarray,
    times: np.ndarray,
    threshold_factor: float = 4.0,
    min_distance_sec: float = 0.3,
) -> List[Tuple[float, float, str]]:
    """Detect structural change-points using two-sided Cumulative Sum (CUSUM).
    
    Returns:
        List of (timestamp, score, direction)
    """
    if len(signal) < 10:
        return []

    mean = np.mean(signal)
    std = np.std(signal) + 1e-9
    norm_sig = (signal - mean) / std

    s_pos = 0.0
    s_neg = 0.0
    drift = 0.5
    changepoints = []
    last_time = -999.0

    for i, (val, t) in enumerate(zip(norm_sig, times)):
        s_pos = max(0.0, s_pos + val - drift)
        s_neg = min(0.0, s_neg + val + drift)

        if s_pos > threshold_factor:
            if t - last_time >= min_distance_sec:
                changepoints.append((float(t), float(s_pos), "POSITIVE_STEP"))
                last_time = t
            s_pos = 0.0

        elif abs(s_neg) > threshold_factor:
            if t - last_time >= min_distance_sec:
                changepoints.append((float(t), float(abs(s_neg)), "NEGATIVE_STEP"))
                last_time = t
            s_neg = 0.0

    return changepoints


def detect_repeated_segments(
    audio: np.ndarray,
    sample_rate: int,
    min_segment_sec: float = 0.4,
    similarity_threshold: float = 0.96,
    max_search_duration_sec: float = 30.0,
) -> List[Tuple[float, float, float, float]]:
    """Detect potential clone-and-paste audio repetitions using normalized cross-correlation.
    
    To avoid excessive compute on long files, searches up to `max_search_duration_sec`.
    Returns list of (start_a, end_a, start_b, similarity).
    """
    max_samples = int(max_search_duration_sec * sample_rate)
    target_audio = audio[:max_samples]

    # Decimate to ~1000 Hz for efficient envelope cross-correlation
    downsample_factor = max(1, sample_rate // 1000)
    sig_low = target_audio[::downsample_factor]
    sr_low = sample_rate / downsample_factor

    seg_len = int(min_segment_sec * sr_low)
    step = int(0.2 * sr_low)  # 200 ms step
    matches = []

    num_segs = max(0, (len(sig_low) - seg_len) // step)
    # Check pairwise correlations for non-overlapping windows
    for i in range(0, min(num_segs, 40)):  # capped to prevent combinatorial stall
        start_i = i * step
        seg_i = sig_low[start_i : start_i + seg_len]
        norm_i = np.linalg.norm(seg_i)
        if norm_i < 1e-4:
            continue

        for j in range(i + int(min_segment_sec * sr_low // step) + 1, min(num_segs, 40)):
            start_j = j * step
            seg_j = sig_low[start_j : start_j + seg_len]
            norm_j = np.linalg.norm(seg_j)
            if norm_j < 1e-4:
                continue

            corr = np.dot(seg_i, seg_j) / (norm_i * norm_j)
            if corr >= similarity_threshold:
                t_i = start_i / sr_low
                t_j = start_j / sr_low
                matches.append((float(t_i), float(t_i + min_segment_sec), float(t_j), float(corr)))

    return matches


def analyze_audio_tampering(
    audio: np.ndarray,
    sample_rate: int,
    evidence_id: str,
    run_id: Optional[str] = None,
    input_sha256: str = "",
    flux_sigma_threshold: float = 3.5,
    rms_step_sigma_threshold: float = 3.5,
    noise_floor_shift_db: float = 6.0,
) -> TamperingAnalysisResult:
    """Run full acoustic tampering and edit-point analysis.
    
    Generates structured findings with MEASUREMENT interpretation flags.
    """
    if run_id is None:
        run_id = f"RUN-TAMP-{uuid.uuid4().hex[:8].upper()}"

    duration_sec = len(audio) / sample_rate
    findings: List[AnalysisFinding] = []

    # 1. Compute STFT and spectral flux
    mag_spec, flux_times = compute_frame_stft(audio, sample_rate)
    flux = compute_spectral_flux(mag_spec)

    # 2. Compute RMS Energy
    rms, rms_times = compute_frame_rms(audio, sample_rate)

    # Ensure length parity
    min_len = min(len(flux), len(rms), len(flux_times))
    times = flux_times[:min_len]
    flux = flux[:min_len]
    rms = rms[:min_len]

    # 3. Compute Noise Floor Profile
    noise_floor = compute_noise_floor_profile(rms, window_frames=50)

    # --- INDICATOR 1: Spectral Flux Discontinuities ---
    flux_mean = np.mean(flux)
    flux_std = np.std(flux) + 1e-9
    flux_thresh = float(flux_mean + flux_sigma_threshold * flux_std)

    flux_anomalies = np.where(flux > flux_thresh)[0]
    # Cluster consecutive frames
    if len(flux_anomalies) > 0:
        clusters = []
        cur_cluster = [flux_anomalies[0]]
        for idx in flux_anomalies[1:]:
            if idx == cur_cluster[-1] + 1:
                cur_cluster.append(idx)
            else:
                clusters.append(cur_cluster)
                cur_cluster = [idx]
        clusters.append(cur_cluster)

        for cl in clusters:
            start_t = float(times[cl[0]])
            end_t = float(times[cl[-1]] + 0.010)
            max_val = float(np.max(flux[cl]))
            severity = "HIGH" if max_val > flux_thresh * 1.5 else "MEDIUM"
            finding = AnalysisFinding(
                finding_id=f"FIND-FLUX-{uuid.uuid4().hex[:8].upper()}",
                run_id=run_id,
                evidence_id=evidence_id,
                start_sec=round(start_t, 3),
                end_sec=round(end_t, 3),
                indicator="SPECTRAL_FLUX_DISCONTINUITY",
                value=round(max_val, 4),
                unit="amplitude_flux",
                threshold=round(flux_thresh, 4),
                severity=severity,
                interpretation_flag="MEASUREMENT",
                description=(
                    f"Sharp spectral flux surge of {max_val:.4f} exceeding threshold ({flux_thresh:.4f}). "
                    "Indicates sudden transient, abrupt acoustic shift, or potential edit splice."
                ),
            )
            findings.append(finding)

    # --- INDICATOR 2: RMS Energy Step Changes ---
    rms_diff = np.abs(np.diff(rms))
    rms_diff_mean = np.mean(rms_diff)
    rms_diff_std = np.std(rms_diff) + 1e-9
    rms_step_thresh = float(rms_diff_mean + rms_step_sigma_threshold * rms_diff_std)

    rms_anomalies = np.where(rms_diff > rms_step_thresh)[0]
    for idx in rms_anomalies:
        t_sec = float(times[idx])
        val = float(rms_diff[idx])
        severity = "HIGH" if val > rms_step_thresh * 1.5 else "MEDIUM"
        finding = AnalysisFinding(
            finding_id=f"FIND-RMS-{uuid.uuid4().hex[:8].upper()}",
            run_id=run_id,
            evidence_id=evidence_id,
            start_sec=round(t_sec, 3),
            end_sec=round(t_sec + 0.015, 3),
            indicator="RMS_ENERGY_STEP",
            value=round(val, 4),
            unit="linear_energy_delta",
            threshold=round(rms_step_thresh, 4),
            severity=severity,
            interpretation_flag="MEASUREMENT",
            description=(
                f"Abrupt RMS energy step change of {val:.4f} across 10ms frame boundary. "
                "Exceeds expected smooth speech transition dynamics."
            ),
        )
        findings.append(finding)

    # --- INDICATOR 3: Noise Floor Shifting ---
    # Convert noise floor to dB
    noise_db = 20 * np.log10(np.maximum(noise_floor, 1e-6))
    noise_db_diff = np.abs(np.diff(noise_db))
    noise_shift_anomalies = np.where(noise_db_diff > noise_floor_shift_db)[0]
    for idx in noise_shift_anomalies:
        t_sec = float(times[idx])
        val = float(noise_db_diff[idx])
        finding = AnalysisFinding(
            finding_id=f"FIND-NOISE-{uuid.uuid4().hex[:8].upper()}",
            run_id=run_id,
            evidence_id=evidence_id,
            start_sec=round(t_sec, 3),
            end_sec=round(t_sec + 0.1, 3),
            indicator="NOISE_FLOOR_SHIFT",
            value=round(val, 2),
            unit="dB",
            threshold=round(noise_floor_shift_db, 2),
            severity="MEDIUM",
            interpretation_flag="MEASUREMENT",
            description=(
                f"Background noise floor level jumped by {val:.2f} dB within a short interval. "
                "Consistent with audio recorded in different environments or spliced background tracks."
            ),
        )
        findings.append(finding)

    # --- INDICATOR 4: Repeated Segments (Splice / Clone Detection) ---
    repeated = detect_repeated_segments(audio, sample_rate)
    for t_a1, t_a2, t_b1, corr in repeated:
        finding = AnalysisFinding(
            finding_id=f"FIND-REP-{uuid.uuid4().hex[:8].upper()}",
            run_id=run_id,
            evidence_id=evidence_id,
            start_sec=round(t_a1, 3),
            end_sec=round(t_a2, 3),
            indicator="REPEATED_SEGMENT",
            value=round(corr, 3),
            unit="correlation_coefficient",
            threshold=0.96,
            severity="HIGH",
            interpretation_flag="MEASUREMENT",
            description=(
                f"Acoustic segment at {t_a1:.2f}s-{t_a2:.2f}s is nearly identical to segment at "
                f"{t_b1:.2f}s (normalized cross-correlation = {corr:.3f}). "
                "Suggests audio replication or copy-paste manipulation."
            ),
        )
        findings.append(finding)

    # Decimate time curves for lightweight dashboard display & JSON serialization
    decimate_step = max(1, len(times) // 500)
    sampled_times = [round(float(t), 4) for t in times[::decimate_step]]
    sampled_flux = [round(float(f), 4) for f in flux[::decimate_step]]
    sampled_rms = [round(float(r), 4) for r in rms[::decimate_step]]
    sampled_noise = [round(float(n), 4) for n in noise_floor[::decimate_step]]

    status = "ANOMALIES_DETECTED" if len(findings) > 0 else "NO_ANOMALIES_DETECTED"
    disclaimer = (
        "FORENSIC NOTICE: Detected signal-level discontinuities, step changes, and repeated segments "
        "represent empirical physical measurements of acoustic dynamics. They serve as investigative indicators "
        "and do NOT constitute conclusive proof of malicious manipulation or deepfake synthesis. "
        "Natural vocal transients (clicks, plosives, breath intakes) or recording artifacts may produce similar indicators."
    )

    return TamperingAnalysisResult(
        run_id=run_id,
        evidence_id=evidence_id,
        input_sha256=input_sha256,
        duration_seconds=round(duration_sec, 3),
        sample_rate=sample_rate,
        findings=findings,
        time_points=sampled_times,
        spectral_flux=sampled_flux,
        rms_energy=sampled_rms,
        noise_floor=sampled_noise,
        flux_threshold=round(flux_thresh, 4),
        rms_step_threshold=round(rms_step_thresh, 4),
        noise_shift_threshold=round(noise_floor_shift_db, 2),
        total_anomalies=len(findings),
        status=status,
        disclaimer=disclaimer,
    )
