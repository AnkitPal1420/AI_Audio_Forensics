"""Synthetic Generator & Vocoder Characterization Engine.

Forensic Rule & Scientific Constraint:
Definitive identification of the specific synthetic voice generator (e.g., ElevenLabs,
Tacotron2, VITS, HiFi-GAN) is strictly UNAVAILABLE because the dataset lacks multi-class
generator annotations and existing models are binary classifiers.

This module provides acoustic artifact profiling (effective bandwidth, high-frequency cutoff,
spectral rolloff, and energy concentration) to hypothesize candidate vocoder/architecture classes.

ALL outputs are classified as EXPERIMENTAL_HYPOTHESIS and require independent confirmation.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from database.models import AnalysisFinding

logger = logging.getLogger(__name__)


@dataclass
class GeneratorAnalysisResult:
    """Container for synthetic generator characterization."""

    run_id: str
    evidence_id: str
    effective_bandwidth_hz: float
    high_freq_cutoff_hz: float
    spectral_rolloff_85_hz: float
    high_freq_energy_ratio: float
    candidate_generator_class: str
    confidence_level: str  # LOW, EXPERIMENTAL
    findings: List[AnalysisFinding]
    prerequisites_missing: List[str]
    scientific_disclaimer: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "evidence_id": self.evidence_id,
            "effective_bandwidth_hz": self.effective_bandwidth_hz,
            "high_freq_cutoff_hz": self.high_freq_cutoff_hz,
            "spectral_rolloff_85_hz": self.spectral_rolloff_85_hz,
            "high_freq_energy_ratio": self.high_freq_energy_ratio,
            "candidate_generator_class": self.candidate_generator_class,
            "confidence_level": self.confidence_level,
            "findings": [f.to_dict() for f in self.findings],
            "prerequisites_missing": self.prerequisites_missing,
            "scientific_disclaimer": self.scientific_disclaimer,
        }


def analyze_spectral_bandwidth(
    audio: np.ndarray,
    sample_rate: int,
    cutoff_energy_db_threshold: float = -50.0,
) -> Tuple[float, float, float]:
    """Measure effective acoustic bandwidth, cutoff frequency, and 85% rolloff frequency."""
    n_fft = 2048
    window = np.hanning(min(len(audio), n_fft))
    step = n_fft // 2

    # Average power spectrum across audio
    power_acc = np.zeros(n_fft // 2 + 1, dtype=np.float64)
    count = 0
    for start in range(0, len(audio) - n_fft, step):
        frame = audio[start : start + n_fft] * window
        spec = np.abs(np.fft.rfft(frame)) ** 2
        power_acc += spec
        count += 1

    if count == 0:
        return 0.0, 0.0, 0.0

    avg_power = power_acc / count
    freqs = np.linspace(0, sample_rate / 2, len(avg_power))

    # Total energy and rolloff (85%)
    total_energy = np.sum(avg_power) + 1e-12
    cum_energy = np.cumsum(avg_power) / total_energy
    rolloff_idx = np.searchsorted(cum_energy, 0.85)
    rolloff_85 = float(freqs[min(rolloff_idx, len(freqs) - 1)])

    # Cutoff frequency (where energy falls below threshold relative to peak)
    peak_power = np.max(avg_power) + 1e-12
    power_db = 10 * np.log10(avg_power / peak_power + 1e-12)

    # Search backwards from Nyquist to find where power exceeds threshold
    cutoff_hz = float(sample_rate / 2)
    for i in range(len(power_db) - 1, 0, -1):
        if power_db[i] > cutoff_energy_db_threshold:
            cutoff_hz = float(freqs[i])
            break

    # Ratio of high-frequency energy (> 7 kHz)
    high_freq_idx = np.searchsorted(freqs, 7000.0)
    high_freq_energy_ratio = float(np.sum(avg_power[high_freq_idx:]) / total_energy)

    return cutoff_hz, rolloff_85, high_freq_energy_ratio


def profile_generator_artifacts(
    audio: np.ndarray,
    sample_rate: int,
    evidence_id: str,
    run_id: Optional[str] = None,
) -> GeneratorAnalysisResult:
    """Analyze acoustic bandwidth and spectral boundaries to hypothesize generator profile."""
    if run_id is None:
        run_id = f"RUN-GEN-{uuid.uuid4().hex[:8].upper()}"

    cutoff_hz, rolloff_hz, high_ratio = analyze_spectral_bandwidth(audio, sample_rate)

    # Acoustic heuristic classification
    if cutoff_hz <= 4200:
        candidate = "NARROWBAND_TELEPHONY_OR_LEGACY_VOCODER (e.g. G.711 / 8kHz Speech)"
        desc = "Effective audio bandwidth is truncated below 4.2 kHz, characteristic of narrow-band telephony or legacy voice synthesis."
    elif cutoff_hz <= 8500:
        candidate = "WIDEBAND_VOCODER (e.g. 16kHz sample rate standard models)"
        desc = "Sharp spectral cutoff near 8.0 kHz indicates native 16 kHz generation or standard wideband resampled vocoder (WaveNet/Tacotron base)."
    elif cutoff_hz <= 12000:
        candidate = "MEDIUM_FIDELITY_TTS (e.g. 22.05kHz / 24kHz older neural vocoders)"
        desc = "Spectral cutoff between 10-12 kHz matches legacy 22.05 kHz or 24 kHz TTS synthesis engines."
    else:
        candidate = "FULL_BAND_MODERN_NEURAL_VOCODER (e.g. HiFi-GAN / modern 44.1k-48k diffusion/autoregressive)"
        desc = "Acoustic energy extends above 12 kHz, characteristic of modern high-fidelity neural vocoders or studio-grade acoustic recordings."

    findings: List[AnalysisFinding] = []
    findings.append(
        AnalysisFinding(
            finding_id=f"FIND-GEN-{uuid.uuid4().hex[:8].upper()}",
            run_id=run_id,
            evidence_id=evidence_id,
            start_sec=0.0,
            end_sec=0.0,
            indicator="ACOUSTIC_BANDWIDTH_CUTOFF",
            value=round(cutoff_hz, 1),
            unit="Hz",
            threshold=8000.0,
            severity="MEDIUM",
            interpretation_flag="INTERPRETATION",
            description=desc,
        )
    )

    prerequisites = [
        "Multi-class synthetic voice dataset with verified labels from 10+ synthesis architectures.",
        "Vocoder fingerprint dictionary (HiFi-GAN, MelGAN, WaveGlow, BigVGAN, ElevenLabs API, Tortoise).",
        "Supervised multi-class acoustic feature classifier or neural embedding space.",
    ]

    disclaimer = (
        "CRITICAL FORENSIC LIMITATION: Specific synthetic generator identification (e.g. 'ElevenLabs', "
        "'Tortoise', 'Tacotron') is UNAVAILABLE in this release. The candidate classification reported above "
        "is strictly an exploratory acoustic heuristic based on spectral bandwidth and sample-rate artifacts. "
        "It MUST NOT be cited as forensic attribution of authoring software."
    )

    return GeneratorAnalysisResult(
        run_id=run_id,
        evidence_id=evidence_id,
        effective_bandwidth_hz=round(cutoff_hz, 1),
        high_freq_cutoff_hz=round(cutoff_hz, 1),
        spectral_rolloff_85_hz=round(rolloff_hz, 1),
        high_freq_energy_ratio=round(high_ratio, 4),
        candidate_generator_class=candidate,
        confidence_level="EXPERIMENTAL_HEURISTIC",
        findings=findings,
        prerequisites_missing=prerequisites,
        scientific_disclaimer=disclaimer,
    )

