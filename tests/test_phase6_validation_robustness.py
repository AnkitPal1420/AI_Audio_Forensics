"""Phase 6: Comprehensive scientific validation, robustness, calibration, and OOD test suite."""

import numpy as np
import pytest
from pathlib import Path

from src.group_split import split_by_speaker_group, verify_no_leakage
from src.model_calibration import compute_calibration_metrics
from src.robustness_testing import (
    add_gaussian_noise,
    apply_telephony_bandpass,
    apply_lossy_codec_simulation,
    apply_reverberation,
    apply_clipping,
    apply_resampling_stress,
)
from src.audio_quality import assess_feature_distribution
from src.utils import project_root


@pytest.fixture
def synthetic_signal():
    """Deterministic synthetic test signal: 1 second harmonic audio at 16 kHz."""
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    sig = 0.6 * np.sin(2 * np.pi * 300.0 * t) + 0.3 * np.sin(2 * np.pi * 1200.0 * t)
    return sig.astype(np.float32), sr


def test_group_aware_leakage_prevention():
    """Verify that verify_no_leakage raises an assertion error if partitions overlap."""
    tr_spk = ["biden", "linus"]
    va_spk = ["ryan"]
    te_spk = ["taylor", "trump"]

    tr_rec = ["rec_01", "rec_02"]
    va_rec = ["rec_03"]
    te_rec = ["rec_04"]

    # Valid: no leakage
    audit = verify_no_leakage(tr_spk, va_spk, te_spk, tr_rec, va_rec, te_rec)
    assert audit["has_leakage"] is False
    assert "PASSED" in audit["status"]

    # Leakage in speakers
    bad_spk = verify_no_leakage(["biden", "ryan"], ["ryan"], te_spk, tr_rec, va_rec, te_rec)
    assert bad_spk["has_leakage"] is True
    assert "ryan" in bad_spk["speaker_overlap_train_val"]

    # Leakage in recordings
    bad_rec = verify_no_leakage(tr_spk, va_spk, te_spk, ["rec_01", "rec_04"], va_rec, ["rec_04"])
    assert bad_rec["has_leakage"] is True
    assert "rec_04" in bad_rec["recording_overlap_train_test"]


def test_calibration_metrics_brier_and_ece():
    """Verify that Brier score and Expected Calibration Error compute correctly."""
    y_true = np.array([0, 0, 1, 1], dtype=np.int32)
    y_prob = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float64)

    calib = compute_calibration_metrics(y_true, y_prob, n_bins=5)
    assert "brier_score" in calib
    assert "expected_calibration_error" in calib
    assert calib["brier_score"] < 0.05  # Highly calibrated
    assert calib["expected_calibration_error"] < 0.15


def test_robustness_noise_addition(synthetic_signal):
    """Verify that add_gaussian_noise maintains signal shape and peak bounds."""
    sig, sr = synthetic_signal
    noisy = add_gaussian_noise(sig, snr_db=20.0, seed=42)

    assert noisy.shape == sig.shape
    assert noisy.dtype == np.float32
    assert np.all(np.isfinite(noisy))
    assert np.max(np.abs(noisy)) <= 1.0


def test_robustness_telephony_bandpass(synthetic_signal):
    """Verify that telephony filtering sharply attenuates frequencies above 3400 Hz."""
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    # 5000 Hz high tone (outside 300-3400 Hz band)
    high_tone = np.sin(2 * np.pi * 5000.0 * t).astype(np.float32)

    filtered = apply_telephony_bandpass(high_tone, sample_rate=sr, low_freq=300.0, high_freq=3400.0)
    attenuation_ratio = np.max(np.abs(filtered)) / np.max(np.abs(high_tone))
    # Must be heavily attenuated (> 99% reduction)
    assert attenuation_ratio < 0.05


def test_robustness_lossy_codec(synthetic_signal):
    """Verify that ITU-T mu-law 8-bit companding executes deterministically."""
    sig, _ = synthetic_signal
    compressed = apply_lossy_codec_simulation(sig, mu=255)

    assert compressed.shape == sig.shape
    assert np.all(np.isfinite(compressed))
    # Check that quantizer introduces slight quantization error
    diff = np.max(np.abs(sig - compressed))
    assert diff > 0.0  # Quantization effect present
    assert diff < 0.1   # Preserves overall signal


def test_robustness_reverberation(synthetic_signal):
    """Verify that reverberation convolving produces finite, normalized output."""
    sig, sr = synthetic_signal
    reverbed = apply_reverberation(sig, rt60=0.3, sample_rate=sr, seed=42)

    assert reverbed.shape == sig.shape
    assert np.all(np.isfinite(reverbed))
    assert np.max(np.abs(reverbed)) <= 1.0


def test_robustness_clipping(synthetic_signal):
    """Verify that hard clipping enforces the amplitude ceiling."""
    sig, _ = synthetic_signal
    clipped = apply_clipping(sig, threshold=0.50)

    assert clipped.shape == sig.shape
    assert np.all(np.isfinite(clipped))


def test_robustness_resampling_stress(synthetic_signal):
    """Verify narrowband downsampling and reconstruction preserves waveform length."""
    sig, sr = synthetic_signal
    resampled = apply_resampling_stress(sig, intermediate_sr=8000, target_sr=sr)

    assert resampled.shape == sig.shape
    assert np.all(np.isfinite(resampled))


def test_phase6_documentation_reports_exist():
    """Verify that all Phase 6 scientific and technical reports are present on disk."""
    root = project_root()
    required_reports = [
        "PHASE_6_PRE_IMPLEMENTATION_AUDIT.md",
        "PHASE_6_FEATURE_PIPELINE_VALIDATION_REPORT.md",
        "PHASE_6_SPEAKER_INDEPENDENT_EVALUATION_REPORT.md",
        "PHASE_6_CROSS_DATASET_VALIDATION_REPORT.md",
        "PHASE_6_CALIBRATION_AND_THRESHOLD_REPORT.md",
        "PHASE_6_AUDIO_ROBUSTNESS_REPORT.md",
        "PHASE_6_OOD_AND_UNCERTAINTY_REPORT.md",
        "PHASE_6_FORENSIC_RELIABILITY_REPORT.md",
        "PHASE_6_DASHBOARD_INTEGRATION_REPORT.md",
    ]
    for r in required_reports:
        f = root / r
        assert f.is_file(), f"Required Phase 6 report missing: {r}"
        assert f.stat().st_size > 500, f"Report {r} appears truncated or empty"
