"""Tests for audio feature extraction using in-memory deterministic signals."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.feature_extraction import (
    extract_basic_features,
    extract_features,
    extract_features_from_file,
)


def test_extract_basic_features_returns_finite_summary():
    sample_rate = 16_000
    time = np.arange(sample_rate // 2, dtype=np.float32) / sample_rate
    waveform = 0.2 * np.sin(2 * np.pi * 440 * time)

    features = extract_basic_features(waveform, sample_rate)

    assert features["duration_seconds"] == pytest.approx(0.5)
    assert "rms_mean" in features
    assert "spectral_centroid_mean" in features
    assert "mfcc_1_mean" in features
    assert all(np.isfinite(value) for value in features.values())


def test_extract_basic_features_rejects_empty_signal():
    with pytest.raises(ValueError, match="non-empty"):
        extract_basic_features(np.array([], dtype=np.float32), 16_000)


def test_extract_features_all_nine_required_types_present():
    """Verify presence of all 9 required forensic feature representations."""
    sample_rate = 16_000
    t = np.arange(sample_rate, dtype=np.float32) / sample_rate
    waveform = 0.3 * np.sin(2 * np.pi * 500 * t) + 0.1 * np.sin(2 * np.pi * 1500 * t)

    features = extract_features(waveform, sample_rate, n_mfcc=20)

    # 1. 20 MFCC coefficients
    for idx in range(1, 21):
        assert f"mfcc_{idx}_mean" in features, f"Missing mfcc_{idx}_mean"
        assert f"mfcc_{idx}_std" in features, f"Missing mfcc_{idx}_std"

    # 2. Spectral centroid
    assert "spectral_centroid_mean" in features
    assert "spectral_centroid_std" in features

    # 3. Spectral bandwidth
    assert "spectral_bandwidth_mean" in features
    assert "spectral_bandwidth_std" in features

    # 4. Spectral rolloff
    assert "spectral_rolloff_mean" in features
    assert "spectral_rolloff_std" in features

    # 5. Zero-crossing rate
    assert "zero_crossing_rate_mean" in features
    assert "zero_crossing_rate_std" in features

    # 6. RMS energy
    assert "rms_mean" in features
    assert "rms_std" in features

    # 7. Chroma
    assert "chroma_mean" in features
    assert "chroma_std" in features
    for c_idx in range(1, 13):
        assert f"chroma_{c_idx}_mean" in features

    # 8. Spectral contrast
    assert "spectral_contrast_mean" in features
    assert "spectral_contrast_std" in features
    assert "spectral_contrast_band_1_mean" in features

    # 9. Spectral flatness
    assert "spectral_flatness_mean" in features
    assert "spectral_flatness_std" in features

    # All values must be finite floats
    assert all(np.isfinite(val) for val in features.values())


def test_extract_features_deterministic_across_runs():
    sample_rate = 16_000
    t = np.arange(sample_rate // 2, dtype=np.float32) / sample_rate
    waveform = 0.4 * np.sin(2 * np.pi * 1000 * t)

    feat1 = extract_features(waveform, sample_rate, n_mfcc=20)
    feat2 = extract_features(waveform, sample_rate, n_mfcc=20)

    for key in feat1:
        assert feat1[key] == pytest.approx(feat2[key], rel=1e-6)


def test_extract_features_rejects_non_finite_values():
    bad_wave = np.array([0.1, np.nan, 0.3], dtype=np.float32)
    with pytest.raises(ValueError, match="finite"):
        extract_features(bad_wave, 16_000)

    inf_wave = np.array([0.1, np.inf, 0.3], dtype=np.float32)
    with pytest.raises(ValueError, match="finite"):
        extract_features(inf_wave, 16_000)


def test_extract_features_rejects_invalid_mfcc():
    wave = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    with pytest.raises(ValueError):
        extract_features(wave, 16_000, n_mfcc=0)
    with pytest.raises(ValueError):
        extract_features(wave, 16_000, n_mfcc=50)


def test_extract_features_spectral_centroid_tracks_frequency():
    sr = 16_000
    t = np.arange(sr // 2, dtype=np.float32) / sr
    low_tone = np.sin(2 * np.pi * 300 * t)
    high_tone = np.sin(2 * np.pi * 3000 * t)

    feat_low = extract_features(low_tone, sr)
    feat_high = extract_features(high_tone, sr)

    assert feat_high["spectral_centroid_mean"] > feat_low["spectral_centroid_mean"]


def test_extract_features_from_file_integration(tmp_path: Path):
    audio_path = tmp_path / "integration_test.wav"
    sr = 16_000
    t = np.arange(sr // 2, dtype=np.float32) / sr
    samples = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    sf.write(str(audio_path), samples, sr)

    features = extract_features_from_file(audio_path, sample_rate=16_000, n_mfcc=20)
    assert features["file_name"] == "integration_test.wav"
    assert features["sample_rate"] == 16_000
    assert "spectral_centroid_mean" in features
    assert "mfcc_20_mean" in features
