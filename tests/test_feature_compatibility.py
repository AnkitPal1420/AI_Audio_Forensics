"""Automated regression tests for feature extraction compatibility and schema validation."""

import numpy as np
import pytest

from src.feature_extraction import (
    CANONICAL_FEATURE_NAMES,
    extract_basic_features,
    extract_features,
    extract_slice_features_from_audio,
    validate_feature_vector,
)


def test_feature_schema_contains_expected_26_features():
    assert len(CANONICAL_FEATURE_NAMES) == 26
    expected_prefixes = ["chroma_stft", "rms", "spectral_centroid", "spectral_bandwidth", "rolloff", "zero_crossing_rate"]
    for prefix in expected_prefixes:
        assert prefix in CANONICAL_FEATURE_NAMES
    for i in range(1, 21):
        assert f"mfcc{i}" in CANONICAL_FEATURE_NAMES


def test_validate_feature_vector_success():
    sr = 16_000
    t = np.arange(sr, dtype=np.float32) / sr
    waveform = 0.25 * np.sin(2 * np.pi * 500 * t)
    feats = extract_basic_features(waveform, sr)

    vector = validate_feature_vector(feats, CANONICAL_FEATURE_NAMES)
    assert vector.shape == (1, 26)
    assert np.all(np.isfinite(vector))


def test_validate_feature_vector_detects_missing_features():
    bad_feats = {"rms": 0.05, "mfcc1": -200.0}
    with pytest.raises(ValueError, match="Missing required features"):
        validate_feature_vector(bad_feats, CANONICAL_FEATURE_NAMES)


def test_validate_feature_vector_detects_non_finite_features():
    sr = 16_000
    t = np.arange(sr, dtype=np.float32) / sr
    waveform = 0.25 * np.sin(2 * np.pi * 500 * t)
    feats = extract_basic_features(waveform, sr)
    feats["mfcc1"] = float(np.nan)

    with pytest.raises(ValueError, match="Non-finite feature values"):
        validate_feature_vector(feats, CANONICAL_FEATURE_NAMES)


def test_slice_feature_extraction():
    sr = 16_000
    t = np.arange(sr * 3, dtype=np.float32) / sr  # 3 seconds
    waveform = 0.3 * np.sin(2 * np.pi * 440 * t)

    slices = extract_slice_features_from_audio(waveform, sample_rate=sr, slice_duration=1.0)
    assert len(slices) == 3
    for s_idx, s in enumerate(slices):
        assert s["slice_start_sec"] == pytest.approx(float(s_idx))
        assert s["slice_end_sec"] == pytest.approx(float(s_idx + 1))
        vec = validate_feature_vector(s, CANONICAL_FEATURE_NAMES)
        assert vec.shape == (1, 26)


def test_mfcc_scale_consistency_regression():
    """Verify that MFCC scale is consistently decibel-scaled (negative base coefficient, realistic dynamic range)."""
    sr = 16_000
    t = np.arange(sr, dtype=np.float32) / sr
    waveform = 0.2 * np.sin(2 * np.pi * 1000 * t)
    feats = extract_basic_features(waveform, sr)

    # MFCC1 for typical amplitude should be negative in standard dB representation
    assert feats["mfcc1"] < 0.0, f"Expected negative mfcc1 in dB-scale, got {feats['mfcc1']}"
    assert -800.0 <= feats["mfcc1"] <= 0.0
