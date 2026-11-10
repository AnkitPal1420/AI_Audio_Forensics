"""Phase 6.1: Feature pipeline mathematical validation and deterministic parity test suite."""

import json
from pathlib import Path
import numpy as np
import pytest

from src.feature_extraction import (
    CANONICAL_FEATURE_NAMES,
    extract_basic_features,
    extract_features,
    validate_feature_vector,
)
from src.audio_prediction import align_features, load_prediction_model
from src.utils import project_root


@pytest.fixture
def test_waveform():
    """Deterministic synthetic test signal: 1 second harmonic tone + noise at 16 kHz."""
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    # 440 Hz fundamental + 880 Hz harmonic + pink/white noise
    sig = 0.5 * np.sin(2 * np.pi * 440.0 * t) + 0.25 * np.sin(2 * np.pi * 880.0 * t)
    rng = np.random.RandomState(42)
    noise = rng.normal(0, 0.05, sr).astype(np.float32)
    waveform = sig + noise
    # Peak normalize
    return waveform / np.max(np.abs(waveform)), sr


def test_feature_schema_matches_saved_models():
    """Verify that CANONICAL_FEATURE_NAMES matches models/feature_names.json exactly."""
    root = project_root()
    feat_names_path = root / "models" / "feature_names.json"
    assert feat_names_path.is_file(), "models/feature_names.json must exist"

    with feat_names_path.open("r", encoding="utf-8") as f:
        saved_names = json.load(f)

    assert len(saved_names) == 26, f"Expected 26 features, found {len(saved_names)}"
    assert saved_names == CANONICAL_FEATURE_NAMES, (
        f"Feature ordering mismatch between canonical schema and model schema:\n"
        f"Canonical: {CANONICAL_FEATURE_NAMES}\nSaved: {saved_names}"
    )


def test_feature_extraction_determinism(test_waveform):
    """Verify that multiple extractions on the identical waveform produce bitwise identical floats."""
    waveform, sr = test_waveform

    feat_run_1 = extract_basic_features(waveform, sample_rate=sr)
    feat_run_2 = extract_basic_features(waveform, sample_rate=sr)

    for k in CANONICAL_FEATURE_NAMES:
        assert k in feat_run_1, f"Missing feature {k} in run 1"
        assert k in feat_run_2, f"Missing feature {k} in run 2"
        np.testing.assert_equal(
            feat_run_1[k],
            feat_run_2[k],
            err_msg=f"Non-deterministic feature calculation for {k}",
        )


def test_feature_vector_validation_passes_valid(test_waveform):
    """Verify that a valid feature dictionary validates into a (1, 26) float32 vector."""
    waveform, sr = test_waveform
    raw_feats = extract_basic_features(waveform, sample_rate=sr)

    vec = validate_feature_vector(raw_feats, expected_names=CANONICAL_FEATURE_NAMES)
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (1, 26)
    assert vec.dtype == np.float32
    assert np.all(np.isfinite(vec))


def test_feature_vector_validation_catches_missing():
    """Verify that missing required features trigger a descriptive ValueError."""
    incomplete_feats = {"chroma_stft": 0.5, "rms": 0.1}
    with pytest.raises(ValueError, match="Missing required features"):
        validate_feature_vector(incomplete_feats, expected_names=CANONICAL_FEATURE_NAMES)


def test_feature_vector_validation_catches_nan_and_inf():
    """Verify that NaN or infinite feature values are rejected."""
    bad_feats = {k: 1.0 for k in CANONICAL_FEATURE_NAMES}
    bad_feats["mfcc1"] = float("nan")
    with pytest.raises(ValueError, match="Non-finite feature values"):
        validate_feature_vector(bad_feats, expected_names=CANONICAL_FEATURE_NAMES)

    bad_feats["mfcc1"] = float("inf")
    with pytest.raises(ValueError, match="Non-finite feature values"):
        validate_feature_vector(bad_feats, expected_names=CANONICAL_FEATURE_NAMES)


def test_align_features_ordering():
    """Verify that align_features maps dictionary keys correctly into the 26-dim vector."""
    sample_dict = {name: float(idx + 1) for idx, name in enumerate(CANONICAL_FEATURE_NAMES)}
    aligned = align_features(sample_dict, CANONICAL_FEATURE_NAMES)
    assert aligned.shape == (1, 26)
    for idx, name in enumerate(CANONICAL_FEATURE_NAMES):
        assert aligned[0, idx] == float(idx + 1)


def test_trained_model_dimensions():
    """Verify that all saved model artifacts expect exactly 26 input dimensions."""
    root = project_root()
    model_files = [
        "best_model.joblib",
        "calibrated_svm_model.joblib",
        "svm_model.joblib",
        "rf_model.joblib",
        "logistic_regression_model.joblib",
    ]
    for mf in model_files:
        path = root / "models" / mf
        if path.is_file():
            model, feat_names, meta = load_prediction_model(path)
            assert len(feat_names) == 26
            # Test inference with synthetic (1, 26) vector
            dummy_x = np.zeros((1, 26), dtype=np.float32)
            pred = model.predict(dummy_x)
            assert pred is not None
