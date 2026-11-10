"""Unit tests for audio prediction, feature alignment, and forensic report generation."""

import json
from pathlib import Path

import joblib
import numpy as np
import pytest
import soundfile as sf
from sklearn.ensemble import RandomForestClassifier

from src.audio_prediction import align_features, load_prediction_model, predict_audio
from src.forensic_report import create_forensic_report


@pytest.fixture
def dummy_model_setup(tmp_path: Path):
    """Create a temporary trained model and feature names list."""
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    expected_features = ["chroma_stft", "rms", "spectral_centroid", "spectral_bandwidth", "rolloff", "zero_crossing_rate"]
    for i in range(1, 21):
        expected_features.append(f"mfcc{i}")

    X_dummy = np.random.randn(20, len(expected_features)).astype(np.float32)
    y_dummy = np.array([0] * 10 + [1] * 10, dtype=np.int32)

    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    rf.fit(X_dummy, y_dummy)

    model_path = models_dir / "best_model.joblib"
    joblib.dump(rf, model_path)

    feat_path = models_dir / "feature_names.json"
    with open(feat_path, "w", encoding="utf-8") as f:
        json.dump(expected_features, f)

    meta_path = models_dir / "model_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"best_model_name": "Random Forest", "phase": 3}, f)

    return model_path, expected_features


def test_align_features_maps_standard_keys():
    raw_features = {
        "chroma_mean": 0.25,
        "rms_mean": 0.05,
        "spectral_centroid_mean": 1500.0,
        "spectral_bandwidth_mean": 1200.0,
        "spectral_rolloff_mean": 2800.0,
        "zero_crossing_rate_mean": 0.08,
    }
    for i in range(1, 21):
        raw_features[f"mfcc_{i}_mean"] = float(i * 1.5)

    expected_names = ["chroma_stft", "rms", "spectral_centroid", "spectral_bandwidth", "rolloff", "zero_crossing_rate"]
    for i in range(1, 21):
        expected_names.append(f"mfcc{i}")

    vec = align_features(raw_features, expected_names)
    assert vec.shape == (1, 26)
    assert vec[0, 0] == pytest.approx(0.25)
    assert vec[0, 1] == pytest.approx(0.05)
    assert vec[0, 6] == pytest.approx(1.5)


def test_predict_audio_end_to_end_on_synthetic_wav(tmp_path: Path, dummy_model_setup):
    model_path, _ = dummy_model_setup

    audio_file = tmp_path / "test_synth.wav"
    sr = 16_000
    t = np.arange(sr, dtype=np.float32) / sr
    samples = (0.4 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    sf.write(str(audio_file), samples, sr)

    result = predict_audio(audio_file, model_path=model_path, return_features=True)

    assert result["file_name"] == "test_synth.wav"
    assert "file_sha256" in result
    assert result["predicted_label"] in {"GENUINE", "SYNTHETIC"}
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["probability_synthetic"] is not None
    assert result["probability_genuine"] is not None
    assert pytest.approx(result["probability_synthetic"] + result["probability_genuine"]) == 1.0
    assert "forensic_assessment" in result
    assert "legal_disclaimer" in result
    assert "extracted_features" in result


def test_predict_audio_rejects_missing_file(dummy_model_setup):
    model_path, _ = dummy_model_setup
    with pytest.raises(FileNotFoundError):
        predict_audio("nonexistent_path_to_audio_123.wav", model_path=model_path)


def test_create_forensic_report_markdown(tmp_path: Path):
    mock_pred = {
        "file_name": "evidence_001.wav",
        "file_sha256": "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
        "file_size_bytes": 1048576,
        "original_format": "WAV",
        "original_sample_rate": 44100,
        "duration_seconds": 15.5,
        "predicted_label": "SYNTHETIC",
        "confidence": 0.945,
        "probability_synthetic": 0.945,
        "probability_genuine": 0.055,
        "model_name": "Support Vector Machine",
        "forensic_assessment": "High probability of synthetic manipulation.",
    }

    report_path = tmp_path / "test_report.md"
    report_text = create_forensic_report(mock_pred, case_number="CASE-2026-TEST", output_path=report_path)

    assert "FORENSIC AUDIO EXAMINATION REPORT" in report_text
    assert "CASE-2026-TEST" in report_text
    assert "evidence_001.wav" in report_text
    assert "SYNTHETIC" in report_text
    assert "94.50%" in report_text
    assert "IMPORTANT LEGAL NOTICE" in report_text
    assert report_path.is_file()

