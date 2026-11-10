"""Comprehensive Phase 4 unit and integration tests for audio prediction & forensic reporting."""

from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf

from src.audio_prediction import predict_audio
from src.audio_quality import assess_audio_quality, assess_feature_distribution, estimate_snr_db
from src.deepfake_detection import DeepfakeDetectionEngine
from src.forensic_integrity import (
    EvidenceIntegrityError,
    compute_file_sha256,
    generate_evidence_id,
    record_analysis_event,
    verify_evidence_immutability,
)
from src.forensic_report import create_forensic_report, generate_forensic_pdf
from src.spectrogram_analysis import (
    compute_spectral_trajectories,
    compute_stft_frames,
    create_mel_filterbank,
    generate_forensic_visualizations,
)
from src.utils import project_root


@pytest.fixture
def clean_speech_signal():
    """Generate a clean synthetic harmonic test audio signal (16 kHz, 2 seconds)."""
    sr = 16000
    t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False)
    # Fundamental + harmonics
    wave = 0.4 * np.sin(2 * np.pi * 220 * t) + 0.2 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.sin(2 * np.pi * 880 * t)
    # Add subtle stationary noise
    noise = np.random.normal(0, 0.005, len(t))
    signal = wave + noise
    return signal.astype(np.float32), sr


def test_audio_quality_normal_signal(clean_speech_signal):
    wave, sr = clean_speech_signal
    res = assess_audio_quality(wave, sr)
    assert res["status"] == "PASS"
    assert res["duration_seconds"] == pytest.approx(2.0, abs=0.05)
    assert res["clipping_ratio"] == 0.0
    assert res["silence_ratio"] < 0.50
    assert res["estimated_snr_db"] > 10.0
    assert res["is_reliable_for_inference"] is True


def test_audio_quality_insufficient_duration():
    sr = 16000
    short_wave = np.random.normal(0, 0.1, int(sr * 0.2)).astype(np.float32)  # 0.2 seconds
    res = assess_audio_quality(short_wave, sr, min_duration_sec=0.5)
    assert res["status"] == "FAIL"
    assert any("INSUFFICIENT_DURATION" in f for f in res["quality_flags"])
    assert res["is_reliable_for_inference"] is False


def test_audio_quality_excessive_clipping():
    sr = 16000
    # Artificially clipped signal
    wave = np.ones(int(sr * 1.0), dtype=np.float32)
    res = assess_audio_quality(wave, sr, max_clipping_ratio=0.05)
    assert res["status"] == "FAIL"
    assert any("EXCESSIVE_CLIPPING" in f for f in res["quality_flags"])


def test_audio_quality_excessive_silence():
    sr = 16000
    silent_wave = np.zeros(int(sr * 2.0), dtype=np.float32)
    res = assess_audio_quality(silent_wave, sr)
    assert res["status"] == "FAIL"
    assert any("EXCESSIVE_SILENCE" in f for f in res["quality_flags"])


def test_feature_distribution_ood():
    # Artificially extreme feature values (e.g. z-score > 10)
    extreme_features = {
        "chroma_stft": 99.0,
        "rms": 999.0,
        "spectral_centroid": 99999.0,
        "mfcc1": -9999.0,
    }
    res = assess_feature_distribution(extreme_features)
    assert res["status"] == "OUT_OF_DISTRIBUTION"
    assert res["is_in_distribution"] is False
    assert res["max_z_score"] > 6.0
    assert len(res["outlier_features"]) > 0


def test_forensic_integrity_hashing(tmp_path):
    test_file = tmp_path / "exhibit.wav"
    data = np.random.uniform(-0.5, 0.5, 16000).astype(np.float32)
    sf.write(str(test_file), data, 16000)

    hash1 = compute_file_sha256(test_file)
    assert len(hash1) == 64
    assert verify_evidence_immutability(test_file, hash1) is True

    # Mutate file
    with test_file.open("ab") as f:
        f.write(b"\x00\x01\x02")

    with pytest.raises(EvidenceIntegrityError):
        verify_evidence_immutability(test_file, hash1)


def test_spectrogram_and_trajectories(clean_speech_signal, tmp_path):
    wave, sr = clean_speech_signal
    trajectories = compute_spectral_trajectories(wave, sr)

    assert "centroid" in trajectories
    assert "bandwidth" in trajectories
    assert "rolloff" in trajectories
    assert "mfcc_matrix" in trajectories
    assert trajectories["mfcc_matrix"].shape[0] == 20

    figs = generate_forensic_visualizations(wave, sr, output_dir=tmp_path, file_stem="unit_test")
    assert "overview" in figs
    assert figs["overview"].is_file()
    assert figs["spectrogram"].is_file()
    assert figs["mel_mfcc"].is_file()


def test_deepfake_detection_engine(clean_speech_signal):
    engine = DeepfakeDetectionEngine()
    models = engine.get_available_models()
    assert len(models) >= 3

    # Create dummy 26-dim feature vector
    feat_vec = np.zeros((1, 26), dtype=np.float32)
    # Give realistic center values
    feat_vec[0, 6] = -200.0  # mfcc1

    res = engine.predict_multi_model(feat_vec)
    assert "individual_models" in res
    assert "consensus_label" in res
    assert "probability_spread" in res


def test_forensic_report_generation(tmp_path):
    mock_prediction = {
        "evidence_id": "EVID-TEST-001",
        "file_name": "test_audio.wav",
        "file_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "file_size_bytes": 10240,
        "original_format": "WAV",
        "original_sample_rate": 16000,
        "original_channels": 1,
        "duration_seconds": 2.5,
        "status": "COMPLETED",
        "predicted_label": "GENUINE",
        "confidence": 0.985,
        "decision_score": 2.345,
        "calibrated_probability_synthetic": 0.015,
        "calibrated_probability_genuine": 0.985,
        "primary_model_name": "Calibrated SVM",
        "quality_assessment": {"estimated_snr_db": 25.0, "clipping_ratio": 0.0, "silence_ratio": 0.1},
        "distribution_assessment": {"status": "IN_DISTRIBUTION", "max_z_score": 1.2},
        "multi_model_analysis": {
            "individual_models": {
                "calibrated_svm": {"predicted_label": "GENUINE", "probability_synthetic": 0.015, "confidence": 0.985},
                "random_forest": {"predicted_label": "GENUINE", "probability_synthetic": 0.03, "confidence": 0.97},
            }
        },
        "visualizations": {},
        "forensic_assessment": "Consistent with genuine human speech.",
        "legal_disclaimer": "Forensic screening notice.",
    }

    rep_md_path = tmp_path / "report.md"
    rep_pdf_path = tmp_path / "report.pdf"

    md_content = create_forensic_report(mock_prediction, output_path=rep_md_path, generate_pdf=True)
    assert rep_md_path.is_file()
    assert rep_pdf_path.is_file()
    assert "FORENSIC AUDIO EXAMINATION REPORT" in md_content
    assert "EVID-TEST-001" in md_content
    assert rep_pdf_path.stat().st_size > 1000  # Valid non-empty PDF


def test_predict_audio_end_to_end_genuine():
    root = project_root()
    demo_audio = root / "dataset" / "real" / "linus-original-DEMO.mp3"
    if not demo_audio.is_file():
        pytest.skip("Demo audio exhibit not present.")

    res = predict_audio(demo_audio, generate_visuals=False)
    assert res["status"] == "COMPLETED"
    assert res["predicted_label"] == "GENUINE"
    assert res["calibrated_probability_genuine"] > 0.80
    assert res["evidence_id"].startswith("EVID-")
