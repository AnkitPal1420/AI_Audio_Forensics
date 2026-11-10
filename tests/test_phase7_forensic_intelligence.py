"""Automated Test Suite for Phase 7: Advanced Forensic Intelligence, Explainability & Manipulation Analysis.

Tests:
1. Signal tampering indicators (spectral flux, RMS step, repeated segments).
2. Acoustic anomaly detection (flatness, centroid, IsolationForest / LOF).
3. Explainable AI framework (occlusion, pure-Python Shapley sampling, unvalidated warning, schema guard).
4. Speaker comparison baseline (strictly EXPERIMENTAL_INDETERMINATE, no identity claim).
5. Synthetic generator characterization (bandwidth cutoff, vocoder heuristic, disclaimer).
6. Multi-evidence correlation & database persistence (runs, findings, comparisons).
"""

import json
import numpy as np
import pytest

from database.database import ForensicDatabase
from database.models import (
    AnalysisFinding,
    AnalysisRun,
    EvidenceComparisonRecord,
    ForensicCase,
    ForensicEvidence,
)
from src.anomaly_detection import (
    compute_pitch_autocorrelation_proxy,
    compute_spectral_centroid,
    compute_spectral_flatness,
    detect_acoustic_anomalies,
)
from src.evidence_correlation import correlate_exhibits, search_temporal_offset
from src.explainable_ai import (
    compute_sampling_shapley,
    explain_instance_occlusion,
    explain_prediction,
)
from src.generator_analysis import analyze_spectral_bandwidth, profile_generator_artifacts
from src.speaker_comparison import compare_acoustic_profiles
from src.tampering_analysis import (
    analyze_audio_tampering,
    compute_frame_rms,
    compute_frame_stft,
    compute_noise_floor_profile,
    compute_spectral_flux,
    detect_repeated_segments,
)


@pytest.fixture
def sample_audio_and_sr():
    """Generate 2 seconds of synthetic speech-like harmonic signal with a transient discontinuity."""
    sr = 16000
    t = np.linspace(0, 2.0, sr * 2, endpoint=False, dtype=np.float32)
    # 200 Hz fundamental + harmonics
    audio = 0.4 * np.sin(2 * np.pi * 200 * t) + 0.2 * np.sin(2 * np.pi * 400 * t)
    # Inject sharp splice / burst at t = 1.0s
    burst_idx = int(1.0 * sr)
    audio[burst_idx : burst_idx + 200] += 0.8
    # Inject repeated segment from 0.2-0.5s into 1.4-1.7s
    seg_len = int(0.4 * sr)
    audio[int(1.4 * sr) : int(1.4 * sr) + seg_len] = audio[int(0.2 * sr) : int(0.2 * sr) + seg_len]
    return audio, sr


@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary SQLite database with Phase 7 tables."""
    db_file = tmp_path / "test_forensics.db"
    return ForensicDatabase(db_path=db_file)


# =============================================================================
# 1. Tampering Analysis Tests
# =============================================================================
def test_signal_tampering_analysis(sample_audio_and_sr):
    audio, sr = sample_audio_and_sr
    res = analyze_audio_tampering(audio, sr, evidence_id="EVID-TEST-001", flux_sigma_threshold=2.5)

    assert res.sample_rate == sr
    assert res.duration_seconds == pytest.approx(2.0, abs=0.05)
    assert len(res.time_points) > 0
    assert len(res.spectral_flux) == len(res.time_points)
    assert "FORENSIC NOTICE" in res.disclaimer

    # Check findings
    assert len(res.findings) > 0
    for finding in res.findings:
        assert finding.interpretation_flag == "MEASUREMENT"
        assert finding.evidence_id == "EVID-TEST-001"
        assert finding.severity in ["LOW", "MEDIUM", "HIGH"]
        assert finding.start_sec >= 0.0


def test_repeated_segment_detection(sample_audio_and_sr):
    audio, sr = sample_audio_and_sr
    repeats = detect_repeated_segments(audio, sr, min_segment_sec=0.3, similarity_threshold=0.90)
    assert len(repeats) >= 1
    t_a1, t_a2, t_b1, corr = repeats[0]
    assert corr >= 0.90
    assert t_b1 > t_a1


# =============================================================================
# 2. Acoustic Anomaly Detection Tests
# =============================================================================
def test_acoustic_anomaly_detection_isolation_forest(sample_audio_and_sr):
    audio, sr = sample_audio_and_sr
    res = detect_acoustic_anomalies(
        audio, sr, evidence_id="EVID-TEST-001", contamination=0.08, method="IsolationForest"
    )

    assert res.algorithm == "IsolationForest"
    assert res.outlier_fraction > 0.0
    assert len(res.anomaly_scores) == len(res.time_points)
    assert "FORENSIC NOTICE" in res.disclaimer
    for finding in res.findings:
        assert finding.indicator == "ACOUSTIC_OUTLIER"
        assert finding.interpretation_flag == "MEASUREMENT"


def test_acoustic_anomaly_spectral_measures(sample_audio_and_sr):
    audio, sr = sample_audio_and_sr
    mag_spec, _ = compute_frame_stft(audio, sr)
    flatness = compute_spectral_flatness(mag_spec)
    centroid = compute_spectral_centroid(mag_spec, sr)
    f0, pitch_stab = compute_pitch_autocorrelation_proxy(audio, sr)

    assert len(flatness) > 0
    assert np.all(flatness >= 0.0) and np.all(flatness <= 1.0)
    assert len(centroid) == len(flatness)
    assert np.all(centroid >= 0.0)
    assert len(f0) > 0


# =============================================================================
# 3. Explainable AI Tests
# =============================================================================
def test_explainable_ai_framework():
    # Construct a valid 26-element feature vector
    feat_vec = np.random.randn(1, 26).astype(np.float32)

    res = explain_prediction(feat_vec, evidence_id="EVID-TEST-001", model_key="calibrated_svm", compute_shapley=True)

    assert res.predicted_label in ["GENUINE", "SYNTHETIC"]
    assert 0.0 <= res.predicted_prob_synthetic <= 1.0
    assert len(res.attributions) == 26
    assert res.shapley_values is not None
    assert len(res.shapley_values) == 26
    assert "MODEL UNVALIDATED WARNING" in res.unvalidated_model_warning
    assert "EXPLAINABLE AI NOTICE" in res.forensic_disclaimer


def test_explainable_ai_schema_rejection():
    # Invalid dimension: 20 features instead of 26
    invalid_feat = np.random.randn(1, 20).astype(np.float32)
    with pytest.raises(ValueError, match="Feature schema mismatch"):
        explain_prediction(invalid_feat, evidence_id="EVID-TEST-001")


# =============================================================================
# 4. Speaker Comparison Scientific Constraint Tests
# =============================================================================
def test_speaker_comparison_baseline():
    vec_a = np.ones(26, dtype=np.float32)
    vec_b = np.ones(26, dtype=np.float32)

    res = compare_acoustic_profiles(vec_a, vec_b, evidence_a_id="EVID-A", evidence_b_id="EVID-B")

    assert res.cosine_similarity == pytest.approx(1.0, abs=1e-3)
    assert res.euclidean_distance == pytest.approx(0.0, abs=1e-3)
    assert res.status == "EXPERIMENTAL_INDETERMINATE"
    assert res.identity_claim_permitted is False
    assert len(res.prerequisites_missing) > 0
    assert "CRITICAL FORENSIC LIMITATION" in res.scientific_disclaimer


# =============================================================================
# 5. Synthetic Generator Characterization Tests
# =============================================================================
def test_generator_analysis_bandwidth(sample_audio_and_sr):
    audio, sr = sample_audio_and_sr
    res = profile_generator_artifacts(audio, sr, evidence_id="EVID-TEST-001")

    assert res.effective_bandwidth_hz > 0.0
    assert res.spectral_rolloff_85_hz > 0.0
    assert res.confidence_level == "EXPERIMENTAL_HEURISTIC"
    assert "CRITICAL FORENSIC LIMITATION" in res.scientific_disclaimer
    assert len(res.prerequisites_missing) > 0


# =============================================================================
# 6. Multi-Evidence Correlation & Database Persistence Tests
# =============================================================================
def test_evidence_correlation_and_db_persistence(temp_db, sample_audio_and_sr):
    audio, sr = sample_audio_and_sr

    # Set up case and evidence in temp DB
    case = ForensicCase(case_id="CASE-P7-001", title="Phase 7 Test Case", investigator="Agent Tester")
    temp_db.create_case(case)

    ev_a = ForensicEvidence(
        evidence_id="EVID-P7-A",
        case_id="CASE-P7-001",
        original_filename="audio_a.wav",
        stored_path="test_a.wav",
        file_size_bytes=len(audio) * 4,
        duration_seconds=2.0,
        sample_rate=sr,
        channels=1,
        container_format="WAV",
        sha256_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    )
    ev_b = ForensicEvidence(
        evidence_id="EVID-P7-B",
        case_id="CASE-P7-001",
        original_filename="audio_b.wav",
        stored_path="test_b.wav",
        file_size_bytes=len(audio) * 4,
        duration_seconds=2.0,
        sample_rate=sr,
        channels=1,
        container_format="WAV",
        sha256_hash="f4c0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b899",
    )
    temp_db.add_evidence(ev_a)
    temp_db.add_evidence(ev_b)

    feat_a = np.random.randn(26).astype(np.float32)
    feat_b = np.random.randn(26).astype(np.float32)

    report = correlate_exhibits(
        evidence_a=ev_a,
        evidence_b=ev_b,
        audio_a=audio,
        audio_b=audio,
        features_a=feat_a,
        features_b=feat_b,
        db=temp_db,
    )

    assert report.case_id == "CASE-P7-001"
    assert report.sha256_match is False
    assert report.correlation_score >= 0.0

    # Verify database persistence
    comps = temp_db.list_evidence_comparisons(case_id="CASE-P7-001")
    assert len(comps) == 1
    assert comps[0].evidence_a_id == "EVID-P7-A"
    assert comps[0].evidence_b_id == "EVID-P7-B"

    # Test AnalysisRun insertion & retrieval
    run = AnalysisRun(
        run_id="RUN-TEST-001",
        case_id="CASE-P7-001",
        evidence_id="EVID-P7-A",
        analysis_type="AUDIO_MANIPULATION",
        algorithm_name="analyze_audio_tampering",
        algorithm_version="1.0.0",
        software_versions_json=json.dumps({"numpy": "2.2.0"}),
        parameters_json=json.dumps({"threshold": 3.5}),
        input_sha256=ev_a.sha256_hash,
        status="COMPLETED",
        warnings_json=json.dumps([]),
        results_json=json.dumps({"anomalies": 2}),
        output_files_json=json.dumps([]),
    )
    temp_db.create_analysis_run(run)
    fetched_run = temp_db.get_analysis_run("RUN-TEST-001")
    assert fetched_run is not None
    assert fetched_run.algorithm_name == "analyze_audio_tampering"

    runs_list = temp_db.list_analysis_runs(case_id="CASE-P7-001")
    assert len(runs_list) == 2
    manip_runs = temp_db.list_analysis_runs(case_id="CASE-P7-001", analysis_type="AUDIO_MANIPULATION")
    assert len(manip_runs) == 1
