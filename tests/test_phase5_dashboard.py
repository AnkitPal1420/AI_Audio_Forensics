"""Comprehensive unit and integration tests for Phase 5 Forensic Web Dashboard and Database."""

from __future__ import annotations

import tempfile
from pathlib import Path
import pytest
import soundfile as sf
import numpy as np

from database.database import ForensicDatabase, init_db
from database.models import ForensicCase, ForensicEvidence, ForensicExamination, IntegrityRecord
from src.storage_manager import get_storage_paths, ingest_evidence_file, sanitize_filename


@pytest.fixture
def temp_db(tmp_path: Path):
    """Create a temporary isolated SQLite database for testing."""
    db_file = tmp_path / "test_forensics.db"
    return ForensicDatabase(db_file)


def test_database_initialization(temp_db):
    assert temp_db.db_path.is_file()
    metrics = temp_db.get_dashboard_metrics()
    assert metrics["total_cases"] == 0
    assert metrics["total_evidence"] == 0


def test_case_lifecycle(temp_db):
    case = ForensicCase(
        case_id="CASE-TEST-2026-001",
        title="Test Deepfake Extortion Case",
        description="Acoustic investigation of suspected synthetic voice calls.",
        investigator="Special Agent Tester",
        status="ACTIVE",
    )
    temp_db.create_case(case)

    # Retrieval
    retrieved = temp_db.get_case("CASE-TEST-2026-001")
    assert retrieved is not None
    assert retrieved["title"] == "Test Deepfake Extortion Case"
    assert retrieved["status"] == "ACTIVE"

    # Status update
    updated = temp_db.update_case_status("CASE-TEST-2026-001", "CLOSED")
    assert updated is True
    assert temp_db.get_case("CASE-TEST-2026-001")["status"] == "CLOSED"

    # Search
    search_res = temp_db.list_cases(search="Extortion")
    assert len(search_res) == 1


def test_filename_sanitization():
    assert sanitize_filename("../../../etc/passwd.wav") == "passwd.wav"
    assert sanitize_filename("..\\..\\malicious\\payload.mp3") == "payload.mp3"
    assert sanitize_filename("safe_recording.wav") == "safe_recording.wav"
    assert sanitize_filename("audio with spaces & symbols !@#.wav") == "audio_with_spaces___symbols____.wav"


def test_evidence_ingestion_and_immutability(temp_db, tmp_path):
    case = ForensicCase(case_id="CASE-001", title="Case 001")
    temp_db.create_case(case)

    # Create dummy WAV
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    wave = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    dummy_wav = tmp_path / "intake_sample.wav"
    sf.write(str(dummy_wav), wave, sr)

    dest_path, evid_id, sha256_hash, meta = ingest_evidence_file(
        dummy_wav, "intake_sample.wav", "CASE-001"
    )

    assert dest_path.is_file()
    assert evid_id.startswith("EVID-")
    assert len(sha256_hash) == 64

    # Register in DB
    ev = ForensicEvidence(
        evidence_id=evid_id,
        case_id="CASE-001",
        original_filename="intake_sample.wav",
        stored_path=str(dest_path),
        file_size_bytes=meta["file_size_bytes"],
        duration_seconds=1.0,
        sample_rate=sr,
        channels=1,
        container_format="WAV",
        sha256_hash=sha256_hash,
    )
    temp_db.add_evidence(ev)

    ev_list = temp_db.list_evidence_by_case("CASE-001")
    assert len(ev_list) == 1
    assert ev_list[0]["evidence_id"] == evid_id


def test_integrity_check_logging(temp_db):
    # Insert parent case and evidence first for referential integrity
    case = ForensicCase(case_id="CASE-INT-001", title="Integrity Test Case")
    temp_db.create_case(case)
    ev = ForensicEvidence(
        evidence_id="EVID-TEST-001",
        case_id="CASE-INT-001",
        original_filename="sample.wav",
        stored_path="storage/sample.wav",
        file_size_bytes=1000,
        duration_seconds=1.0,
        sample_rate=16000,
        channels=1,
        container_format="WAV",
        sha256_hash="aaaa1111bbbb2222cccc3333dddd4444eeee5555ffff6666aaaa1111bbbb2222",
    )
    temp_db.add_evidence(ev)

    check = IntegrityRecord(
        evidence_id="EVID-TEST-001",
        recorded_sha256="aaaa1111bbbb2222cccc3333dddd4444eeee5555ffff6666aaaa1111bbbb2222",
        checked_sha256="aaaa1111bbbb2222cccc3333dddd4444eeee5555ffff6666aaaa1111bbbb2222",
        status="VERIFIED",
        verified_by="Analyst Jane",
        notes="All clear",
    )
    temp_db.record_integrity_check(check)

    checks = temp_db.list_integrity_checks(evidence_id="EVID-TEST-001")
    assert len(checks) == 1
    assert checks[0]["status"] == "VERIFIED"


def test_examination_recording_and_kpis(temp_db):
    # Insert parent case and evidence first
    case = ForensicCase(case_id="CASE-001", title="Case 001")
    temp_db.create_case(case)
    ev = ForensicEvidence(
        evidence_id="EVID-001",
        case_id="CASE-001",
        original_filename="sample.wav",
        stored_path="storage/sample.wav",
        file_size_bytes=1000,
        duration_seconds=1.0,
        sample_rate=16000,
        channels=1,
        container_format="WAV",
        sha256_hash="1111222233334444555566667777888899990000111122223333444455556666",
    )
    temp_db.add_evidence(ev)

    exam = ForensicExamination(
        examination_id="EXAM-001",
        evidence_id="EVID-001",
        case_id="CASE-001",
        model_name="ensemble",
        predicted_label="GENUINE",
        confidence=0.985,
        status="COMPLETED",
        forensic_assessment="Consistent with genuine speech.",
        prob_synthetic=0.015,
        prob_genuine=0.985,
    )
    temp_db.add_examination(exam)

    exams = temp_db.list_examinations(label="GENUINE")
    assert len(exams) == 1
    assert exams[0]["confidence"] == 0.985

    metrics = temp_db.get_dashboard_metrics()
    assert metrics["genuine_count"] == 1


def test_audit_event_logging(temp_db):
    temp_db.log_audit_event("TEST_EVENT", {"detail": "verification test"}, case_id="CASE-001")
    events = temp_db.list_audit_events()
    assert len(events) >= 1
    assert events[0]["event_type"] == "TEST_EVENT"
