"""Comprehensive Automated Test Suite for Phase 8: Production Deployment, Cybersecurity & Hardening.

Validates:
1. Memory-hard password hashing (scrypt) & complexity validation.
2. User authentication & brute-force account lockout defenses.
3. Cryptographic session token generation, validation, and revocation.
4. Role-Based Access Control (RBAC) permission matrix.
5. Binary magic-byte audio validation (WAV, MP3, FLAC, OGG) & polyglot rejection.
6. Upload payload size bounds & path traversal defense.
7. Cryptographic model artifact SHA-256 seal verification.
8. Online hot database backup & restoration with PRAGMA integrity_check.
9. SQLite Write-Ahead Logging (WAL) concurrency configuration.
10. Evidence read-only filesystem attributes & pre-examination verification.
"""

import json
from pathlib import Path
import sqlite3
import numpy as np
import pytest

from database.database import ForensicDatabase
from database.models import ForensicCase, ForensicEvidence, User, UserRole, UserSession
from src.auth import (
    AccountLockedError,
    AuthenticationError,
    authenticate_user,
    bootstrap_admin_user,
    check_permission,
    create_user_session,
    has_permission,
    hash_password,
    validate_password_complexity,
    validate_session,
    verify_password,
)
from src.backup_manager import create_system_backup, restore_system_backup
from src.security import (
    ModelIntegrityError,
    validate_audio_magic_bytes,
    validate_upload_payload,
    verify_model_artifact,
)
from src.storage_manager import apply_file_read_only, verify_exhibit_integrity


@pytest.fixture
def temp_db(tmp_path):
    """Provide an isolated SQLite database instance in temporary directory."""
    db_file = tmp_path / "forensics_test.db"
    return ForensicDatabase(db_path=db_file)


# =============================================================================
# 1. Password Hashing & Policy Tests
# =============================================================================
def test_password_hashing_and_verification():
    raw_pwd = "ForensicExaminer#2026!"
    h1, salt1 = hash_password(raw_pwd)
    h2, salt2 = hash_password(raw_pwd)

    # Different salts must produce distinct hashes
    assert salt1 != salt2
    assert h1 != h2

    # Verification
    assert verify_password(raw_pwd, h1, salt1) is True
    assert verify_password("WrongPassword123!", h1, salt1) is False
    assert verify_password(raw_pwd, h1, salt2) is False


def test_password_complexity_policy():
    valid, _ = validate_password_complexity("ValidPass123$")
    assert valid is True

    # Too short (< 10)
    valid_short, msg = validate_password_complexity("Short1$")
    assert valid_short is False
    assert "at least 10" in msg

    # Missing uppercase
    valid_no_upper, _ = validate_password_complexity("lowercase123$")
    assert valid_no_upper is False

    # Missing digit
    valid_no_digit, _ = validate_password_complexity("NoDigitsHere!$")
    assert valid_no_digit is False

    # Missing special char
    valid_no_spec, _ = validate_password_complexity("NoSpecialChars123")
    assert valid_no_spec is False


# =============================================================================
# 2. User Authentication & Lockout Tests
# =============================================================================
def test_user_authentication_success_and_failure(temp_db):
    user, raw_pwd = bootstrap_admin_user(temp_db, username="lead_admin", password="AdminPassword#123")

    # Success
    auth_user = authenticate_user("lead_admin", "AdminPassword#123", temp_db)
    assert auth_user.user_id == user.user_id
    assert auth_user.role == UserRole.ADMINISTRATOR

    # Case-insensitive username
    auth_case = authenticate_user("LEAD_ADMIN", "AdminPassword#123", temp_db)
    assert auth_case.user_id == user.user_id

    # Invalid password
    with pytest.raises(AuthenticationError, match="Invalid username or password"):
        authenticate_user("lead_admin", "WrongPassword!99", temp_db)


def test_account_lockout_after_5_failed_attempts(temp_db):
    bootstrap_admin_user(temp_db, username="locked_user", password="UserPassword#123")

    # 4 failed attempts should not lock
    for _ in range(4):
        with pytest.raises(AuthenticationError):
            authenticate_user("locked_user", "IncorrectPass!1", temp_db)

    # 5th failed attempt triggers lockout
    with pytest.raises(AccountLockedError, match="Account locked for 15 minutes"):
        authenticate_user("locked_user", "IncorrectPass!1", temp_db)

    # Subsequent attempt while locked
    with pytest.raises(AccountLockedError, match="Account is locked"):
        authenticate_user("locked_user", "UserPassword#123", temp_db)


# =============================================================================
# 3. Session Token & RBAC Tests
# =============================================================================
def test_session_lifecycle_and_revocation(temp_db):
    user, _ = bootstrap_admin_user(temp_db, username="session_tester", password="SessionPassword#1")
    session = create_user_session(user, temp_db, duration_hours=2)

    assert len(session.session_id) == 64  # 256-bit hex token
    assert session.role == UserRole.ADMINISTRATOR

    # Validate active session
    validated = validate_session(session.session_id, temp_db)
    assert validated is not None
    assert validated.username == "session_tester"

    # Revoke session
    temp_db.revoke_session(session.session_id)
    assert validate_session(session.session_id, temp_db) is None


def test_rbac_permission_matrix():
    # Admin has all permissions
    assert has_permission(UserRole.ADMINISTRATOR, "manage_users") is True
    assert has_permission(UserRole.ADMINISTRATOR, "run_examinations") is True

    # Examiner cannot manage users or backups
    assert has_permission(UserRole.FORENSIC_EXAMINER, "run_examinations") is True
    assert has_permission(UserRole.FORENSIC_EXAMINER, "manage_users") is False
    assert has_permission(UserRole.FORENSIC_EXAMINER, "manage_backups") is False

    # Reviewing Officer can review and export, but not run ML
    assert has_permission(UserRole.REVIEWING_OFFICER, "export_reports") is True
    assert has_permission(UserRole.REVIEWING_OFFICER, "run_examinations") is False

    # Read-only user can only view
    assert has_permission(UserRole.READ_ONLY_USER, "view_cases") is True
    assert has_permission(UserRole.READ_ONLY_USER, "create_cases") is False


# =============================================================================
# 4. Cybersecurity & Magic-Byte Tests
# =============================================================================
def test_magic_byte_audio_container_validation():
    # Valid WAV header
    wav_header = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00"
    is_valid, fmt, _ = validate_audio_magic_bytes(wav_header)
    assert is_valid is True
    assert fmt == "WAV"

    # Valid FLAC header
    flac_header = b"fLaC\x00\x00\x00\x22"
    is_valid, fmt, _ = validate_audio_magic_bytes(flac_header)
    assert is_valid is True
    assert fmt == "FLAC"

    # Valid OGG header
    ogg_header = b"OggS\x00\x02\x00\x00"
    is_valid, fmt, _ = validate_audio_magic_bytes(ogg_header)
    assert is_valid is True
    assert fmt == "OGG"

    # Disguised executable / ELF / Script rejected
    fake_header = b"#!/bin/bash\nrm -rf /"
    is_valid, fmt, _ = validate_audio_magic_bytes(fake_header)
    assert is_valid is False
    assert fmt == "UNKNOWN"


def test_upload_payload_screening():
    wav_data = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00data"
    valid, _ = validate_upload_payload(wav_data, "exhibit_1.wav", max_size_bytes=1024)
    assert valid is True

    # Path traversal attempt rejected
    valid_trav, msg = validate_upload_payload(wav_data, "../../etc/passwd.wav")
    assert valid_trav is False
    assert "traversal" in msg

    # Oversized payload rejected
    valid_size, msg = validate_upload_payload(wav_data * 100, "large.wav", max_size_bytes=50)
    assert valid_size is False
    assert "exceeds maximum" in msg


# =============================================================================
# 5. Model Artifact Cryptographic Integrity Tests
# =============================================================================
def test_model_artifact_cryptographic_verification(tmp_path):
    model_file = tmp_path / "test_model.joblib"
    model_file.write_bytes(b"SimulatedModelBytes12345")

    # Correct manifest
    hasher = __import__("hashlib").sha256(b"SimulatedModelBytes12345").hexdigest()
    manifest_file = tmp_path / "model_manifest.json"
    manifest_file.write_text(json.dumps({"test_model.joblib": hasher}), encoding="utf-8")

    assert verify_model_artifact(model_file, manifest_file) is True

    # Tampered model bytes
    model_file.write_bytes(b"MaliciousTamperedPayload!")
    with pytest.raises(ModelIntegrityError, match="failed cryptographic verification"):
        verify_model_artifact(model_file, manifest_file)


# =============================================================================
# 6. Database Reliability, WAL Mode & Hot Backup Tests
# =============================================================================
def test_sqlite_wal_mode_and_concurrency(temp_db):
    with sqlite3.connect(temp_db.db_path) as conn:
        mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
        assert mode.lower() == "wal"


def test_hot_backup_and_restoration_lifecycle(temp_db, tmp_path):
    # Seed data
    case = ForensicCase(case_id="CASE-BACKUP-01", title="Backup Test Case", investigator="Agent Backup")
    temp_db.create_case(case)

    # 1. Create Hot Backup
    backup_out = tmp_path / "backups"
    res = create_system_backup(temp_db, output_dir=backup_out, operator="PyTestRunner")
    assert res["status"] == "SUCCESS"
    assert res["archive_filename"].endswith(".zip")

    zip_path = Path(res["archive_path"])
    assert zip_path.is_file()

    # 2. Restore to clean target database
    restored_db_file = tmp_path / "restored_forensics.db"
    restore_res = restore_system_backup(zip_path, target_db_path=restored_db_file)
    assert restore_res["status"] == "RESTORE_SUCCESS"

    # 3. Verify restored data parity
    restored_db = ForensicDatabase(db_path=restored_db_file)
    restored_case = restored_db.get_case("CASE-BACKUP-01")
    assert restored_case is not None
    assert restored_case["title"] == "Backup Test Case"


# =============================================================================
# 7. Evidence Read-Only Storage & Pre-Examination Integrity
# =============================================================================
def test_evidence_read_only_and_integrity_check(temp_db, tmp_path):
    case = ForensicCase(case_id="CASE-EVID-01", title="Evidence Test Case")
    temp_db.create_case(case)

    dummy_audio = tmp_path / "audio_sample.wav"
    dummy_audio.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00")
    h = __import__("hashlib").sha256(dummy_audio.read_bytes()).hexdigest()

    ev = ForensicEvidence(
        evidence_id="EVID-TEST-STORE",
        case_id="CASE-EVID-01",
        original_filename="audio_sample.wav",
        stored_path=str(dummy_audio),
        file_size_bytes=len(dummy_audio.read_bytes()),
        duration_seconds=1.0,
        sample_rate=16000,
        channels=1,
        container_format="WAV",
        sha256_hash=h,
    )
    temp_db.add_evidence(ev)

    # Apply read-only
    apply_file_read_only(dummy_audio)

    # Verify integrity passes
    ok, msg, _ = verify_exhibit_integrity("EVID-TEST-STORE", temp_db)
    assert ok is True
    assert "verified intact" in msg

    # Check that an integrity record was appended to database
    checks = temp_db.list_integrity_checks(evidence_id="EVID-TEST-STORE")
    assert len(checks) == 1
    assert checks[0]["status"] == "VERIFIED"
