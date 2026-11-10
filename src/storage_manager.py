"""Storage management and evidence preservation layer.

Ensures strict segregation of digital assets adhering to ISO/IEC 27037 principles:
- storage/original_evidence/ : Bitstream-exact immutable exhibits (read-only)
- storage/processed_audio/   : Normalized 16 kHz mono derivatives
- storage/visualizations/    : Forensic spectrograms, waveforms, heatmaps
- storage/reports/           : Formal Markdown and PDF examination reports
"""

from __future__ import annotations

import ctypes
import logging
import os
import re
import shutil
import stat
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from database.database import ForensicDatabase
from database.models import ForensicCase, ForensicEvidence, IntegrityRecord
from src.audio_preprocessing import compute_file_hash, get_audio_metadata
from src.forensic_integrity import compute_file_sha256, generate_evidence_id
from src.utils import project_root

logger = logging.getLogger(__name__)


def apply_file_read_only(path: Union[str, Path]) -> None:
    """Apply OS-level read-only file attributes to protect digital evidence from accidental modification."""
    p = Path(path).resolve()
    if not p.is_file():
        return

    try:
        # Standard POSIX / cross-platform read-only mode
        current_perms = os.stat(p).st_mode
        read_only_perms = current_perms & ~(stat.S_IWRITE | stat.S_IWGRP | stat.S_IWOTH)
        os.chmod(p, read_only_perms)

        # Windows-specific API enforcement (FILE_ATTRIBUTE_READONLY = 0x01)
        if sys.platform == "win32":
            FILE_ATTRIBUTE_READONLY = 0x01
            ctypes.windll.kernel32.SetFileAttributesW(str(p), FILE_ATTRIBUTE_READONLY)
    except Exception as e:
        logger.warning("Could not apply read-only file attribute to %s: %s", p, e)


def verify_exhibit_integrity(evidence_id: str, db: ForensicDatabase) -> Tuple[bool, str, Dict[str, Any]]:
    """Cryptographically verify an exhibit on disk against its intake SHA-256 record."""
    evidence = db.get_evidence(evidence_id)
    if not evidence:
        return False, f"Evidence record '{evidence_id}' not found in database.", {}

    stored_path = Path(evidence["stored_path"])
    if not stored_path.is_file():
        return False, f"Physical evidence file missing on disk at: {stored_path}", evidence

    # Compute current live hash
    live_hash = compute_file_sha256(stored_path)
    recorded_hash = evidence["sha256_hash"].strip().lower()

    is_intact = live_hash.lower() == recorded_hash
    status = "VERIFIED" if is_intact else "HASH_MISMATCH"

    # Log to integrity_checks table
    rec = IntegrityRecord(
        evidence_id=evidence_id,
        recorded_sha256=recorded_hash,
        checked_sha256=live_hash,
        status=status,
        verified_by="StorageManager.verify_exhibit_integrity",
        notes="Automated pre-examination cryptographic audit check.",
    )
    db.record_integrity_check(rec)

    if not is_intact:
        logger.critical(
            "INTEGRITY BREACH: Evidence %s hash mismatch! Recorded %s, Current %s",
            evidence_id,
            recorded_hash,
            live_hash,
        )
        return False, f"CRITICAL INTEGRITY BREACH: SHA-256 mismatch for exhibit {evidence_id}!", evidence

    return True, f"Exhibit {evidence_id} verified intact (SHA-256: {live_hash[:16]}...)", evidence


def get_storage_paths() -> Dict[str, Path]:
    """Retrieve and ensure existence of standardized forensic storage directories."""
    root = project_root()
    base = root / "storage"

    paths = {
        "root": base,
        "original_evidence": base / "original_evidence",
        "processed_audio": base / "processed_audio",
        "visualizations": base / "visualizations",
        "reports": base / "reports",
    }

    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)

    return paths


def sanitize_filename(filename: str) -> str:
    """Sanitize user-provided filename to prevent path traversal and shell injection."""
    name = Path(filename).name
    # Remove any dangerous or non-standard characters
    clean = re.sub(r"[^\w\.-]", "_", name)
    # Prevent leading dots or empty names
    clean = clean.lstrip(".-")
    return clean or "exhibit_audio.wav"


def ingest_evidence_file(
    file_bytes_or_path: Union[bytes, Path, str],
    filename: str,
    case_id: str,
    make_read_only: bool = True,
) -> Tuple[Path, str, str, Dict[str, Any]]:
    """Securely store an uploaded or ingested audio exhibit in original_evidence storage.

    Parameters
    ----------
    file_bytes_or_path : bytes or Path or str
        Raw binary data or path to source audio file.
    filename : str
        Original filename.
    case_id : str
        Associated forensic case reference ID.
    make_read_only : bool, default=True
        Whether to mark the stored file with read-only permissions.

    Returns
    -------
    Tuple[Path, str, str, Dict[str, Any]]
        stored_path, evidence_id, sha256_hash, metadata
    """
    storage = get_storage_paths()
    clean_name = sanitize_filename(filename)
    unique_prefix = uuid.uuid4().hex[:8]
    dest_path = storage["original_evidence"] / f"{unique_prefix}_{clean_name}"

    if isinstance(file_bytes_or_path, (bytes, bytearray)):
        dest_path.write_bytes(file_bytes_or_path)
    else:
        src = Path(file_bytes_or_path).resolve()
        if not src.is_file():
            raise FileNotFoundError(f"Source file not found for ingestion: {src}")
        shutil.copy2(src, dest_path)

    # Compute immediate cryptographic SHA-256 digest
    sha256_hash = compute_file_sha256(dest_path)
    evidence_id = generate_evidence_id(clean_name, sha256_hash)

    # Extract audio metadata
    meta = get_audio_metadata(dest_path)

    # Apply OS-level read-only file permissions
    if make_read_only:
        apply_file_read_only(dest_path)

    logger.info("Ingested digital evidence exhibit %s -> %s (SHA-256: %s)", evidence_id, dest_path, sha256_hash)
    return dest_path, evidence_id, sha256_hash, meta


def seed_default_cases_and_exhibits(db: ForensicDatabase) -> None:
    """Seed initial forensic cases and exhibits if database is empty."""
    existing_cases = db.list_cases()
    if existing_cases:
        return  # Database already seeded or populated

    root = project_root()

    # Case 1: Active Impersonation Case
    c1 = ForensicCase(
        case_id="CASE-20261110-001",
        title="Voice Conversion Impersonation Investigation",
        description="Investigation into suspected real-time voice conversion mimicking executive personnel.",
        investigator="Special Agent R. Vance (Lead Audio Forensic Analyst)",
        status="ACTIVE",
    )
    db.create_case(c1)

    # Ingest Linus Genuine Demo
    linus_real = root / "dataset" / "real" / "linus-original-DEMO.mp3"
    if linus_real.is_file():
        dest, evid_id, sha, meta = ingest_evidence_file(linus_real, "linus-original-DEMO.mp3", c1.case_id)
        ev1 = ForensicEvidence(
            evidence_id=evid_id,
            case_id=c1.case_id,
            original_filename="linus-original-DEMO.mp3",
            stored_path=str(dest),
            file_size_bytes=meta.get("file_size_bytes", 1024),
            duration_seconds=round(meta.get("duration_seconds", 60.0), 2),
            sample_rate=meta.get("sample_rate", 44100),
            channels=meta.get("channels", 2),
            container_format=meta.get("format", "MP3"),
            sha256_hash=sha,
        )
        db.add_evidence(ev1)

    # Ingest Linus Fake Demo
    linus_fake = root / "dataset" / "fake" / "linus-to-musk-DEMO.mp3"
    if linus_fake.is_file():
        dest, evid_id, sha, meta = ingest_evidence_file(linus_fake, "linus-to-musk-DEMO.mp3", c1.case_id)
        ev2 = ForensicEvidence(
            evidence_id=evid_id,
            case_id=c1.case_id,
            original_filename="linus-to-musk-DEMO.mp3",
            stored_path=str(dest),
            file_size_bytes=meta.get("file_size_bytes", 1024),
            duration_seconds=round(meta.get("duration_seconds", 60.0), 2),
            sample_rate=meta.get("sample_rate", 44100),
            channels=meta.get("channels", 2),
            container_format=meta.get("format", "MP3"),
            sha256_hash=sha,
        )
        db.add_evidence(ev2)

    # Case 2: VIP Audio Verification
    c2 = ForensicCase(
        case_id="CASE-20261110-002",
        title="Public Official Speech Authenticity Screening",
        description="Forensic baseline evaluation of public address audio recordings against synthetic cloning models.",
        investigator="Senior Examiner D. Miller",
        status="ACTIVE",
    )
    db.create_case(c2)

    biden_real = root / "dataset" / "real" / "biden-original.wav"
    if biden_real.is_file():
        dest, evid_id, sha, meta = ingest_evidence_file(biden_real, "biden-original.wav", c2.case_id)
        ev3 = ForensicEvidence(
            evidence_id=evid_id,
            case_id=c2.case_id,
            original_filename="biden-original.wav",
            stored_path=str(dest),
            file_size_bytes=meta.get("file_size_bytes", 1024),
            duration_seconds=round(meta.get("duration_seconds", 300.0), 2),
            sample_rate=meta.get("sample_rate", 44100),
            channels=meta.get("channels", 1),
            container_format=meta.get("format", "WAV"),
            sha256_hash=sha,
        )
        db.add_evidence(ev3)

    logger.info("Successfully seeded default forensic cases and exhibits into database.")

