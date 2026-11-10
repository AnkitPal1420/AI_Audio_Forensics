"""Storage management and evidence preservation layer.

Ensures strict segregation of digital assets adhering to ISO/IEC 27037 principles:
- storage/original_evidence/ : Bitstream-exact immutable exhibits (read-only)
- storage/processed_audio/   : Normalized 16 kHz mono derivatives
- storage/visualizations/    : Forensic spectrograms, waveforms, heatmaps
- storage/reports/           : Formal Markdown and PDF examination reports
"""

from __future__ import annotations

import logging
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from database.database import ForensicDatabase
from database.models import ForensicCase, ForensicEvidence
from src.audio_preprocessing import compute_file_hash, get_audio_metadata
from src.forensic_integrity import compute_file_sha256, generate_evidence_id
from src.utils import project_root

logger = logging.getLogger(__name__)


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
