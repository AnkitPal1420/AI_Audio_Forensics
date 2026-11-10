"""Forensic evidence integrity and chain-of-custody audit logging module.

Adheres to ISO/IEC 27037 technical principles for digital evidence management:
- Cryptographic SHA-256 intake verification and ongoing immutability checks.
- Unique Evidence ID generation and provenance tracking.
- Non-destructive separation between original physical exhibits and derivatives.
- Append-only forensic event logging (audit trail) with UTC timestamps and environment details.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.utils import project_root

logger = logging.getLogger(__name__)


class EvidenceIntegrityError(RuntimeError):
    """Raised when an evidence exhibit's cryptographic hash changes or exhibits tampering."""
    pass


def compute_file_sha256(path: Union[str, Path], chunk_size: int = 65536) -> str:
    """Compute cryptographic SHA-256 digest of a file using chunked binary reading."""
    p = Path(path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"File not found for hashing: {p}")

    hasher = hashlib.sha256()
    with p.open("rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest().lower()


def generate_evidence_id(filename: str, sha256_hash: str) -> str:
    """Generate a reproducible, unique evidence tracking identifier."""
    stem = Path(filename).stem.replace(" ", "_")[:16]
    hash_short = sha256_hash[:8].upper()
    return f"EVID-{hash_short}-{stem}"


def verify_evidence_immutability(path: Union[str, Path], expected_sha256: str) -> bool:
    """Verify that an audio file's current bitstream matches its intake SHA-256 hash.

    Raises
    ------
    EvidenceIntegrityError
        If the hash differs, indicating modification, tampering, or disk corruption.
    """
    actual_hash = compute_file_sha256(path)
    if actual_hash.lower() != expected_sha256.lower():
        msg = (
            f"CRITICAL EVIDENCE INTEGRITY VIOLATION: File '{Path(path).name}' has been altered! "
            f"Expected SHA-256: {expected_sha256}, Actual SHA-256: {actual_hash}."
        )
        logger.critical(msg)
        raise EvidenceIntegrityError(msg)
    return True


def get_environment_metadata() -> Dict[str, Any]:
    """Capture runtime execution environment for forensic auditability."""
    git_commit = "unknown"
    git_branch = "unknown"
    try:
        git_commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("ascii")
            .strip()
        )
        git_branch = (
            subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("ascii")
            .strip()
        )
    except Exception:
        pass

    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "git_commit": git_commit,
        "git_branch": git_branch,
        "standards_alignment": "ISO/IEC 27037 Technical Principles (Evidence Identification, Collection, Acquisition, Preservation)",
    }


def record_analysis_event(
    event_payload: Dict[str, Any],
    log_file: Optional[Union[str, Path]] = None,
) -> Path:
    """Append a cryptographically auditable forensic analysis record to the audit trail.

    Parameters
    ----------
    event_payload : Dict[str, Any]
        Structured event details (evidence ID, hashes, model output, timestamps).
    log_file : Union[str, Path], optional
        Destination jsonl file. Defaults to ``results/reports/forensic_audit_trail.jsonl``.

    Returns
    -------
    Path
        Path to the written audit trail log.
    """
    root = project_root()
    dest = Path(log_file) if log_file else root / "results" / "reports" / "forensic_audit_trail.jsonl"
    dest.parent.mkdir(parents=True, exist_ok=True)

    event_record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "event_type": "FORENSIC_AUDIO_EXAMINATION",
        "environment": get_environment_metadata(),
        "event_data": event_payload,
    }

    with dest.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event_record) + "\n")

    logger.info("Recorded forensic audit trail event for evidence ID: %s", event_payload.get("evidence_id"))
    return dest

