"""Forensic Backup, Archive Verification & Disaster Recovery Module.

Implements court-grade disaster recovery procedures:
- Non-blocking online hot backup using SQLite Online Backup API (sqlite3.Connection.backup).
- Comprehensive bundling of database, evidence manifests, configuration, model checksums, and reports.
- Automated cryptographic SHA-256 fingerprinting of all backup archives.
- Automated restoration engine with PRAGMA integrity_check verification.
"""

from __future__ import annotations

import hashlib
import json
import logging
import secrets
import shutil
import sqlite3
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from database.database import ForensicDatabase
from src.utils import project_root

logger = logging.getLogger(__name__)


class BackupError(RuntimeError):
    """Raised when backup creation or restoration fails."""
    pass


class RestoreError(RuntimeError):
    """Raised when backup restoration fails integrity validation."""
    pass


def get_default_backup_dir() -> Path:
    """Return default backup directory in database/backups/."""
    root = project_root()
    b_dir = root / "database" / "backups"
    b_dir.mkdir(parents=True, exist_ok=True)
    return b_dir


def create_hot_database_backup(db_path: Path, target_path: Path) -> str:
    """Perform hot online SQLite backup using native sqlite3.Connection.backup() API."""
    target_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as src_conn:
        with sqlite3.connect(target_path) as dst_conn:
            # Use SQLite online backup API
            src_conn.backup(dst_conn, pages=100)

    # Compute SHA-256 of the created snapshot
    hasher = hashlib.sha256()
    with target_path.open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest().lower()


def create_system_backup(
    db: ForensicDatabase,
    output_dir: Optional[Union[str, Path]] = None,
    include_reports: bool = True,
    operator: str = "AutomatedBackupEngine",
) -> Dict[str, Any]:
    """Create a complete, versioned forensic backup bundle with cryptographic verification."""
    root = project_root()
    dest_dir = Path(output_dir) if output_dir else get_default_backup_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_stem = f"forensic_backup_{timestamp}"
    temp_stage = dest_dir / f"stage_{timestamp}"
    temp_stage.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Hot SQLite Database Backup
        stage_db = temp_stage / "forensics.db"
        db_hash = create_hot_database_backup(db.db_path, stage_db)

        # 2. Manifests & Config Backup
        manifest_files = [
            root / "config" / "config.json",
            root / "models" / "model_manifest.json",
            root / "models" / "model_metadata.json",
            root / "dataset" / "evidence_manifest.json",
        ]
        manifest_hashes = {}
        stage_meta = temp_stage / "metadata"
        stage_meta.mkdir(parents=True, exist_ok=True)

        for mf in manifest_files:
            if mf.is_file():
                shutil.copy2(mf, stage_meta / mf.name)
                h = hashlib.sha256(mf.read_bytes()).hexdigest().lower()
                manifest_hashes[mf.name] = h

        # 3. Reports Backup
        if include_reports:
            rep_dir = root / "storage" / "reports"
            if rep_dir.is_dir():
                stage_rep = temp_stage / "reports"
                stage_rep.mkdir(parents=True, exist_ok=True)
                for rf in rep_dir.glob("*.md"):
                    shutil.copy2(rf, stage_rep / rf.name)
                for rf in rep_dir.glob("*.pdf"):
                    shutil.copy2(rf, stage_rep / rf.name)

        # 4. Generate Backup Inventory Metadata
        inventory = {
            "backup_version": "8.0.0",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "operator": operator,
            "database_sha256": db_hash,
            "manifest_hashes": manifest_hashes,
            "source_db": str(db.db_path),
        }
        inv_path = temp_stage / "backup_manifest.json"
        with inv_path.open("w", encoding="utf-8") as f:
            json.dump(inventory, f, indent=2)

        # 5. Compress into ZIP archive
        zip_path = dest_dir / f"{backup_stem}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for item in temp_stage.rglob("*"):
                if item.is_file():
                    arcname = item.relative_to(temp_stage)
                    zf.write(item, arcname)

        # 6. Compute Archive SHA-256
        archive_hash = hashlib.sha256(zip_path.read_bytes()).hexdigest().lower()
        hash_file = dest_dir / f"{backup_stem}.sha256"
        hash_file.write_text(f"{archive_hash}  {zip_path.name}\n", encoding="utf-8")

        # Log audit event in database
        db.log_audit_event(
            "BACKUP_CREATED",
            {
                "backup_file": zip_path.name,
                "archive_sha256": archive_hash,
                "database_sha256": db_hash,
                "operator": operator,
            },
        )

        logger.info("Successfully created system backup: %s (SHA-256: %s)", zip_path.name, archive_hash)
        return {
            "status": "SUCCESS",
            "archive_path": str(zip_path),
            "archive_filename": zip_path.name,
            "archive_sha256": archive_hash,
            "database_sha256": db_hash,
            "timestamp_utc": inventory["timestamp_utc"],
        }

    finally:
        # Cleanup temporary staging folder
        if temp_stage.is_dir():
            shutil.rmtree(temp_stage, ignore_errors=True)


def restore_system_backup(
    backup_zip_path: Union[str, Path],
    target_db_path: Optional[Union[str, Path]] = None,
    verify_archive_hash: bool = True,
) -> Dict[str, Any]:
    """Safely restore database and metadata from a verified backup archive."""
    zip_path = Path(backup_zip_path).resolve()
    if not zip_path.is_file():
        raise FileNotFoundError(f"Backup archive not found at: {zip_path}")

    # Verify SHA-256 checksum if sidecar .sha256 file exists
    hash_file = zip_path.with_suffix(".sha256")
    actual_archive_hash = hashlib.sha256(zip_path.read_bytes()).hexdigest().lower()
    if verify_archive_hash and hash_file.is_file():
        expected_hash = hash_file.read_text(encoding="utf-8").split()[0].strip().lower()
        if actual_archive_hash != expected_hash:
            raise RestoreError(
                f"BACKUP CORRUPTION: Archive SHA-256 mismatch! Expected {expected_hash}, got {actual_archive_hash}"
            )

    temp_extract = zip_path.parent / f"restore_stage_{secrets.token_hex(4)}"
    temp_extract.mkdir(parents=True, exist_ok=True)

    try:
        # Extract archive
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(temp_extract)

        # Validate extracted database
        extracted_db = temp_extract / "forensics.db"
        if not extracted_db.is_file():
            raise RestoreError("Invalid backup archive: forensics.db missing from archive root.")

        # Read manifest
        inv_file = temp_extract / "backup_manifest.json"
        manifest_data = {}
        if inv_file.is_file():
            with inv_file.open("r", encoding="utf-8") as f:
                manifest_data = json.load(f)

        # Verify DB hash
        extracted_db_hash = hashlib.sha256(extracted_db.read_bytes()).hexdigest().lower()
        expected_db_hash = manifest_data.get("database_sha256")
        if expected_db_hash and extracted_db_hash != expected_db_hash.lower():
            raise RestoreError("Database SHA-256 within archive does not match backup manifest!")

        # Perform SQLite PRAGMA integrity_check on extracted DB
        with sqlite3.connect(extracted_db) as conn:
            row = conn.execute("PRAGMA integrity_check;").fetchone()
            if not row or row[0].lower() != "ok":
                raise RestoreError(f"Database integrity check failed: {row[0] if row else 'Unknown error'}")

        # If target DB specified, restore to it using SQLite hot backup
        target_path = Path(target_db_path).resolve() if target_db_path else project_root() / "database" / "forensics.db"
        create_hot_database_backup(extracted_db, target_path)

        logger.info("Successfully restored forensic database to %s", target_path)
        return {
            "status": "RESTORE_SUCCESS",
            "restored_db_path": str(target_path),
            "restored_db_sha256": extracted_db_hash,
            "manifest": manifest_data,
        }

    finally:
        if temp_extract.is_dir():
            shutil.rmtree(temp_extract, ignore_errors=True)
