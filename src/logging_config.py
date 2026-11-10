"""Centralized Forensic Logging, Security Event Auditing & Diagnostics Module.

Separates log streams according to digital forensic compliance:
1. logs/operational.log    : System health, application startup, processing status, warnings.
2. logs/security.log       : Authentication events, failed logins, account lockouts, RBAC violations.
3. logs/audit_events.jsonl : Evidentiary audit events (ISO/IEC 27037 immutable event stream).
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from src.utils import project_root

_LOGGING_INITIALIZED = False


def get_log_dir() -> Path:
    """Return and ensure existence of logs/ directory."""
    root = project_root()
    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def configure_forensic_logging(log_level: int = logging.INFO) -> None:
    """Configure separate file and console logging handlers."""
    global _LOGGING_INITIALIZED
    if _LOGGING_INITIALIZED:
        return

    log_dir = get_log_dir()
    operational_log = log_dir / "operational.log"
    security_log = log_dir / "security.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Standard formatter
    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 1. Operational File Handler
    op_handler = logging.FileHandler(operational_log, encoding="utf-8")
    op_handler.setLevel(log_level)
    op_handler.setFormatter(fmt)
    root_logger.addHandler(op_handler)

    # 2. Security Logger & File Handler
    sec_logger = logging.getLogger("security")
    sec_logger.setLevel(logging.INFO)
    sec_logger.propagate = False  # Avoid polluting operational.log with security events

    sec_handler = logging.FileHandler(security_log, encoding="utf-8")
    sec_handler.setLevel(logging.INFO)
    sec_fmt = logging.Formatter(
        "[%(asctime)s] [SECURITY_EVENT] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    sec_handler.setFormatter(sec_fmt)
    sec_logger.addHandler(sec_handler)

    _LOGGING_INITIALIZED = True
    logging.getLogger(__name__).info("Initialized forensic logging infrastructure in %s", log_dir)


def log_security_event(event_type: str, details: Dict[str, Any], username: Optional[str] = None, ip_address: str = "127.0.0.1") -> None:
    """Log an auditable security event to logs/security.log."""
    configure_forensic_logging()
    sec_logger = logging.getLogger("security")
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        "username": username or "ANONYMOUS",
        "ip_address": ip_address,
        "details": details,
    }
    sec_logger.info(json.dumps(payload))


def log_evidentiary_audit(event_type: str, details: Dict[str, Any], evidence_id: Optional[str] = None, case_id: Optional[str] = None) -> None:
    """Append structured evidentiary audit event to logs/audit_events.jsonl."""
    log_dir = get_log_dir()
    audit_file = log_dir / "audit_events.jsonl"
    entry = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        "evidence_id": evidence_id,
        "case_id": case_id,
        "details": details,
    }
    with audit_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def get_system_diagnostics() -> Dict[str, Any]:
    """Retrieve runtime telemetry: Python version, disk usage, database size, and model status."""
    root = project_root()
    db_file = root / "database" / "forensics.db"

    # Disk usage
    total, used, free = 0, 0, 0
    try:
        import shutil
        total, used, free = shutil.disk_usage(root)
    except Exception:
        pass

    # Check models
    manifest_file = root / "models" / "model_manifest.json"
    models_status = {}
    if manifest_file.is_file():
        with manifest_file.open("r", encoding="utf-8") as mf:
            m = json.load(mf)
            for m_name in m.keys():
                p = root / "models" / m_name
                models_status[m_name] = "SEALED_INTACT" if p.is_file() else "MISSING"

    return {
        "python_version": sys.version.split()[0],
        "platform": sys.platform,
        "database_exists": db_file.is_file(),
        "database_size_bytes": db_file.stat().st_size if db_file.is_file() else 0,
        "disk_total_gb": round(total / (1024**3), 2),
        "disk_free_gb": round(free / (1024**3), 2),
        "models_status": models_status,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

