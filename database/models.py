"""Dataclasses and schemas for forensic database entities."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class ForensicCase:
    case_id: str
    title: str
    description: str = ""
    investigator: str = "Lead Forensic Examiner"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    status: str = "ACTIVE"  # ACTIVE, PENDING_REVIEW, CLOSED, ARCHIVED

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ForensicEvidence:
    evidence_id: str
    case_id: str
    original_filename: str
    stored_path: str
    file_size_bytes: int
    duration_seconds: float
    sample_rate: int
    channels: int
    container_format: str
    sha256_hash: str
    processed_path: Optional[str] = None
    uploaded_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ForensicExamination:
    examination_id: str
    evidence_id: str
    case_id: str
    model_name: str
    predicted_label: str  # GENUINE, SYNTHETIC, INDETERMINATE
    confidence: float
    status: str  # COMPLETED, QUALITY_CHECK_FAILED, OUT_OF_DISTRIBUTION, MODEL_DISAGREEMENT
    forensic_assessment: str
    prob_synthetic: Optional[float] = None
    prob_genuine: Optional[float] = None
    decision_score: Optional[float] = None
    estimated_snr_db: Optional[float] = None
    clipping_ratio: Optional[float] = None
    silence_ratio: Optional[float] = None
    is_in_distribution: bool = True
    max_z_score: Optional[float] = None
    multi_model_consensus: Optional[str] = None
    overview_figure_path: Optional[str] = None
    report_md_path: Optional[str] = None
    report_pdf_path: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class IntegrityRecord:
    evidence_id: str
    recorded_sha256: str
    checked_sha256: str
    status: str  # VERIFIED, HASH_MISMATCH, NOT_VERIFIED
    verified_by: str = "Automated Integrity Monitor"
    notes: str = ""
    checked_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

