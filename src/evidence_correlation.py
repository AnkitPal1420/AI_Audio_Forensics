"""Multi-Evidence Correlation & Cross-Exhibit Comparison Module.

Enables forensic investigators to correlate two audio exhibits within a case:
- Cryptographic SHA-256 identity verification.
- Audio container and recording metadata divergence analysis.
- Acoustic spectral feature distance matrix (Euclidean and cosine metrics).
- Temporal alignment and cross-correlation offset search (identifying cropped or shifted excerpts).
- Relational persistence in SQLite evidence_comparisons table.
"""

import json
import logging
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from database.database import ForensicDatabase
from database.models import AnalysisFinding, AnalysisRun, EvidenceComparisonRecord, ForensicEvidence
from src.forensic_integrity import compute_file_sha256
from src.speaker_comparison import compare_acoustic_profiles

logger = logging.getLogger(__name__)


@dataclass
class EvidenceCorrelationReport:
    """Detailed correlation report between two audio exhibits."""

    comparison_id: str
    case_id: str
    evidence_a_id: str
    evidence_b_id: str
    filename_a: str
    filename_b: str
    sha256_match: bool
    metadata_diff: Dict[str, Any]
    feature_distance: Dict[str, Any]
    temporal_alignment: Dict[str, Any]
    correlation_score: float
    findings: List[AnalysisFinding]
    notes: str
    disclaimer: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "comparison_id": self.comparison_id,
            "case_id": self.case_id,
            "evidence_a_id": self.evidence_a_id,
            "evidence_b_id": self.evidence_b_id,
            "filename_a": self.filename_a,
            "filename_b": self.filename_b,
            "sha256_match": self.sha256_match,
            "metadata_diff": self.metadata_diff,
            "feature_distance": self.feature_distance,
            "temporal_alignment": self.temporal_alignment,
            "correlation_score": self.correlation_score,
            "findings": [f.to_dict() for f in self.findings],
            "notes": self.notes,
            "disclaimer": self.disclaimer,
        }


def search_temporal_offset(
    audio_a: np.ndarray,
    audio_b: np.ndarray,
    sample_rate: int,
    max_search_sec: float = 20.0,
) -> Tuple[float, float]:
    """Estimate temporal alignment offset (in seconds) between two audio signals via envelope cross-correlation."""
    downsample = max(1, sample_rate // 1000)
    sr_low = sample_rate / downsample

    max_len = int(max_search_sec * sr_low)
    sig_a = np.abs(audio_a[::downsample][:max_len])
    sig_b = np.abs(audio_b[::downsample][:max_len])

    if len(sig_a) == 0 or len(sig_b) == 0:
        return 0.0, 0.0

    # Cross-correlate
    corr = np.correlate(sig_a - np.mean(sig_a), sig_b - np.mean(sig_b), mode="full")
    norm = (np.linalg.norm(sig_a) * np.linalg.norm(sig_b)) + 1e-9
    norm_corr = corr / norm

    peak_idx = int(np.argmax(norm_corr))
    peak_val = float(norm_corr[peak_idx])
    lag_samples = peak_idx - (len(sig_b) - 1)
    offset_sec = float(lag_samples / sr_low)

    return round(offset_sec, 3), round(peak_val, 4)


def correlate_exhibits(
    evidence_a: ForensicEvidence,
    evidence_b: ForensicEvidence,
    audio_a: Optional[np.ndarray] = None,
    audio_b: Optional[np.ndarray] = None,
    features_a: Optional[np.ndarray] = None,
    features_b: Optional[np.ndarray] = None,
    db: Optional[ForensicDatabase] = None,
) -> EvidenceCorrelationReport:
    """Execute end-to-end multi-evidence correlation analysis."""
    comparison_id = f"CMP-{uuid.uuid4().hex[:8].upper()}"
    findings: List[AnalysisFinding] = []

    # 1. SHA-256 match
    sha_match = evidence_a.sha256_hash.strip().lower() == evidence_b.sha256_hash.strip().lower()

    if sha_match:
        findings.append(
            AnalysisFinding(
                finding_id=f"FIND-SHA-{uuid.uuid4().hex[:8].upper()}",
                run_id=comparison_id,
                evidence_id=evidence_a.evidence_id,
                start_sec=0.0,
                end_sec=0.0,
                indicator="CRYPTOGRAPHIC_DUPLICATE",
                value=1.0,
                unit="boolean",
                threshold=1.0,
                severity="HIGH",
                interpretation_flag="MEASUREMENT",
                description=(
                    f"Exhibit {evidence_a.evidence_id} and {evidence_b.evidence_id} share an "
                    "identical SHA-256 hash. The digital audio bitstreams are byte-for-byte identical."
                ),
            )
        )

    # 2. Metadata differences
    meta_diff = {
        "duration_delta_sec": round(abs(evidence_a.duration_seconds - evidence_b.duration_seconds), 3),
        "same_sample_rate": evidence_a.sample_rate == evidence_b.sample_rate,
        "sample_rate_a": evidence_a.sample_rate,
        "sample_rate_b": evidence_b.sample_rate,
        "same_channels": evidence_a.channels == evidence_b.channels,
        "channels_a": evidence_a.channels,
        "channels_b": evidence_b.channels,
        "same_container": evidence_a.container_format.lower() == evidence_b.container_format.lower(),
        "container_a": evidence_a.container_format,
        "container_b": evidence_b.container_format,
    }

    # 3. Acoustic feature distance (if features available)
    feature_dist: Dict[str, Any] = {}
    if features_a is not None and features_b is not None:
        spk_res = compare_acoustic_profiles(
            features_a, features_b, evidence_a.evidence_id, evidence_b.evidence_id, comparison_id
        )
        feature_dist = {
            "cosine_similarity": spk_res.cosine_similarity,
            "euclidean_distance": spk_res.euclidean_distance,
            "divergence_pct": spk_res.acoustic_divergence_pct,
            "band_similarity": spk_res.band_similarity,
        }
        findings.extend(spk_res.findings)

    # 4. Temporal Alignment (if audio signals available)
    temporal_align: Dict[str, Any] = {"offset_seconds": 0.0, "peak_envelope_correlation": 0.0}
    if audio_a is not None and audio_b is not None:
        offset, peak_corr = search_temporal_offset(audio_a, audio_b, evidence_a.sample_rate)
        temporal_align = {
            "offset_seconds": offset,
            "peak_envelope_correlation": peak_corr,
        }
        if peak_corr > 0.85 and not sha_match:
            findings.append(
                AnalysisFinding(
                    finding_id=f"FIND-TIME-{uuid.uuid4().hex[:8].upper()}",
                    run_id=comparison_id,
                    evidence_id=evidence_a.evidence_id,
                    start_sec=0.0,
                    end_sec=0.0,
                    indicator="TEMPORAL_ALIGNMENT_MATCH",
                    value=peak_corr,
                    unit="correlation_coefficient",
                    threshold=0.85,
                    severity="HIGH",
                    interpretation_flag="MEASUREMENT",
                    description=(
                        f"Envelope cross-correlation peak of {peak_corr:.3f} at temporal offset {offset:+.3f}s. "
                        "Strong evidence that one recording is a trimmed, shifted, or re-recorded version of the other."
                    ),
                )
            )

    # Overall correlation score (0.0 to 1.0)
    if sha_match:
        overall_corr = 1.0
    else:
        cos = feature_dist.get("cosine_similarity", 0.5)
        temp = temporal_align.get("peak_envelope_correlation", 0.0)
        overall_corr = round(float(0.5 * max(0.0, cos) + 0.5 * max(0.0, temp)), 4)

    notes = (
        f"Correlation analysis between Exhibit {evidence_a.evidence_id} and {evidence_b.evidence_id}. "
        f"SHA-256 match: {sha_match}. Acoustic similarity: {feature_dist.get('cosine_similarity', 'N/A')}."
    )
    disclaimer = (
        "FORENSIC NOTICE: Multi-evidence correlation assesses digital identity and acoustic similarity. "
        "High acoustic correlation demonstrates shared spectral or temporal characteristics, but does NOT "
        "prove individual speaker identity or verify recording chain custody without forensic provenance."
    )

    report = EvidenceCorrelationReport(
        comparison_id=comparison_id,
        case_id=evidence_a.case_id,
        evidence_a_id=evidence_a.evidence_id,
        evidence_b_id=evidence_b.evidence_id,
        filename_a=evidence_a.original_filename,
        filename_b=evidence_b.original_filename,
        sha256_match=sha_match,
        metadata_diff=meta_diff,
        feature_distance=feature_dist,
        temporal_alignment=temporal_align,
        correlation_score=overall_corr,
        findings=findings,
        notes=notes,
        disclaimer=disclaimer,
    )

    # Persist to database if db manager provided
    if db is not None:
        # Create an AnalysisRun record so findings satisfy the foreign key constraint
        run_record = AnalysisRun(
            run_id=comparison_id,
            case_id=evidence_a.case_id,
            evidence_id=evidence_a.evidence_id,
            analysis_type="EVIDENCE_CORRELATION",
            algorithm_name="correlate_exhibits",
            algorithm_version="1.0.0",
            software_versions_json=json.dumps({"numpy": np.__version__}),
            parameters_json=json.dumps({"evidence_b_id": evidence_b.evidence_id}),
            input_sha256=evidence_a.sha256_hash,
            status="COMPLETED",
            warnings_json=json.dumps([]),
            results_json=json.dumps({
                "sha256_match": sha_match,
                "correlation_score": overall_corr,
                "evidence_b_id": evidence_b.evidence_id,
            }),
            output_files_json=json.dumps([]),
        )
        db.create_analysis_run(run_record)

        rec = EvidenceComparisonRecord(
            comparison_id=comparison_id,
            case_id=evidence_a.case_id,
            evidence_a_id=evidence_a.evidence_id,
            evidence_b_id=evidence_b.evidence_id,
            sha_equal=sha_match,
            metadata_diff_json=json.dumps(meta_diff),
            feature_distance_json=json.dumps(feature_dist),
            referenced_run_ids_json=json.dumps([f.finding_id for f in findings]),
            notes=notes,
        )
        db.create_evidence_comparison(rec)
        db.create_analysis_findings_batch(findings)

    return report
