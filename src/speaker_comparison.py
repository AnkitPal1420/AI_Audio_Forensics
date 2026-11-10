"""Speaker Acoustic Consistency & Comparison Baseline.

Forensic Rule & Scientific Constraint:
Biometric speaker verification and identity confirmation are strictly UNAVAILABLE
because deep speaker embedding networks (e.g. ECAPA-TDNN, x-vectors via SpeechBrain/PyTorch)
are not present in this runtime, and the training dataset lacks cross-channel speaker identity labels.

This module provides an exploratory acoustic feature distance baseline between two audio
recordings (MFCC profile cosine similarity and Euclidean distance).

ALL results are marked EXPERIMENTAL_INDETERMINATE and explicitly prohibit speaker identity claims.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from database.models import AnalysisFinding

logger = logging.getLogger(__name__)


@dataclass
class SpeakerComparisonResult:
    """Container for pairwise acoustic speaker comparison."""

    comparison_id: str
    evidence_a_id: str
    evidence_b_id: str
    cosine_similarity: float
    euclidean_distance: float
    acoustic_divergence_pct: float
    band_similarity: Dict[str, float]
    status: str  # Always EXPERIMENTAL_INDETERMINATE
    identity_claim_permitted: bool  # False
    findings: List[AnalysisFinding]
    prerequisites_missing: List[str]
    scientific_disclaimer: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "comparison_id": self.comparison_id,
            "evidence_a_id": self.evidence_a_id,
            "evidence_b_id": self.evidence_b_id,
            "cosine_similarity": self.cosine_similarity,
            "euclidean_distance": self.euclidean_distance,
            "acoustic_divergence_pct": self.acoustic_divergence_pct,
            "band_similarity": self.band_similarity,
            "status": self.status,
            "identity_claim_permitted": self.identity_claim_permitted,
            "findings": [f.to_dict() for f in self.findings],
            "prerequisites_missing": self.prerequisites_missing,
            "scientific_disclaimer": self.scientific_disclaimer,
        }


def compare_acoustic_profiles(
    features_a: np.ndarray,
    features_b: np.ndarray,
    evidence_a_id: str,
    evidence_b_id: str,
    comparison_id: Optional[str] = None,
) -> SpeakerComparisonResult:
    """Compare two 26-dimensional acoustic feature vectors.
    
    Args:
        features_a: shape (26,) or (1, 26) float32 vector.
        features_b: shape (26,) or (1, 26) float32 vector.
        evidence_a_id: identifier for evidence A.
        evidence_b_id: identifier for evidence B.
        comparison_id: optional comparison ID.
    """
    if comparison_id is None:
        comparison_id = f"CMP-SPK-{uuid.uuid4().hex[:8].upper()}"

    vec_a = np.asarray(features_a, dtype=np.float32).reshape(-1)
    vec_b = np.asarray(features_b, dtype=np.float32).reshape(-1)

    if len(vec_a) != len(vec_b):
        raise ValueError(f"Feature dimension mismatch: {len(vec_a)} vs {len(vec_b)}")

    # 1. Cosine Similarity
    norm_a = np.linalg.norm(vec_a) + 1e-9
    norm_b = np.linalg.norm(vec_b) + 1e-9
    cos_sim = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
    cos_sim = max(-1.0, min(1.0, cos_sim))

    # 2. Euclidean Distance
    euc_dist = float(np.linalg.norm(vec_a - vec_b))

    # 3. Acoustic divergence percentage
    max_possible_dist = norm_a + norm_b
    divergence_pct = float(min(100.0, (euc_dist / max_possible_dist) * 100.0))

    # 4. Band-wise similarity breakdown (Low MFCCs 0-3, Mid 4-7, High 8-12)
    # Assumes first 13 elements are means, next 13 are stds
    n_mfcc = len(vec_a) // 2
    band_sim = {}
    if n_mfcc >= 12:
        low_a, low_b = vec_a[:4], vec_b[:4]
        mid_a, mid_b = vec_a[4:8], vec_b[4:8]
        high_a, high_b = vec_a[8:13], vec_b[8:13]

        def _sub_sim(v1: np.ndarray, v2: np.ndarray) -> float:
            n1 = np.linalg.norm(v1) + 1e-9
            n2 = np.linalg.norm(v2) + 1e-9
            return float(np.dot(v1, v2) / (n1 * n2))

        band_sim["low_frequency_profile"] = round(_sub_sim(low_a, low_b), 3)
        band_sim["mid_frequency_profile"] = round(_sub_sim(mid_a, mid_b), 3)
        band_sim["high_frequency_profile"] = round(_sub_sim(high_a, high_b), 3)

    findings: List[AnalysisFinding] = []
    severity = "LOW" if cos_sim > 0.85 else ("MEDIUM" if cos_sim > 0.6 else "HIGH")
    findings.append(
        AnalysisFinding(
            finding_id=f"FIND-SPK-{uuid.uuid4().hex[:8].upper()}",
            run_id=comparison_id,
            evidence_id=evidence_a_id,
            start_sec=0.0,
            end_sec=0.0,
            indicator="ACOUSTIC_PROFILE_COSINE_SIMILARITY",
            value=round(cos_sim, 4),
            unit="cosine_score",
            threshold=0.85,
            severity=severity,
            interpretation_flag="INTERPRETATION",
            description=(
                f"Acoustic spectral distance between Exhibit {evidence_a_id} and Exhibit {evidence_b_id} "
                f"yields a cosine similarity of {cos_sim:.3f} and Euclidean distance of {euc_dist:.2f}. "
                "This indicates overall spectral resemblance but CANNOT determine speaker identity."
            ),
        )
    )

    prerequisites = [
        "Pre-trained deep speaker embedding architecture (e.g., ECAPA-TDNN, VoxCeleb-tuned x-vectors).",
        "Deep learning inference runtime (PyTorch / ONNX Runtime).",
        "Enrollment database with verified speaker reference recordings.",
        "Calibrated forensic likelihood ratio (LR) scoring framework with channel compensation.",
    ]

    disclaimer = (
        "CRITICAL FORENSIC LIMITATION: True biometric speaker verification is UNAVAILABLE in this system. "
        "The computed scores reflect acoustic spectrum similarity, which is influenced by recording microphone, "
        "room reverberation, language, and background noise. It is scientifically invalid and legally impermissible "
        "to assert that two recordings share the same speaker identity based on these acoustic metrics."
    )

    return SpeakerComparisonResult(
        comparison_id=comparison_id,
        evidence_a_id=evidence_a_id,
        evidence_b_id=evidence_b_id,
        cosine_similarity=round(cos_sim, 4),
        euclidean_distance=round(euc_dist, 4),
        acoustic_divergence_pct=round(divergence_pct, 2),
        band_similarity=band_sim,
        status="EXPERIMENTAL_INDETERMINATE",
        identity_claim_permitted=False,
        findings=findings,
        prerequisites_missing=prerequisites,
        scientific_disclaimer=disclaimer,
    )

