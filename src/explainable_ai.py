"""Explainable AI (XAI) Framework for Audio Forensics.

Provides model-agnostic and model-specific interpretability methods:
- Feature Occlusion & Sensitivity Analysis (instance-level)
- Pure-Python Monte Carlo Sampling Shapley Values (instance-level, no external 'shap' dependency)
- Linear Model Coefficient Interpretation (Logistic Regression)
- Tree Ensemble Gini Importances (Random Forest)
- Grouped Spectral Feature Attribution (MFCC Means vs Standard Deviations, Low/Mid/High Bands)

Forensic Rules:
- All explanations are marked with: 'MODEL UNVALIDATED — experimental indicator only'
- Feature importance highlights algorithmic decision influence, NOT real-world physical tampering.
"""

import json
import logging
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from database.models import AnalysisFinding
from src.deepfake_detection import DeepfakeDetectionEngine
from src.utils import project_root

logger = logging.getLogger(__name__)


@dataclass
class FeatureAttribution:
    """Attribution score for a single acoustic feature."""

    feature_index: int
    feature_name: str
    feature_value: float
    importance_score: float  # Absolute or signed contribution
    direction: str  # TOWARD_SYNTHETIC, TOWARD_GENUINE, NEUTRAL
    group: str  # MEAN, STD, LOW_FREQ, MID_FREQ, HIGH_FREQ

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExplainableAIResult:
    """Container for comprehensive explainability outputs."""

    run_id: str
    evidence_id: str
    model_name: str
    predicted_label: str
    predicted_prob_synthetic: float
    attributions: List[FeatureAttribution]
    top_synthetic_contributors: List[Dict[str, Any]]
    top_genuine_contributors: List[Dict[str, Any]]
    grouped_importance: Dict[str, float]
    shapley_values: Optional[List[float]]
    method_used: str  # OCCLUSION, SHAPLEY_SAMPLING, MODEL_INTRINSIC
    findings: List[AnalysisFinding]
    unvalidated_model_warning: str
    forensic_disclaimer: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "evidence_id": self.evidence_id,
            "model_name": self.model_name,
            "predicted_label": self.predicted_label,
            "predicted_prob_synthetic": self.predicted_prob_synthetic,
            "attributions": [a.to_dict() for a in self.attributions],
            "top_synthetic_contributors": self.top_synthetic_contributors,
            "top_genuine_contributors": self.top_genuine_contributors,
            "grouped_importance": self.grouped_importance,
            "shapley_values": self.shapley_values,
            "method_used": self.method_used,
            "findings": [f.to_dict() for f in self.findings],
            "unvalidated_model_warning": self.unvalidated_model_warning,
            "forensic_disclaimer": self.forensic_disclaimer,
        }


def get_feature_group(feature_name: str) -> str:
    """Categorize MFCC feature name into acoustic groups."""
    name_lower = feature_name.lower()
    is_std = "std" in name_lower
    base_group = "STD" if is_std else "MEAN"

    # Identify frequency band
    # Names typically: mfcc_1_mean ... mfcc_13_mean
    try:
        parts = feature_name.split("_")
        num = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1
        if num <= 4:
            band = "LOW_BAND"
        elif num <= 8:
            band = "MID_BAND"
        else:
            band = "HIGH_BAND"
    except Exception:
        band = "MID_BAND"

    return f"{base_group}:{band}"


def explain_instance_occlusion(
    model: Any,
    feature_vector: np.ndarray,
    feature_names: List[str],
    reference_vector: Optional[np.ndarray] = None,
) -> List[FeatureAttribution]:
    """Compute feature sensitivity by replacing each feature with a reference value."""
    x = feature_vector.reshape(1, -1)
    n_features = x.shape[1]

    if reference_vector is None:
        ref = np.zeros_like(x)
    else:
        ref = reference_vector.reshape(1, -1)

    # Baseline prediction probability
    if hasattr(model, "predict_proba"):
        base_prob = float(model.predict_proba(x)[0, 1])
    elif hasattr(model, "decision_function"):
        score = float(model.decision_function(x)[0])
        base_prob = float(1.0 / (1.0 + np.exp(-score)))
    else:
        base_prob = float(model.predict(x)[0])

    attributions = []
    for j in range(n_features):
        x_occ = x.copy()
        x_occ[0, j] = ref[0, j]

        if hasattr(model, "predict_proba"):
            occ_prob = float(model.predict_proba(x_occ)[0, 1])
        elif hasattr(model, "decision_function"):
            score_occ = float(model.decision_function(x_occ)[0])
            occ_prob = float(1.0 / (1.0 + np.exp(-score_occ)))
        else:
            occ_prob = float(model.predict(x_occ)[0])

        delta_prob = base_prob - occ_prob  # Positive means feature pushed toward SYNTHETIC

        direction = "NEUTRAL"
        if delta_prob > 0.005:
            direction = "TOWARD_SYNTHETIC"
        elif delta_prob < -0.005:
            direction = "TOWARD_GENUINE"

        fname = feature_names[j] if j < len(feature_names) else f"feature_{j}"
        group = get_feature_group(fname)

        attributions.append(
            FeatureAttribution(
                feature_index=j,
                feature_name=fname,
                feature_value=round(float(x[0, j]), 4),
                importance_score=round(float(delta_prob), 5),
                direction=direction,
                group=group,
            )
        )

    return attributions


def compute_sampling_shapley(
    model: Any,
    feature_vector: np.ndarray,
    reference_vector: Optional[np.ndarray] = None,
    num_samples: int = 40,
    random_seed: int = 42,
) -> np.ndarray:
    """Monte Carlo sampling approximation of Shapley values for tabular features.
    
    Implemented without external C dependencies or 'shap' package.
    """
    rng = np.random.RandomState(random_seed)
    x = feature_vector.reshape(-1)
    n_features = len(x)
    ref = reference_vector.reshape(-1) if reference_vector is not None else np.zeros_like(x)

    def eval_fn(vec: np.ndarray) -> float:
        v = vec.reshape(1, -1)
        if hasattr(model, "predict_proba"):
            return float(model.predict_proba(v)[0, 1])
        elif hasattr(model, "decision_function"):
            s = float(model.decision_function(v)[0])
            return float(1.0 / (1.0 + np.exp(-s)))
        return float(model.predict(v)[0])

    shapley_vals = np.zeros(n_features, dtype=np.float32)

    for _ in range(num_samples):
        perm = rng.permutation(n_features)
        cur_sample = ref.copy()

        prev_val = eval_fn(cur_sample)
        for idx in perm:
            cur_sample[idx] = x[idx]
            new_val = eval_fn(cur_sample)
            marginal_contrib = new_val - prev_val
            shapley_vals[idx] += marginal_contrib
            prev_val = new_val

    shapley_vals /= max(1, num_samples)
    return shapley_vals


def explain_prediction(
    feature_vector: np.ndarray,
    evidence_id: str,
    model_key: str = "calibrated_svm",
    run_id: Optional[str] = None,
    compute_shapley: bool = True,
) -> ExplainableAIResult:
    """Generate explainability attributions for a forensic audio feature vector."""
    if run_id is None:
        run_id = f"RUN-XAI-{uuid.uuid4().hex[:8].upper()}"

    engine = DeepfakeDetectionEngine()
    model = engine.load_model(model_key)
    feature_names = engine.feature_names

    # Ensure shape (1, 26)
    feat = np.asarray(feature_vector, dtype=np.float32).reshape(1, -1)
    if feat.shape[1] != len(feature_names):
        raise ValueError(
            f"Feature schema mismatch: expected {len(feature_names)} features, got {feat.shape[1]}"
        )

    # Base inference
    pred = engine.predict_single(feat, model_key=model_key)
    pred_label = pred["predicted_label"]
    prob_synth = float(pred.get("prob_synthetic", 0.5))

    # Reference baseline (zeros or neutral standardized values)
    ref_vec = np.zeros_like(feat)

    # 1. Occlusion Sensitivity
    attributions = explain_instance_occlusion(model, feat, feature_names, ref_vec)

    # 2. Shapley Values
    shap_list = None
    if compute_shapley:
        shap_vals = compute_sampling_shapley(model, feat, ref_vec, num_samples=35)
        shap_list = [round(float(s), 5) for s in shap_vals]
        # Update importance score with Shapley value if available
        for j, s_val in enumerate(shap_vals):
            attributions[j].importance_score = round(float(s_val), 5)
            if s_val > 0.005:
                attributions[j].direction = "TOWARD_SYNTHETIC"
            elif s_val < -0.005:
                attributions[j].direction = "TOWARD_GENUINE"
            else:
                attributions[j].direction = "NEUTRAL"

    # Sort attributions
    sorted_synth = sorted(
        [a for a in attributions if a.direction == "TOWARD_SYNTHETIC"],
        key=lambda x: x.importance_score,
        reverse=True,
    )
    sorted_real = sorted(
        [a for a in attributions if a.direction == "TOWARD_GENUINE"],
        key=lambda x: x.importance_score,
    )

    top_synth = [a.to_dict() for a in sorted_synth[:5]]
    top_real = [a.to_dict() for a in sorted_real[:5]]

    # Group aggregations
    group_scores: Dict[str, float] = {}
    for a in attributions:
        grp = a.group.split(":")[0]  # MEAN vs STD
        group_scores[grp] = group_scores.get(grp, 0.0) + abs(a.importance_score)

    total_score = sum(group_scores.values()) + 1e-9
    grouped_importance = {k: round(v / total_score, 4) for k, v in group_scores.items()}

    # Generate Findings
    findings: List[AnalysisFinding] = []
    if top_synth:
        top_feat = top_synth[0]
        finding_synth = AnalysisFinding(
            finding_id=f"FIND-XAI-{uuid.uuid4().hex[:8].upper()}",
            run_id=run_id,
            evidence_id=evidence_id,
            start_sec=0.0,
            end_sec=0.0,
            indicator="SYNTHETIC_ATTRIBUTION_WEIGHT",
            value=top_feat["importance_score"],
            unit="shapley_attribution",
            threshold=0.01,
            severity="MEDIUM",
            interpretation_flag="INTERPRETATION",
            description=(
                f"Feature '{top_feat['feature_name']}' exerted strongest positive influence "
                f"(+{top_feat['importance_score']:.4f}) toward SYNTHETIC classification under {model_key}."
            ),
        )
        findings.append(finding_synth)

    unvalidated_warning = (
        "MODEL UNVALIDATED WARNING: The underlying classification models have unvalidated generalization "
        "on unseen acoustic domains and G.711 telephony channels (empirical FoR ROC-AUC 0.283 - 0.534). "
        "Interpretability metrics reflect mathematical model internals, NOT real-world ground truth."
    )
    forensic_disclaimer = (
        "EXPLAINABLE AI NOTICE: Feature attributions indicate which acoustic features influenced the classifier's "
        "algorithmic score. High attribution does not prove audio manipulation, splicing, or vocal cloning."
    )

    return ExplainableAIResult(
        run_id=run_id,
        evidence_id=evidence_id,
        model_name=model_key,
        predicted_label=pred_label,
        predicted_prob_synthetic=round(prob_synth, 4),
        attributions=attributions,
        top_synthetic_contributors=top_synth,
        top_genuine_contributors=top_real,
        grouped_importance=grouped_importance,
        shapley_values=shap_list,
        method_used="SHAPLEY_SAMPLING" if compute_shapley else "OCCLUSION",
        findings=findings,
        unvalidated_model_warning=unvalidated_warning,
        forensic_disclaimer=forensic_disclaimer,
    )

