"""Deepfake detection interface with multi-model inference and disagreement analysis.

Supports:
- Support Vector Machine (RBF kernel, Platt scaled or post-hoc calibrated)
- Random Forest Classifier (ensemble of decision trees)
- Logistic Regression (linear baseline)
- Multi-model consensus and disagreement resolution.
- Extensible interface for future neural models.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np

from src.security import verify_model_artifact
from src.utils import project_root

logger = logging.getLogger(__name__)

# Standard model filenames in models/ directory
MODEL_REGISTRY = {
    "svm": "svm_model.joblib",
    "calibrated_svm": "calibrated_svm_model.joblib",
    "random_forest": "rf_model.joblib",
    "logistic_regression": "logistic_regression_model.joblib",
    "ensemble": "ensemble_voting_model.joblib",
    "best_model": "best_model.joblib",
}


class DeepfakeDetectionEngine:
    """Manages loading, feature validation, and inference across validated forensic models."""

    def __init__(self, models_dir: Optional[Union[str, Path]] = None) -> None:
        root = project_root()
        self.models_dir = Path(models_dir) if models_dir else root / "models"
        self._loaded_models: Dict[str, Any] = {}
        self.feature_names: List[str] = []
        self.metadata: Dict[str, Any] = {}
        self._load_schema()

    def _load_schema(self) -> None:
        """Load expected feature schema and model metadata."""
        feat_path = self.models_dir / "feature_names.json"
        meta_path = self.models_dir / "model_metadata.json"

        if feat_path.is_file():
            with feat_path.open("r", encoding="utf-8") as f:
                self.feature_names = json.load(f)
        else:
            logger.warning("Feature schema file not found at %s.", feat_path)

        if meta_path.is_file():
            with meta_path.open("r", encoding="utf-8") as f:
                self.metadata = json.load(f)

    def get_available_models(self) -> List[str]:
        """List model identifiers available on disk in models directory."""
        available = []
        for key, fname in MODEL_REGISTRY.items():
            if (self.models_dir / fname).is_file():
                available.append(key)
        return available

    def load_model(self, model_key: str = "best_model") -> Any:
        """Retrieve or lazy-load a model by its registry key."""
        if model_key in self._loaded_models:
            return self._loaded_models[model_key]

        if model_key not in MODEL_REGISTRY:
            raise KeyError(
                f"Unknown model key '{model_key}'. Available options: {list(MODEL_REGISTRY.keys())}"
            )

        model_file = self.models_dir / MODEL_REGISTRY[model_key]
        if not model_file.is_file():
            raise FileNotFoundError(f"Model artifact not found at {model_file}")

        # Cryptographic verification against model manifest to prevent insecure deserialization
        verify_model_artifact(model_file)

        model = joblib.load(model_file)
        self._loaded_models[model_key] = model
        return model

    def predict_single(
        self,
        feature_vector: np.ndarray,
        model_key: str = "best_model",
    ) -> Dict[str, Any]:
        """Execute inference using a single specified model artifact.

        Parameters
        ----------
        feature_vector : np.ndarray
            2D feature array of shape (1, n_features).
        model_key : str, default="best_model"
            Key from MODEL_REGISTRY.

        Returns
        -------
        Dict[str, Any]
            Single-model prediction outcome dictionary.
        """
        model = self.load_model(model_key)

        pred_idx = int(model.predict(feature_vector)[0])
        pred_label = "SYNTHETIC" if pred_idx == 1 else "GENUINE"

        prob_synth = None
        prob_real = None
        confidence = 1.0
        decision_score = None

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(feature_vector)[0]
            prob_real = float(probs[0])
            prob_synth = float(probs[1])
            confidence = float(np.max(probs))
        elif hasattr(model, "decision_function"):
            decision_score = float(model.decision_function(feature_vector)[0])
            # Sigmoid distance approximation
            prob_synth = float(1.0 / (1.0 + np.exp(-decision_score)))
            prob_real = float(1.0 - prob_synth)
            confidence = float(max(prob_synth, prob_real))

        # Check if model has a decision function even if it has predict_proba
        if decision_score is None and hasattr(model, "decision_function"):
            try:
                decision_score = float(model.decision_function(feature_vector)[0])
            except Exception:
                pass

        return {
            "model_key": model_key,
            "model_class": type(model).__name__,
            "predicted_label": pred_label,
            "prediction_index": pred_idx,
            "confidence": round(confidence, 4),
            "probability_synthetic": round(prob_synth, 4) if prob_synth is not None else None,
            "probability_genuine": round(prob_real, 4) if prob_real is not None else None,
            "decision_score": round(decision_score, 4) if decision_score is not None else None,
        }

    def predict_multi_model(
        self,
        feature_vector: np.ndarray,
        models_to_run: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Execute multi-model inference and perform model disagreement analysis.

        Parameters
        ----------
        feature_vector : np.ndarray
            Standardized feature vector of shape (1, n_features).
        models_to_run : Optional[List[str]]
            List of model keys. If None, runs available core models:
            ['svm', 'random_forest', 'logistic_regression'].

        Returns
        -------
        Dict[str, Any]
            Consolidated multi-model evaluation including consensus and disagreement flags.
        """
        if models_to_run is None:
            models_to_run = [m for m in ["svm", "random_forest", "logistic_regression"] if m in self.get_available_models()]
            if not models_to_run:
                models_to_run = ["best_model"]

        model_results: Dict[str, Any] = {}
        synth_probs: List[float] = []
        labels: List[str] = []

        for m_key in models_to_run:
            try:
                res = self.predict_single(feature_vector, model_key=m_key)
                model_results[m_key] = res
                labels.append(res["predicted_label"])
                if res["probability_synthetic"] is not None:
                    synth_probs.append(res["probability_synthetic"])
            except Exception as e:
                logger.warning("Inference failed for model '%s': %s", m_key, e)
                model_results[m_key] = {"error": str(e)}

        if not model_results:
            raise RuntimeError("No models were successfully evaluated.")

        # Disagreement analysis
        unique_labels = set(labels)
        is_unanimous = len(unique_labels) == 1
        mean_synth_prob = float(np.mean(synth_probs)) if synth_probs else 0.5

        # Check for contradiction between primary high-capacity non-linear classifiers
        svm_res = model_results.get("calibrated_svm") or model_results.get("svm")
        rf_res = model_results.get("random_forest")

        primary_conflict = False
        if svm_res and rf_res and "predicted_label" in svm_res and "predicted_label" in rf_res:
            primary_conflict = (svm_res["predicted_label"] != rf_res["predicted_label"])

        max_prob = max(synth_probs) if synth_probs else 0.5
        min_prob = min(synth_probs) if synth_probs else 0.5
        prob_spread = max_prob - min_prob

        # Significant disagreement occurs if primary classifiers disagree or spread is critical across all models without strong majority
        significant_disagreement = primary_conflict or (prob_spread >= 0.65 and len(unique_labels) > 1)

        if significant_disagreement:
            consensus_label = "INDETERMINATE_MODEL_DISAGREEMENT"
            consensus_assessment = (
                f"Significant model disagreement detected (Spread: {prob_spread:.1%}). "
                f"Primary classifiers produced conflicting classifications ({', '.join([f'{k}: {v.get('predicted_label')}' for k, v in model_results.items() if 'predicted_label' in v])}). "
                "The recording is flagged for specialized biometric examination rather than automated classification."
            )
        elif is_unanimous:
            consensus_label = labels[0]
            consensus_assessment = (
                f"Unanimous consensus across all {len(labels)} evaluated models ({consensus_label}). "
                f"Average synthetic probability estimate: {mean_synth_prob:.2%}."
            )
        else:
            # Majority vote supported by primary classifiers
            synth_count = labels.count("SYNTHETIC")
            real_count = labels.count("GENUINE")
            consensus_label = "SYNTHETIC" if synth_count > real_count else "GENUINE"
            consensus_assessment = (
                f"Primary consensus ({consensus_label}) confirmed by top classifiers with minor baseline variance (Spread: {prob_spread:.1%}). "
                f"Average synthetic probability estimate: {mean_synth_prob:.2%}."
            )

        return {
            "individual_models": model_results,
            "consensus_label": consensus_label,
            "is_unanimous": is_unanimous,
            "significant_disagreement": significant_disagreement,
            "mean_synthetic_probability": round(mean_synth_prob, 4),
            "probability_spread": round(prob_spread, 4),
            "consensus_assessment": consensus_assessment,
        }
