"""Soft voting and stacking ensemble classifier module for deepfake audio forensics.

Combines calibrated predictions across multiple diverse architectures:
1. Calibrated Support Vector Machine (RBF kernel, Platt scaled)
2. Random Forest Classifier (100 estimators)
3. Logistic Regression (L2 regularized linear baseline)

Implements weighted posterior probability combination to minimize variance
and provide robust cross-speaker generalization.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np

from src.utils import project_root

logger = logging.getLogger(__name__)


class SoftVotingEnsembleClassifier:
    """Weighted Soft Voting Ensemble combining pre-fitted classifiers."""

    def __init__(
        self,
        estimators: List[Tuple[str, Any]],
        weights: Optional[List[float]] = None,
    ) -> None:
        self.estimators = estimators  # List of (name, model)
        self.weights = weights if weights else [1.0] * len(estimators)
        self.classes_ = np.array([0, 1])

        # Normalize weights
        total_w = sum(self.weights)
        self._norm_weights = [w / total_w for w in self.weights]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Calculate weighted average posterior class probabilities."""
        weighted_probs = np.zeros((X.shape[0], len(self.classes_)), dtype=np.float64)

        for (name, model), w in zip(self.estimators, self._norm_weights):
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X)
            elif hasattr(model, "decision_function"):
                scores = model.decision_function(X)
                p_synth = 1.0 / (1.0 + np.exp(-scores))
                probs = np.column_stack([1.0 - p_synth, p_synth])
            else:
                preds = model.predict(X)
                probs = np.zeros((X.shape[0], 2))
                for idx, p in enumerate(preds):
                    probs[idx, int(p)] = 1.0

            weighted_probs += probs * w

        return weighted_probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels with highest weighted probability."""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Derive an ensemble logit decision score from synthetic probability."""
        proba = self.predict_proba(X)[:, 1]
        eps = 1e-7
        clipped = np.clip(proba, eps, 1.0 - eps)
        return np.log(clipped / (1.0 - clipped))


def build_and_save_ensemble(
    models_dir: Optional[Union[str, Path]] = None,
    weights: Optional[List[float]] = None,
) -> Path:
    """Load trained base models and serialize the weighted soft voting ensemble artifact.

    Parameters
    ----------
    models_dir : Union[str, Path], optional
        Directory containing trained models. Defaults to project models/ folder.
    weights : List[float], optional
        Weights for [calibrated_svm, random_forest, logistic_regression].
        Defaults to [2.0, 1.5, 0.5].

    Returns
    -------
    Path
        Path to the saved ensemble artifact.
    """
    root = project_root()
    m_dir = Path(models_dir) if models_dir else root / "models"

    svm_path = m_dir / "calibrated_svm_model.joblib"
    rf_path = m_dir / "rf_model.joblib"
    lr_path = m_dir / "logistic_regression_model.joblib"

    if not svm_path.is_file() or not rf_path.is_file() or not lr_path.is_file():
        raise FileNotFoundError(
            f"Base models missing in {m_dir}. Ensure calibrated_svm, rf, and lr are trained."
        )

    svm = joblib.load(svm_path)
    rf = joblib.load(rf_path)
    lr = joblib.load(lr_path)

    default_weights = weights if weights else [2.0, 1.5, 0.5]
    estimators = [
        ("calibrated_svm", svm),
        ("random_forest", rf),
        ("logistic_regression", lr),
    ]

    ensemble = SoftVotingEnsembleClassifier(estimators, weights=default_weights)
    out_path = m_dir / "ensemble_voting_model.joblib"
    joblib.dump(ensemble, out_path)
    logger.info("Successfully serialized soft voting ensemble to %s", out_path)
    return out_path

