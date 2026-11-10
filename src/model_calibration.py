"""Post-hoc probability calibration and forensic reliability analysis.

Implements rigorous probability calibration using CalibratedClassifierCV
(Platt scaling sigmoid and Isotonic regression), evaluating Brier score,
Expected Calibration Error (ECE), log loss, and reliability diagrams.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from src.utils import project_root

logger = logging.getLogger(__name__)


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Calculate forensic probability calibration statistics: Brier score, ECE, log loss, slope/intercept."""
    y_t = np.asarray(y_true, dtype=np.int32)
    y_p = np.clip(np.asarray(y_prob, dtype=np.float64), 1e-7, 1.0 - 1e-7)

    # 1. Brier score
    brier = float(brier_score_loss(y_t, y_p))

    # 2. Log loss
    ll = float(log_loss(y_t, y_p))

    # 3. Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    bin_stats = []

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (y_p >= low) & (y_p <= high)
        else:
            mask = (y_p >= low) & (y_p < high)

        count = int(np.sum(mask))
        if count > 0:
            bin_acc = float(np.mean(y_t[mask]))
            bin_conf = float(np.mean(y_p[mask]))
            abs_err = abs(bin_acc - bin_conf)
            ece += (count / len(y_t)) * abs_err
            mce = max(mce, abs_err)
            bin_stats.append({
                "bin": i,
                "range": [float(low), float(high)],
                "count": count,
                "confidence": bin_conf,
                "accuracy": bin_acc,
                "error": abs_err,
            })

    # 4. Calibration Slope and Intercept (via logistic regression on log-odds)
    try:
        eps = 1e-6
        clipped_p = np.clip(y_p, eps, 1.0 - eps)
        log_odds = np.log(clipped_p / (1.0 - clipped_p)).reshape(-1, 1)
        lr = LogisticRegression(solver="lbfgs")
        lr.fit(log_odds, y_t)
        calib_slope = float(lr.coef_[0][0])
        calib_intercept = float(lr.intercept_[0])
    except Exception as exc:
        logger.warning("Could not compute calibration slope/intercept: %s", exc)
        calib_slope = 1.0
        calib_intercept = 0.0

    return {
        "brier_score": brier,
        "log_loss": ll,
        "expected_calibration_error": float(ece),
        "maximum_calibration_error": float(mce),
        "calibration_slope": calib_slope,
        "calibration_intercept": calib_intercept,
        "n_bins": n_bins,
        "bin_details": bin_stats,
    }


def fit_calibrated_model(
    base_model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    method: str = "sigmoid",
) -> Tuple[CalibratedClassifierCV, Dict[str, Any]]:
    """Train or post-hoc calibrate a model using CalibratedClassifierCV.

    Parameters
    ----------
    base_model : Any
        Fitted or unfitted scikit-learn classifier pipeline.
    X_train, y_train : np.ndarray
        Training features and labels.
    X_val, y_val : np.ndarray, optional
        Holdout calibration partition. If provided, uses ``cv='prefit'``.
        If omitted, uses 5-fold cross-validation.
    method : {'sigmoid', 'isotonic'}, default='sigmoid'
        Calibration mapping method.
    """
    if X_val is not None and y_val is not None:
        # Pre-fit base model if not fitted
        if not hasattr(base_model, "classes_"):
            base_model.fit(X_train, y_train)
        try:
            from sklearn.frozen import FrozenEstimator
            estimator_to_use = FrozenEstimator(base_model)
        except ImportError:
            estimator_to_use = base_model
        calibrator = CalibratedClassifierCV(estimator=estimator_to_use, method=method, cv=None)
        calibrator.fit(X_val, y_val)
        eval_X, eval_y = X_val, y_val
    else:
        calibrator = CalibratedClassifierCV(estimator=base_model, method=method, cv=5)
        calibrator.fit(X_train, y_train)
        eval_X, eval_y = X_train, y_train

    val_probs = calibrator.predict_proba(eval_X)[:, 1]
    metrics = compute_calibration_metrics(eval_y, val_probs)
    return calibrator, metrics


def plot_reliability_diagram(
    y_true: np.ndarray,
    uncalibrated_prob: np.ndarray,
    calibrated_prob: np.ndarray,
    model_name: str = "SVM",
    output_path: Optional[Union[str, Path]] = None,
    n_bins: int = 10,
) -> str:
    """Generate and save dual reliability diagram comparing raw vs calibrated probabilities."""
    root = project_root()
    out_file = Path(output_path) if output_path else root / "results" / "graphs" / f"{model_name.lower().replace(' ', '_')}_reliability_diagram.png"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 1. Reliability Curves
    ax1.plot([0, 1], [0, 1], "k--", label="Perfect Calibration (Ideal)", alpha=0.7)

    prob_true_uncal, prob_pred_uncal = calibration_curve(y_true, uncalibrated_prob, n_bins=n_bins)
    ax1.plot(prob_pred_uncal, prob_true_uncal, "s-", color="#d62728", label=f"Uncalibrated (Platt/Raw)", lw=2)

    prob_true_cal, prob_pred_cal = calibration_curve(y_true, calibrated_prob, n_bins=n_bins)
    ax1.plot(prob_pred_cal, prob_true_cal, "o-", color="#2ca02c", label=f"Calibrated (Sigmoid)", lw=2)

    ax1.set_xlabel("Mean Predicted Probability (Confidence)")
    ax1.set_ylabel("Empirical Fraction of Positives (Accuracy)")
    ax1.set_title(f"Reliability Diagram — {model_name}")
    ax1.legend(loc="lower right")
    ax1.grid(alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])

    # 2. Probability Histograms
    ax2.hist(uncalibrated_prob, range=(0, 1), bins=n_bins, histtype="step", lw=2, color="#d62728", label="Uncalibrated")
    ax2.hist(calibrated_prob, range=(0, 1), bins=n_bins, histtype="step", lw=2, color="#2ca02c", label="Calibrated")
    ax2.set_xlabel("Predicted Probability")
    ax2.set_ylabel("Count")
    ax2.set_title("Probability Distribution")
    ax2.legend(loc="upper center")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_file, dpi=200)
    plt.close(fig)

    logger.info("Saved reliability diagram to %s", out_file)
    return str(out_file)
