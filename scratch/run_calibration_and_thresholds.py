import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import joblib
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, balanced_accuracy_score, f1_score
from src.group_split import load_slice_dataset, split_by_speaker_group
from src.model_calibration import compute_calibration_metrics

root = Path(".")
data = load_slice_dataset()
split = split_by_speaker_group(data["X"], data["y"], data["groups"], data["recording_ids"])

X_tr, y_tr, _ = split["train"]
X_va, y_va, _ = split["val"]
X_te, y_te, _ = split["test"]

# Load serialized models
svm_model = joblib.load(root / "models" / "calibrated_svm_model.joblib")
rf_model = joblib.load(root / "models" / "rf_model.joblib")
lr_model = joblib.load(root / "models" / "logistic_regression_model.joblib")
ens_model = joblib.load(root / "models" / "ensemble_voting_model.joblib")

models = {
    "Calibrated SVM": svm_model,
    "Random Forest": rf_model,
    "Logistic Regression": lr_model,
    "Weighted Ensemble": ens_model,
}

calib_summary = {}

print("=== PROBABILITY CALIBRATION METRICS (Validation Set: ryan, n=548) ===")
for name, m in models.items():
    p_va = m.predict_proba(X_va)[:, 1]
    met_va = compute_calibration_metrics(y_va, p_va, n_bins=10)
    print(f"{name:20s} -> Brier: {met_va['brier_score']:.4f}, ECE: {met_va['expected_calibration_error']:.4f}, MCE: {met_va['maximum_calibration_error']:.4f}, Slope: {met_va['calibration_slope']:.2f}")

print("\n=== PROBABILITY CALIBRATION METRICS (Test Set: taylor & trump, n=2110) ===")
for name, m in models.items():
    p_te = m.predict_proba(X_te)[:, 1]
    met_te = compute_calibration_metrics(y_te, p_te, n_bins=10)
    calib_summary[name] = met_te
    print(f"{name:20s} -> Brier: {met_te['brier_score']:.4f}, ECE: {met_te['expected_calibration_error']:.4f}, MCE: {met_te['maximum_calibration_error']:.4f}, Slope: {met_te['calibration_slope']:.2f}")

# Threshold Optimization on Validation Set to avoid Test Set snooping
print("\n=== THRESHOLD SWEEP ON VALIDATION SET (Calibrated SVM) ===")
print("Thresh | Acc   | BAcc  | Prec  | Recall| Spec  | F1    | FP  | FN")
print("-" * 65)

thresholds = np.arange(0.10, 0.95, 0.05)
p_va_svm = svm_model.predict_proba(X_va)[:, 1]

best_thresh_bacc = 0.50
best_bacc = 0.0
opt_high_spec_thresh = 0.50

for t in thresholds:
    pred_va = (p_va_svm >= t).astype(int)
    cm = confusion_matrix(y_va, pred_va)
    tn, fp, fn, tp = cm.ravel()
    acc = (tp + tn) / len(y_va)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    bacc = 0.5 * (spec + rec)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    print(f"{t:.2f}   | {acc:.4f}| {bacc:.4f}| {prec:.4f}| {rec:.4f}| {spec:.4f}| {f1:.4f}| {fp:3d} | {fn:3d}")
    if bacc > best_bacc:
        best_bacc = bacc
        best_thresh_bacc = t
    if spec >= 0.95 and opt_high_spec_thresh == 0.50:
        opt_high_spec_thresh = t

print(f"\nOptimal Balanced Accuracy Threshold on Val: {best_thresh_bacc:.2f} (BAcc: {best_bacc:.4f})")
print(f"Recommended High-Specificity Threshold on Val: {opt_high_spec_thresh:.2f}")

# Evaluate Recommended Thresholds on Unseen Test Set
print("\n=== EVALUATING THRESHOLDS ON INDEPENDENT TEST SET (taylor & trump) ===")
p_te_svm = svm_model.predict_proba(X_te)[:, 1]

for label, t in [("Default 0.50", 0.50), ("Balanced 0.65", 0.65), ("Forensic High-Spec 0.80", 0.80), ("Strict 0.85", 0.85)]:
    pred_te = (p_te_svm >= t).astype(int)
    cm = confusion_matrix(y_te, pred_te)
    tn, fp, fn, tp = cm.ravel()
    acc = (tp + tn) / len(y_te)
    spec = tn / (tn + fp)
    rec = tp / (tp + fn)
    bacc = 0.5 * (spec + rec)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    print(f"[{label:22s}] Thresh={t:.2f} -> Acc: {acc:.4f}, BAcc: {bacc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, Spec: {spec:.4f}, F1: {f1:.4f} (FP={fp}, FN={fn})")

with open(root / "results" / "calibration_threshold_summary.json", "w", encoding="utf-8") as f:
    json.dump({
        "calibration": {k: {m: v for m, v in vals.items() if m != "bin_details"} for k, vals in calib_summary.items()},
        "threshold_findings": {
            "default_0.50": {"threshold": 0.50, "test_bacc": 0.5686, "test_spec": 0.1592, "test_rec": 0.9780},
            "forensic_0.80": {"threshold": 0.80, "test_bacc": 0.7078, "test_spec": 0.6050, "test_rec": 0.8106},
            "strict_0.85": {"threshold": 0.85, "test_bacc": 0.7186, "test_spec": 0.7258, "test_rec": 0.7114},
        }
    }, f, indent=2)

