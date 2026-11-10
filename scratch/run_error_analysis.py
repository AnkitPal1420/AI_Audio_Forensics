import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, precision_recall_curve, auc, confusion_matrix,
    brier_score_loss
)
from src.group_split import load_slice_dataset, split_by_speaker_group
from src.model_calibration import compute_calibration_metrics

root = Path(".")
data = load_slice_dataset()
split = split_by_speaker_group(data["X"], data["y"], data["groups"], data["recording_ids"])

X_te, y_te, te_idx = split["test"]
te_rows = [data["rows"][i] for i in te_idx]
te_recordings = np.array([r["recording_id"] for r in te_rows])

# Load calibrated SVM
svm = joblib.load(root / "models" / "calibrated_svm_model.joblib")
p_te = svm.predict_proba(X_te)[:, 1]

# Base metrics at threshold 0.50 and forensic threshold 0.80
def compute_all_metrics(y_true, y_prob, thresh=0.50):
    y_pred = (y_prob >= thresh).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    acc = (tp + tn) / len(y_true)
    spec = tn / (tn + fp)
    rec = tp / (tp + fn)
    bacc = 0.5 * (spec + rec)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    roc_auc = roc_auc_score(y_true, y_prob)
    
    p_curve, r_curve, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(r_curve, p_curve)
    
    # EER calculation
    fpr = fp / (tn + fp)
    fnr = fn / (tp + fn)
    
    cal = compute_calibration_metrics(y_true, y_prob)
    
    return {
        "accuracy": acc,
        "balanced_accuracy": bacc,
        "precision": prec,
        "recall": rec,
        "specificity": spec,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "far": fpr,
        "frr": fnr,
        "brier": cal["brier_score"],
        "ece": cal["expected_calibration_error"],
        "cm": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    }

base_metrics = compute_all_metrics(y_te, p_te, thresh=0.50)
forensic_metrics = compute_all_metrics(y_te, p_te, thresh=0.80)

# Grouped Bootstrap Confidence Intervals (Resampling by recording_id)
unique_recs = np.unique(te_recordings)
n_bootstraps = 500
rng = np.random.RandomState(42)

boot_acc = []
boot_bacc = []
boot_auc = []
boot_f1 = []

for _ in range(n_bootstraps):
    sampled_recs = rng.choice(unique_recs, size=len(unique_recs), replace=True)
    sample_indices = []
    for r in sampled_recs:
        sample_indices.extend(np.where(te_recordings == r)[0])
    
    sample_y = y_te[sample_indices]
    sample_p = p_te[sample_indices]
    
    if len(np.unique(sample_y)) < 2:
        continue
    
    sample_pred = (sample_p >= 0.80).astype(int)
    boot_acc.append(accuracy_score(sample_y, sample_pred))
    boot_bacc.append(balanced_accuracy_score(sample_y, sample_pred))
    boot_auc.append(roc_auc_score(sample_y, sample_p))
    boot_f1.append(f1_score(sample_y, sample_pred, zero_division=0))

ci_acc = [float(np.percentile(boot_acc, 2.5)), float(np.percentile(boot_acc, 97.5))]
ci_bacc = [float(np.percentile(boot_bacc, 2.5)), float(np.percentile(boot_bacc, 97.5))]
ci_auc = [float(np.percentile(boot_auc, 2.5)), float(np.percentile(boot_auc, 97.5))]
ci_f1 = [float(np.percentile(boot_f1, 2.5)), float(np.percentile(boot_f1, 97.5))]

print("=== FORENSIC RELIABILITY METRICS (Test Set, n=2110) ===")
print("Default Threshold (0.50):")
print(f"  Accuracy: {base_metrics['accuracy']:.4f}, Balanced Acc: {base_metrics['balanced_accuracy']:.4f}")
print(f"  Precision: {base_metrics['precision']:.4f}, Recall: {base_metrics['recall']:.4f}, Specificity: {base_metrics['specificity']:.4f}")
print(f"  ROC-AUC: {base_metrics['roc_auc']:.4f}, PR-AUC: {base_metrics['pr_auc']:.4f}")
print(f"  FAR (False Acceptance Rate): {base_metrics['far']:.4f}, FRR (False Rejection Rate): {base_metrics['frr']:.4f}")
print(f"  Brier Score: {base_metrics['brier']:.4f}, ECE: {base_metrics['ece']:.4f}")

print("\nForensic Calibrated Threshold (0.80):")
print(f"  Accuracy: {forensic_metrics['accuracy']:.4f} (95% Grouped CI: [{ci_acc[0]:.4f}, {ci_acc[1]:.4f}])")
print(f"  Balanced Acc: {forensic_metrics['balanced_accuracy']:.4f} (95% Grouped CI: [{ci_bacc[0]:.4f}, {ci_bacc[1]:.4f}])")
print(f"  ROC-AUC: {forensic_metrics['roc_auc']:.4f} (95% Grouped CI: [{ci_auc[0]:.4f}, {ci_auc[1]:.4f}])")
print(f"  F1-Score: {forensic_metrics['f1']:.4f} (95% Grouped CI: [{ci_f1[0]:.4f}, {ci_f1[1]:.4f}])")
print(f"  FAR: {forensic_metrics['far']:.4f}, FRR: {forensic_metrics['frr']:.4f}")

# Feature Error Analysis: Compare True Negatives vs False Positives
genuine_indices = np.where(y_te == 0)[0]
fp_indices = [i for i in genuine_indices if p_te[i] >= 0.80]
tn_indices = [i for i in genuine_indices if p_te[i] < 0.80]

print(f"\nGenuine Slices Analysis (Total={len(genuine_indices)}):")
print(f"  True Negatives (Correctly Genuine): {len(tn_indices)}")
print(f"  False Positives (Incorrectly Synthetic): {len(fp_indices)}")

if fp_indices and tn_indices:
    fp_feats = X_te[fp_indices]
    tn_feats = X_te[tn_indices]
    print("\nFeature Drift in False Positives vs True Negatives (Top Divergences):")
    divergences = []
    for idx, name in enumerate(data["feature_names"]):
        fp_mean = np.mean(fp_feats[:, idx])
        tn_mean = np.mean(tn_feats[:, idx])
        std = np.std(X_te[:, idx]) + 1e-6
        cohen_d = abs(fp_mean - tn_mean) / std
        divergences.append((name, cohen_d, fp_mean, tn_mean))
    
    divergences.sort(key=lambda x: x[1], reverse=True)
    for name, d, fpm, tnm in divergences[:6]:
        print(f"  {name:20s} -> Cohen's d: {d:.2f} (FP Mean: {fpm:.2f}, TN Mean: {tnm:.2f})")

# Save summary
with open(root / "results" / "forensic_reliability_metrics.json", "w", encoding="utf-8") as f:
    json.dump({
        "base_metrics_0_50": base_metrics,
        "forensic_metrics_0_80": forensic_metrics,
        "bootstrap_95_ci": {
            "accuracy": ci_acc,
            "balanced_accuracy": ci_bacc,
            "roc_auc": ci_auc,
            "f1_score": ci_f1,
            "resampling_unit": "recording_id",
            "n_bootstraps": n_bootstraps,
        }
    }, f, indent=2)

print("\nSaved full forensic reliability report data to results/forensic_reliability_metrics.json")

