import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
from scipy.spatial.distance import mahalanobis
from sklearn.covariance import EmpiricalCovariance, LedoitWolf
from sklearn.ensemble import IsolationForest
from src.group_split import load_slice_dataset, split_by_speaker_group
from src.audio_quality import assess_feature_distribution, load_training_distribution_stats

root = Path(".")
data = load_slice_dataset()
split = split_by_speaker_group(data["X"], data["y"], data["groups"], data["recording_ids"])

X_tr = split["train"][0]
X_va = split["val"][0]
X_te = split["test"][0]

print(f"X_tr: {X_tr.shape}, X_va: {X_va.shape}, X_te: {X_te.shape}")

# Fit Ledoit-Wolf regularized covariance for stable matrix inversion
cov_estimator = LedoitWolf().fit(X_tr)
mean_tr = cov_estimator.location_
inv_cov = cov_estimator.precision_

def calc_mahalanobis_batch(X_mat):
    diff = X_mat - mean_tr
    # d^2 = sum((diff @ precision) * diff, axis=1)
    md2 = np.sum((diff @ inv_cov) * diff, axis=1)
    return np.sqrt(np.maximum(0.0, md2))

md_tr = calc_mahalanobis_batch(X_tr)
md_va = calc_mahalanobis_batch(X_va)
md_te = calc_mahalanobis_batch(X_te)

# Fit Isolation Forest
iso = IsolationForest(n_estimators=100, contamination=0.01, random_state=42).fit(X_tr)
iso_tr = iso.score_samples(X_tr)
iso_va = iso.score_samples(X_va)
iso_te = iso.score_samples(X_te)

# Empirical 99th percentile threshold on training
md_thresh_99 = np.percentile(md_tr, 99.0)
iso_thresh_99 = np.percentile(iso_tr, 1.0)

print(f"Training Mahalanobis Distance: Mean={np.mean(md_tr):.2f}, 95th={np.percentile(md_tr, 95):.2f}, 99th={md_thresh_99:.2f}")
print(f"Validation Mahalanobis Distance: Mean={np.mean(md_va):.2f}, >99th Thresh={np.mean(md_va > md_thresh_99):.2%}")
print(f"Test Mahalanobis Distance: Mean={np.mean(md_te):.2f}, >99th Thresh={np.mean(md_te > md_thresh_99):.2%}")

# Load FoR external benchmark features
for_features_path = root / "dataset" / "external_evaluation" / "for_test" / "for_manifest.json"
with open(for_features_path, "r", encoding="utf-8") as f:
    for_entries = json.load(f)

from src.audio_preprocessing import load_audio
from src.feature_extraction import extract_basic_features, validate_feature_vector, CANONICAL_FEATURE_NAMES

for_vecs = []
for e in for_entries[:50]:
    w, sr = load_audio(e["file_path"], sample_rate=16000, mono=True, normalize=True)
    f = extract_basic_features(w, sample_rate=sr)
    v = validate_feature_vector(f, CANONICAL_FEATURE_NAMES)
    for_vecs.append(v[0])

for_X = np.array(for_vecs, dtype=np.float32)
md_for = calc_mahalanobis_batch(for_X)

print(f"FoR External Benchmark Mahalanobis: Mean={np.mean(md_for):.2f}, >99th Thresh={np.mean(md_for > md_thresh_99):.2%}")

# Save OOD empirical comparison
with open(root / "results" / "ood_validation_summary.json", "w", encoding="utf-8") as f:
    json.dump({
        "mahalanobis_stats": {
            "train_mean": float(np.mean(md_tr)),
            "train_99th_threshold": float(md_thresh_99),
            "val_mean": float(np.mean(md_va)),
            "val_ood_flagged_ratio": float(np.mean(md_va > md_thresh_99)),
            "test_mean": float(np.mean(md_te)),
            "test_ood_flagged_ratio": float(np.mean(md_te > md_thresh_99)),
            "external_for_mean": float(np.mean(md_for)),
            "external_for_ood_flagged_ratio": float(np.mean(md_for > md_thresh_99)),
        }
    }, f, indent=2)

print("\nSaved OOD validation summary to results/ood_validation_summary.json")
