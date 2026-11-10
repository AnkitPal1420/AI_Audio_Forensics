"""Recompute ALL Phase 6 headline numbers from real model outputs (corrective pass).

Writes results/phase6_verified_metrics.json. Nothing here is hand-typed.
"""
import sys, json, warnings, platform
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import numpy as np, joblib, sklearn, soundfile as sf
from sklearn.metrics import (roc_auc_score, roc_curve, average_precision_score,
                             brier_score_loss, confusion_matrix)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest
from src.group_split import load_slice_dataset, split_by_speaker_group, leave_one_speaker_out_folds
from src.model_calibration import compute_calibration_metrics
from src.audio_preprocessing import load_audio
from src.feature_extraction import extract_basic_features, validate_feature_vector, CANONICAL_FEATURE_NAMES

root = Path(".")
out = {"software": {"python": platform.python_version(), "numpy": np.__version__,
                    "sklearn": sklearn.__version__}, "seed": 42}

def eer(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    i = np.nanargmin(np.abs((1 - tpr) - fpr))
    return float((fpr[i] + (1 - tpr[i])) / 2)

def rates(y, p, t):
    pred = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if tn + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    prec = tp / (tp + fp) if tp + fp else float("nan")
    return dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp), specificity=float(spec),
                recall=float(rec), precision=float(prec), balanced_accuracy=float((spec + rec) / 2),
                accuracy=float((tp + tn) / len(y)), far=float(1 - spec), frr=float(1 - rec))

data = load_slice_dataset()
X, y, groups = data["X"], data["y"], data["groups"]
rows = data["rows"]
sp = split_by_speaker_group(X, y, groups, data["recording_ids"])
X_tr, y_tr, _ = sp["train"]; X_va, y_va, _ = sp["val"]; X_te, y_te, te_idx = sp["test"]
te_rec = np.array([rows[i]["recording_id"] for i in te_idx])

# ---- A. Saved models on TEST (the models the app actually uses) ----
models = {n: joblib.load(root / "models" / f) for n, f in [
    ("calibrated_svm", "calibrated_svm_model.joblib"), ("random_forest", "rf_model.joblib"),
    ("logistic_regression", "logistic_regression_model.joblib"),
    ("ensemble", "ensemble_voting_model.joblib")]}
out["saved_models"] = {}
for n, m in models.items():
    pv, pt = m.predict_proba(X_va)[:, 1], m.predict_proba(X_te)[:, 1]
    cv_, ct_ = compute_calibration_metrics(y_va, pv), compute_calibration_metrics(y_te, pt)
    out["saved_models"][n] = {
        "val": {"brier": cv_["brier_score"], "ece": cv_["expected_calibration_error"],
                "slope": cv_["calibration_slope"], "intercept": cv_["calibration_intercept"]},
        "test": {"brier": ct_["brier_score"], "ece": ct_["expected_calibration_error"],
                 "slope": ct_["calibration_slope"], "intercept": ct_["calibration_intercept"],
                 "roc_auc": float(roc_auc_score(y_te, pt)), "pr_auc": float(average_precision_score(y_te, pt)),
                 "eer": eer(y_te, pt)}}

svm = models["calibrated_svm"]
pv, pt = svm.predict_proba(X_va)[:, 1], svm.predict_proba(X_te)[:, 1]
ths = [round(t, 2) for t in np.arange(0.10, 0.951, 0.05)]
out["threshold_sweep_val"] = {str(t): rates(y_va, pv, t) for t in ths}
out["threshold_sweep_test_REPORTING_ONLY"] = {str(t): rates(y_te, pt, t) for t in ths}
# Threshold chosen on VALIDATION ONLY (max balanced accuracy), then applied once to test.
best_t = max(ths, key=lambda t: out["threshold_sweep_val"][str(t)]["balanced_accuracy"])
out["threshold_selected_on_val"] = best_t
out["test_at_val_selected_threshold"] = rates(y_te, pt, best_t)
out["test_at_0.50"] = rates(y_te, pt, 0.5)
out["test_at_0.80"] = rates(y_te, pt, 0.8)

# ---- B. Recording-grouped bootstrap CIs (all metrics, at val-selected threshold) ----
rng = np.random.RandomState(42); urec = np.unique(te_rec)
idx_by = {r: np.where(te_rec == r)[0] for r in urec}
acc = {k: [] for k in ["balanced_accuracy", "specificity", "recall", "precision", "roc_auc", "pr_auc", "brier", "far", "frr"]}
for _ in range(1000):
    s = np.concatenate([idx_by[r] for r in rng.choice(urec, len(urec), replace=True)])
    ys, ps = y_te[s], pt[s]
    if len(np.unique(ys)) < 2: continue
    r_ = rates(ys, ps, best_t)
    for k in ["balanced_accuracy", "specificity", "recall", "precision", "far", "frr"]: acc[k].append(r_[k])
    acc["roc_auc"].append(roc_auc_score(ys, ps)); acc["pr_auc"].append(average_precision_score(ys, ps))
    acc["brier"].append(brier_score_loss(ys, ps))
out["bootstrap_ci95_grouped_by_recording"] = {
    k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in acc.items()}
out["bootstrap_meta"] = {"n_test_recordings": int(len(urec)), "iterations": 1000,
                         "note": "Only 16 test recordings / 2 speakers: CIs are wide and speaker-count-limited."}

# ---- C. LOSO with FRESH SVM (uncalibrated Platt via probability=True) incl. spec/recall/EER ----
loso = []
for f in leave_one_speaker_out_folds(X, y, groups):
    m = Pipeline([("s", StandardScaler()), ("c", SVC(C=10.0, gamma="scale", probability=True, random_state=42))])
    m.fit(X[f["train_idx"]], y[f["train_idx"]])
    yt, p = y[f["test_idx"]], m.predict_proba(X[f["test_idx"]])[:, 1]
    r_ = rates(yt, p, 0.5)
    r_.update(speaker=f["held_out_speaker"], n=int(len(yt)), roc_auc=float(roc_auc_score(yt, p)), eer=eer(yt, p))
    loso.append(r_)
out["loso"] = loso
out["loso_macro"] = {k: float(np.mean([r[k] for r in loso])) for k in
                     ["accuracy", "balanced_accuracy", "roc_auc", "specificity", "recall", "eer"]}
out["loso_note"] = ("LOSO folds hold out one 'group_id' (acoustic source). Genuine slices of the held-out "
                    "speaker are unseen, but synthetic slices whose TARGET voice is the held-out speaker "
                    "also appear in training under other groups -> LOSO is NOT fully speaker/target independent.")

# ---- D. Feature parity: same audio, training-generator vs inference extractor ----
wav = root / "dataset/preprocessed/real/biden-original_16k_mono.wav"
w, sr = sf.read(str(wav)); w = np.asarray(w, dtype=np.float32)
csvrows = [r for r in rows if r["recording_id"] == "biden-original_16k_mono"][:5]
par = []
for r in csvrows:
    s0 = int(float(r["slice_start_sec"]) * sr)
    f_inf = extract_basic_features(w[s0:s0 + sr], sr)
    d = np.array([f_inf[n] - float(r[n]) for n in CANONICAL_FEATURE_NAMES])
    rel = np.array([abs(f_inf[n] - float(r[n])) / (abs(float(r[n])) + 1e-9) for n in CANONICAL_FEATURE_NAMES])
    par.append({"slice": r["slice_id"], "max_abs_diff": float(np.abs(d).max()),
                "median_rel_diff": float(np.median(rel)), "max_rel_diff": float(rel.max())})
out["feature_parity_training_vs_inference"] = par

# ---- E. FoR external: ALL 300 files, saved calibrated SVM, AUC/EER too ----
ent = json.load(open(root / "dataset/external_evaluation/for_test/for_manifest.json", encoding="utf-8"))
FV, FY, FS, FM = [], [], [], []
for e in ent:
    wf, s = load_audio(e["file_path"], sample_rate=16000, mono=True, normalize=True)
    FV.append(validate_feature_vector(extract_basic_features(wf, s), CANONICAL_FEATURE_NAMES)[0])
    FY.append(e["label_idx"]); FS.append(e["subset_type"])
FV, FY, FS = np.array(FV), np.array(FY), np.array(FS)
out["for_external"] = {}
for sub in sorted(set(FS)):
    mk = FS == sub; pe = svm.predict_proba(FV[mk])[:, 1]
    out["for_external"][sub] = {"n": int(mk.sum()), "roc_auc": float(roc_auc_score(FY[mk], pe)),
                                "eer": eer(FY[mk], pe), "at_0.50": rates(FY[mk], pe, 0.5),
                                "at_0.80": rates(FY[mk], pe, 0.8)}

# ---- F. OOD: Mahalanobis (LW) vs Isolation Forest, per-partition, ALL 300 FoR ----
lw = LedoitWolf().fit(X_tr)
md = lambda A: np.sqrt(np.maximum(0, np.sum(((A - lw.location_) @ lw.precision_) * (A - lw.location_), 1)))
mtr = md(X_tr); thr = float(np.percentile(mtr, 99))
iso = IsolationForest(n_estimators=200, contamination=0.01, random_state=42).fit(X_tr)
ithr = float(np.percentile(iso.score_samples(X_tr), 1))
def ood(A):
    m = md(A); s = iso.score_samples(A)
    return {"n": int(len(A)), "maha_mean": float(m.mean()), "maha_p95": float(np.percentile(m, 95)),
            "maha_flag_rate": float((m > thr).mean()), "iso_flag_rate": float((s < ithr).mean())}
out["ood"] = {"maha_threshold_p99_train": thr, "train": ood(X_tr), "val": ood(X_va), "test": ood(X_te),
              "for_all": ood(FV)}
for sub in sorted(set(FS)): out["ood"]["for_" + sub] = ood(FV[FS == sub])

(root / "results").mkdir(exist_ok=True)
json.dump(out, open(root / "results/phase6_verified_metrics.json", "w", encoding="utf-8"), indent=2)
print("OK -> results/phase6_verified_metrics.json")
print("val-selected threshold:", best_t)

