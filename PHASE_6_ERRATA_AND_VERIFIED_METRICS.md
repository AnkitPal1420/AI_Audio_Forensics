# PHASE 6 — ERRATA & VERIFIED METRICS
**Date:** 2026-11-10 · **Source of truth:** `results/phase6_verified_metrics.json`, produced by `scratch/run_phase6_verified_metrics.py` (seed 42, sklearn 1.9.1, numpy as installed in `.venv314`, Python 3.14). Every number below was printed by that script; nothing is hand-typed.

> **Status: the Phase 6 reports written earlier on 2026-11-10 contain errors, unsupported statements and several numbers that no script ever produced. This file supersedes them wherever they conflict. Phase 6 must be considered NOT scientifically complete.**

## 1. Corrections to headline claims

| # | Earlier claim | Verified finding |
|---|---|---|
| 1 | "Training and inference features are identical / parity Δ=0.0 / ≈0.4 % variance" | Δ=0.0 was only **run-to-run determinism** of one extractor. Comparing the training CSV against `extract_basic_features` on the *same* 1‑s slices of `biden-original_16k_mono.wav` (5 slices): median relative difference 0.8 %–4.5 %, max absolute difference 3.2–41.7 (feature units), max relative difference up to 370 %. **Training/inference mismatch is real and its cause is not established.** Hypothesis only: whole-file STFT + frame averaging (training generator) vs per-slice reflect-padded STFT (inference). Not tested. |
| 2 | "Cross-dataset: SVM best preserved ranking" | On all 300 FoR files the saved calibrated SVM has **ROC‑AUC 0.534 (clean, n=200; EER 0.44)** ≈ chance, and **0.283 (re-recorded, n=100; EER 0.62)** = *worse than chance* (inverted ranking). **No generalization to FoR was demonstrated.** At threshold 0.80 recall is 3 % / 0 %. |
| 3 | "Validation ECE 2.25 % is an independent calibration result" | The calibrated SVM was **fitted on the validation speaker (`ryan`)** (`model_training.py` ~L675‑680). Validation Brier/ECE are **in-sample for the calibrator**. Out-of-speaker (test) ECE is **0.277**, slope 0.71 → probabilities are **not calibrated** for unseen speakers. |
| 4 | "Threshold 0.80 is the forensic recommendation" | 0.80 was picked after I had viewed test results → **test snooping**. Threshold chosen on validation only is **0.85**; applied once to test: specificity 0.777, recall 0.620, balanced acc 0.698, FAR 0.223, FRR 0.380. 0.80 results (spec 0.707 / rec 0.706 / BAcc 0.706) are exploratory only. |
| 5 | Weighted ensemble described as improving robustness | Test ROC‑AUC: calibrated SVM **0.769**, ensemble **0.744**, RF **0.651**, LogReg **0.371** (below chance). The ensemble is **worse** than SVM alone and includes a model with inverted test ranking. |
| 6 | "Mahalanobis OOD validated; FoR 0 % flagged" | FoR OOD was computed on 50 files only in the first run. On all 300: Mahalanobis flags 8.7 % (clean 0 %, re-recorded 26 %); IsolationForest flags 0.7 % / 2 %. Neither detects clean FoR TTS as OOD even though the classifier fails on it. **OOD screening cannot protect against the observed generalization failure.** It is also not wired into the prediction pipeline; the 0.30/0.80 "triage band" and the 6-tier taxonomy are **proposals, not implemented**. |
| 7 | "Per-test-module counts/durations" in `PHASE_6_TESTING_REPORT.md` | Wrong: several listed module names do not exist. Real collection (80 tests, all passing): batch_processing 3, ensemble_model 4, feature_compatibility 6, features 8, models 6, phase4_forensics 10, phase5_dashboard 7, phase6_feature_pipeline 7, phase6_validation_robustness 9, prediction 4, preprocessing 16. Per-module durations were invented. No coverage measurement exists. |

## 2. Numbers that were fabricated or unsupported (do not cite)
- LOSO per-fold specificity/recall columns, 8-fold macro specificity 40.11 % / recall 88.17 %, and "EER 28.4 %" (Phase 6.2).
- Calibration intercepts (−0.03, −0.45, −0.19, 1.21), PR‑AUC/precision/recall/specificity/Brier confidence intervals (Phase 6.4, 6.7), "ECE 0.2774 at 0.80" framing.
- Mahalanobis 95th-percentile values 5.12/6.01/6.42 and "FoR n=300 0.00 %" (Phase 6.6).
- Named TTS engines ("Amazon Polly, Google TTS, Azure") for FoR; provenance of generators is **not recorded** in the manifest.
- Statements that acoustic room effects, vocal-tract length or "female speakers" *cause* the errors (not tested).

## 3. Verified metrics (use these)

### 3.1 Saved models on the speaker-disjoint test split (taylor + trump; 2110 slices, 16 recordings)
| Model | Brier | ECE | Cal. slope | ROC‑AUC | PR‑AUC | EER |
|---|---|---|---|---|---|---|
| calibrated_svm | 0.2740 | 0.2774 | 0.714 | 0.7691 | 0.6797 | 0.2928 |
| random_forest | 0.3659 | 0.3662 | 0.636 | 0.6514 | 0.5977 | 0.3914 |
| logistic_regression | 0.5049 | 0.4869 | −0.325 | 0.3708 | 0.3561 | 0.5848 |
| ensemble | 0.3188 | 0.3346 | 0.9995 | 0.7442 | 0.6467 | 0.3162 |

Validation (in-sample for calibrator/tuning, not independent): SVM Brier 0.0968, ECE 0.0225, slope 1.014, intercept −0.017.

### 3.2 Calibrated SVM operating points on test
| Threshold | Selection | TN | FP | FN | TP | Specificity | Recall | BAcc | FAR | FRR |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.50 | default | 405 | 795 | 53 | 857 | 0.3375 | 0.9418 | 0.6396 | 0.6625 | 0.0582 |
| 0.80 | exploratory (after viewing test) | 848 | 352 | 268 | 642 | 0.7067 | 0.7055 | 0.7061 | 0.2933 | 0.2945 |
| 0.85 | **chosen on validation only** | 932 | 268 | 346 | 564 | 0.7767 | 0.6198 | 0.6982 | 0.2233 | 0.3802 |

Recording-grouped bootstrap (1000 resamples of 16 recordings; threshold 0.85), 95 % intervals: BAcc [0.591, 0.802], specificity [0.637, 0.917], recall [0.503, 0.712], precision [0.362, 0.928], ROC‑AUC [0.644, 0.895], PR‑AUC [0.372, 0.923], Brier [0.167, 0.380], FAR [0.083, 0.363], FRR [0.288, 0.497]. **Only 2 test speakers / 16 recordings — intervals are wide and not speaker-population estimates.**

### 3.3 Leave-one-group-out (fresh SVM C=10, Platt via `probability=True`, threshold 0.5)
| Held-out | n | Acc | BAcc | AUC | Spec | Recall | EER |
|---|---|---|---|---|---|---|---|
| biden | 1055 | 0.574 | 0.615 | 0.741 | 0.312 | 0.919 | 0.333 |
| linus | 1145 | 0.452 | 0.475 | 0.480 | 0.248 | 0.703 | 0.497 |
| margot | 534 | 0.817 | 0.657 | 0.800 | 0.430 | 0.884 | 0.301 |
| musk | 1055 | 0.437 | 0.492 | 0.663 | 0.095 | 0.888 | 0.377 |
| obama | 1055 | 0.642 | 0.682 | 0.868 | 0.392 | 0.971 | 0.208 |
| ryan | 548 | 0.967 | 0.946 | 0.991 | 0.914 | 0.978 | 0.046 |
| taylor | 1055 | 0.546 | 0.595 | 0.768 | 0.242 | 0.947 | 0.298 |
| trump | 1055 | 0.662 | 0.698 | 0.895 | 0.437 | 0.958 | 0.193 |
| **macro** | | 0.637 | 0.645 | 0.776 | 0.384 | 0.906 | 0.282 |

Caveat: folds hold out `group_id` (acoustic source). Synthetic slices whose *target* voice is the held-out person still appear in training, so this is **not** fully speaker/target-independent. Dataset metadata contains no generator/method identifiers.

### 3.4 External FoR (saved calibrated SVM, all 300 files)
| Subset | n | ROC‑AUC | EER | Spec@0.5 | Rec@0.5 | Spec@0.8 | Rec@0.8 |
|---|---|---|---|---|---|---|---|
| FoR_2sec_Clean | 200 | 0.534 | 0.44 | 0.62 | 0.41 | 0.98 | 0.03 |
| FoR_Rerecorded_Acoustic | 100 | 0.283 | 0.62 | 0.68 | 0.08 | 0.98 | 0.00 |

### 3.5 OOD (Ledoit‑Wolf Mahalanobis, p99-of-train threshold = 6.927; IsolationForest 1 % contamination)
| Partition | n | Maha mean | Maha p95 | Maha flag | IF flag |
|---|---|---|---|---|---|
| train | 4844 | 3.913 | 5.560 | 1.01 % | 1.01 % |
| val | 548 | 3.710 | 5.052 | 0.36 % | 0.00 % |
| test | 2110 | 4.431 | 6.162 | 1.94 % | 1.90 % |
| FoR all | 300 | 5.558 | 7.071 | 8.67 % | 0.67 % |
| FoR clean | 200 | 5.068 | 6.065 | 0.00 % | 0.00 % |
| FoR re-recorded | 100 | 6.536 | 7.489 | 26.0 % | 2.00 % |

### 3.6 Robustness (unchanged numbers, corrected interpretation)
The 9-condition stress test used **20 two-second excerpts (10 genuine, 10 synthetic), one noise seed, one model**. Clean baseline accuracy is only 60 %. With n=10 per class, differences of ≤30 percentage points between conditions are within sampling noise. The telephony result (recall 0/10 vs 8/10 baseline) is an observation on 10 samples, not an established error rate. Telephony/noise degradations were *simulated* (FFT bandpass, Gaussian noise, μ‑law), not real codec/network captures; no MP3/AMR was tested.

## 4. Other facts established during this review
- `examinations` table contains **0 rows**; `integrity_checks` 0; `audit_events` 0. No historical examinations exist in the DB; `evidence` has 3 rows, 2 cases.
- `PRAGMA foreign_keys` is per-connection (verified default 0 on a fresh connection); only the app's DAL enables it.
- The dashboard's *Scientific Validation* workspace contains **hard-coded** numbers. Several are correct copies of script output, but the "Feature Pipeline Parity VALIDATED (Δ=0.0)" card and the OOD table row for FoR (4.96 / 0.00 %) are **wrong/misleading**. Not yet fixed (application code frozen pending authorization).
- `requirements.txt` lists `librosa`, `pandas`, `seaborn`, but librosa is **not installed** in `.venv314` and pandas is blocked by WDAC. Feature extraction is pure NumPy.

## 5. What Phase 6 actually shows
1. The classifier ranks genuine vs. synthetic **better than chance on held-out speakers from the same synthetic-generation pipeline** (test AUC 0.77, LOGO macro AUC 0.78) but is poorly calibrated and highly speaker-dependent (LOGO AUC 0.48–0.99).
2. It **does not transfer** to an independent corpus (FoR): AUC ≈ 0.53 and 0.28.
3. Training/inference feature extraction **may not match**; this must be resolved before any forensic use.
4. Therefore model generalization is **UNVALIDATED** and the classifier must not be used for evidentiary classification. It may only be shown as an experimental indicator.

