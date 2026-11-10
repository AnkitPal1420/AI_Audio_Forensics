# PHASE 7 — PRE-IMPLEMENTATION AUDIT
**Project:** AI-Based Voice Analysis for Detecting Synthetic and Deepfake Audio in Criminal Investigations
**Date:** 2026-11-10 · **Branch:** `phase-6-forensic-validation` · **Scope:** inspection only. No application code was modified for this audit.
**Verified data sources:** `results/phase6_verified_metrics.json`, `PHASE_6_ERRATA_AND_VERIFIED_METRICS.md`, live inspection of `.venv314`, `database/forensics.db`, `src/`, `tests/`, `storage/`.

---

## 0. Gate decision (read this first)

**Phase 6 is not complete and the model is not validated.** Phase 7 therefore cannot treat classifier output as trustworthy.

| Phase 6 item | Verified status |
|---|---|
| Training ↔ inference feature parity | **UNRESOLVED.** Same 1‑s slices differ by median 0.8–4.5 % (max abs 41.7) between the training CSV and `extract_basic_features`. Cause unknown. |
| Source-recording leakage | Recordings and `group_id` disjoint across train/val/test (verified). **Target-voice leakage remains** (synthetic slices of held-out speakers appear in training). Generalization to unseen speakers/generators is **UNVALIDATED**. |
| Cross-dataset (FoR, 300 files) | **FAILED.** AUC 0.534 (clean), 0.283 (re-recorded) — chance / inverted. |
| Calibration | Calibrator fitted on the validation speaker → no independent evidence. Test ECE 0.277. **Not calibrated out-of-speaker.** |
| Threshold | Validation-selected 0.85 (test BAcc 0.698, FAR 0.223, FRR 0.380). Not a validated forensic threshold. |
| Models | SVM test AUC 0.769; ensemble 0.744; RF 0.651; **LogReg 0.371 (below chance)**. |
| OOD | Does not flag the clean-FoR failure; not wired into the pipeline. |
| Earlier Phase 6 reports | Contained unsupported numbers; now bannered and superseded by the errata. Dashboard Phase‑6 workspace has hard-coded values, two of them misleading. |

**Consequence for Phase 7 (applied to every module below):**
1. Any module that consumes classifier output (XAI, generator analysis, correlation of model results) must carry a persistent **"MODEL UNVALIDATED — experimental indicator only"** banner and must never emit a REAL/SYNTHETIC verdict as a finding.
2. XAI must refuse inputs that fail the feature-schema check (existing `validate_feature_vector`) or are OOD/quality-rejected, as required.
3. Modules that do not depend on the classifier (signal analysis, tampering indicators, anomaly detection, evidence correlation) can proceed, but none has labelled ground truth → they are **experimental** and report measurements, not conclusions.

---

## 1. Repository inventory (verified)

**Source modules (`src/`):** `audio_preprocessing`, `audio_quality`, `feature_extraction` (pure NumPy; 26 canonical features), `spectrogram_analysis` (6‑panel STFT/Mel/MFCC/trajectory figures), `audio_prediction` (12‑step pipeline), `deepfake_detection` (model registry), `ensemble_model`, `model_training`, `model_calibration`, `group_split`, `dataset_manifest`, `cross_dataset_evaluation`, `robustness_testing` (noise, bandpass, μ‑law, reverb, clipping, resampling), `forensic_integrity` (SHA‑256, audit JSONL), `forensic_report` (Markdown + ReportLab PDF), `batch_processing`, `storage_manager`, `utils`. **App:** `app.py` (10 workspaces, pandas-free HTML tables). **DB layer:** `database/database.py`, `models.py`.

**Tests:** 80 collected, 80 passing (Phase 6 run): batch 3, ensemble 4, feature_compat 6, features 8, models 6, phase4_forensics 10, phase5_dashboard 7, phase6_feature_pipeline 7, phase6_validation_robustness 9, prediction 4, preprocessing 16. No coverage measured. None test `app.py` rendering (no Streamlit AppTest).

**Database (`database/forensics.db`):** tables `cases`(2 rows), `evidence`(3), `examinations`(**0**), `integrity_checks`(0), `audit_events`(0). `examinations` has no model-version / feature-schema-version / software-version columns. FK enforcement is a per-connection pragma (default OFF; only the DAL turns it on). **There are no historical examination records to preserve yet.**

**Storage:** `storage/original_evidence/` holds 5 files vs 3 `evidence` rows (two ~32 KB `intake_sample.wav` copies look like test artefacts) → likely orphan files; needs reconciliation before any correlation feature (not yet verified by hash).

**Dependencies in `.venv314`:** available: numpy, scipy 1.18.1, scikit-learn 1.9.1, soundfile, matplotlib 3.11.2, reportlab 5.1.0, streamlit 1.67.0. **Missing:** `shap`, `lime`, `torch`, `torchaudio`, `speechbrain`, `pyannote.audio`, `resemblyzer`, `onnxruntime`, `librosa` (listed in `requirements.txt` but not installed), `ruptures`, `statsmodels`. `pandas` is blocked by Windows application control. System policy has already blocked compiled extensions once (pandas), so any new compiled wheel (torch, numba/llvmlite for SHAP) is at real risk of being blocked.

**Available audio data/metadata:** 66 recordings / 7502 slices (8 subjects; label, `speaker_id`, `acoustic_source_speaker`, recording id; **no generator/method, device, or session fields**); 300 FoR files (label, subset; **no generator or speaker IDs**); 3 ingested exhibits. Existing speaker analysis: **none** (no embeddings, diarization or similarity code).

---

## 2. Feasibility by Phase 7 module

Legend: **Feasible** = implementable now with installed packages · **Partial** · **Blocked** = needs unavailable dependency/data.

| Module | Status | Basis and constraints |
|---|---|---|
| **7.1 Explainable AI** | **Partial** | SHAP/LIME not installed. Feasible with sklearn/NumPy: permutation importance (grouped by recording), logistic coefficients, RF impurity importance (known bias), occlusion/sensitivity on the 26 features, and a custom sampling Shapley estimator (must be labelled *custom*, not "SHAP"). Explaining the LogReg model (AUC 0.37) and an uncalibrated ensemble is of little scientific value; explanations describe the model, not the audio, and the model is unvalidated. Installing `shap` needs numba/llvmlite — likely blocked. |
| **7.2 Tampering / manipulation analysis** | **Feasible (experimental)** | STFT, spectral flux, RMS/noise-floor, cross-correlation for repeats, change-point (CUSUM / sliding-window statistic implemented in NumPy/SciPy; `ruptures` absent). **No labelled tampered corpus exists**, so detection rate/false-alarm on real edits cannot be measured. Validation limited to (a) self-generated synthetic splices (synthetic evidence, not independent) and (b) false-alarm rate on the genuine recordings already on disk (flagging counts, no ground truth that they are unedited). Output = timestamped *measurements*; never "tampering confirmed". |
| **7.3 Speaker comparison** | **Blocked (as specified)** | ECAPA‑TDNN/x‑vector need `torch` + `speechbrain` + pretrained weights (network download, model licence check) and none are installed. A hand-built MFCC-statistics cosine score is not a speaker embedding and has no validated reliability. Data to evaluate reliability (same speaker across mics/noise) is not in the repository (8 speakers, no channel metadata). **Recommend: mark UNAVAILABLE** until dependencies are approved and a licensed model + evaluation set are identified; implement only the INDETERMINATE/"insufficient reliability" framework if authorized. |
| **7.4 Synthetic generator characterization** | **Blocked** | No generator/method/family labels anywhere (dataset and FoR manifest). Unseen-generator evaluation impossible. Licensing/provenance of FoR subset not recorded in-repo. **Recommend:** deliver only a documented "unknown / out of scope" output with data-requirements report; no generator classifier. |
| **7.5 Acoustic anomaly detection** | **Feasible (experimental)** | Spectral flatness, flux, RMS, pitch (autocorrelation), HNR (autocorrelation-based), crude formant proxy (LPC via NumPy/SciPy), IsolationForest/LOF in sklearn. No labelled anomalies → no detection accuracy; false-alarm rate measurable on available genuine audio, per recording. Formant/pitch estimators are unvalidated approximations (no Praat/librosa). |
| **7.6 Advanced signal workspace** | **Feasible** | Reuse `spectrogram_analysis` and `feature_extraction`. Needs windowed/segment processing and downsampling: one stored exhibit is a **115 MB WAV** (`biden-original.wav`, ~1 h at 16 kHz) — full-resolution plotting must be chunked/decimated. Audio-segment playback via `st.audio` on cropped working copies. Exploratory features (flux, bandwidth etc.) must be visibly labelled non-model features. |
| **7.7 Multi-evidence correlation** | **Feasible** | All needed fields exist (ids, SHA‑256, metadata, case links). Hash-duplicate and metadata comparisons are factual; model/voice/timestamp comparisons only available after those analyses are run. Currently 3 exhibits and 0 examinations, so there is little to correlate; orphan-file reconciliation needed. |
| **7.8 Dashboard** | **Feasible** | 8 new workspaces on the existing Streamlit app; tables must stay pandas-free. Must read results from DB/files only (no hard-coded values — the Phase 6 workspace currently violates this and should be refactored to read `results/phase6_verified_metrics.json`). |
| **DB & audit trail** | **Feasible** | Add new tables only (no ALTER of historical tables) — see §4. |

---

## 3. Unresolved scientific limitations carried into Phase 7
1. Feature-extractor mismatch between training and inference (unresolved).
2. No independent generalization (FoR AUC ≈ chance/inverted); calibration not independent; thresholds not independently validated.
3. Only 8 subjects / 2 test speakers; LOGO still leaks target voices; no generator metadata.
4. No labelled tampering, anomaly, speaker-channel or generator ground truth → Phase 7 modules 7.2/7.5 can report measurements only; 7.3/7.4 cannot be validated.
5. Pitch/HNR/formant/change-point methods will be self-implemented approximations (no librosa/Praat/ruptures).
6. Robustness evidence is based on 20 clips and simulated degradations.
7. Windows application control may block compiled dependencies.

---

## 4. Proposed schema extension (new tables only; FK-enforced; append-only)

- `analysis_runs`(id, run_id UNIQUE, case_id FK, evidence_id FK, examination_id FK NULL, analysis_type, algorithm_name, algorithm_version, software_versions_json, model_id, model_version, feature_schema_version, parameters_json, input_sha256, status, warnings_json, results_json, output_files_json, created_at_utc) — one generic table for XAI / manipulation / anomaly / signal / generator runs; type-specific detail stays in `results_json`.
- `analysis_findings`(id, run_id FK, start_sec, end_sec, indicator, value, unit, threshold, severity_label, interpretation_flag `MEASUREMENT|INTERPRETATION`) — timestamped measurable findings.
- `evidence_comparisons`(id, comparison_id UNIQUE, case_id FK, evidence_a FK, evidence_b FK, sha_equal, metadata_diff_json, referenced_run_ids_json, created_at_utc, notes) — never mutates exhibits.
- `speaker_comparisons` — **deferred/unavailable** (blocked, §2).
- No UPDATE/DELETE paths; supersession by new rows. Existing `audit_events` reused for every action.

---

## 5. Proposed implementation plan (pending authorization)

| Step | Work | Depends on |
|---|---|---|
| **P6‑R (recommended first)** | Resolve feature mismatch: determine cause (same audio, stage-by-stage diff), then either regenerate training features with the inference extractor or align the extractor — **requires retraining and re-evaluation**; refactor Phase‑6 dashboard to read real result files; reconcile orphan files. | Your decision (changes models) |
| 7.6 | Signal workspace (chunked, configurable, export) | none |
| 7.5 | Anomaly framework + per-recording false-alarm report on genuine audio | none |
| 7.2 | Tampering indicators + synthetic-splice unit tests + false-alarm measurement | none |
| 7.1 | XAI: permutation importance (recording-grouped), coefficients, occlusion, custom Shapley sampler; refuse invalid inputs; UNVALIDATED banner | P6‑R preferred |
| 7.7 | Evidence correlation + DB tables + audit events | schema |
| 7.8 | Dashboard (8 workspaces), PDF/Markdown exports | above |
| 7.3 / 7.4 | Mark **UNAVAILABLE**; ship data-requirements notes and INDETERMINATE framework only | dependency + data approval |
| Docs | 14 required documents, each limited to what was actually built/tested | — |

## 6. Decisions requested from you
1. **Run P6‑R first** (fix/measure the feature mismatch before building more on the classifier)? Recommended: yes.
2. **7.3 Speaker comparison:** may I attempt installing `torch`/`speechbrain` (network, large, may be blocked, model licence/weights check), or mark it unavailable? Recommended: unavailable for now.
3. **7.1 XAI:** accept custom, clearly-labelled estimators instead of SHAP/LIME, or try `pip install shap lime` first (SHAP likely blocked by numba)?
4. **7.4 Generator analysis:** accept "unavailable — no labelled data" as the Phase 7 outcome?
5. Authorise modifying `app.py` (including replacing hard-coded Phase 6 workspace values) and adding the DB tables above.

**Not started (per instructions):** no Phase 7 code, schema or UI changes have been made. Awaiting explicit authorization.

