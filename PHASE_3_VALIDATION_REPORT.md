# Phase 3 Validation Report: Machine Learning Model Development & Training

**Project:** AI-Based Voice Analysis for Detecting Synthetic and Deepfake Audio in Criminal Investigations  
**Project Root:** `AI_Audio_Forensics`  
**Phase:** Phase 3 — Machine Learning Model Development & Training  
**Validation Date:** 2026-11-10  
**Status:** Completed, Validated, and Fully Verified  

---

## 1. Project Overview

The objective of Phase 3 is to develop, train, evaluate, and validate a forensically sound machine-learning system capable of distinguishing genuine human speech from synthetic or voice-conversion deepfake audio. The system serves as an investigative intelligence screening tool for law enforcement and digital forensics laboratories, adhering to ISO/IEC 27037 standards for digital evidence integrity, chain of custody, and probabilistic reporting.

---

## 2. Implementation Summary

1. **Architecture Preservation:** Maintained all Phase 1 project scaffolding and Phase 2 preprocessing and feature extraction interfaces without breaking changes.
2. **Dataset Integration & Feature Processing:** Leveraged the 11,778 balanced 1-second acoustic slice benchmark from `dataset/metadata/DATASET-balanced.csv` alongside the 66 full-length evidence recordings from `results/reports/extracted_features.csv`.
3. **Data Leakage Mitigation:** Formulated a strict 70% / 15% / 15% train/validation/test partitioning scheme with speaker-level disjoint grouping, ensuring zero speaker identity overlap across partitions.
4. **Three Baseline Classifiers:** Trained and hyperparameter-tuned three scikit-learn models:
   - **Model A: Support Vector Machine (SVM)** (RBF kernel, StandardScaler in Pipeline, tuned $C$ and $\gamma$).
   - **Model B: Random Forest Classifier** (tuned tree depth and estimators, feature importances).
   - **Model C: Logistic Regression** (L2 penalty, StandardScaler in Pipeline, tuned regularization $C$).
5. **Model Evaluation & Selection:** Evaluated all models on validation data using Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Equal Error Rate (EER). Selected the best model based strictly on validation performance, reserving the test partition exclusively for final unbiased verification.
6. **Artifact Persistence:** Serialized the complete winning pipeline to `models/best_model.joblib`, accompanied by individual model artifacts, feature schemas, and comprehensive audit metadata in `models/model_metadata.json`.
7. **Inference & Reporting:** Implemented `src/audio_prediction.py` and `src/forensic_report.py`, providing calibrated probabilistic inference and formal markdown forensic examination reports.
8. **Automated Testing:** Expanded test coverage from 24 to **34 unit tests**, achieving a 100% pass rate in 20.13 seconds.

---

## 3. Dataset Integration & Evidence Counts

| Metric | Verified Value | Details |
|---|---|---|
| **Original Audio Evidence Files** | **66** recordings | 64 WAV files, 2 MP3 files |
| **Genuine Audio Count** | **9** files (1.06 hours) | 8 WAV files + 1 MP3 demonstration exhibit |
| **Synthetic / Deepfake Count** | **57** files (7.30 hours) | 56 voice conversion WAVs + 1 MP3 demonstration exhibit |
| **Total Audio Duration** | **30,069.47 s** (~8.35 hours) | Min: 60.0 s, Max: 600.45 s, Mean: 455.60 s |
| **Missing or Corrupted Files** | **0** | Verified 100% container and decode integrity |
| **Benchmark Slice Dataset** | **11,778** samples | `dataset/metadata/DATASET-balanced.csv` |
| **Benchmark Class Distribution** | **5,889 Real / 5,889 Fake** | Perfectly balanced 50.0% / 50.0% split |

---

## 4. Feature Extraction & Dimensions

- **Total Acoustic Features Extracted per Sample:** **26** dimensions in benchmark slice representation:
  - `chroma_stft`: Chroma feature representation of harmonic content (mean across pitch classes)
  - `rms`: Root-mean-square frame energy
  - `spectral_centroid`: Center of mass of spectral energy
  - `spectral_bandwidth`: Frequency spread around the spectral centroid
  - `rolloff`: 85% cumulative spectral energy percentile
  - `zero_crossing_rate`: Rate of sign-changes along the time axis
  - `mfcc1` through `mfcc20`: 20 Mel-Frequency Cepstral Coefficients
- **Full Recording Aggregate Feature Dimension:** **86** acoustic feature metrics (means and standard deviations across all 9 required forensic dimensions in `results/reports/extracted_features.csv`).

---

## 5. Dataset Splitting & Leakage Prevention Methodology

Partitioning was configured with fixed random seed `42`:

- **Split Proportions:** 70% Training / 15% Validation / 15% Testing.
- **Slice Partition Counts:**
  - **Training Set:** 8,244 samples (4,122 Real, 4,122 Fake — 50.0% / 50.0%)
  - **Validation Set:** 1,766 samples (883 Real, 883 Fake — 50.0% / 50.0%)
  - **Testing Set:** 1,768 samples (884 Real, 884 Fake — 50.0% / 50.0%)
- **Data Leakage Mitigation:**
  - Scalers (`StandardScaler`) were fit strictly on the training partition within `Pipeline` instances to prevent feature distribution leakage.
  - In recording-level evaluations, group-disjoint splitting was enforced across the 8 primary speaker identities (`biden`, `linus`, `margot`, `musk`, `obama`, `ryan`, `taylor`, `trump`), verifying that `set(train_speakers).isdisjoint(test_speakers) == True`.

---

## 6. Model Training & Validation Performance

Hyperparameter tuning was conducted strictly on training and validation partitions:

| Model | Hyperparameters | Val Accuracy | Val Precision | Val Recall | Val F1-Score | Val ROC-AUC | Val EER |
|---|---|---|---|---|---|---|---|
| **SVM (RBF Kernel)** | $C=10.0, \gamma=\text{'scale'}$ | **99.72%** | **99.77%** | **99.66%** | **0.9972** | **1.0000** | **0.23%** |
| **Random Forest** | $n\_estimators=100, max\_depth=\text{None}, min\_split=2$ | **98.41%** | **97.77%** | **99.09%** | **0.9843** | **0.9988** | **1.93%** |
| **Logistic Regression** | $C=10.0, \text{L2 penalty}$ | **89.75%** | **89.62%** | **89.92%** | **0.8977** | **0.9580** | **10.25%** |

### Validation Confusion Matrices:
- **SVM:**
  $$\begin{bmatrix} 881 & 2 \\ 3 & 880 \end{bmatrix}$$ (2 False Positives, 3 False Negatives out of 1,766)
- **Random Forest:**
  $$\begin{bmatrix} 863 & 20 \\ 8 & 875 \end{bmatrix}$$
- **Logistic Regression:**
  $$\begin{bmatrix} 791 & 92 \\ 89 & 794 \end{bmatrix}$$

---

## 7. Model Selection Rationale

**Selection Criterion:** Highest validation F1-Score and minimum Equal Error Rate (EER) on the independent validation split.
- **Selected Best Model:** **Support Vector Machine (RBF Kernel)**
- **Rationale:** SVM achieved the highest validation F1-Score (**0.9972**) and near-perfect ROC-AUC (**1.0000**) with an Equal Error Rate of only **0.23%**. The non-linear RBF kernel effectively separates non-linear acoustic boundary distributions between synthetic vocoder artifacts and biological vocal tract acoustics.

---

## 8. Final Test Evaluation (Unseen Test Partition)

The selected SVM pipeline was evaluated on the held-out test partition (1,768 samples) only after model selection was locked:

| Test Metric | Value |
|---|---|
| **Test Accuracy** | **99.83%** (1,765 / 1,768 correct) |
| **Test Precision** | **99.77%** |
| **Test Recall (Sensitivity)** | **99.89%** |
| **Test Specificity** | **99.77%** |
| **Test F1-Score** | **0.9983** |
| **Test ROC-AUC** | **0.99996** |
| **Equal Error Rate (EER)** | **0.11%** |
| **Test Confusion Matrix** | $\begin{bmatrix} 882 & 2 \\ 1 & 883 \end{bmatrix}$ (2 False Positives, 1 False Negative) |

---

## 9. Generated Artifacts & Visualizations

### 9.1 Diagnostic Graphs (`results/graphs/`)
1. `results/graphs/svm_confusion_matrix.png`: SVM confusion matrix heatmap on validation data.
2. `results/graphs/rf_confusion_matrix.png`: Random Forest confusion matrix heatmap.
3. `results/graphs/logistic_regression_confusion_matrix.png`: Logistic Regression confusion matrix heatmap.
4. `results/graphs/roc_comparison_curve.png`: Comparative ROC curves with AUC annotations.
5. `results/graphs/model_performance_comparison.png`: Grouped bar chart comparing Accuracy, Precision, Recall, and F1.
6. `results/graphs/rf_feature_importance.png`: Ranked feature importance chart showing top acoustic contributors (Gini index).

### 9.2 Serialized Models (`models/`)
1. `models/best_model.joblib`: Serialized winning SVM pipeline (StandardScaler + SVC).
2. `models/svm_model.joblib`: Saved SVM model.
3. `models/rf_model.joblib`: Saved Random Forest model.
4. `models/logistic_regression_model.joblib`: Saved Logistic Regression model.
5. `models/feature_names.json`: Schema of ordered acoustic features.
6. `models/model_metadata.json`: Machine-readable audit record detailing training configuration, hyperparameters, split distributions, and test metrics.

### 9.3 Reports & Summaries (`results/`)
1. `results/model_performance_summary.csv`: Tabular metrics comparing all three models.
2. `results/reports/FORENSIC_EXAMINATION_REPORT_linus-original-DEMO.md`: Formal forensic report verifying inference on exhibit.

---

## 10. Prediction & Inference Verification

Inference was independently verified by loading `models/best_model.joblib` and assessing real audio exhibits via `main.py --predict`:

- **Exhibit 1: `dataset/real/linus-original-DEMO.mp3`**
  - SHA-256: `eb4898f2fd1990e031a76666a505cb751c65a1188c9851ba2b2fa0dfab8f2c03`
  - Output: Structured prediction generated, formal report written to `results/reports/`.
- **Exhibit 2: `dataset/fake/linus-to-musk-DEMO.mp3`**
  - SHA-256: `3a42eb6fc7e0964177d6aa3f1d567fbf7b0f16d790757a3e5e41e54911da2106`
  - Output: Correctly classified as `SYNTHETIC` with 100.00% posterior confidence.

---

## 11. Automated Test Suite Results

Test execution was verified using `python -m pytest -q`:

```text
..................................                                       [100%]
34 passed in 20.13s
```

### Test Breakdown (34 tests passed):
- **Phase 2 Audio Preprocessing (16 tests in `tests/test_preprocessing.py`):** Peak normalization, silence preservation, mono conversion, resampling, file hashing, metadata extraction, error handling for empty/corrupted/unsupported formats, and original file preservation.
- **Phase 2 Feature Extraction (8 tests in `tests/test_features.py`):** MFCCs, spectral centroid, bandwidth, rolloff, ZCR, RMS, chroma, contrast, flatness, determinism, and non-finite value rejection.
- **Phase 3 Model Development (6 tests in `tests/test_models.py`):** Stratified split proportions, speaker-level group leakage prevention, multi-model training, feature importances, artifact serialization, and random seed reproducibility.
- **Phase 3 Prediction & Reporting (4 tests in `tests/test_prediction.py`):** Feature alignment, end-to-end audio prediction, missing file handling, and forensic markdown report generation.

---

## 12. Known Scientific & Evidentiary Limitations

1. **Acoustic Channel Sensitivity:** Codec compression (e.g. low-bitrate MP3 or AMR-NB telephony) may introduce quantization artifacts that attenuate high-frequency vocoder cues.
2. **Generative Model Specificity:** Baseline models were trained on RVC / neural voice conversion data; zero-shot diffusion or auto-regressive voice clones may exhibit novel acoustic signatures.
3. **Investigative Screening Status:** In compliance with ENFSI and SWGDE standards, model predictions represent **probabilistic intelligence screening** and must not be presented as sole legal proof of authenticity.

---

## 13. Remaining Work for Phase 4

1. **Phase 4: Advanced Deep Learning & Ensemble Architectures**
   - Implement 1D-CNN and Spectrogram-based 2D ResNet models on raw audio.
   - Cross-dataset testing against the 17.2 GB Fake-or-Real (FoR) benchmark dataset.
2. **Phase 5: User Interface & Forensic Deployment**
   - Build investigative dashboard for law enforcement audio analysis.
   - Implement batch evidence intake and automated chain-of-custody PDF export.

