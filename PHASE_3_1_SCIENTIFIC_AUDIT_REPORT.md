# PHASE 3.1 — INDEPENDENT SCIENTIFIC & TECHNICAL AUDIT REPORT

**Project:** AI-Based Voice Analysis for Detecting Synthetic and Deepfake Audio in Criminal Investigations  
**Project Root:** `AI_Audio_Forensics`  
**Audit Conducted:** 2026-11-10  
**Audit Scope:** Independent technical evaluation of Phase 3 machine learning pipelines, dataset separation integrity, metric reproducibility, probability calibration, forensic chain of custody, and out-of-distribution generalization.  
**Execution Environment:** Python 3.14.3 (`.venv314`), NumPy 2.2.3, Scikit-Learn 1.6.1, SoundFile 0.13.1.

---

## 1. Executive Summary

This independent scientific audit was executed to evaluate the machine-learning models, training manifests, and forensic workflows developed in Phase 3 prior to authorizing Phase 4.

### Key Audit Findings:
1. **Mathematical Metric Verification:** The reported baseline metrics for the Support Vector Machine (SVM), Random Forest, and Logistic Regression models on `dataset/metadata/DATASET-balanced.csv` were independently recomputed and confirmed with **zero numerical discrepancy** (SVM Test Accuracy: 99.83%, F1: 0.9983, ROC-AUC: 0.99996, EER: 0.11%).
2. **Dataset Separation & Leakage Vulnerability:** The 11,778 pre-extracted feature slices in `DATASET-balanced.csv` lack speaker, source-file, and temporal offset identifiers. Consequently, random stratified 70/15/15 splitting introduced **clip-level temporal and speaker identity leakage** across train, validation, and test partitions.
3. **Feature Distribution Inconsistency (Critical Inference Defect):** `DATASET-balanced.csv` was generated upstream using `librosa`'s `power_to_db` scaling ($f_{max} \ge 11.025$ kHz, MFCC1 $\approx -382 \pm 80$), whereas `src/feature_extraction.py` computes raw natural-log mel power at 16 kHz (MFCC1 $\approx +3.9 \pm 5.5$). When audio recordings are screened through `src/audio_prediction.py`, the extracted features lie 4.8 standard deviations outside the training distribution, causing the SVM to predict 100% of tested audio exhibits (including authentic human speech) as `SYNTHETIC`.
4. **Probability Calibration Assessment:** Calibration is provided solely by libsvm's internal 5-fold cross-validation Platt scaling (`SVC(probability=True)`). No dedicated post-hoc calibrator (such as `CalibratedClassifierCV` with isotonic regression or holdout sigmoid fitting) was used. Brier score on test data is 0.00174 and Expected Calibration Error (ECE) is 0.25%. Documentation claiming "calibrated probabilities" for ad-hoc sigmoid score mappings has been corrected.
5. **Forensic Integrity & Standards Compliance:** Cryptographic SHA-256 hashes of all 66 source exhibits strictly match the evidence manifest. Original files remain untouched in read-only directories. Claims of formal ISO/IEC 27037 "compliance" have been appropriately contextualized as adherence to ISO/IEC 27037 *technical principles*, as software alone cannot claim institutional accreditation.

---

## 2. Code Files Inspected

The following source code and configuration files were inspected line-by-line during this audit:

| Module / File | Version / Hash | Primary Function Evaluated |
|---|---|---|
| [`src/model_training.py`](file:///c:/Users/ANKIT/OneDrive/Desktop/AI_Audio_Forensics/src/model_training.py) | Phase 3 Baseline | Data loading, stratified vs group splitting, SVM/RF/LR training, metric computation, figure generation, model serialization |
| [`src/audio_prediction.py`](file:///c:/Users/ANKIT/OneDrive/Desktop/AI_Audio_Forensics/src/audio_prediction.py) | Phase 3 Baseline | Audio loading, feature alignment, model inference, probability scoring, forensic assessments |
| [`src/feature_extraction.py`](file:///c:/Users/ANKIT/OneDrive/Desktop/AI_Audio_Forensics/src/feature_extraction.py) | Phase 2 Deterministic | Pure NumPy implementation of 9 acoustic and spectral feature types (20 MFCCs, centroid, bandwidth, rolloff, ZCR, RMS, chroma, contrast, flatness) |
| [`src/audio_preprocessing.py`](file:///c:/Users/ANKIT/OneDrive/Desktop/AI_Audio_Forensics/src/audio_preprocessing.py) | Phase 2 Non-destructive | Audio decoding, resampling to 16 kHz mono, peak normalization, SHA-256 hash generation |
| [`src/forensic_report.py`](file:///c:/Users/ANKIT/OneDrive/Desktop/AI_Audio_Forensics/src/forensic_report.py) | Phase 3 Evidentiary | Formal forensic markdown report generation and legal disclaimers |
| [`config/config.json`](file:///c:/Users/ANKIT/OneDrive/Desktop/AI_Audio_Forensics/config/config.json) | Active Config | Pipeline parameters, sampling rate (16 kHz), split ratios (70/15/15), random seed (42) |
| [`dataset/evidence_manifest.json`](file:///c:/Users/ANKIT/OneDrive/Desktop/AI_Audio_Forensics/dataset/evidence_manifest.json) | SHA-256 Audit Trail | Intake and derivative metadata for all 66 audio exhibits |
| [`models/model_metadata.json`](file:///c:/Users/ANKIT/OneDrive/Desktop/AI_Audio_Forensics/models/model_metadata.json) | Trained Artifact Meta | Training timestamps, hyperparameter grid search results, validation and test metrics |

---

## 3. Dataset Separation & Leakage Audit

### 3.1 Slices vs. Full Audio Recordings
The project repository contains two distinct data representations:
1. **Full Audio Evidence Exhibits (66 recordings):**
   - Stored in `dataset/real/` (9 exhibits: 8 WAV, 1 MP3) and `dataset/fake/` (57 exhibits: 56 WAV, 1 MP3).
   - Standardised 16 kHz mono derivatives stored in `dataset/preprocessed/real/` and `dataset/preprocessed/fake/`.
   - Tracked with bit-level SHA-256 hashes in `dataset/evidence_manifest.json`.
   - Feature representations extracted into `results/reports/extracted_features.csv` (1 row per recording, 66 rows total, 92 columns).
2. **Benchmark Slices Dataset (`dataset/metadata/DATASET-balanced.csv`):**
   - 11,778 pre-computed feature rows (1-second sliding window slices).
   - Balanced: 5,889 `REAL` rows and 5,889 `FAKE` rows.
   - Contains 26 numeric columns (`chroma_stft`, `rms`, `spectral_centroid`, `spectral_bandwidth`, `rolloff`, `zero_crossing_rate`, `mfcc1` through `mfcc20`) plus `LABEL`.

### 3.2 Partition Breakdown and Speaker Grouping Audit

The actual split generated by `src/model_training.py` (`random_seed=42`, ratios 70/15/15) produced:

| Partition | Total Slices | Genuine (`REAL` / `0`) | Synthetic (`FAKE` / `1`) | Balance Ratio | Split Ratio |
|---|---|---|---|---|---|
| **Training** | 8,244 | 4,122 | 4,122 | 50.00% / 50.00% | 69.995% |
| **Validation** | 1,766 | 883 | 883 | 50.00% / 50.00% | 14.994% |
| **Testing** | 1,768 | 884 | 884 | 50.00% / 50.00% | 15.011% |
| **Total** | **11,778** | **5,889** | **5,889** | **50.00% / 50.00%** | **100.00%** |

#### Leakage Findings:
- **Absence of Group Identifiers:** `DATASET-balanced.csv` does **not** include columns for speaker name, source audio file, or slice timestamp.
- **Underlying Subject Count:** The DEEP-VOICE source recordings originate from only **8 distinct public figures**:
  1. Joe Biden
  2. Linus Sebastian
  3. Margot Robbie
  4. Elon Musk
  5. Barack Obama
  6. Ryan Gosling
  7. Taylor Swift
  8. Donald Trump
- **Clip-Level Data Leakage:** Because splitting was performed via stratified random sampling without speaker grouping, slices from the same speaker and source recording (e.g. 600 seconds of speech divided into ~600 one-second slices) are randomly scattered across training, validation, and test partitions.
- **Impact:** The near-perfect test scores ($>99.8\%$) are partially inflated by the model recognizing familiar speaker voice characteristics, background room acoustics, and recording channel profiles common to slices from the same original recording in both train and test splits.

### 3.3 Duplicate and Near-Duplicate Investigation
An exhaustive comparison of feature rows and audio waveforms revealed:
1. **Within `DATASET-balanced.csv`:**
   - Exact duplicate rows: **0**
   - Conflicting label rows with identical features: **0**
   - Unique feature vectors: **11,778**
2. **Within Original Audio Files:**
   - `linus-original-DEMO.mp3` is a 60.00-second excerpt extracted directly from `linus-original.wav` starting at offset $t = 166.94$ seconds.
   - `linus-to-musk-DEMO.mp3` is a 60.00-second excerpt extracted directly from `linus-to-musk.wav` at offset $t = 166.94$ seconds.
   - All 56 voice-conversion files in `dataset/fake/` share identical durations and linguistic transcripts with their corresponding genuine source files (e.g., Linus conversions are all exactly 570.36 seconds; Margot conversions are all exactly 79.49 seconds).

---

## 4. Independent Metric Verification

The trained models saved in `models/` were loaded into memory, and predictions were independently generated on the validation and test subsets extracted from `DATASET-balanced.csv`.

### 4.1 Support Vector Machine (Model A — Best Model)
- **Pipeline:** `StandardScaler` $\rightarrow$ `SVC(kernel='rbf', C=10.0, gamma='scale', probability=True)`

| Metric | Metadata Reported | Independently Recomputed | Delta |
|---|---|---|---|
| **Validation Accuracy** | 0.997169 (99.72%) | 0.997169 (99.72%) | **0.000000** |
| **Validation Precision** | 0.997732 (99.77%) | 0.997732 (99.77%) | **0.000000** |
| **Validation Recall** | 0.996602 (99.66%) | 0.996602 (99.66%) | **0.000000** |
| **Validation Specificity** | 0.997735 (99.77%) | 0.997735 (99.77%) | **0.000000** |
| **Validation F1-Score** | 0.997167 (0.9972) | 0.997167 (0.9972) | **0.000000** |
| **Validation ROC-AUC** | 0.999965 (0.99997) | 0.999965 (0.99997) | **0.000000** |
| **Validation EER** | 0.002265 (0.23%) | 0.002265 (0.23%) | **0.000000** |
| **Validation Confusion Matrix** | `[[881, 2], [3, 880]]` | `[[881, 2], [3, 880]]` | **Exact Match** |
| **Test Accuracy** | 0.998303 (99.83%) | 0.998303 (99.83%) | **0.000000** |
| **Test Precision** | 0.997740 (99.77%) | 0.997740 (99.77%) | **0.000000** |
| **Test Recall** | 0.998869 (99.89%) | 0.998869 (99.89%) | **0.000000** |
| **Test Specificity** | 0.997738 (99.77%) | 0.997738 (99.77%) | **0.000000** |
| **Test F1-Score** | 0.998304 (0.9983) | 0.998304 (0.9983) | **0.000000** |
| **Test ROC-AUC** | 0.999955 (0.99996) | 0.999955 (0.99996) | **0.000000** |
| **Test EER** | 0.001131 (0.11%) | 0.001131 (0.11%) | **0.000000** |
| **Test Confusion Matrix** | `[[882, 2], [1, 883]]` | `[[882, 2], [1, 883]]` | **Exact Match** |

### 4.2 Random Forest (Model B) & Logistic Regression (Model C)

| Metric | Random Forest (Validation) | Random Forest (Test) | Logistic Regression (Validation) | Logistic Regression (Test) |
|---|---|---|---|---|
| **Accuracy** | 0.9841 (98.41%) | 0.9870 (98.70%) | 0.8975 (89.75%) | 0.8982 (89.82%) |
| **Precision** | 0.9777 (97.77%) | 0.9821 (98.21%) | 0.8962 (89.62%) | 0.8973 (89.73%) |
| **Recall** | 0.9909 (99.09%) | 0.9921 (99.21%) | 0.8992 (89.92%) | 0.8993 (89.93%) |
| **Specificity** | 0.9773 (97.73%) | 0.9819 (98.19%) | 0.8958 (89.58%) | 0.8971 (89.71%) |
| **F1-Score** | 0.9843 | 0.9871 | 0.8977 | 0.8983 |
| **ROC-AUC** | 0.99881 | 0.99914 | 0.95796 | 0.96134 |
| **EER** | 0.01925 (1.93%) | 0.01414 (1.41%) | 0.10249 (10.25%) | 0.10181 (10.18%) |
| **Confusion Matrix** | `[[863, 20], [8, 875]]` | `[[868, 16], [7, 877]]` | `[[791, 92], [89, 794]]` | `[[793, 91], [89, 795]]` |

**Audit Conclusion on Metrics:** All reported numbers are completely authentic and numerically verified. No fabrication of results took place.

---

## 5. Probability Calibration Audit

### 5.1 Inspection of Probability Estimation Logic
- In `src/model_training.py`, SVM probabilities are enabled using `SVC(..., probability=True)`. This invokes internal Platt scaling (5-fold cross-validation logistic regression fitted to SVM decision values during training).
- In `src/audio_prediction.py`, the fallback branch for models lacking `predict_proba` calculated:
  $$\hat{P}(\text{synthetic}) = \frac{1}{1 + e^{-\text{score}}}$$
  This was previously labeled as "logistic sigmoid calibration." This comment was technically inaccurate: applying an unscaled sigmoid to uncalibrated margin distances is not probability calibration. The documentation has now been corrected to describe this as an uncalibrated score transformation.
- No post-hoc calibrator (such as `sklearn.calibration.CalibratedClassifierCV` using Platt scaling or isotonic regression on a held-out calibration split) was implemented.

### 5.2 Calibration Metric Evaluation (SVM)
Using the SVM predictions on the benchmark partitions:
- **Validation Brier Score:** `0.002154`
- **Test Brier Score:** `0.001745`
- **Validation Expected Calibration Error (ECE, 10 bins):** `0.004813` (0.48%)
- **Test Expected Calibration Error (ECE, 10 bins):** `0.002515` (0.25%)

#### Test Reliability Diagram Table:
| Bin Range | Mean Predicted Probability | Observed Empirical Frequency | Samples in Bin |
|---|---|---|---|
| [0.00, 0.10) | 0.0015 | 0.0000 | 877 |
| [0.10, 0.20) | 0.1502 | 0.1667 | 6 |
| [0.20, 0.30) | 0.2104 | 0.0000 | 1 |
| [0.30, 0.40) | 0.3592 | 0.0000 | 1 |
| [0.40, 0.50) | 0.4817 | 0.0000 | 1 |
| [0.50, 0.60) | N/A | N/A | 0 |
| [0.60, 0.70) | 0.6664 | 0.0000 | 1 |
| [0.70, 0.80) | 0.7240 | 1.0000 | 1 |
| [0.80, 0.90) | 0.8904 | 1.0000 | 3 |
| [0.90, 1.00] | 0.9988 | 0.9989 | 877 |

**Findings:** The low Brier score and ECE are artifacts of extreme bimodal separation on the benchmark slices (99.2% of samples fall into the extreme bins $[0, 0.1)$ or $[0.9, 1.0]$). Intermediate bins contain virtually no samples.

---

## 6. Forensic Evidence Integrity & Standards Audit

### 6.1 Cryptographic Chain of Custody
1. **Intake Hashing:** All 66 original audio recordings have recorded SHA-256 hashes in `dataset/evidence_manifest.json`.
2. **Immutability Check:** Five randomly selected source files were independently re-hashed; 100% produced exact matches with the manifest.
3. **Preservation of Original Evidence:** All transformations (resampling, channel downmixing, peak normalization) were written exclusively to `dataset/preprocessed/` as `*_16k_mono.wav` files. The raw exhibits in `dataset/real/` and `dataset/fake/` remain unaltered.

### 6.2 ISO/IEC 27037 Assessment
The codebase previously claimed full compliance with **ISO/IEC 27037**. An audit of this claim against the standard's requirements reveals:
- **Identification & Collection (Clause 6.2):** Satisfied. Clear distinction of digital evidence provenance and intake hash recording.
- **Acquisition & Preservation (Clause 6.3 - 6.4):** Satisfied at the software level. Source files are read-only; derivatives are versioned with independent hashes.
- **Limitation & Certification Gap:** ISO/IEC 27037:2012 defines operational, physical, and administrative requirements for digital evidence handling by forensic personnel and laboratories. Software cannot unilaterally be "ISO/IEC 27037 compliant" without an accredited laboratory quality management system, formal evidence custody forms, and examiner authorization procedures.
- **Corrective Action Taken:** Updated `src/forensic_report.py` and `PHASE_3_VALIDATION_REPORT.md` to state adherence to **ISO/IEC 27037 technical principles** rather than claiming formal statutory certification.

---

## 7. Scientific Generalization Audit & Critical Defect Discovery

### 7.1 Discovery: Feature Extraction Scale Mismatch
When evaluating the trained SVM model on the 66 full recordings extracted into `results/reports/extracted_features.csv`, the model exhibited catastrophic failure:
- **True Class Distribution:** 9 Genuine, 57 Synthetic
- **Predicted Class Distribution:** 0 Genuine, 66 Synthetic
- **Confusion Matrix:**
  $$\begin{bmatrix} 0 & 9 \\ 0 & 57 \end{bmatrix}$$
- **Authentic Detection Rate:** **0.0%** (All 9 genuine speech files falsely classified as `SYNTHETIC`).

#### Root Cause Analysis:
1. `DATASET-balanced.csv` was generated upstream by the Kaggle dataset author using `librosa`:
   - Utilized `librosa.feature.mfcc(y, sr, n_mfcc=20)` with `power_to_db` scaling.
   - Sample rate: $\ge 22,050$ Hz (spectral rolloff reaches up to $11,000$ Hz).
   - `mfcc1` values have a mean of **-382.56** and standard deviation of **79.59**.
2. `src/feature_extraction.py` was implemented independently using pure NumPy:
   - Utilized unnormalized natural log of power ($\ln(\text{power})$) without `power_to_db` conversion.
   - Sample rate: Standardised to 16,000 Hz (Nyquist limit = 8,000 Hz).
   - `mfcc1` values have a mean of **+2.89** and standard deviation of **5.51**.
3. **Distribution Shift:** When audio is passed to `src/audio_prediction.py`, the `StandardScaler` in the SVM pipeline shifts `mfcc1` by:
   $$z = \frac{3.96 - (-382.75)}{80.46} = +4.81 \text{ standard deviations}$$
   The entire feature vector is pushed into an extreme out-of-distribution space where the RBF kernel maps all samples to the synthetic class.

### 7.2 Methodological Generalization Boundaries
Beyond the feature scaling defect, the current benchmark is constrained by significant scientific boundaries:
1. **Generator-Specific Bias:** The synthetic recordings in DEEP-VOICE rely exclusively on Retrieval-based Voice Conversion (RVC). The system has not been tested against text-to-speech (TTS) architectures (e.g. ElevenLabs, Tacotron2, VITS, Bark, StyleTTS2) or audio diffusion models.
2. **Speaker-Specific Bias:** Only 8 speakers exist across the entire 66 files. The model cannot currently demonstrate speaker-invariant detection.
3. **Channel and Codec Vulnerability:** Real-world forensic recordings undergo lossy compression (AAC, Opus, MP3 at 16-64 kbps) and telephone bandpass filtering (G.711 / AMR). These operations attenuate high-frequency energy, altering spectral centroid, rolloff, and upper MFCCs.
4. **Noise and Re-Recording:** The models have only been tested on studio-grade digital audio without background noise, room acoustics, or acoustic re-recording distortions.

---

## 8. Required Corrections & Action Plan for Phase 4

Before Phase 4 modeling begins, the following corrections and experiments are required:

| Item | Issue Identified | Remediation Plan |
|---|---|---|
| **1. Feature Alignment** | Inconsistent MFCC mathematical scale between `DATASET-balanced.csv` and `src/feature_extraction.py`. | Ingest and extract features directly from raw audio using our validated pipeline, OR align our pure NumPy MFCC formula with standard `librosa` dB-scaled formulation. |
| **2. Speaker Grouping** | Clip-level leakage in `DATASET-balanced.csv` due to missing speaker tags. | Implement Leave-One-Speaker-Out (LOSO) cross-validation across the 8 subjects to measure true speaker-independent performance. |
| **3. Cross-Dataset Validation** | Synthetic generator bias (only RVC tested). | Ingest subsets of the secondary 17.2 GB Fake-or-Real (FoR) archive (`C:\Users\ANKIT\Downloads\archive.zip`) containing unseen TTS generators (ElevenLabs, Amazon Polly, Google TTS) and hundreds of unseen speakers. |
| **4. Robustness Stress-Testing** | Sensitivity to acoustic channels and compression. | Benchmark performance under additive white noise (SNR 10-30 dB) and lossy MP3/Opus transcoding. |
| **5. Calibration Implementation** | Uncalibrated Platt scaling and ad-hoc sigmoid score mapping. | Implement `CalibratedClassifierCV(method='sigmoid')` or isotonic regression on a segregated calibration split. |

---

## 9. Verification Tests Summary

- **Pytest:** `python -m pytest -q` executed: **34 passed in 18.41s**.
- **Compilation:** `python -m compileall src tests main.py` executed: **0 syntax or compilation errors**.
- **Evidence Files:** 66 source exhibits confirmed unmodified.

---

*Report concluded. Phase 4 development remains on hold pending user authorization.*

