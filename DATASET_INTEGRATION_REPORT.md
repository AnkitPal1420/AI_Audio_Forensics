# Dataset Integration and Verification Report

**Project:** AI-Based Voice Analysis for Detecting Synthetic and Deepfake Audio in Criminal Investigations  
**Project Root:** `AI_Audio_Forensics`  
**Report Generated:** 2026-11-10  
**Phase:** Pre-Phase 3 Dataset Integrity and Verification  
**Status:** Successfully Integrated and Fully Verified (Read-Only)  

---

## 1. Executive Summary

Prior to initiating Phase 3 (Model Training and Evaluation), a comprehensive forensic inspection of the dataset was conducted. All original audio recordings have been verified and integrated into the project's `dataset/` structure without modifying, overwriting, or deleting any original audio evidence files. Cryptographic SHA-256 hashes confirm bit-level immutability across all files.

---

## 2. Key Dataset Metrics

| Metric | Verified Value | Details |
|---|---|---|
| **Total Original Audio Evidence Files** | **66** files | 64 WAV files, 2 MP3 files |
| **Genuine Audio Count** | **9** files | 8 WAV files + 1 MP3 demonstration file |
| **Fake / Synthetic Audio Count** | **57** files | 56 WAV files + 1 MP3 demonstration file |
| **Total Audio Duration** | **30,069.47 s** (~8.35 hours) | Genuine: 1.06 hours; Fake: 7.30 hours |
| **Audio File Formats** | **WAV** (64 files), **MP3** (2 files) | Decoded safely via `soundfile` / `libsndfile` |
| **Original Sampling Rates** | 40,000 Hz (4), 44,100 Hz (59), 48,000 Hz (3) | Verified via audio container headers |
| **Missing Files** | **0** | All expected source recordings are accounted for |
| **Corrupted Files** | **0** | All 66 files pass audio decoding and header checks |
| **Derivative Preprocessed Files** | **66** files | Standardised 16 kHz mono WAVs in `dataset/preprocessed/` |
| **Companion Feature Metadata** | **11,778** balanced slices | `dataset/metadata/DATASET-balanced.csv` (5,889 Real, 5,889 Fake) |

---

## 3. Dataset Folder Structure

The project's `dataset/` directory is organized into distinct partitions separating raw evidence exhibits, preprocessed derivatives, and forensic metadata:

```text
dataset/
├── real/                                     # Original bona fide audio evidence (read-only exhibits)
│   ├── biden-original.wav                    # (48,000 Hz, stereo, 600.00 s)
│   ├── linus-original.wav                    # (44,100 Hz, stereo, 570.36 s)
│   ├── linus-original-DEMO.mp3               # (44,100 Hz, stereo, 60.00 s)
│   ├── margot-original.wav                   # (44,100 Hz, stereo, 79.49 s)
│   ├── musk-original.wav                     # (44,100 Hz, stereo, 600.14 s)
│   ├── obama-original.wav                    # (48,000 Hz, stereo, 600.45 s)
│   ├── ryan-original.wav                     # (44,100 Hz, stereo, 93.69 s)
│   ├── taylor-original.wav                   # (44,100 Hz, stereo, 600.18 s)
│   └── trump-original.wav                    # (48,000 Hz, stereo, 600.43 s)
├── fake/                                     # Original synthetic / deepfake audio evidence (read-only exhibits)
│   ├── biden-to-linus.wav                    # (44,100 Hz, stereo, 600.00 s)
│   ├── biden-to-margot.wav                   # (44,100 Hz, stereo, 600.00 s)
│   ├── biden-to-musk.wav                     # (44,100 Hz, stereo, 600.00 s)
│   ├── biden-to-Obama.wav                    # (40,000 Hz, stereo, 600.00 s)
│   ├── biden-to-ryan.wav                     # (44,100 Hz, stereo, 600.00 s)
│   ├── biden-to-taylor.wav                   # (44,100 Hz, stereo, 600.00 s)
│   ├── biden-to-Trump.wav                    # (40,000 Hz, stereo, 600.00 s)
│   ├── linus-to-musk-DEMO.mp3                # (44,100 Hz, stereo, 60.00 s)
│   └── ... (49 additional pairwise voice-conversion deepfake WAVs)
├── metadata/                                 # Provenance and benchmark metadata
│   └── DATASET-balanced.csv                  # 11,778 balanced 1-sec slices (5,889 Fake, 5,889 Real)
├── preprocessed/                             # Standardised working derivatives (never overwrite originals)
│   ├── real/                                 # 9 mono 16 kHz WAV files (*_16k_mono.wav)
│   └── fake/                                 # 57 mono 16 kHz WAV files (*_16k_mono.wav)
├── evidence_manifest.json                    # Forensic audit trail with source and derivative SHA-256 hashes
└── README.md                                 # Dataset handling policy and terms
```

---

## 4. Label Verification Methodology

To strictly adhere to forensic integrity standards, **labels were not assigned based on filename heuristics or parent folder names alone**. Instead, ground-truth labels were validated against the source dataset documentation and experimental pipeline:

### 4.1 Provenance of Downloaded Archives
1. **Primary Dataset: DEEP-VOICE (DeepFake Voice Recognition)**
   - **Source:** Kaggle benchmark dataset for voice conversion deepfake detection.
   - **Original Layout:** Contained `KAGGLE/AUDIO/REAL/`, `KAGGLE/AUDIO/FAKE/`, and `DEMONSTRATION/DEMONSTRATION/`.
2. **Identification of Folder Ambiguity in `DEMONSTRATION/DEMONSTRATION/`:**
   - In the downloaded archive, the folder `DEMONSTRATION/DEMONSTRATION/` contains two files:
     - `linus-original-DEMO.mp3`
     - `linus-to-musk-DEMO.mp3`
   - If folder-level labeling had been used, both files would have been assigned the same label.
   - Cross-referencing the dataset author's tutorial notebook and audio provenance confirmed that `linus-original-DEMO.mp3` was recorded directly from the authentic speaker (Linus Sebastian) as a genuine benchmark, while `linus-to-musk-DEMO.mp3` was synthesized using Retrieval-based Voice Conversion (RVC) targeting Elon Musk's vocal profile.
   - Consequently, `linus-original-DEMO.mp3` was integrated into `dataset/real/` and `linus-to-musk-DEMO.mp3` into `dataset/fake/`.

### 4.2 Cross-Validation with `DATASET-balanced.csv`
- The companion file `DATASET-balanced.csv` contains 11,778 feature rows extracted across 1-second sliding windows from these exact source files.
- The ground-truth distribution in `DATASET-balanced.csv` is:
  - `REAL`: 5,889 samples (50.0%)
  - `FAKE`: 5,889 samples (50.0%)
- The acoustic feature distributions (20 MFCCs, spectral centroid, spectral bandwidth, spectral rolloff, zero-crossing rate, RMS energy, and chroma) match the feature ranges extracted by our independent pipeline in `src/feature_extraction.py`.

---

## 5. File Integrity and Forensic Evidence Chain of Custody

1. **Original Evidence Immutability:**
   - Original audio files in `dataset/real/` and `dataset/fake/` are treated as read-only exhibits.
   - No original file was resampled, re-encoded, renamed, or modified in place.
2. **Cryptographic SHA-256 Manifest:**
   - Every file was hashed in streaming 64 KB binary chunks.
   - An immutable record is preserved in `dataset/evidence_manifest.json`, linking each original file to its derivative:
     - Example (Genuine): `dataset/real/biden-original.wav`  
       SHA-256: `2732f505f4ede866451a48b069f7a505b0ec3166fabae9a43ab03035f47f2a45`  
       Derivative: `dataset/preprocessed/real/biden-original_16k_mono.wav`  
       Derivative SHA-256: `93627e45d4903e240ba102a54b54c5ccca6ff89d8287047ffe26b2382818081a`
     - Example (Synthetic): `dataset/fake/Obama-to-Biden.wav`  
       SHA-256: `a937a4c7e6c98caef793d508933b9148dffabdf7ec79409e51cce2a6be851c22`  
       Derivative: `dataset/preprocessed/fake/Obama-to-Biden_16k_mono.wav`  
       Derivative SHA-256: `cbe67cb88da87eb13fe909d93fc909a8f4c278c772cb1b09b555e783a3d2e389`
3. **Absence of Corrupted / Missing Files:**
   - Every audio container was parsed using `soundfile.info()` without errors.
   - Zero files were 0 bytes or contained unreadable audio streams.

---

## 6. Issues Requiring Your Attention Before Phase 3

Before model training begins in Phase 3, the following three methodological and architectural points require consideration:

### Issue 1: Class Imbalance in Full-Length Raw Audio Files
- **Observation:** The full-length recordings consist of **9 genuine** and **57 synthetic** files (13.6% vs 86.4%). This imbalance occurred because the dataset creators took 8 source speaker voices and performed all pairwise voice conversions (8 × 7 = 56 synthetic pairs).
- **Recommendation:** 
  - For frame/slice-based training, utilize the balanced representation (such as the 11,778 1-second balanced slices in `dataset/metadata/DATASET-balanced.csv` or frame-level extraction), which provides an exact 50/50 balance (5,889 Real vs 5,889 Fake).
  - If training on aggregate file-level representations, class-weighting (e.g. `compute_class_weight('balanced')` or SMOTE) and threshold-independent metrics (ROC-AUC, Precision-Recall AUC, EER) should be enforced.

### Issue 2: Speaker-Disjoint Splitting (Strict Data Leakage Prevention)
- **Observation:** All 56 fake files are conversions between the 8 real speakers (Biden, Linus, Margot Robbie, Musk, Obama, Ryan Reynolds, Taylor Swift, Trump).
- **Risk:** If audio from the same speaker appears in both training and test sets, models will memorize individual vocal characteristics rather than learning generic synthetic voice conversion artifacts.
- **Recommendation:** Phase 3 dataset splitting must employ **GroupKFold** or **Leave-One-Speaker-Group-Out** cross-validation, keeping held-out speakers strictly unseen during training.

### Issue 3: Secondary Benchmark Dataset (`archive.zip`)
- **Observation:** The 17.2 GB archive (`archive.zip`) containing the Fake-or-Real (FoR) dataset (169,754 files) is present in `Downloads`.
- **Recommendation:** Keep `archive.zip` available for secondary cross-dataset out-of-distribution generalization testing after establishing baseline models on the primary dataset.

---

## 7. Verification Sign-Off

The dataset integration into `AI_Audio_Forensics/dataset/` is **verified, forensically sound, and ready for Phase 3**. 

- Model training has **not** been started.
- All 24 unit tests pass (`python -m pytest -q`).
- Code compilation verified with zero errors (`python -m compileall src tests main.py`).

