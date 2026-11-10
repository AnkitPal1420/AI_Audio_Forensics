# Phase 7: Forensic Scientific Limitations & Court Admissibility Report

**Project:** AI-Based Voice Analysis for Detecting Synthetic and Deepfake Audio in Criminal Investigations  
**Project Root:** `AI_Audio_Forensics`  
**Phase:** 7 — Scientific Reliability & Limitations Audit  
**Date:** November 2026  
**Standards:** Daubert Standard (US Federal Rule of Evidence 702), Frye Standard, ENFSI Guidelines  

---

## 1. Executive Summary

This report establishes the formal boundary conditions, scientific caveats, and legal limitations of the Phase 7 forensic intelligence pipeline. Forensic software used in criminal proceedings must adhere to strict principles of transparency, known error rates, and empirical validation.

---

## 2. Comprehensive Inventory of Scientific Limitations

### 2.1 Limitation 1: Unvalidated Classifier Generalization & Telephony Blind Spot
- **Empirical Reality:** In Phase 6 auditing against the external Fake-or-Real (FoR) benchmark, the Calibrated SVM achieved chance-level or inverted ROC-AUC:
  - Clean TTS Sub-benchmark: **ROC-AUC = 0.534**
  - Acoustic Re-recorded Sub-benchmark: **ROC-AUC = 0.283** (inverted performance)
- **Telephony Vulnerability:** Under G.711 narrowband telephony simulation (300–3,400 Hz bandpass filtering), synthetic detection recall dropped to **0.00%** (100% false negative rate).
- **Forensic Guidance:** Binary classification predictions (REAL vs SYNTHETIC) **must not be presented as definitive fact**. Classification outputs should be described strictly as investigative screening indicators.

### 2.2 Limitation 2: Explainability is Not Causality
- **Empirical Reality:** Feature attribution methods (Shapley sampling, occlusion sensitivity, logistic regression coefficients) explain the internal mathematical mechanics of the machine-learning model on that specific feature vector.
- **Forensic Guidance:** High feature attribution for `mfcc_10_mean` means that this feature shifted the classifier's probability toward "SYNTHETIC". It **does not prove** that the recording was synthesized or that a vocoder modified that frequency band.

### 2.3 Limitation 3: Transient Signal Discontinuities vs Edit Splicing
- **Empirical Reality:** Natural human speech contains physical acoustic transients:
  - Bilabial and alveolar plosives ($/p/$, $/t/$, $/k/$).
  - Glottal stops and sharp voice onsets.
  - Saliva clicks, tongue releases, and abrupt breath intakes.
- **Forensic Guidance:** Signal tampering indicators (`SPECTRAL_FLUX_DISCONTINUITY`, `RMS_ENERGY_STEP`) represent objective physical measurements of acoustic rates of change. They **must not be cited as conclusive proof of audio editing** without manual spectrogram verification and background noise phase continuity analysis.

### 2.4 Limitation 4: Strict Biometric Speaker Identity Prohibition
- **Empirical Reality:** Biometric speaker verification requires deep neural embedding networks (ECAPA-TDNN) trained on millions of multi-speaker utterances, evaluated with probabilistic linear discriminant analysis (PLDA). The runtime environment lacks PyTorch/SpeechBrain, and the dataset lacks cross-channel speaker identity labels.
- **Forensic Guidance:** The computed acoustic similarity scores (cosine similarity, Euclidean distance) evaluate spectral envelope resemblance only. Asserting that two recordings originate from the same speaker based on these metrics is **scientifically invalid and legally inadmissible**.

### 2.5 Limitation 5: Absence of Multi-Class Generator Attribution
- **Empirical Reality:** The system models synthetic detection as a binary problem (REAL vs FAKE). The dataset contains no vocoder family labels.
- **Forensic Guidance:** Cutoff frequency analysis categorizes effective bandwidth tiers (e.g., Narrowband vs Wideband), but **cannot attribute audio to specific commercial or open-source software** (e.g., "ElevenLabs" or "VITS").

---

## 3. Court Admissibility Checklist (Federal Rule of Evidence 702)

| Daubert Factor | Project Status | Compliance Notes |
| :--- | :--- | :--- |
| **1. Empirical Testing** | **Compliant** | Rigorously evaluated in Phase 6 across 8-fold LOSO and 300 external benchmark files. |
| **2. Known Error Rates** | **Compliant** | Documented false positive rates (FPR), false negative rates (FNR), and ECE are published. |
| **3. Peer Review & Standards** | **Compliant** | Standardized MFCC signal processing and ISO/IEC 27037 cryptographic hashing. |
| **4. Maintenance of Standards** | **Compliant** | Full software versioning, immutable SQLite audit trails, and strict schema validation. |
| **5. General Acceptance** | **Partial** | Acoustic signal analysis is widely accepted; ML generalization remains unvalidated. |
