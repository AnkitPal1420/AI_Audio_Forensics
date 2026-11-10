# AI-Based Voice Analysis for Detecting Synthetic and Deepfake Audio in Criminal Investigations

**Project Root:** `AI_Audio_Forensics`  
**Current Phase:** Phase 4 — Audio Prediction & Forensic Reporting  
**Environment:** Python 3.14.3 (`.venv314`)  
**Standard Alignment:** ISO/IEC 27037:2012 Technical Principles & SWGDE Guidelines  

---

## 1. System Overview

This platform provides an auditable, scientifically validated digital forensic system designed to assist criminal investigators and forensic analysts in detecting synthetic voice conversions and deepfake speech.

### Key Capabilities:
- **12-Step Forensic Examination Pipeline:** Enforces signal validation, SHA-256 evidence hashing, non-destructive 16 kHz standardisation, acoustic quality screening, and immutable audit logging.
- **Empirical Out-of-Distribution (OOD) Screening:** Validates feature vectors against empirical training bounds ($z$-scores) to withhold automated classification on out-of-domain signals.
- **Multi-Model Consensus & Disagreement Engine:** Evaluates exhibits across Calibrated SVM (Platt-scaled RBF), Random Forest (100 trees), and Logistic Regression, flagging contradictory classifications as `INDETERMINATE (MODEL_DISAGREEMENT)`.
- **High-Resolution Acoustic Visualizations:** Generates publication-grade 6-panel overview figures (Waveform, STFT linear spectrogram, 128-band Mel spectrogram, MFCC 1-20 heatmap, and spectral trajectory dynamics).
- **Formal 10-Section Forensic Reports:** Compiles comprehensive, court-ready examination documents in both GitHub-flavored Markdown (`.md`) and styled PDF (`.pdf`) via ReportLab.
- **Interactive Web Interface:** Streamlit-powered examination dashboard (`app.py`) for file upload, live audio playback, visual inspection, and one-click report export.

---

## 2. Quick Start

### 2.1 Activate Virtual Environment (PowerShell)
```powershell
.\.venv314\Scripts\Activate.ps1
```

### 2.2 Launch Interactive Web Interface
```powershell
python main.py --ui
# or directly:
streamlit run app.py
```
Access the application at **`http://localhost:8501`**.

### 2.3 Run Forensic CLI Assessment on an Exhibit
```powershell
python main.py --predict dataset/real/linus-original-DEMO.mp3
```

### 2.4 Run Visual Spectrogram Analysis
```powershell
python main.py --analyze-audio dataset/fake/linus-to-musk-DEMO.mp3
```

### 2.5 Run Automated Verification Test Suite
```powershell
python -m pytest -q
```
*(All 50 unit and integration tests passing in ~7s).*

---

## 3. Project Directory Structure

| Path | Purpose |
|---|---|
| `app.py` | Interactive Streamlit web interface for forensic examination. |
| `main.py` | Unified command-line interface supporting prediction, training, and UI. |
| `src/audio_prediction.py` | 12-step forensic prediction coordinator and quality screening. |
| `src/deepfake_detection.py` | Multi-model inference, schema alignment, and disagreement protocol. |
| `src/spectrogram_analysis.py` | STFT, Mel, and MFCC spectral trajectory visual generation. |
| `src/audio_quality.py` | Pre-inference acoustic quality (SNR, clipping, silence) and OOD checks. |
| `src/forensic_integrity.py` | SHA-256 intake hashing, ongoing immutability checks, and audit logging. |
| `src/forensic_report.py` | 10-section formal report generator in Markdown and court-ready PDF. |
| `src/audio_preprocessing.py` | Non-destructive audio loading and 16 kHz mono standardisation. |
| `src/feature_extraction.py` | Pure NumPy Slaney 128-band dB-scaled acoustic feature extraction. |
| `src/model_training.py` | Group-labeled Leave-One-Speaker-Out (LOSO) training pipeline. |
| `models/` | Trained model artifacts (`calibrated_svm`, `rf`, `logistic_regression`). |
| `results/graphs/` | High-resolution generated acoustic figures. |
| `results/reports/` | Generated forensic Markdown reports, PDFs, and append-only audit trail. |
| `tests/` | 50 comprehensive unit and integration test suites. |

---

## 4. Forensic & Evidentiary Limitations

> **IMPORTANT LEGAL NOTICE:**  
> This system provides automated investigative screening assessments based on statistical signal processing. In accordance with ISO/IEC 27037 technical principles, its outputs represent analytical indicators and do **not** constitute definitive legal proof of authentic human origin or criminal forgery. All findings must be corroborated by chain of custody, witness testimony, contextual evidence, and independent expert review.
