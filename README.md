# AI-Based Voice Analysis for Detecting Synthetic and Deepfake Audio in Criminal Investigations

**Project Root:** `AI_Audio_Forensics`  
**Current Phase:** Phase 5 — Forensic Web Dashboard & Evidence Case Management  
**Environment:** Python 3.14.3 (`.venv314`), SQLite 3, Streamlit 1.67.0  
**Standard Alignment:** ISO/IEC 27037:2012 Technical Principles & SWGDE Guidelines  

---

## 1. System Overview

This platform provides an auditable, scientifically validated digital forensic system designed to assist criminal investigators, national security analysts, and forensic examiners in detecting synthetic voice conversions and deepfake speech.

### Key Capabilities:
- **Centralized Forensic Database (SQLite):** Persistent relational tracking of cases, evidence exhibits, multi-model examinations, on-demand integrity audits, and event logs.
- **12-Step Forensic Examination Pipeline:** Enforces signal validation, SHA-256 evidence hashing, non-destructive 16 kHz standardisation, acoustic quality screening, and immutable audit logging.
- **Empirical Out-of-Distribution (OOD) Screening:** Validates feature vectors against empirical training bounds ($z$-scores) to withhold automated classification on out-of-domain signals.
- **Multi-Model Consensus & Disagreement Protocol:** Evaluates exhibits across a Weighted Soft Voting Ensemble, Calibrated SVM (Platt-scaled RBF), Random Forest (100 trees), and Logistic Regression, flagging contradictory classifications as `INDETERMINATE (MODEL_DISAGREEMENT)`.
- **Publication-Grade Acoustic Spectrograms:** Generates 6-panel overview figures (Waveform, STFT linear spectrogram, 128-band Mel spectrogram, MFCC 1-20 heatmap, and spectral trajectory dynamics).
- **Automated Dual-Format Reports:** Compiles formal 10-section examination documents in both GitHub-flavored Markdown (`.md`) and court-ready styled PDF (`.pdf`) via ReportLab.
- **Interactive Multi-Page Web Dashboard:** 9 specialized workspaces (Home, Cases, Ingestion, Workspace, AI Detection, Integrity Verification, Report Center, History, Settings).

---

## 2. Quick Start

### 2.1 One-Click Windows Launchers
- **Launch Forensic Web Dashboard:** Double-click [`run_ui.bat`](file:///c:/Users/ANKIT/OneDrive/Desktop/AI_Audio_Forensics/run_ui.bat).
- **Run Automated Test Suite:** Double-click [`run_tests.bat`](file:///c:/Users/ANKIT/OneDrive/Desktop/AI_Audio_Forensics/run_tests.bat).

### 2.2 PowerShell Manual Launch
```powershell
.\.venv314\Scripts\Activate.ps1
python -m streamlit run app.py
```
Access the application at **`http://localhost:8501`**.

### 2.3 Run Forensic CLI Assessment on an Exhibit
```powershell
python main.py --predict dataset/real/linus-original-DEMO.mp3
```

### 2.4 Run Automated Verification Test Suite
```powershell
python -m pytest -q
```
*(All 64 unit, integration, and regression tests passing).*

---

## 3. Project Directory Structure

| Path | Purpose |
|---|---|
| `app.py` | Master Streamlit forensic web application with 9 specialized workspaces. |
| `main.py` | Unified command-line interface supporting prediction, training, and UI. |
| `run_ui.bat` | One-click Windows desktop launcher for the web dashboard. |
| `run_tests.bat` | One-click Windows test suite verification launcher. |
| `database/forensics.db` | SQLite relational database tracking cases, evidence, exams, and audits. |
| `database/database.py` | SQLite data access layer and referential integrity manager. |
| `database/models.py` | Schema dataclasses (ForensicCase, ForensicEvidence, ForensicExamination). |
| `storage/original_evidence/` | Segregated read-only storage for raw physical exhibits. |
| `storage/processed_audio/` | Standardized 16 kHz mono floating-point derivatives. |
| `storage/visualizations/` | Generated publication-grade acoustic figures. |
| `storage/reports/` | Generated Markdown and court-ready PDF examination documents. |
| `src/audio_prediction.py` | 12-step forensic prediction coordinator and quality screening. |
| `src/deepfake_detection.py` | Multi-model inference, schema alignment, and disagreement protocol. |
| `src/spectrogram_analysis.py` | STFT, Mel, and MFCC spectral trajectory visual generation. |
| `src/audio_quality.py` | Pre-inference acoustic quality (SNR, clipping, silence) and OOD checks. |
| `src/forensic_integrity.py` | SHA-256 intake hashing, ongoing immutability checks, and audit logging. |
| `src/forensic_report.py` | 10-section formal report generator in Markdown and court-ready PDF. |
| `src/batch_processing.py` | Batch evidence processor and Master Case PDF generator. |
| `src/ensemble_model.py` | Weighted soft voting ensemble classifier. |
| `src/storage_manager.py` | Path traversal protection and evidence ingestion layer. |
| `tests/` | 64 comprehensive unit and integration test suites. |

---

## 4. Forensic & Evidentiary Limitations

> **IMPORTANT LEGAL NOTICE:**  
> This system provides automated investigative screening assessments based on statistical signal processing. In accordance with ISO/IEC 27037 technical principles, its outputs represent analytical indicators and do **not** constitute definitive legal proof of authentic human origin or criminal forgery. All findings must be corroborated by chain of custody, witness testimony, contextual evidence, and independent expert review.
