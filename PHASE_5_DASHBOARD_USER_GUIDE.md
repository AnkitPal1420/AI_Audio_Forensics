# PHASE 5 — FORENSIC DASHBOARD USER GUIDE
## AI-Based Voice Analysis for Detecting Synthetic and Deepfake Audio in Criminal Investigations

**System:** Forensic Audio Web Dashboard  
**Phase:** 5 — Evidence Case Management & Forensic Analytics  
**Environment:** Python 3.14.3 (`.venv314`)  
**Standard Alignment:** ISO/IEC 27037:2012 Technical Principles  

---

## 1. Quick Launch Instructions

### Method A: One-Click Windows Launcher
In your file explorer, navigate to the project directory and double-click:
```text
run_ui.bat
```

### Method B: PowerShell Terminal Launch
Open PowerShell in `AI_Audio_Forensics`:
```powershell
.\.venv314\Scripts\Activate.ps1
python -m streamlit run app.py
```
Open **`http://localhost:8501`** in your web browser.

---

## 2. Navigating the Dashboard

The dashboard provides 9 specialized forensic workspaces accessible via the left sidebar:

### 1. 🏠 Dashboard Overview
- View high-level metrics: Active Cases, Total Exhibits, Completed Examinations, Genuine vs. Synthetic counts.
- Inspect the active status of the machine learning engines (Ensemble, Calibrated SVM, Random Forest, Logistic Regression).
- Review recent investigative cases.

### 2. 📁 Case Management
- **Create a Case:** Enter a unique Case ID (e.g. `CASE-20261110-001`), descriptive title, lead examiner name, and investigative context notes.
- **Browse & Filter:** Search cases by ID, title, or examiner name. Filter cases by status (`ACTIVE`, `PENDING_REVIEW`, `CLOSED`, `ARCHIVED`).
- **Update Status:** Change case status as the investigation progresses.

### 3. 📤 Evidence Ingestion & Upload
- Select an active forensic case from the dropdown.
- Drag and drop one or more audio exhibits (`.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`).
- Click **"Securely Ingest Evidence Files"**. The system will:
  - Sanitize the filename to prevent path traversal.
  - Copy the raw file to `storage/original_evidence/`.
  - Calculate the cryptographic SHA-256 intake digest immediately.
  - Generate a unique Evidence Tracking ID (`EVID-...`).
  - Record the exhibit metadata in the SQLite database.

### 4. 🔬 Audio Examination Workspace
- Select any ingested exhibit from the dropdown.
- Listen to the audio recording using the in-browser playback controls.
- Review audio container specifications, duration, sampling rate, and SHA-256 hash.
- Inspect the **Master 6-Panel Spectrogram**:
  1. Time-domain waveform.
  2. STFT linear frequency spectrogram (dBFS).
  3. 128-band Log Mel-frequency spectrogram.
  4. MFCC 1–20 temporal heatmap.
  5. Centroid and 85% Rolloff dynamics curves.
  6. RMS energy and Zero-Crossing Rate curves.

### 5. 🤖 AI Detection & Consensus
- Select an exhibit and choose the primary classification model (`ensemble`, `calibrated_svm`, `random_forest`, `logistic_regression`).
- Click **"Execute Calibrated Forensic Detection"**.
- Review the primary finding (`GENUINE`, `SYNTHETIC`, `INDETERMINATE`).
- Review the calibrated posterior probabilities and confidence score.
- Review the multi-model comparison table: shows individual model predictions and checks for consensus vs. disagreement.
- Automatically compiles formal Markdown and PDF reports and records the examination into the database.

### 6. 🔒 Evidence Integrity Verification
- Select an exhibit to audit.
- Click **"Recalculate SHA-256 and Verify Bitstream Immutability"**.
- The system reads the stored file, recalculates the SHA-256 hash in real-time, and compares it to the intake record.
- **VERIFIED:** Green confirmation that the file has remained byte-for-byte unaltered.
- **HASH MISMATCH:** Red alert if any byte of the file on disk was modified or corrupted.
- Logs the verification timestamp and examiner notes into the database audit trail.

### 7. 📄 Forensic Report Center
- Select any completed examination.
- One-click download for the formal 10-section **Markdown Report (`.md`)**.
- One-click download for the court-ready **PDF Report (`.pdf`)** with embedded figures.
- Live expandable preview of the complete report text.

### 8. 📜 Examination Audit Trail
- Filter historical examinations by finding (`ALL`, `GENUINE`, `SYNTHETIC`, `INDETERMINATE`) or Case ID.
- Inspect detailed system audit events capturing all administrative and computational actions.

### 9. ⚙️ System Settings & Diagnostics
- Review execution environment details (Python version, OS architecture, git commit hash).
- Inspect data-driven Out-of-Distribution (OOD) threshold bounds.
- Execute an on-demand SQLite database integrity check (`PRAGMA integrity_check`).

---

## 3. Mandatory Forensic Disclaimer

> **IMPORTANT LEGAL NOTICE:**  
> This system provides automated investigative screening indicators based on statistical signal processing. In accordance with ISO/IEC 27037 technical principles, its outputs represent analytical indicators and do **not** constitute definitive legal proof of authentic human origin or criminal forgery. All findings must be corroborated by chain of custody, witness testimony, contextual evidence, and independent expert review.
