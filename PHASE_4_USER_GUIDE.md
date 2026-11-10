# PHASE 4 — USER GUIDE & OPERATIONAL MANUAL
## AI-Based Voice Analysis for Detecting Synthetic and Deepfake Audio in Criminal Investigations

**System:** AI Audio Forensics Investigation Platform  
**Phase:** 4 — Audio Prediction, Visual Analysis & Forensic Reporting  
**Environment:** Python 3.14.3 (`.venv314`)  

---

## 1. Quick Start

Ensure your virtual environment is active in PowerShell:

```powershell
.\.venv314\Scripts\Activate.ps1
```

---

## 2. Interactive Web Interface (Streamlit)

Launch the full graphical forensic examination interface in your default web browser:

```powershell
python main.py --ui
```
*Or directly via Streamlit:*
```powershell
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

### Key Features in the Web Interface:
1. **Evidence Ingestion:**
   - Drag and drop any `.wav`, `.mp3`, `.flac`, or `.ogg` recording.
   - Or select directly from the 66 cataloged case exhibits (`linus-original-DEMO.mp3`, `linus-to-musk-DEMO.mp3`, etc.).
2. **Instant Metadata & Hashing:**
   - Automatic SHA-256 digest calculation upon file selection.
   - Real-time Evidence Tracking ID generation (`EVID-...`).
   - In-browser audio playback player (`st.audio`).
3. **Run Forensic Examination:**
   - Click **"🚀 Run Complete Forensic Authenticity Examination"**.
   - Displays real-time status badges (`COMPLETED`, `INDETERMINATE`, `OUT_OF_DISTRIBUTION`, `QUALITY_CHECK_FAILED`).
4. **Visual Evidence Tabs:**
   - **Spectrogram & Visual Evidence:** High-resolution 6-panel overview (Waveform, STFT, Mel, MFCC, Dynamics).
   - **Multi-Model Consensus:** Side-by-side comparison of Calibrated SVM, Random Forest, and Logistic Regression.
   - **Acoustic Quality & OOD Screen:** Estimated SNR, Clipping Ratio, Silence Percentage, and Feature Outlier metrics.
   - **Formal Report & Export:** Preview the 10-section report and click **"Download Markdown"** or **"Download PDF"** with one click.

---

## 3. Command-Line Interface (CLI)

The command-line interface in `main.py` provides scriptable forensic examination:

### 3.1 Run Complete Forensic Prediction on an Exhibit
```powershell
python main.py --predict dataset/real/linus-original-DEMO.mp3
```
- Performs quality screening.
- Extracts 26 forensic features.
- Runs multi-model inference (SVM, Random Forest, Logistic Regression).
- Generates 6-panel spectrogram plots in `results/graphs/`.
- Generates formal Markdown report in `results/reports/`.
- Compiles formal companion PDF in `results/reports/`.
- Appends immutable event to `results/reports/forensic_audit_trail.jsonl`.

### 3.2 Generate Visual Spectrogram Figures Only
```powershell
python main.py --analyze-audio dataset/fake/linus-to-musk-DEMO.mp3
```
Saves high-resolution PNG figures to `results/graphs/`.

### 3.3 Inspect Model Artifacts & Validation Status
```powershell
python main.py --models
```

### 3.4 Execute Automated Test Suite
```powershell
python -m pytest -q
```
Executes all 50 unit and integration tests.

---

## 4. Understanding System Statuses & Results

| Status Code | Meaning | Forensic Guidance |
|---|---|---|
| `COMPLETED` | Examination completed within validated domain. | Review posterior probability, multi-model consensus, and visual figures. |
| `INDETERMINATE (QUALITY_CHECK_FAILED)` | Signal failed quality thresholds (duration < 0.5s, clipping > 5%, or silence > 80%). | Re-request original recording or inspect physical capture media. |
| `INDETERMINATE (OUT_OF_DISTRIBUTION)` | Acoustic features fall outside the validated empirical training bounds ($z > 6.47$). | Automated classification withheld to prevent false alarms. Specialized biometric review required. |
| `INDETERMINATE (MODEL_DISAGREEMENT)` | Primary classifiers (SVM vs Random Forest) produced contradictory findings. | Boundary ambiguity detected; do not treat as definitive genuine or synthetic. |
| `INTEGRITY_ERROR` | Cryptographic SHA-256 changed during analysis. | Digital evidence altered; result invalidated. |

---

*User guide concluded.*
