# Phase 8: Forensic Examiner & Investigator User Operating Guide

**Project:** AI-Based Voice Analysis for Detecting Synthetic and Deepfake Audio in Criminal Investigations  
**Project Root:** `AI_Audio_Forensics`  
**Phase:** 8.13 — User Operating Guide  
**Audience:** Digital Audio Examiners, Forensic Analysts, Reviewing Officers  
**Date:** November 2026  

---

## 1. Accessing the Forensic Portal

1. Open your web browser and navigate to the laboratory interface:
   ```
   http://127.0.0.1:8501
   ```
2. Enter your assigned **Examiner Username** and **Password** in the access portal.
3. Click **`🔒 Authenticate & Access Evidence`**.
   *Note: After 5 failed password attempts, your account will be locked for 15 minutes.*
4. Upon authentication, your active role and username will be displayed in the left navigation sidebar.

---

## 2. Standard Forensic Casework Workflow

```
[1. Case Creation] ──► [2. Evidence Ingestion] ──► [3. Signal Inspection]
                                                          │
                                                          ▼
[6. Forensic Report] ◄── [5. Phase 7 Intelligence] ◄── [4. AI Prediction]
```

### Step 1: Case Creation
1. Select **`📁 Case Management`** in the sidebar.
2. Switch to the **`➕ Create New Forensic Case`** tab.
3. Input the official **Case Reference ID** (e.g., `CASE-20261110-002`), **Case Title**, **Description**, and **Lead Investigator**.
4. Click **`Create Forensic Case`**. The case is registered in the immutable SQLite registry.

### Step 2: Evidence Ingestion & Cryptographic Intake
1. Select **`📤 Evidence Ingestion & Upload`**.
2. Select your target Case ID from the dropdown.
3. Submit the audio evidence exhibits (drag & drop or browse).
   - Supported containers: **WAV, MP3, FLAC, OGG, M4A**.
   - Maximum file size: **250 MB**.
   - Security Notice: All uploads undergo automated binary magic-byte inspection. Executables or polyglot files are rejected immediately.
4. Click **`🔒 Securely Ingest Evidence Files`**.
   - The file is stored in `storage/original_evidence/`.
   - OS-level read-only file attributes are applied.
   - An immediate SHA-256 intake digest is calculated and logged.

### Step 3: Acoustic Signal Inspection
1. Select **`🔬 Audio Examination Workspace`**.
2. Select the exhibit from the dropdown.
3. Review the publication-grade visualizations across 3 tabs:
   - **Master 6-Panel Spectrogram:** Time-domain waveform, STFT linear spectrogram, 128-band Mel spectrogram, MFCC heatmap, and energy curves.
   - **Trajectory Dynamics:** Pitch proxy, spectral centroid, and spectral rolloff curves.
   - **Numerical Summary:** Detailed acoustic statistics (SNR, clipping, silence ratio).

### Step 4: AI Synthetic Audio Detection & Consensus
1. Select **`🤖 AI Detection & Consensus`**.
2. Select the exhibit.
3. Choose the model (Recommended: **`calibrated_svm`** or **`ensemble`**).
4. Click **`Execute Deepfake Detection`**.
5. **Evaluating Output:**
   - **Classification Verdict:** `GENUINE`, `SYNTHETIC`, or `INDETERMINATE`.
   - **Multi-Model Consensus:** Displays agreement across Calibrated SVM, Random Forest, and Logistic Regression.
   - *Mandatory Caveat:* Review the displayed Phase 6 Scientific Disclaimer. External generalization on unseen acoustic channels is unvalidated (FoR ROC-AUC 0.283–0.534). Do not present automated classification as definitive legal fact.

### Step 5: Advanced Forensic Intelligence (Phase 7)
Navigate to **`🧠 Forensic Intelligence & Manipulation`** to access 7 specialized forensic workspaces:
- **`🔍 Explainable AI (XAI)`:** Inspect Monte Carlo Shapley attributions explaining which acoustic parameters pushed the model toward real or synthetic.
- **`⚡ Manipulation & Splicing`:** Detect millisecond-level spectral flux surges, RMS energy jumps, noise floor shifts, and cloned audio segments.
- **`👥 Speaker Comparison Baseline`:** Evaluate exploratory acoustic distance between two exhibits. *(Note: Biometric speaker identity confirmation is strictly prohibited).*
- **`🤖 Generator Characterization`:** Inspect vocoder bandwidth cutoffs and candidate architecture tiers.
- **`🚨 Acoustic Anomaly Detection`:** Identify statistical vocal outliers using unsupervised Isolation Forest or LOF.
- **`🔬 Advanced Signal Workspace`:** Zoom into suspected regions with the 100ms temporal cropper and listen to audio playback.
- **`🔗 Multi-Evidence Correlation`:** Cross-correlate exhibits for cryptographic identity, metadata divergence, and temporal alignment offsets.

### Step 6: Generating Forensic Examination Reports
1. Navigate to **`📄 Forensic Report Center`**.
2. Select the completed examination record.
3. Click **`Generate Forensic Report`**.
4. Download the generated **Court-Ready PDF Report** or **Markdown Source Document**.

---

## 3. Ending Your Session
When casework is completed, click **`🚪 Sign Out`** in the left sidebar. This invalidates your session token in the database and clears browser memory.

