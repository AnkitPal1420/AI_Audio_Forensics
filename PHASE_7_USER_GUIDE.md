# Phase 7: Forensic Investigator Operating User Guide

**Project:** AI-Based Voice Analysis for Detecting Synthetic and Deepfake Audio in Criminal Investigations  
**Project Root:** `AI_Audio_Forensics`  
**Phase:** 7 — Forensic User Guide  
**Date:** November 2026  
**Audience:** Digital Audio Examiners, Forensic Investigators, Legal Counsel  

---

## 1. Introduction

The **Phase 7 Forensic Intelligence Suite** extends the AI Audio Forensics dashboard with explainability, physical edit detection, anomaly screening, and cross-evidence correlation.

To access these capabilities:
1. Launch the dashboard:
   ```powershell
   .venv314\Scripts\python.exe -m streamlit run app.py --server.port 8501
   ```
2. Navigate to **`🧠 Forensic Intelligence & Manipulation`** in the left sidebar.
3. Select the primary evidence exhibit you wish to examine from the dropdown menu.

---

## 2. Forensic Workflows

### Workflow 1: Explaining an AI Classification Decision
1. Navigate to the **`🔍 Explainable AI (XAI)`** sub-tab.
2. Select the classifier model you wish to interpret (Recommended: `calibrated_svm`).
3. Click **`🚀 Run Explainability Analysis`**.
4. **Interpreting Results:**
   - Review the predicted label and probability.
   - Inspect the **Top Feature Drivers toward SYNTHETIC Classification**: these features exerted the strongest positive Shapley attribution.
   - Inspect the **Top Feature Drivers toward GENUINE Classification**: these features pulled the model toward genuine speech.
   - Check the **Grouped Feature Attribution Share** to understand whether MFCC means or variance (temporal stability) drove the decision.
5. *Legal Note:* Always include the displayed "Model Unvalidated" notice in your notes.

### Workflow 2: Detecting Splicing, Cuts & Energy Discontinuities
1. Navigate to the **`⚡ Manipulation & Splicing`** sub-tab.
2. Adjust detection sensitivity sliders (defaults are calibrated for general speech):
   - *Spectral Flux Sigma Threshold (σ):* 3.5
   - *RMS Step Sigma Threshold (σ):* 3.5
   - *Noise Floor Shift Threshold (dB):* 6.0
3. Click **`⚡ Execute Signal Tampering Analysis`**.
4. **Interpreting Results:**
   - Review summary metrics for flagged anomalies.
   - Inspect the **Frame-by-Frame Signal Trajectories** to visually locate spikes.
   - Read the **Timestamped Acoustic Findings** table for exact millisecond start/end intervals.
   - Listen to the flagged interval in the Advanced Signal Workspace to verify whether the anomaly is a natural transient (e.g., mouth click) or an edit splice.

### Workflow 3: Comparing Acoustic Profiles of Two Exhibits
1. Navigate to the **`👥 Speaker Comparison Baseline`** sub-tab.
2. Select **Exhibit B** from the dropdown menu.
3. Click **`🔬 Compute Acoustic Distance Baseline`**.
4. **Interpreting Results:**
   - **Cosine Similarity (0 to 1):** Higher values indicate similar overall spectral shapes.
   - **Euclidean Distance:** Lower values indicate close proximity in acoustic feature space.
   - **Sub-Band Resemblance:** Breaks down similarity across Low, Mid, and High vocal frequencies.
5. *Critical Prohibition:* You **must not** claim in investigative reports that the two exhibits share the same speaker identity based on this metric.

### Workflow 4: Profiling Vocoder Bandwidth & Cutoff
1. Navigate to the **`🤖 Generator Characterization`** sub-tab.
2. Click **`🔍 Profile Vocoder Bandwidth & Artifacts`**.
3. Review the **Effective Bandwidth (Hz)** and **Candidate Architecture Class**.
   - If bandwidth is $\le 4.2 \text{ kHz}$, the recording has undergone narrowband telephony compression or legacy synthesis.
   - If bandwidth is $> 12.0 \text{ kHz}$, the audio preserves full-band acoustic harmonics.

### Workflow 5: Unsupervised Acoustic Anomaly Detection
1. Navigate to the **`🚨 Acoustic Anomaly Detection`** sub-tab.
2. Choose between **`IsolationForest`** (global manifold) or **`LOF`** (local density).
3. Set expected contamination fraction (default: 0.05 / 5%).
4. Click **`🚨 Run Anomaly Screening`**.
5. Examine the anomaly score trajectory graph and review the flagged cluster intervals.

### Workflow 6: Advanced Signal Inspection & Audio Playback
1. Navigate to the **`🔬 Advanced Signal Workspace`** sub-tab.
2. Listen to the recording using the integrated audio player.
3. Use the **Interactive Temporal Region Selector** sliders to isolate any region of interest (down to 100ms precision).
4. Inspect the zoomed waveform to examine individual pitch periods or abrupt phase reversals.

### Workflow 7: Multi-Evidence Case Correlation
1. Navigate to the **`🔗 Multi-Evidence Correlation`** sub-tab.
2. Select the target exhibit you wish to correlate against.
3. Click **`🔗 Execute Multi-Evidence Correlation`**.
4. Review the **SHA-256 Identity**, **Metadata Differences**, and **Temporal Alignment Offset**.
   - If temporal alignment peak correlation $\ge 0.85$, the two exhibits are identical or excerpted copies with a measured time shift.
   - All comparison records are automatically saved to the case file in SQLite.

