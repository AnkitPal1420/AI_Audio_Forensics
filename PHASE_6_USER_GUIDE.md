> **⚠ SUPERSEDED IN PART — read PHASE_6_ERRATA_AND_VERIFIED_METRICS.md first. This report contains numbers/claims later found to be unverified or incorrect (e.g. feature parity, cross-dataset generalization, calibration independence, threshold selection, test counts). Where they conflict, the errata and results/phase6_verified_metrics.json govern.**

# PHASE 6 — OPERATOR & INVESTIGATOR USER GUIDE
## Scientific Validation, Error Disclosures & Operational Workspaces
**Project:** AI-Based Voice Analysis for Detecting Synthetic and Deepfake Audio in Criminal Investigations  
**Project Root:** `AI_Audio_Forensics`  
**Phase:** Phase 6 — Advanced Forensic Validation & Scientific Reliability  
**Date:** November 10, 2026  

---

## 1. ACCESSING THE SCIENTIFIC VALIDATION WORKSPACE

1. Launch the forensic web interface via [`run_ui.bat`](file:///c:/Users/ANKIT/OneDrive/Desktop/AI_Audio_Forensics/run_ui.bat) or browse to:
   **`http://localhost:8501`**
2. In the left-hand navigation sidebar, select:
   **`📊 Scientific Validation & Reliability`**
3. Review the top status cards:
   - **Feature Pipeline Parity:** Confirms active $(1, 26)$ schema match.
   - **Speaker Independence:** Shows 8-fold LOSO mean AUC ($77.58\%$).
   - **Telephony Vulnerability:** Warning indicator for G.711 telephony audio.

---

## 2. INTERPRETING SCIENTIFIC VALIDATION TABS

### Tab 1: 🧑‍🔬 Speaker Independence & LOSO
- Review how the model performs on each of the 8 individual subjects when held out during training.
- Use this data to determine whether your exhibit's vocal profile aligns with well-performing or high-shift acoustic categories.

### Tab 2: 🌐 Cross-Dataset Evaluation (FoR Benchmark)
- Demonstrates performance on multi-speaker neural text-to-speech ($51.5\%$) vs. room re-recordings ($38.0\%$).
- **Rule of Thumb:** If your evidence exhibit is suspected to be a recording of a loudspeaker playing in a room, automated analysis is inconclusive.

### Tab 3: ⚖️ Probability Calibration & Threshold Tuning
- **Calibrated Posterior:** In forensic casework, examine the Platt-scaled probability rather than raw decision distances.
- **Operating Thresholds:**
  - $P \ge 0.80$: High-confidence forensic synthetic threshold.
  - $0.30 \le P < 0.80$: Indeterminate triage band (recommend expert human acoustic inspection).
  - $P < 0.30$: Genuine human speech indicator.

### Tab 4: 🔊 Channel Robustness Stress Tests
- Inspect the 9 degradation conditions.
- If audio exhibits $\text{SNR} < 15$ dB or bandwidth $\le 3,400$ Hz, consult the degradation table before reporting findings.

### Tab 5: 🛡️ OOD Manifold Bounds
- Inspect Ledoit-Wolf Mahalanobis distance thresholds ($D_M > 6.93$).

### Tab 6: 📥 Phase 6 Report Exporter
- Download formal markdown validation documents directly to your local computer with 1 click.

---

## 3. COMMAND-LINE AUDITING & VERIFICATION

```powershell
# Run the complete 80-test regression suite
.\.venv314\Scripts\python.exe -m pytest -q

# Run feature extraction determinism tests
.\.venv314\Scripts\python.exe -m pytest tests/test_phase6_feature_pipeline.py -v

# Run channel robustness and calibration tests
.\.venv314\Scripts\python.exe -m pytest tests/test_phase6_validation_robustness.py -v
```

