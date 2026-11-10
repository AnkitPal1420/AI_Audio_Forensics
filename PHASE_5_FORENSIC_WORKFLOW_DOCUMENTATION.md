# PHASE 5 — FORENSIC WORKFLOW & EVIDENTIARY PROTOCOL
## AI-Based Voice Analysis for Detecting Synthetic and Deepfake Audio in Criminal Investigations

**Standards Alignment:** ISO/IEC 27037:2012 Technical Principles & SWGDE Guidelines  
**Application Architecture:** Phase 5 Forensic Web Dashboard & SQLite Database  
**Author:** Forensic Software Architect & Machine Learning Engineer  

---

## 1. End-to-End Investigative Examination Protocol

The Phase 5 platform implements a standardized, non-destructive 7-stage investigative protocol designed to produce scientifically defensible evidence for legal proceedings:

```mermaid
flowchart TD
    A[Digital Audio Exhibit Intake] --> B[Stage 1: Secure Storage & Intake SHA-256 Hashing]
    B --> C[Stage 2: Non-Destructive Pre-Screening & In-Memory Standardisation]
    C --> D{Pass Quality Screen?}
    D -- No --> E["Halt: Return INDETERMINATE (QUALITY_CHECK_FAILED)"]
    D -- Yes --> F[Stage 3: Feature Extraction & Empirical OOD Domain Check]
    F --> G{In-Distribution?}
    G -- No --> H["Halt: Return INDETERMINATE (OUT_OF_DISTRIBUTION)"]
    G -- Yes --> I[Stage 4: Multi-Model Inference & Disagreement Analysis]
    I --> J{Model Disagreement?}
    J -- Yes --> K["Halt: Return INDETERMINATE (MODEL_DISAGREEMENT)"]
    J -- No --> L[Stage 5: Consensus Verdict & Calibrated Probabilities]
    L --> M[Stage 6: Re-Verify SHA-256 Bitstream Immutability]
    M --> N[Stage 7: Compile 10-Section Formal Report (MD & PDF) & DB Record]
```

---

## 2. Stage-by-Stage Forensic Procedures

### Stage 1: Intake, Sanitization & Cryptographic Fingerprinting
- **File Ingestion:** Uploaded files undergo path sanitization to eliminate directory traversal vectors (`../../`).
- **Segregated Storage:** Raw audio files are copied directly to `storage/original_evidence/`.
- **Intake Hashing:** A cryptographic SHA-256 hash is computed immediately before any processing:
  $$H = \text{SHA-256}(\text{Raw Evidence Bitstream})$$
- **Tracking ID:** An immutable Evidence Tracking ID is generated:
  $$\text{Evidence ID} = \text{EVID-}[H_{1:8}]\text{-}[\text{Sanitized Stem}]$$

### Stage 2: Signal Quality Screening & In-Memory Normalization
- The exhibit is loaded into a floating-point memory buffer. The original exhibit on disk is **never** altered.
- **Duration Check:** Audio must satisfy $t \ge 0.50$ seconds. Shorter recordings are halted.
- **Clipping Detection:** Measures proportion of samples hitting full scale ($|y| \ge 0.999$). Files with $>5\%$ clipping are halted.
- **Silence Check:** Measures proportion of silent frames ($\text{RMS} < 0.005$). Files with $>80\%$ silence are halted.
- **SNR Estimation:** Frame percentile energy ratio ($95\text{th}$ percentile speech vs $5\text{th}$ percentile noise floor). Signals with $\text{SNR} < 5$ dB are flagged.
- **Standardisation:** Audio is resampled to 16,000 Hz single-channel mono and peak-normalized to 1.0.

### Stage 3: Feature Extraction & OOD Domain Validation
- 26 standardized acoustic features are extracted (20 MFCCs with Slaney 128-band area normalization and $10\log_{10}$ dB scaling, Centroid, Rolloff, Bandwidth, ZCR, RMS, Chroma).
- **Out-of-Distribution (OOD) Screening:** Features are standardized against empirical training bounds ($z$-scores):
  $$z_i = \left|\frac{x_i - \mu_i}{\sigma_i}\right|$$
  If $\max(z_i) > 6.47$ or $\text{mean}(z_i) > 2.11$, the signal is classified as **OUT_OF_DISTRIBUTION**. Inference is halted, and `INDETERMINATE` is returned to prevent false positive classifications.

### Stage 4: Multi-Model Inference & Disagreement Protocol
- Evaluates the exhibit across the **Weighted Soft Voting Ensemble**, **Calibrated SVM**, **Random Forest**, and **Logistic Regression**.
- If primary non-linear classifiers contradict each other (e.g. SVM predicts SYNTHETIC with 95% while Random Forest predicts GENUINE with 90%), the system raises a **Model Disagreement Alert** and marks the verdict as `INDETERMINATE_MODEL_DISAGREEMENT`.

### Stage 5: Consensus Verdict & Calibrated Posterior Estimation
- For consensus cases, posterior probabilities are reported using post-hoc Platt scaling (`CalibratedClassifierCV`).
- Uncalibrated decision function scores are explicitly distinguished from calibrated probabilities.

### Stage 6: Bitstream Immutability Audit
- The stored exhibit on disk is re-read and its SHA-256 digest is recomputed.
- If $H_{\text{current}} \ne H_{\text{intake}}$, the system halts with an `EvidenceIntegrityError` and logs a tampering violation in the audit trail.

### Stage 7: Reporting, Database Logging & Archiving
- Compiles the formal 10-section report in both Markdown (`.md`) and court-ready PDF (`.pdf`) formats.
- Inserts comprehensive examination records into `database/forensics.db`.
- Appends structured JSON lines to the immutable audit log (`results/reports/forensic_audit_trail.jsonl`).

---

## 3. Adherence to ISO/IEC 27037:2012 Technical Principles

| ISO/IEC 27037 Principle | Standard Requirement | Phase 5 Technical Implementation |
|---|---|---|
| **Identification (Clause 6.2)** | Distinguish original evidence provenance, recording environment, and container metadata. | `evidence` table stores container, bit depth, native sampling rate, channel count, duration, and investigator notes. |
| **Collection & Acquisition (Clause 6.3 - 6.4)** | Establish cryptographic baseline at the earliest acquisition phase. | Immediate SHA-256 digest generation upon ingestion into `storage/original_evidence/`. |
| **Preservation (Clause 6.4)** | Ensure digital evidence bitstreams remain unaltered throughout analysis. | Original files are marked read-only; all signal transformations occur on floating-point memory buffers or separate derivative directories. |
| **Repeatability & Reproducibility (Clause 5.4)** | Ensure independent examiners obtain identical results using documented procedures. | Deterministic NumPy feature extraction, fixed filterbank specifications, documented model seeds, and automated test suite verification. |

---

## 4. Mandatory Forensic & Legal Disclaimer

> **IMPORTANT LEGAL NOTICE:**  
> This system provides automated investigative screening assessments based on statistical signal processing. In accordance with ISO/IEC 27037 technical principles, its outputs represent analytical indicators and do **not** constitute definitive legal proof of authentic human origin or criminal forgery. All findings must be corroborated by chain of custody, witness testimony, contextual evidence, and independent expert review.

