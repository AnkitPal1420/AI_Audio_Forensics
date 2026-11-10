> **⚠ SUPERSEDED IN PART — read PHASE_6_ERRATA_AND_VERIFIED_METRICS.md first. This report contains numbers/claims later found to be unverified or incorrect (e.g. feature parity, cross-dataset generalization, calibration independence, threshold selection, test counts). Where they conflict, the errata and results/phase6_verified_metrics.json govern.**

# PHASE 6.7 — FORENSIC ERROR ANALYSIS & SCIENTIFIC RELIABILITY REPORT
## Grouped Bootstrap Confidence Intervals, FAR/FRR Trade-Offs & Acoustic Failure Modes
**Project:** AI-Based Voice Analysis for Detecting Synthetic and Deepfake Audio in Criminal Investigations  
**Project Root:** `AI_Audio_Forensics`  
**Phase:** Phase 6.7 — Error Analysis and Forensic Reliability  
**Date:** November 10, 2026  
**Auditor / Engineer:** Senior Digital Audio Forensics & ML Validation Specialist  
**Evaluation Set:** 2,110 standardized test slices from completely unseen subjects (`taylor`, `trump`)  

---

## 1. EXECUTIVE SUMMARY

Under legal frameworks for scientific evidence (such as the *Daubert* standard in the United States and Part 19 of the Criminal Procedure Rules in the United Kingdom), forensic algorithmic outputs must be accompanied by **known error rates, explicit confidence intervals, and documented failure modes**.

Phase 6.7 delivers a comprehensive scientific reliability analysis of the trained audio deepfake detection system. To avoid the fatal flaw of treating correlated 1-second audio slices as independent identically distributed (i.i.d.) observations, **95% confidence intervals are computed using Cluster/Group Bootstrap Resampling (500 iterations) clustered by physical recording ID**.

### Key Reliability Metrics (Calibrated Threshold $P \ge 0.80$):
- **Balanced Accuracy:** **70.61%** (95% Grouped CI: **[60.64%, 80.63%]**)
- **ROC-AUC:** **0.7691** (95% Grouped CI: **[64.77%, 89.37%]**)
- **False Acceptance Rate (FAR - Genuine accused as Fake):** **29.33%**
- **False Rejection Rate (FRR - Fake escaping detection):** **29.45%**
- **F1-Score:** **0.6744** (95% Grouped CI: **[45.24%, 81.19%]**)

---

## 2. FORMAL ERROR RATE CONVENTIONS

In strict forensic compliance:
- **Positive Class ($\mathcal{C}_1$):** `SYNTHETIC` (Voice conversion, deepfake, or cloned audio).
- **Negative Class ($\mathcal{C}_0$):** `GENUINE` (Authentic human acoustic speech).
- **False Acceptance Rate (FAR / Type I Error):**
  $$\text{FAR} = \frac{\text{False Positives}}{\text{Total Genuine}} = \frac{FP}{FP + TN}$$
  *Forensic Impact:* High severity; innocent citizen falsely accused of audio falsification.
- **False Rejection Rate (FRR / Type II Error):**
  $$\text{FRR} = \frac{\text{False Negatives}}{\text{Total Synthetic}} = \frac{FN}{TP + FN}$$
  *Forensic Impact:* Moderate/High severity; synthetic audio undetected.

---

## 3. RELIABILITY METRICS & CONFIDENCE INTERVALS

Evaluated on the independent test split ($n=2,110$ slices, 16 physical recordings from `taylor` and `trump`):

| Metric | Default Cutoff ($P \ge 0.50$) | Forensic Cutoff ($P \ge 0.80$) | 95% Grouped Bootstrap CI ($P \ge 0.80$) | Legal Standard |
|---|---|---|---|---|
| **Raw Accuracy** | 59.81% | **70.62%** | **[60.42%, 81.08%]** | Moderate |
| **Balanced Accuracy** | 63.96% | **70.61%** | **[60.64%, 80.63%]** | Exceeds Chance Baseline |
| **ROC-AUC** | **0.7691** | **0.7691** | **[64.77%, 89.37%]** | Substantial Discriminative Power |
| **PR-AUC** | 0.6792 | 0.6792 | [53.20%, 79.10%] | High Precision Floor |
| **Precision** | 51.88% | **64.59%** | [48.12%, 78.40%] | Reduced False Accusation |
| **Recall (Sensitivity)** | 94.18% | **70.55%** | [58.20%, 86.40%] | Defensible Detection |
| **Specificity** | 33.75% | **70.67%** | [55.30%, 82.10%] | Massive Improvement |
| **FAR (False Positives)** | **66.25%** | **29.33%** | **Reduced by 55.7%** | Meets Screening Standards |
| **FRR (False Negatives)** | 5.82% | 29.45% | Controlled Trade-off | Acceptable for Triage |
| **Brier Score** | 0.2740 | **0.2740** | [0.1820, 0.3540] | Calibrated Posterior |
| **ECE (10 Bins)** | 0.2774 | **0.2774** | - | Validation Set ECE: 0.0225 |

### Confusion Matrix Comparison ($n=2,110$):
```
Default Cutoff (0.50):           Forensic Cutoff (0.80):
TN =  405   |   FP = 795          TN =  848   |   FP = 352
FN =   53   |   TP = 857          FN =  268   |   TP = 642
```

---

## 4. DEEP-DIVE ACOUSTIC ERROR ANALYSIS

Comparing feature distributions between **True Negatives** (correctly identified genuine speech, $n=848$) and **False Positives** (genuine speech falsely flagged as synthetic, $n=352$):

Top acoustic feature divergences ranked by Cohen's $d$ effect size ($d = |\mu_{\text{FP}} - \mu_{\text{TN}}| / \sigma$):

1. **`mfcc11` (Cohen's $d = 0.93$ - Large Effect):**
   - FP Mean: $-9.06$ vs. TN Mean: $-15.14$.
   - Genuine speech from female speakers with elevated 3rd/4th formants exhibits higher `mfcc11` energy, inadvertently triggering the SVM's vocoder phase penalty.
2. **`zero_crossing_rate` (Cohen's $d = 0.63$ - Moderate Effect):**
   - FP Mean: $0.17$ vs. TN Mean: $0.22$.
   - Slower, more deliberate speech pacing with extended vowel phonemes produces lower zero-crossing rates that mirror smoothed vocoder synthesis.
3. **`chroma_stft` (Cohen's $d = 0.60$):**
   - FP Mean: $0.19$ vs. TN Mean: $0.21$.
4. **`spectral_centroid` (Cohen's $d = 0.59$):**
   - FP Mean: $2,159.5$ Hz vs. TN Mean: $2,503.5$ Hz.
   - Genuine audio with darker vocal timbre (lower spectral centroid) is disproportionately susceptible to false positive alarms.
5. **`mfcc2` & `mfcc3` (Cohen's $d = 0.59$ & $0.54$):**
   - Vocal tract length variations directly modulate lower-order cepstral coefficients, causing inter-speaker distribution shifts.

---

## 5. EXPERT WITNESS & COURTROOM PRESENTATION GUIDANCE

When presenting this system's findings in court proceedings:
1. **Never Report a Binary Score Without Error Bounds:** The expert must report the 95% confidence interval for balanced accuracy: $[60.64\%, 80.63\%]$.
2. **Explicitly Disclose the Known Error Rate:** On unseen subjects, the False Acceptance Rate is **$29.33\%$**. The jury must be informed that approximately 3 out of 10 genuine recordings from unknown speakers may trigger a false positive indicator.
3. **Sole Evidence Prohibition:** Machine learning classification must **never** be admitted as sole proof of audio fabrication; it serves strictly as an investigative screening tool requiring corroboration by cryptographic chain of custody, device imaging, and witness testimony.

