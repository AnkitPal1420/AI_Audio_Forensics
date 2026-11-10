# PHASE 6.2 — SPEAKER-INDEPENDENT & SOURCE-INDEPENDENT EVALUATION REPORT
## Disjoint Group Evaluation, Cross-Validation & Voice Conversion Provenance
**Project:** AI-Based Voice Analysis for Detecting Synthetic and Deepfake Audio in Criminal Investigations  
**Project Root:** `AI_Audio_Forensics`  
**Phase:** Phase 6.2 — Speaker-Independent and Source-Independent Evaluation  
**Date:** November 10, 2026  
**Auditor / Engineer:** Senior Digital Audio Forensics & ML Validation Specialist  
**Dataset:** 7,502 standardized 1.0s slices from 66 source recordings (8 subjects)  

---

## 1. EXECUTIVE SUMMARY

In scientific audio forensics, models frequently exhibit inflated laboratory performance if speech recordings from the same speaker or recording environment leak between training and evaluation splits. 

Phase 6.2 executes a rigorous, multi-tiered group-aware evaluation to establish the **true generalization boundary** of the existing MFCC-based classifiers when applied to:
1. **Unseen Acoustic Speakers** (Disjoint Test Partition: `taylor`, `trump`).
2. **Leave-One-Speaker-Out (LOSO) Cross-Validation** across all 8 dataset subjects.
3. **Voice Conversion Provenance Analysis** (Target Voice Timbre vs. Acoustic Utterance Source).

### Major Scientific Finding:
When evaluated on completely unseen subjects, the Support Vector Machine achieves a strong **ROC-AUC of 0.7691** (and 8-fold LOSO mean AUC of **0.7758**), confirming meaningful discriminative ranking power. 
However, at the default decision threshold ($P=0.50$), raw classification accuracy collapses to **51.23%** (Balanced Accuracy: **56.86%**, Specificity: **15.92%**) due to severe false positive rates on unseen genuine speech ($FP = 1,009$). 

*Forensic Conclusion:* The default $0.50$ threshold is completely uncalibrated for out-of-speaker deployment. The model must never be deployed with an unadjusted $0.50$ threshold on unknown subjects in criminal casework.

---

## 2. DATASET METADATA & PROVENANCE STRUCTURE

The experimental dataset comprises 66 distinct physical recordings segmented into 7,502 one-second standardized 16 kHz slices:

```
Total Slices: 7,502
├── Genuine Human Speech:   3,802 slices (50.68%)
└── Synthetic Speech:       3,700 slices (49.32%)
```

### Subject Distribution:
| Subject ID | Total Slices | Genuine Slices | Synthetic Slices | Recordings |
|---|---|---|---|---|
| `musk` | 1,115 | 600 | 515 | 8 |
| `linus` | 1,085 | 630 | 455 | 9 |
| `biden` | 1,055 | 600 | 455 | 8 |
| `obama` | 1,055 | 600 | 455 | 8 |
| `taylor` | 1,055 | 600 | 455 | 8 |
| `trump` | 1,055 | 600 | 455 | 8 |
| `ryan` | 548 | 93 | 455 | 7 |
| `margot` | 534 | 79 | 455 | 8 |

---

## 3. VOICE CONVERSION TWO-PARTY LEAKAGE AUDIT

A critical forensic insight uncovered during Phase 6.2 is the dual-identity nature of voice conversion (VC):
1. **Target Voice Identity ($S_{\text{target}}$):** The vocal timbre being mimicked/synthesized.
2. **Acoustic Source Speaker ($S_{\text{source}}$):** The human speaker providing the original phonetic utterance, prosody, and timing.

### Partitioning Structure:
- In the primary benchmark partition, `group_id` was assigned to $S_{\text{source}}$:
  - **Train ($S_{\text{source}} \in \{\text{biden, linus, margot, musk, obama}\}$):** 4,844 slices
  - **Validation ($S_{\text{source}} = \text{ryan}$):** 548 slices
  - **Test ($S_{\text{source}} \in \{\text{taylor, trump}\}$):** 2,110 slices

### Provenance Audit Findings:
- In the Test split, all 910 synthetic slices have $S_{\text{source}} \in \{\text{taylor, trump}\}$, meaning the model **never heard the acoustic source speaker during training**.
- However, the synthetic targets $S_{\text{target}}$ include voices like `biden` and `musk` that were present in the training set.
- When $S_{\text{source}}$ is completely unseen, the model assigns synthetic speech a very high synthetic probability ($0.9656$ mean), correctly detecting vocoder artifacts regardless of who spoke.
- Conversely, on **genuine unseen speech** from `taylor` and `trump`, the model suffers from extreme false-positive bias ($FP=1009$ out of $1200$ genuine slices), misidentifying idiosyncratic female or idiosyncratic vocal tract acoustics as synthetic vocoder artifacts.

---

## 4. PRIMARY TEST SET EVALUATION (`taylor` & `trump`)

Model: Support Vector Machine (RBF kernel, $C=10.0$, $\gamma=\text{scale}$, with `StandardScaler`):

| Metric | Score | Forensic Interpretation |
|---|---|---|
| **Accuracy** | 51.23% | Near chance level at default 0.50 threshold |
| **Balanced Accuracy** | 56.86% | Slight edge over random baseline |
| **Precision** | 46.87% | Substantial risk of false accusations |
| **Recall (Sensitivity)** | **97.80%** | Excellent synthetic detection ($TP=890 / 910$) |
| **Specificity** | **15.92%** | Severe failure on genuine speech ($TN=191 / 1200$) |
| **F1-Score** | 63.37% | Artificially elevated by high recall |
| **ROC-AUC** | **0.7691** | **Strong ranking capacity (separable distributions)** |
| **Equal Error Rate (EER)** | **28.4%** | Operating point achievable via threshold tuning |

### Confusion Matrix (Test Split: 2,110 slices):
```
                 Predicted Genuine    Predicted Synthetic
Actual Genuine         191 (TN)           1,009 (FP)
Actual Synthetic        20 (FN)             890 (TP)
```

---

## 5. LEAVE-ONE-SPEAKER-OUT (LOSO) CROSS-VALIDATION

To prevent conclusions from resting on a single arbitrary split, full 8-fold Leave-One-Speaker-Out (LOSO) cross-validation was conducted across all 8 subjects:

| Fold (Held-Out Subject) | Test Slices | Accuracy | Balanced Acc | ROC-AUC | Specificity | Recall |
|---|---|---|---|---|---|---|
| **Holdout: `biden`** | 1,055 | 57.35% | 61.52% | 0.7414 | 33.67% | 88.57% |
| **Holdout: `linus`** | 1,145 | 45.24% | 47.53% | 0.4798 | 25.40% | 72.75% |
| **Holdout: `margot`** | 534 | 81.65% | 65.69% | 0.7996 | 40.51% | 88.79% |
| **Holdout: `musk`** | 1,055 | 43.70% | 49.15% | 0.6626 | 13.83% | 82.91% |
| **Holdout: `obama`** | 1,055 | 64.17% | 68.15% | 0.8684 | 45.83% | 88.35% |
| **Holdout: `ryan`** | 548 | 96.72% | 94.60% | 0.9912 | 90.32% | 98.02% |
| **Holdout: `taylor`** | 1,055 | 54.60% | 59.45% | 0.7682 | 23.33% | 95.82% |
| **Holdout: `trump`** | 1,055 | 66.16% | 69.75% | 0.8952 | 48.00% | 90.11% |
| **8-Fold Macro Mean** | **7,502** | **63.70%** | **64.48%** | **0.7758** | **40.11%** | **88.17%** |

### Critical Takeaways:
1. **High Inter-Speaker Variance:** Performance varies dramatically from **0.4798 AUC** (`linus`) to **0.9912 AUC** (`ryan`).
2. **Generalization Gap:** On known training speakers, the model achieved $> 95\%$ accuracy in Phase 3. On held-out speakers, average balanced accuracy drops to **64.48%**.
3. **Threshold Pathology:** Across all folds, Recall is consistently high ($\approx 88\%$) while Specificity is persistently low ($\approx 40\%$). This proves that the feature space of genuine speech expands substantially across new individuals, triggering synthetic false alarms.

---

## 6. SCIENTIFIC RECOMMENDATIONS FOR CRIMINAL INVESTIGATIONS

1. **Threshold Recalibration (Phase 6.4):** The default classification threshold of $0.50$ must be shifted upwards (e.g. to $0.75 - 0.85$) to enforce high specificity ($> 90\%$) and protect authentic speech from false accusation.
2. **Multi-Model Consensus:** The Soft-Voting Ensemble and Logistic Regression must be incorporated to temper SVM edge-case overconfidence.
3. **Forensic Evidentiary Disclaimer:** The forensic report template must explicitly state that automated classifiers have a baseline False Acceptance Rate of $\sim 28\%$ on uncalibrated out-of-domain speakers.
