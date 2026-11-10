# FORENSIC AUDIO EXAMINATION REPORT

**Document Type:** Formal Forensic Intelligence Examination  
**Adherence Standard:** ISO/IEC 27037 Technical Principles  
**Laboratory:** AI Audio Forensics Investigation & Research Unit  

---

## 1. Case Information

| Case Attribute | Value |
|---|---|
| **Case Identifier** | `CASE-20261110-001` |
| **Evidence Tracking ID** | `EVID-D299A9E4-linus-original` |
| **Lead Examiner** | Forensic Analyst |
| **Examination Timestamp** | 2026-11-10T07:23:07.912493+00:00 |
| **Analysis Status** | **COMPLETED** |

---

## 2. Evidence Information & Chain of Custody

The submitted digital audio exhibit was cataloged and secured prior to analysis:

| Evidence Attribute | Recorded Specification |
|---|---|
| **Exhibit Filename** | `linus-original.wav` |
| **Intake Cryptographic Hash (SHA-256)** | `d299a9e4c2c664ea69f03086c7b44c8a057740703b3df2ccda282be5292d260d` |
| **Container Format** | WAV |
| **File Size** | 100,611,372 bytes |
| **Acquisition Duration** | 570.36 seconds |
| **Original Sampling Rate** | 44100 Hz |
| **Channel Count** | 2 channel(s) |
| **Bitstream Immutability** | **CONFIRMED UNALTERED** (Cryptographic verification matched intake hash) |

---

## 3. Examination Methodology

1. **Signal Ingestion & Preprocessing:** Non-destructive audio standardisation was conducted in memory. Signals were converted to single-channel (mono) floating-point arrays and resampled to 16,000 Hz. Peak normalization was applied while preserving acoustic silence.
2. **Deterministic Feature Extraction:** 26 standardized acoustic and spectral features were computed:
   - 20 Mel-Frequency Cepstral Coefficients (MFCCs, Slaney 128-band area normalization, $10\log_{10}$ dB scaling).
   - Spectral Centroid, Bandwidth, and Rolloff (85% energy percentile).
   - Zero-Crossing Rate (ZCR) and Root-Mean-Square (RMS) Energy.
   - Chroma Pitch Class Distribution.
3. **Domain Validation:** Standardized distance ($z$-score) evaluated against empirical training distribution bounds to detect out-of-distribution (OOD) artifacts.
4. **Machine Learning Inference:** Evaluated via trained forensic classifiers (`random_forest`) with calibrated probability estimation.

---

## 4. Acoustic Quality & Feature Distribution Analysis

- **Signal-to-Noise Ratio (SNR):** 26.4 dB
- **Clipping Ratio:** 0.00%
- **Silence Fraction:** 0.61%
- **Domain Status:** `IN_DISTRIBUTION` (Max feature $z$-score: 1.221)

---

## 5. Machine Learning Results

### Primary Classification Finding:
- **Class Determination:** **GENUINE**
- **Decision Confidence:** **91.80%**
- **Decision Function Score:** `N/A`
- **Calibrated Posterior Probability (Synthetic):** **8.20%**
- **Calibrated Posterior Probability (Genuine):** **91.80%**

### Multi-Model Consensus Analysis:
| Model Name | Classification | Synthetic Posterior | Confidence |
|---|---|---|---|
| `calibrated_svm` | **GENUINE** | 0.52% | 99.48% |
| `random_forest` | **GENUINE** | 8.20% | 91.80% |
| `logistic_regression` | **GENUINE** | 28.59% | 71.41% |


### Forensic Assessment Statement:
> The machine-learning system classified the submitted recording as consistent with genuine human speech within the validated evaluation domain (posterior probability estimate: 91.80%). This result is an analytical indicator and does not provide definitive proof of authenticity.

---

## 6. Visual Evidence & Spectrogram Analysis

High-resolution visual acoustic representations generated during examination:
- **Overview Figure:** `[forensic_overview_linus-original.png](C:/Users/ANKIT/OneDrive/Desktop/AI_Audio_Forensics/results/graphs/forensic_overview_linus-original.png)`
- **Spectrogram Figure:** `[spectrogram_linus-original.png](C:/Users/ANKIT/OneDrive/Desktop/AI_Audio_Forensics/results/graphs/spectrogram_linus-original.png)`
- **Mel_Mfcc Figure:** `[mel_mfcc_linus-original.png](C:/Users/ANKIT/OneDrive/Desktop/AI_Audio_Forensics/results/graphs/mel_mfcc_linus-original.png)`


---

## 7. Evidentiary Limitations & Technical Boundaries

Forensic voice analysis is inherently bounded by signal acquisition conditions:
1. **Telephony & Bandpass Filtering:** G.711 telephony codecs attenuate acoustic energy above 3,400 Hz, removing vocoder phase cues and increasing false positive risk.
2. **Acoustic Noise:** Signal-to-noise ratios below 20 dB degrade spectral centroid and higher MFCC stability.
3. **Cross-Speaker Generalization:** Statistical classifiers may exhibit elevated false alarm rates on unfamiliar voice profiles not represented in the training corpus.

---

## 8. Forensic Conclusion

The machine-learning system classified the submitted recording as consistent with genuine human speech within the validated evaluation domain (posterior probability estimate: 91.80%). This result is an analytical indicator and does not provide definitive proof of authenticity.

> **IMPORTANT LEGAL NOTICE:**  
> This examination report provides an investigative intelligence screening assessment based on automated acoustic signal analysis. In accordance with digital forensics principles (referencing ISO/IEC 27037 technical guidelines and SWGDE recommendations), this finding does **not** constitute definitive legal proof of authentic human origin or criminal forgery. It must be evaluated in conjunction with corroborating investigative evidence, witness testimony, and verified chain of custody.

---

## 9. Integrity Statement

The undersigned examiner certifies:
1. The evidence was ingested and analyzed using non-destructive, read-only protocols.
2. The cryptographic SHA-256 hash was verified prior to and immediately following all computational transformations.
3. No modification or overwriting of original evidence was performed.

---

## 10. Technical References

- **ISO/IEC 27037:2012:** Information technology — Security techniques — Guidelines for identification, collection, acquisition and preservation of digital evidence.
- **Scientific Working Group on Digital Evidence (SWGDE):** Best Practices for Forensic Audio Analysis.
- **Signal Processing Framework:** NumPy 2.2.3, Scikit-Learn 1.6.1, SoundFile 0.13.1, Python 3.14.3.

---
*End of Formal Forensic Report.*
