# FORENSIC AUDIO EXAMINATION REPORT

**Document Type:** Formal Forensic Intelligence Examination  
**Adherence Standard:** ISO/IEC 27037 Technical Principles  
**Laboratory:** AI Audio Forensics Investigation & Research Unit  

---

## 1. Case Information

| Case Attribute | Value |
|---|---|
| **Case Identifier** | `CASE-20261110-001` |
| **Evidence Tracking ID** | `EVID-EB4898F2-linus-original-D` |
| **Lead Examiner** | Automated Audio Forensics Unit |
| **Examination Timestamp** | 2026-11-10T07:08:46.933558+00:00 |
| **Analysis Status** | **COMPLETED** |

---

## 2. Evidence Information & Chain of Custody

The submitted digital audio exhibit was cataloged and secured prior to analysis:

| Evidence Attribute | Recorded Specification |
|---|---|
| **Exhibit Filename** | `linus-original-DEMO.mp3` |
| **Intake Cryptographic Hash (SHA-256)** | `eb4898f2fd1990e031a76666a505cb751c65a1188c9851ba2b2fa0dfab8f2c03` |
| **Container Format** | MP3 |
| **File Size** | 1,188,736 bytes |
| **Acquisition Duration** | 60.00 seconds |
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
4. **Machine Learning Inference:** Evaluated via trained forensic classifiers (`calibrated_svm`) with calibrated probability estimation.

---

## 4. Acoustic Quality & Feature Distribution Analysis

- **Signal-to-Noise Ratio (SNR):** 27.1 dB
- **Clipping Ratio:** 0.00%
- **Silence Fraction:** 0.37%
- **Domain Status:** `IN_DISTRIBUTION` (Max feature $z$-score: 1.45)

---

## 5. Machine Learning Results

### Primary Classification Finding:
- **Class Determination:** **GENUINE**
- **Decision Confidence:** **99.54%**
- **Decision Function Score:** `N/A`
- **Calibrated Posterior Probability (Synthetic):** **0.46%**
- **Calibrated Posterior Probability (Genuine):** **99.54%**

### Multi-Model Consensus Analysis:
| Model Name | Classification | Synthetic Posterior | Confidence |
|---|---|---|---|
| `calibrated_svm` | **GENUINE** | 0.46% | 99.54% |
| `random_forest` | **GENUINE** | 12.23% | 87.77% |
| `logistic_regression` | **GENUINE** | 17.67% | 82.33% |


### Forensic Assessment Statement:
> The machine-learning system classified the submitted recording as consistent with genuine human speech within the validated evaluation domain (posterior probability estimate: 99.54%). This result is an analytical indicator and does not provide definitive proof of authenticity.

---

## 6. Visual Evidence & Spectrogram Analysis

High-resolution visual acoustic representations generated during examination:
- **Overview Figure:** `[forensic_overview_linus-original-DEMO.png](C:/Users/ANKIT/OneDrive/Desktop/AI_Audio_Forensics/results/graphs/forensic_overview_linus-original-DEMO.png)`
- **Spectrogram Figure:** `[spectrogram_linus-original-DEMO.png](C:/Users/ANKIT/OneDrive/Desktop/AI_Audio_Forensics/results/graphs/spectrogram_linus-original-DEMO.png)`
- **Mel_Mfcc Figure:** `[mel_mfcc_linus-original-DEMO.png](C:/Users/ANKIT/OneDrive/Desktop/AI_Audio_Forensics/results/graphs/mel_mfcc_linus-original-DEMO.png)`


---

## 7. Evidentiary Limitations & Technical Boundaries

Forensic voice analysis is inherently bounded by signal acquisition conditions:
1. **Telephony & Bandpass Filtering:** G.711 telephony codecs attenuate acoustic energy above 3,400 Hz, removing vocoder phase cues and increasing false positive risk.
2. **Acoustic Noise:** Signal-to-noise ratios below 20 dB degrade spectral centroid and higher MFCC stability.
3. **Cross-Speaker Generalization:** Statistical classifiers may exhibit elevated false alarm rates on unfamiliar voice profiles not represented in the training corpus.

---

## 8. Forensic Conclusion

The machine-learning system classified the submitted recording as consistent with genuine human speech within the validated evaluation domain (posterior probability estimate: 99.54%). This result is an analytical indicator and does not provide definitive proof of authenticity.

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
