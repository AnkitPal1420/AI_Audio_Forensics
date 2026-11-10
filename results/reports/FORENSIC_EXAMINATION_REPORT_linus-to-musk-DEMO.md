# FORENSIC AUDIO EXAMINATION REPORT

**Document Type:** Formal Forensic Intelligence Examination  
**Adherence Standard:** ISO/IEC 27037 Technical Principles  
**Laboratory:** AI Audio Forensics Investigation & Research Unit  

---

## 1. Case Information

| Case Attribute | Value |
|---|---|
| **Case Identifier** | `CASE-20261110-001` |
| **Evidence Tracking ID** | `EVID-C97FBF43-linus-to-musk-DE` |
| **Lead Examiner** | Automated Audio Forensics Unit |
| **Examination Timestamp** | 2026-11-10T07:10:01.540674+00:00 |
| **Analysis Status** | **COMPLETED** |

---

## 2. Evidence Information & Chain of Custody

The submitted digital audio exhibit was cataloged and secured prior to analysis:

| Evidence Attribute | Recorded Specification |
|---|---|
| **Exhibit Filename** | `linus-to-musk-DEMO.mp3` |
| **Intake Cryptographic Hash (SHA-256)** | `c97fbf4328367ddd82c527f1bd2b05860b177bd573105e697525d86f0c844f2a` |
| **Container Format** | MP3 |
| **File Size** | 1,084,026 bytes |
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

- **Signal-to-Noise Ratio (SNR):** 24.7 dB
- **Clipping Ratio:** 0.00%
- **Silence Fraction:** 0.43%
- **Domain Status:** `IN_DISTRIBUTION` (Max feature $z$-score: 1.56)

---

## 5. Machine Learning Results

### Primary Classification Finding:
- **Class Determination:** **SYNTHETIC**
- **Decision Confidence:** **98.35%**
- **Decision Function Score:** `N/A`
- **Calibrated Posterior Probability (Synthetic):** **98.35%**
- **Calibrated Posterior Probability (Genuine):** **1.65%**

### Multi-Model Consensus Analysis:
| Model Name | Classification | Synthetic Posterior | Confidence |
|---|---|---|---|
| `calibrated_svm` | **SYNTHETIC** | 98.35% | 98.35% |
| `random_forest` | **SYNTHETIC** | 75.32% | 75.32% |
| `logistic_regression` | **GENUINE** | 45.41% | 54.59% |


### Forensic Assessment Statement:
> The machine-learning system classified the submitted recording as exhibiting characteristics associated with synthetic speech within the validated evaluation domain (posterior probability estimate: 98.35%). This result is an analytical indicator and does not independently establish that the recording has been manipulated.

---

## 6. Visual Evidence & Spectrogram Analysis

High-resolution visual acoustic representations generated during examination:
- **Overview Figure:** `[forensic_overview_linus-to-musk-DEMO.png](C:/Users/ANKIT/OneDrive/Desktop/AI_Audio_Forensics/results/graphs/forensic_overview_linus-to-musk-DEMO.png)`
- **Spectrogram Figure:** `[spectrogram_linus-to-musk-DEMO.png](C:/Users/ANKIT/OneDrive/Desktop/AI_Audio_Forensics/results/graphs/spectrogram_linus-to-musk-DEMO.png)`
- **Mel_Mfcc Figure:** `[mel_mfcc_linus-to-musk-DEMO.png](C:/Users/ANKIT/OneDrive/Desktop/AI_Audio_Forensics/results/graphs/mel_mfcc_linus-to-musk-DEMO.png)`


---

## 7. Evidentiary Limitations & Technical Boundaries

Forensic voice analysis is inherently bounded by signal acquisition conditions:
1. **Telephony & Bandpass Filtering:** G.711 telephony codecs attenuate acoustic energy above 3,400 Hz, removing vocoder phase cues and increasing false positive risk.
2. **Acoustic Noise:** Signal-to-noise ratios below 20 dB degrade spectral centroid and higher MFCC stability.
3. **Cross-Speaker Generalization:** Statistical classifiers may exhibit elevated false alarm rates on unfamiliar voice profiles not represented in the training corpus.

---

## 8. Forensic Conclusion

The machine-learning system classified the submitted recording as exhibiting characteristics associated with synthetic speech within the validated evaluation domain (posterior probability estimate: 98.35%). This result is an analytical indicator and does not independently establish that the recording has been manipulated.

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
