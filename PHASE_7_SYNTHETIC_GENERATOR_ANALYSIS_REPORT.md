# Phase 7.4: Synthetic Generator & Vocoder Characterization Report

**Project:** AI-Based Voice Analysis for Detecting Synthetic and Deepfake Audio in Criminal Investigations  
**Project Root:** `AI_Audio_Forensics`  
**Phase:** 7.4 — Synthetic Generator Characterization  
**Date:** November 2026  
**Status:** IMPLEMENTED AS HEURISTIC PROFILING (ATTRIBUTION UNAVAILABLE)  

---

> [!WARNING]
> **FORENSIC NOTICE: DEFINITIVE GENERATOR IDENTIFICATION IS UNAVAILABLE**  
> Exact attribution of an audio recording to a specific synthetic speech generation platform (e.g., ElevenLabs, Tortoise-TTS, Amazon Polly, Tacotron2, VITS, HiFi-GAN) is **UNAVAILABLE** in this system.  
> 1. The existing machine-learning models were trained exclusively as binary classifiers (REAL vs FAKE).  
> 2. The project dataset does not contain multi-class annotations identifying specific generative architectures or vocoders.  
> 
> The analysis described below performs **objective physical bandwidth and spectral cutoff measurement**. The resulting candidate classification is strictly an **EXPLORATORY ACOUSTIC HYPOTHESIS** (`EXPERIMENTAL_HEURISTIC`) and must not be presented as definitive attribution of authoring software.

---

## 1. Executive Summary

Phase 7.4 delivers an acoustic profiling engine (`src/generator_analysis.py`) that characterizes physical frequency boundaries, effective bandwidth, and high-frequency spectral rolloff. Different generations of text-to-speech (TTS) and voice conversion (VC) architectures leave characteristic frequency-domain fingerprints based on their native sampling rates and neural vocoder reconstruction filters.

---

## 2. Physical Acoustic Profiling Methodology

### 2.1 Power Spectral Estimation
The average power spectrum $P(f)$ is computed across the entire signal using Hann-windowed 2048-point discrete Fourier transforms (DFT):
$$P(f) = \frac{1}{M} \sum_{m=1}^M |X_m(f)|^2$$
where $f \in [0, f_s/2]$.

### 2.2 Forensic Metrics
1. **Effective Bandwidth & High-Frequency Cutoff ($f_{\text{cutoff}}$):** Measured by identifying the highest frequency at which normalized power spectrum energy exceeds a $-50 \text{ dB}$ threshold relative to peak spectral power. A steep drop-off at a specific sub-Nyquist frequency indicates anti-aliasing filtering from lower-sample-rate vocoder synthesis.
2. **85% Spectral Rolloff Frequency ($f_{85}$):** The frequency below which 85% of total signal energy is concentrated:
   $$\sum_{f=0}^{f_{85}} P(f) = 0.85 \sum_{f=0}^{f_s/2} P(f)$$
3. **High-Frequency Energy Ratio ($R_{\text{HF}}$):** Proportion of power residing above 7,000 Hz:
   $$R_{\text{HF}} = \frac{\sum_{f \ge 7000} P(f)}{\sum_{f=0}^{f_s/2} P(f)}$$

---

## 3. Acoustic Heuristic Classification Tiers

Based on physical signal constraints, the engine categorizes recordings into four architectural tiers:

| Effective Cutoff ($f_{\text{cutoff}}$) | Candidate Architecture Class | Typical Physical Origin |
| :--- | :--- | :--- |
| **$\le 4,200 \text{ Hz}$** | `NARROWBAND_TELEPHONY_OR_LEGACY_VOCODER` | G.711 / AMR telephony bandpass, LPC-10, 8 kHz legacy speech synthesis. |
| **$4,200 \text{ Hz} - 8,500 \text{ Hz}$** | `WIDEBAND_VOCODER (16 kHz Native)` | Standard 16 kHz TTS vocoders (early WaveNet, Griffin-Lim, MelGAN at 16k). |
| **$8,500 \text{ Hz} - 12,000 \text{ Hz}$** | `MEDIUM_FIDELITY_TTS (22.05 kHz / 24 kHz)` | 22.05k or 24k neural vocoders (Tacotron 2 + WaveGlow, FastSpeech). |
| **$> 12,000 \text{ Hz}$** | `FULL_BAND_MODERN_NEURAL_VOCODER` | Modern high-fidelity vocoders (HiFi-GAN, BigVGAN, ElevenLabs 44.1k/48k) or authentic studio microphones. |

---

## 4. Prerequisites for Forensic Multi-Class Generator Attribution

To upgrade from exploratory heuristic profiling to court-admissible generator attribution, the following engineering requirements must be satisfied:
1. **Benchmark Corpus:** Curate a multi-class dataset containing $\ge 1,000$ verified recordings per architecture across at least 10 modern generator families (ElevenLabs, Tortoise-TTS, Bark, XTTS-v2, Vall-E, VITS, HiFi-GAN, StyleTTS2, Diff-TTS, RVC).
2. **Vocoder Fingerprint Dictionary:** Phase-consistency and bispectral higher-order statistical models to detect synthetic phase artifact fingerprints.
3. **Calibrated Multi-Class Classifier:** Neural or ensemble multi-class architecture with verified cross-speaker validation and out-of-domain rejection.

---

## 5. Verification & Testing

Tested in `tests/test_phase7_forensic_intelligence.py`:
- `test_generator_analysis_bandwidth`: Verifies bandwidth calculation, spectral rolloff estimation, candidate tier hypothesis, and presence of mandatory scientific limitation notices. Passed cleanly.
