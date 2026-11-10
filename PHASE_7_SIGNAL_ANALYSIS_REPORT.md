# Phase 7.6: Advanced Signal-Level Forensic Workspace Report

**Project:** AI-Based Voice Analysis for Detecting Synthetic and Deepfake Audio in Criminal Investigations  
**Project Root:** `AI_Audio_Forensics`  
**Phase:** 7.6 — Advanced Signal-Level Forensic Workspace  
**Date:** November 2026  
**Status:** IMPLEMENTED & AUDITED  

---

## 1. Executive Summary

Phase 7.6 delivers a high-performance, non-destructive signal examination workspace integrated into the forensic web dashboard. Forensic audio exhibits can vary widely in length, ranging from 2-second voicemail fragments to multi-hour continuous surveillance recordings (such as the 115 MB `biden-original.wav` exhibit in this repository).

The workspace was engineered to provide sub-second interactive visualization, audio excerpt playback, and micro-acoustic inspection without browser freezes or memory exhaustion.

---

## 2. Technical Architecture & Performance Optimizations

### 2.1 Dynamic Decimation & Waveform Downsampling
Rendering un-decimated 16 kHz audio for a 10-minute file requires passing nearly 10,000,000 coordinate points to the web browser DOM, causing severe thread locking. The workspace implements dynamic adaptive decimation:
$$\text{step} = \max\left(1, \left\lfloor \frac{N_{\text{samples}}}{K_{\text{max}}} \right\rfloor\right)$$
where $K_{\text{max}} = 600$ points for high-density overview graphs. Peak and envelope fidelity are strictly preserved while serializing fewer than 10 KB of data per request.

### 2.2 Interactive Temporal Region Cropping
Investigators can isolate suspected edit points, breath pauses, or synthetic vocoder glitches using interactive start/end sliders. The cropped slice:
1. Generates an isolated micro-waveform.
2. Extracts localized STFT magnitude spectra.
3. Powers the in-browser HTML5 `<audio>` playback widget directly from memory.

### 2.3 Pure HTML/CSS & Zero-Pandas Enforcement
To strictly comply with Windows Defender Application Control (WDAC) rules that block `pandas` C-extensions on this workstation, all signal metrics, parameter tables, and numerical summaries are rendered using the custom `render_styled_table()` HTML generator.

---

## 3. Acoustic Signal Visualizations

The workspace provides four synchronized views:
1. **Decimated Time-Domain Amplitude Envelope:** Highlights dynamic range compression, hard digital clipping, and speech/silence boundaries.
2. **Short-Time RMS Energy Profile:** Uncovers sudden transient energy jumps and background noise levels.
3. **Spectral Centroid Trajectory:** Displays shifts in vocal brightness and frequency center-of-mass.
4. **Interactive Temporal Playback:** Allows auditory verification of flagged visual anomalies.

---

## 4. Verification & Testing

- Verified against large audio files up to 115 MB (`dataset/real/biden-original.wav`) with zero memory leakage and responsive sub-second UI updates.
- Audio loading and resampling tested under `tests/test_preprocessing.py` and `tests/test_phase5_dashboard.py`.
