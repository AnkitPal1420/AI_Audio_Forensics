# Phase 7.3: Speaker Consistency & Comparison Baseline Report

**Project:** AI-Based Voice Analysis for Detecting Synthetic and Deepfake Audio in Criminal Investigations  
**Project Root:** `AI_Audio_Forensics`  
**Phase:** 7.3 — Speaker Comparison Framework  
**Date:** November 2026  
**Status:** IMPLEMENTED AS EXPERIMENTAL BASELINE (IDENTITY CLAIMS PROHIBITED)  

---

> [!CAUTION]
> **CRITICAL SCIENTIFIC & LEGAL PROHIBITION: BIOMETRIC IDENTITY CLAIMS ARE STRICTLY PROHIBITED**  
> True forensic speaker verification requires deep neural embedding architectures (e.g., ECAPA-TDNN, x-vectors via SpeechBrain / PyTorch) trained on massive multi-speaker corpora (e.g., VoxCeleb 1 & 2), evaluated under calibrated forensic likelihood ratios (LR) with probabilistic linear discriminant analysis (PLDA).  
> In this environment:
> 1. Neither PyTorch nor SpeechBrain is installed (due to runtime WDAC security constraints).
> 2. The project's training dataset contains no cross-channel speaker identity ground truth.  
> 
> Therefore, this module provides **ONLY AN EXPLORATORY ACOUSTIC SPECTRAL DISTANCE BASELINE**. It **CANNOT AND MUST NOT** be used to assert, confirm, or refute speaker identity in legal or criminal investigative contexts. All outputs are hard-locked to status `EXPERIMENTAL_INDETERMINATE` with `identity_claim_permitted = False`.

---

## 1. Executive Summary

Phase 7.3 implements an exploratory acoustic comparison baseline (`src/speaker_comparison.py`) allowing investigators to assess the global spectral distance between two audio exhibits. It enforces rigid scientific safeguards to prevent investigators from over-interpreting spectral resemblance as biometric identity.

---

## 2. Mathematical Methodology

### 2.1 Feature Representation
Each recording is represented by its canonical 26-dimensional feature vector $\mathbf{v} \in \mathbb{R}^{26}$, consisting of 13 MFCC temporal means and 13 MFCC temporal standard deviations extracted from speech-active frames.

### 2.2 Distance Metrics
1. **Cosine Similarity:** Evaluates the angle between the two acoustic vectors independent of global magnitude:
   $$\cos(\theta) = \frac{\mathbf{v}_A \cdot \mathbf{v}_B}{\|\mathbf{v}_A\|_2 \|\mathbf{v}_B\|_2}$$
   Range: $[-1.0, 1.0]$. A score of $1.0$ indicates identical spectral shape proportions.
2. **Euclidean Distance:** Evaluates absolute geometric separation in feature space:
   $$d_E(\mathbf{v}_A, \mathbf{v}_B) = \|\mathbf{v}_A - \mathbf{v}_B\|_2 = \sqrt{\sum_{i=1}^{26} (v_{A, i} - v_{B, i})^2}$$
3. **Sub-Band Resemblance:** Computes localized cosine similarity across three acoustic frequency ranges:
   - *Low Frequency (MFCCs 1–4):* Fundamental vocal resonance, chest resonance.
   - *Mid Frequency (MFCCs 5–8):* Formant transitions, vowel articulation.
   - *High Frequency (MFCCs 9–13):* Fricatives, upper spectral slope.

---

## 3. Mandatory Output Architecture

The function `compare_acoustic_profiles()` returns a strictly constrained container:

```json
{
  "comparison_id": "CMP-SPK-91B4C02E",
  "evidence_a_id": "EVID-A",
  "evidence_b_id": "EVID-B",
  "cosine_similarity": 0.8842,
  "euclidean_distance": 2.1405,
  "acoustic_divergence_pct": 11.58,
  "band_similarity": {
    "low_frequency_profile": 0.941,
    "mid_frequency_profile": 0.892,
    "high_frequency_profile": 0.812
  },
  "status": "EXPERIMENTAL_INDETERMINATE",
  "identity_claim_permitted": false,
  "prerequisites_missing": [
    "Pre-trained deep speaker embedding architecture (e.g., ECAPA-TDNN, VoxCeleb-tuned x-vectors).",
    "Deep learning inference runtime (PyTorch / ONNX Runtime).",
    "Enrollment database with verified speaker reference recordings.",
    "Calibrated forensic likelihood ratio (LR) scoring framework with channel compensation."
  ]
}
```

---

## 4. Verification & Testing

Tested in `tests/test_phase7_forensic_intelligence.py`:
- `test_speaker_comparison_baseline`: Verifies calculation of cosine similarity and Euclidean distance, asserts that `status == "EXPERIMENTAL_INDETERMINATE"`, confirms `identity_claim_permitted is False`, and verifies that the prerequisite list and scientific limitation disclaimer are present.

Passed cleanly with 100% assertion compliance.
