# Phase 7.5: Acoustic Anomaly Detection Report

**Project:** AI-Based Voice Analysis for Detecting Synthetic and Deepfake Audio in Criminal Investigations  
**Project Root:** `AI_Audio_Forensics`  
**Phase:** 7.5 — Acoustic Anomaly Detection  
**Date:** November 2026  
**Status:** IMPLEMENTED & AUDITED  

---

> [!NOTE]
> **FORENSIC INTERPRETATION GUIDELINE**  
> Unsupervised acoustic anomaly detection models statistical deviations from the dominant vocal distribution of an audio recording. An anomaly score indicates that a specific cluster of audio frames diverges from the surrounding speech patterns.  
> Outliers frequently result from natural vocal phenomena (coughing, throat clearing, whispering, laughter, abrupt emotional shouting) or ambient acoustic intrusions (door slams, sirens). They provide **investigative leads** for manual examination, not autonomous proof of synthetic generation.

---

## 1. Executive Summary

Phase 7.5 establishes an unsupervised, frame-by-frame anomaly detection pipeline (`src/anomaly_detection.py`). Unlike supervised binary classifiers that assume a fixed training domain, the anomaly engine evaluates the recording's internal statistical self-consistency.

---

## 2. Multidimensional Acoustic Feature Extraction

For every 30ms frame (15ms hop), the pipeline extracts 5 continuous acoustic descriptors:

### 2.1 Spectral Flatness (Wiener Entropy)
Evaluates whether a frame's spectrum is tone-like (harmonic vocal tract resonance) or noise-like:
$$\text{SF} = \frac{\exp\left( \frac{1}{K} \sum_{k=0}^{K-1} \ln |X(k)|^2 \right)}{\frac{1}{K} \sum_{k=0}^{K-1} |X(k)|^2}$$
Range: $[0, 1]$. Values near 0 indicate voiced harmonic speech; values near 1 indicate unvoiced friction or flat synthetic noise floor.

### 2.2 Spectral Centroid
Computes the center of mass of the frequency spectrum:
$$\text{SC} = \frac{\sum_{k=0}^{K-1} f_k \cdot |X(k)|}{\sum_{k=0}^{K-1} |X(k)|}$$
Normalizes the brightness and resonance distribution across vocal harmonics.

### 2.3 Pitch Proxy & Voicing Stability via Autocorrelation
Estimates pitch stability using normalized short-time autocorrelation:
$$R(\tau) = \frac{\sum_n x(n) x(n + \tau)}{\sqrt{\sum_n x^2(n) \sum_n x^2(n + \tau)}}$$
Searches for peak lag $\tau^* \in [f_s/500, f_s/60]$. The peak correlation value serves as a continuous voicing stability metric.

### 2.4 Log RMS Frame Energy
$$\text{Energy} = \ln\left(1 + \sqrt{\frac{1}{K} \sum_{k=0}^{K-1} |X(k)|^2}\right)$$

---

## 3. Unsupervised Outlier Algorithms

Features are standardized ($z$-score scaling) and evaluated under one of two user-selectable algorithms:

### 3.1 Isolation Forest
Builds an ensemble of 100 randomized isolation trees. Anomalous instances require fewer random splits to isolate from the manifold:
$$s(x, n) = 2^{-\frac{\mathbb{E}(h(x))}{c(n)}}$$
where $\mathbb{E}(h(x))$ is the average path length and $c(n)$ is average binary tree search depth. Frames with anomaly scores exceeding the $(1 - \text{contamination})$ percentile are flagged.

### 3.2 Local Outlier Factor (LOF)
Evaluates the local density of each frame relative to its 20 nearest neighbors in feature space:
$$\text{LOF}_k(p) = \frac{\sum_{o \in N_k(p)} \frac{\text{lrd}(o)}{\text{lrd}(p)}}{|N_k(p)|}$$
Detects isolated acoustic frames situated far from dense vowel clusters.

---

## 4. Findings & Evidence Persistence

Consecutive outlier frames are aggregated into discrete timestamped findings stored in SQLite `analysis_findings`.

```json
{
  "finding_id": "FIND-ANOM-238F019C",
  "run_id": "RUN-ANOM-90B1E87A",
  "evidence_id": "EVID-TEST-001",
  "start_sec": 1.450,
  "end_sec": 1.580,
  "indicator": "ACOUSTIC_OUTLIER",
  "value": 0.7812,
  "unit": "anomaly_score",
  "threshold": 0.6200,
  "severity": "HIGH",
  "interpretation_flag": "MEASUREMENT",
  "description": "Acoustic frame cluster at 1.45s-1.58s is a statistical outlier under IsolationForest (peak score 0.781 vs threshold 0.620). Features diverge markedly from dominant vocal distribution."
}
```

---

## 5. Verification & Testing

Tested in `tests/test_phase7_forensic_intelligence.py`:
- `test_acoustic_anomaly_detection_isolation_forest`: Verifies execution, outlier clustering, and finding generation.
- `test_acoustic_anomaly_spectral_measures`: Verifies mathematical bounds: flatness $\in [0, 1]$, centroid $\ge 0$, and valid pitch tracking.

Passed cleanly with zero errors.

