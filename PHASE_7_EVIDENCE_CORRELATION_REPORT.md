# Phase 7.7: Multi-Evidence Case Correlation Report

**Project:** AI-Based Voice Analysis for Detecting Synthetic and Deepfake Audio in Criminal Investigations  
**Project Root:** `AI_Audio_Forensics`  
**Phase:** 7.7 — Multi-Evidence Case Correlation  
**Date:** November 2026  
**Status:** IMPLEMENTED & AUDITED  

---

## 1. Executive Summary

Phase 7.7 establishes a multi-evidence correlation engine (`src/evidence_correlation.py`) enabling forensic investigators to cross-examine two audio exhibits within a case. Real-world criminal investigations rarely involve a single isolated file; investigators frequently receive intercepted phone calls, social media re-uploads, voicemail recordings, and purported reference samples.

The correlation engine integrates cryptographic, container-level, temporal, and acoustic comparisons into an automated forensic workflow.

---

## 2. Multi-Evidence Correlation Pipeline

```
Exhibit A (Source) ──┐
                     ├─► [1. SHA-256 Bitstream Equality] ────► Cryptographic Match
                     ├─► [2. Container & Audio Metadata] ────► Format/Rate Divergence
                     ├─► [3. Acoustic Profile Distance] ─────► Spectral Resemblance
Exhibit B (Target) ──┘─► [4. Envelope Cross-Correlation] ───► Temporal Alignment / Offset
```

### 2.1 Cryptographic Bitstream Equality (SHA-256)
Computes block-wise SHA-256 digests. If hashes match exactly:
- `sha_equal = True`
- `correlation_score = 1.0`
- Generates `CRYPTOGRAPHIC_DUPLICATE` finding. The bitstreams are provably identical.

### 2.2 Container & Acquisition Metadata Divergence
Evaluates differences in:
- Duration delta: $|\text{dur}_A - \text{dur}_B|$
- Sample rate consistency ($16\text{k}$ vs $44.1\text{k}$ vs $8\text{k}$)
- Channel format (mono vs stereo)
- Container format (`WAV`, `MP3`, `FLAC`, `OGG`)

### 2.3 Acoustic Spectral Feature Distance
Computes cosine similarity and Euclidean distance across 26 canonical MFCC statistics using `compare_acoustic_profiles()`.

### 2.4 Temporal Alignment & Envelope Cross-Correlation
Performs normalized cross-correlation on decimated absolute amplitude envelopes to determine if Exhibit B is a trimmed, delayed, or excerpted copy of Exhibit A:
$$R_{xy}(\tau) = \frac{\sum_t (x(t) - \bar{x})(y(t + \tau) - \bar{y})}{\sigma_x \sigma_y}$$
Peak correlation $\rho^* \ge 0.85$ identifies significant temporal alignment and outputs the precise temporal offset in seconds ($\pm \Delta t$).

---

## 3. SQLite Relational Persistence

Results are persisted to the SQLite `evidence_comparisons` table:

```sql
CREATE TABLE IF NOT EXISTS evidence_comparisons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    comparison_id TEXT UNIQUE NOT NULL,
    case_id TEXT NOT NULL,
    evidence_a_id TEXT NOT NULL,
    evidence_b_id TEXT NOT NULL,
    sha_equal INTEGER NOT NULL,
    metadata_diff_json TEXT NOT NULL,
    feature_distance_json TEXT NOT NULL,
    referenced_run_ids_json TEXT NOT NULL,
    notes TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (case_id) REFERENCES cases(case_id) ON DELETE CASCADE,
    FOREIGN KEY (evidence_a_id) REFERENCES evidence(evidence_id) ON DELETE CASCADE,
    FOREIGN KEY (evidence_b_id) REFERENCES evidence(evidence_id) ON DELETE CASCADE
);
```

An associated `AnalysisRun` record is created simultaneously to maintain full relational referential integrity for all child findings in `analysis_findings`.

---

## 4. Verification & Testing

Tested in `tests/test_phase7_forensic_intelligence.py`:
- `test_evidence_correlation_and_db_persistence`:
  - Ingests two simulated exhibits into a test database.
  - Executes `correlate_exhibits()`.
  - Verifies database record insertion into `evidence_comparisons`, `analysis_runs`, and `analysis_findings`.
  - Asserts foreign key enforcement and retrieval.

All assertions passed cleanly in automated CI testing.

