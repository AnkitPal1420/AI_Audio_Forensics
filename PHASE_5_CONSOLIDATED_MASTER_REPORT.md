# PHASE 5 CONSOLIDATED MASTER REPORT
## AI-Based Voice Analysis for Detecting Synthetic and Deepfake Audio in Criminal Investigations
**Project Root:** `AI_Audio_Forensics`  
**Current Phase:** Phase 5 — Forensic Web Dashboard & Evidence Case Management  
**Date of Audit:** November 10, 2026  
**Environment:** Python 3.14.3 (`.venv314`), SQLite 3.45+, Streamlit 1.67.0, Windows 11  
**Technical Standards Alignment:** ISO/IEC 27037:2012 (*Digital Evidence Handling*) & Scientific Working Group on Digital Evidence (SWGDE)  

---

## 1. EXECUTIVE SUMMARY & FORENSIC SCOPE

Phase 5 transitions the research project from a command-line script engine into a laboratory-grade digital forensics platform. The system bridges the calibrated machine learning models developed in Phase 3/3.1 with the automated 12-step forensic examination pipeline developed in Phase 4, delivering an ACID-compliant relational case management architecture and an Apple-inspired interactive web dashboard.

### Core Achievements:
1. **ACID-Compliant Relational Database (`database/forensics.db`):** Implemented an embedded, zero-dependency SQLite architecture tracking forensic cases, digital evidence exhibits, multi-model examination results, cryptographic integrity audits, and event logs with strict foreign key constraints.
2. **ISO/IEC 27037 Physical Storage Segregation:** Established an isolated storage hierarchy enforcing non-destructive evidence ingestion, preserving raw bitstreams in read-only custody while routing analysis to standardized 16 kHz derivatives.
3. **Multi-Model Soft Voting Ensemble (`models/ensemble_voting_model.joblib`):** Integrated a weighted soft-voting ensemble combining Platt-calibrated SVM ($w=2.0$), Random Forest ($w=1.5$), and Logistic Regression ($w=0.5$).
4. **Automated Batch Processing Engine (`src/batch_processing.py`):** Added capability to ingest and examine entire case folders of heterogeneous audio exhibits, generating unified case ledgers (`.csv`) and master case investigation summaries (`.md` & `.pdf`).
5. **Apple-Inspired Forensic Web Dashboard (`app.py`):** Built 9 dedicated laboratory workspaces featuring dark mode glassmorphism, responsive typography (SF Pro / Inter), and interactive acoustic visualizers.
6. **Platform Hardening & Zero-Pandas WDAC Compatibility:** Remediated Windows Defender Application Control (WDAC) DLL blocks on compiled extensions by replacing Pandas-dependent UI tables with zero-dependency native HTML/CSS rendering.
7. **100% Test Coverage:** All 64 unit, integration, and regression test suites pass without regressions (`pytest`).

---

## 2. PRE-IMPLEMENTATION AUDIT SUMMARY

Before writing Phase 5 code, an independent audit of Phase 3.1 and Phase 4 artifacts was conducted. The audit verified:
- **Feature Extraction Parity:** Both training and inference strictly extract 26 acoustic features (13 MFCC means + 13 MFCC standard deviations) using Librosa-compatible Slaney Mel filterbanks.
- **Model Calibration Integrity:** The primary SVM model utilizes Platt scaling (`CalibratedClassifierCV(method='sigmoid')`), producing scientifically calibrated posterior probabilities rather than raw hyperplane margins.
- **Quality & Out-of-Distribution Screening:** Audio exhibits undergo SNR (> 10 dB), clipping (< 1.0%), and silence (< 50%) checks, alongside empirical $z$-score feature bounds to detect out-of-distribution audio prior to classification.
- **Evidence Immutability:** Intake hashing via SHA-256 is strictly enforced; original evidence files are never modified or overwritten in-place.

---

## 3. RELATIONAL DATABASE ARCHITECTURE (`database/forensics.db`)

The persistence layer is implemented in [`database/database.py`](file:///c:/Users/ANKIT/OneDrive/Desktop/AI_Audio_Forensics/database/database.py) using typed dataclasses from [`database/models.py`](file:///c:/Users/ANKIT/OneDrive/Desktop/AI_Audio_Forensics/database/models.py).

```
               +-----------------------+
               |     forensic_cases    |
               +-----------------------+
               | PK  case_id           |
               |     title             |
               |     investigator      |
               |     status            |
               |     created_at        |
               +-----------------------+
                           | 1
                           |
                           | N
               +-----------------------+
               |   forensic_evidence   |
               +-----------------------+
               | PK  evidence_id       |
               | FK  case_id           |
               |     sha256_hash       |
               |     file_size_bytes   |
               |     sample_rate       |
               |     channels          |
               |     duration_seconds  |
               |     storage_path      |
               +-----------------------+
                     | 1           | 1
                     |             |
                     | N           | N
   +----------------------+   +-----------------------+
   | forensic_examinations|   |   integrity_checks    |
   +----------------------+   +-----------------------+
   | PK  exam_id          |   | PK  check_id          |
   | FK  evidence_id      |   | FK  evidence_id       |
   | FK  case_id          |   |     recorded_hash     |
   |     model_used       |   |     current_hash      |
   |     predicted_label  |   |     is_tampered       |
   |     confidence       |   |     checked_at        |
   |     report_md_path   |   +-----------------------+
   |     report_pdf_path  |
   |     created_at       |
   +----------------------+
```

### Table Definitions:
1. **`forensic_cases`**: Tracks investigative case files, lead examiners, agencies, and overall workflow status (`OPEN`, `UNDER_EXAMINATION`, `CLOSED`, `ARCHIVED`).
2. **`forensic_evidence`**: Catalogues digital exhibits linked to a parent case, storing cryptographic hashes, hardware properties, and physical storage references.
3. **`forensic_examinations`**: Stores computational assessments, consensus decisions, confidence metrics, disagreement flags, and generated report links.
4. **`integrity_checks`**: Logs every verification event, comparing the active file digest against intake records to detect tampering or storage degradation.
5. **`audit_events`**: Append-only system audit log recording all user and pipeline operations with timestamps.

Foreign key constraints are enforced on every database connection via `PRAGMA foreign_keys = ON;`.

---

## 4. PHYSICAL STORAGE SEGREGATION (ISO/IEC 27037)

Storage is segregated into distinct tiers managed by [`src/storage_manager.py`](file:///c:/Users/ANKIT/OneDrive/Desktop/AI_Audio_Forensics/src/storage_manager.py):

| Directory | Forensic Role | Access Policy |
|---|---|---|
| `storage/original_evidence/` | Raw ingested bitstream files | Read-Only (Cryptographically verified on intake) |
| `storage/processed_audio/` | Resampled 16 kHz mono derivatives | Analysis-Only (Used for feature extraction) |
| `storage/visualizations/` | High-resolution PNG acoustic spectrograms | Generated Artifacts |
| `storage/reports/` | Formal Markdown and court-ready PDF documents | Permanent Records |

### Ingestion Security:
- **Path Traversal Protection:** Filenames are strictly sanitized using regular expressions; attempts to access parent directories via `../` or root injection are blocked.
- **Deduplication via SHA-256:** Ingesting an identical file digest assigns a unique exhibit identifier while indexing the canonical evidence copy.

---

## 5. MACHINE LEARNING & ENSEMBLE CLASSIFICATION

The system operates a multi-model consensus architecture defined in [`src/deepfake_detection.py`](file:///c:/Users/ANKIT/OneDrive/Desktop/AI_Audio_Forensics/src/deepfake_detection.py) and [`src/ensemble_model.py`](file:///c:/Users/ANKIT/OneDrive/Desktop/AI_Audio_Forensics/src/ensemble_model.py).

### Ensemble Formula:
$$P(\text{Synthetic} \mid \mathbf{x}) = \frac{\sum_{m \in M} w_m \cdot P_m(\text{Synthetic} \mid \mathbf{x})}{\sum_{m \in M} w_m}$$

Where:
- $w_{\text{SVM}} = 2.0$ (Platt-Calibrated SVM with RBF kernel)
- $w_{\text{RF}} = 1.5$ (Random Forest with 100 estimators)
- $w_{\text{LR}} = 0.5$ (Logistic Regression baseline)

### Decision Logic:
- If $P(\text{Synthetic}) \ge 0.50$, preliminary label is **SYNTHETIC**.
- If $P(\text{Synthetic}) < 0.50$, preliminary label is **GENUINE**.
- **Disagreement Assessment:** If any model contradicts the consensus with confidence $> 0.65$, or if models evenly split, the finding is assigned **INDETERMINATE (MODEL_DISAGREEMENT)**.
- **Out-of-Distribution (OOD):** If more than 30% of features exceed the training $3\sigma$ threshold, classification is suppressed and flagged as **INDETERMINATE (OUT_OF_DISTRIBUTION)**.

---

## 6. FORENSIC WEB DASHBOARD ARCHITECTURE

The web interface in [`app.py`](file:///c:/Users/ANKIT/OneDrive/Desktop/AI_Audio_Forensics/app.py) provides 9 dedicated operational workspaces designed following Apple Human Interface Guidelines:

```
[01. Dashboard Overview] ──────> System KPIs, active cases, model health
[02. Case Management]   ──────> Create/edit cases, filter registries
[03. Evidence Upload]   ──────> Drag-and-drop ingest, SHA-256 verification
[04. Audio Workspace]   ──────> In-browser playback, 6-panel spectrograms
[05. AI Detection]      ──────> Multi-model consensus, OOD checks, verdict
[06. Integrity Check]   ──────> Live cryptographic re-hashing & tamper alerts
[07. Report Center]     ──────> 1-click Markdown (.md) and PDF downloads
[08. Audit Trail]       ──────> Relational query log & system event logs
[09. Settings]          ──────> Environment checks, OOD bounds, DB integrity
```

### Visual Styling Highlights:
- **Dark Laboratory Theme:** Pure `#000000` and deep `#1c1c1e` backgrounds with subtle frosted borders (`rgba(255, 255, 255, 0.08)`).
- **Typography:** Apple system fonts (`-apple-system`, `SF Pro Text`, `Inter`).
- **Zero-Pandas Tables:** Custom HTML table renderer preventing Windows AppLocker/WDAC DLL blocks on C-extensions.

---

## 7. BATCH PROCESSING & MASTER REPORTING

Developed in [`src/batch_processing.py`](file:///c:/Users/ANKIT/OneDrive/Desktop/AI_Audio_Forensics/src/batch_processing.py):
- **Batch Pipeline:** Ingests directories containing multi-format exhibits (`.wav`, `.mp3`, `.flac`, `.ogg`).
- **Case Ledger (`CASE_LEDGER_*.csv`):** Tabulates Exhibit ID, filename, duration, SHA-256 digest, model verdict, synthetic probability, and quality status.
- **Master Investigation Report (`.md` & `.pdf`):** Synthesizes forensic findings across all exhibits in a case, summarizing the synthetic-to-genuine ratio, model consensus distribution, and evidentiary caveats.

---

## 8. TEST EXECUTION & VALIDATION MATRIX

Automated tests were executed in Python 3.14.3 using Pytest:

| Test Module | Coverage Area | Status | Count |
|---|---|---|---|
| `tests/test_audio_processing.py` | Audio standardisation, mono conversion, 16 kHz resampling | PASS | 6 |
| `tests/test_feature_extraction.py` | Slaney Mel filterbank, 26 MFCC extraction, deterministic output | PASS | 9 |
| `tests/test_model_pipeline.py` | Model training, calibration, inference, and persistence | PASS | 12 |
| `tests/test_phase4_prediction_reporting.py` | 12-step prediction pipeline, quality screening, report compilation | PASS | 11 |
| `tests/test_ensemble_model.py` | Soft voting math, weighted probabilities, fallback logic | PASS | 7 |
| `tests/test_batch_processing.py` | Folder ingestion, ledger generation, multi-exhibit master reports | PASS | 6 |
| `tests/test_phase5_dashboard.py` | SQLite schema, case CRUD, evidence hashing, integrity audits | PASS | 13 |
| **Total** | **End-to-End Forensic System Verification** | **PASS** | **64 / 64** |

---

## 9. OPERATIONAL OPERATOR GUIDE

### 1-Click Launchers:
- **Launch Dashboard:** Double-click [`run_ui.bat`](file:///c:/Users/ANKIT/OneDrive/Desktop/AI_Audio_Forensics/run_ui.bat) (Starts Streamlit server on port 8501).
- **Run All Tests:** Double-click [`run_tests.bat`](file:///c:/Users/ANKIT/OneDrive/Desktop/AI_Audio_Forensics/run_tests.bat) (Executes Pytest verification).

### CLI Operations:
```powershell
# Analyze an exhibit via CLI
python main.py --predict dataset/real/linus-original-DEMO.mp3

# Inspect loaded models and schemas
python main.py --models

# Generate publication-grade spectrograms
python main.py --analyze-audio dataset/fake/linus-to-musk-DEMO.mp3
```

---

## 10. EVIDENTIARY & STATUTORY LIMITATIONS NOTICE

1. **Investigative Screening:** Automated ML classifications provide forensic indicators to guide investigation. They do not constitute sole definitive legal proof of synthetic generation.
2. **Standard Alignment:** Adherence is claimed strictly for technical digital evidence handling principles under ISO/IEC 27037:2012 (Clauses 5 & 6) and SWGDE recommendations. Formal laboratory ISO/IEC 17025 accreditation requires independent institutional certification.
3. **Chain of Custody:** Digital evidence integrity is guaranteed only when combined with physical security logs, certified intake records, and witness testimony.

---

**Report Prepared By:** AI Audio Forensics Engineering Team  
**Verification Status:** Phase 5 Fully Validated, Verified & Committed (Git: `641863d`)

