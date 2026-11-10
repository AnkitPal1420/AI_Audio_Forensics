# PHASE 3.1 — DATASET LEAKAGE & SPEAKER GROUPING AUDIT REPORT

**Project:** AI-Based Voice Analysis for Detecting Synthetic and Deepfake Audio in Criminal Investigations  
**Phase:** 3.1 Audit Remediation — Priority 2  
**Author:** AI Audio Forensics Investigation Unit  
**Status:** Completed & Validated (Zero Leakage Verified)

---

## 1. Executive Summary & Problem Formulation

In the initial Phase 3 implementation, models trained on `dataset/metadata/DATASET-balanced.csv` reported near-perfect accuracy (SVM Test Accuracy: 99.83%, ROC-AUC: 0.99996). 

The Phase 3.1 Scientific Audit revealed that:
1. `DATASET-balanced.csv` is an anonymous CSV of 11,778 pre-extracted 1-second feature slices without speaker IDs, source recording IDs, or timestamp metadata.
2. The entire DEEP-VOICE dataset was derived from only **8 distinct public figures**:
   - Joe Biden
   - Linus Sebastian
   - Margot Robbie
   - Elon Musk
   - Barack Obama
   - Ryan Gosling
   - Taylor Swift
   - Donald Trump
3. Stratified random splitting (70% train, 15% val, 15% test) at the slice level distributed adjacent 1-second slices from the same speaker and source recording across all three partitions.
4. **Data Leakage Mechanism:** The classifier was not learning universal deepfake acoustic boundaries; it was recognizing specific speaker vocal tracts, room acoustics, and microphone frequency profiles common to slices from the same source file across partitions.

---

## 2. Methodology & Ingestion of Verified Audio Exhibits

To eliminate data leakage, we built a comprehensive recording-level manifest and generated a new, group-aware 1-second slice dataset directly from the 66 standardized audio recordings.

### 2.1 Subject and Acoustic Source Identification
Every recording was mapped to its true acoustic source and target voice profile:

| Subject / Speaker | Genuine Exhibits | Synthetic Conversions Derived from Source | Target Voice Conversions of Other Speakers | Total Audio Duration | Slices in Dataset |
|---|---|---|---|---|---|
| **Joe Biden** | `biden-original.wav` | 7 conversions (`biden-to-*`) | 7 conversions (`*-to-biden`) | 4,800.0 s | 1,055 |
| **Linus Sebastian** | `linus-original.wav`, `linus-original-DEMO.mp3` | 8 conversions (`linus-to-*`, `linus-to-musk-DEMO`) | 7 conversions (`*-to-linus`) | 5,193.3 s | 1,145 |
| **Margot Robbie** | `margot-original.wav` | 7 conversions (`margot-to-*`) | 7 conversions (`*-to-margot`) | 635.9 s | 534 |
| **Elon Musk** | `musk-original.wav` | 7 conversions (`musk-to-*`) | 7 conversions (`*-to-musk`) | 4,801.0 s | 1,055 |
| **Barack Obama** | `obama-original.wav` | 7 conversions (`obama-to-*`) | 7 conversions (`*-to-obama`) | 4,803.6 s | 1,055 |
| **Ryan Gosling** | `ryan-original.wav` | 7 conversions (`ryan-to-*`) | 7 conversions (`*-to-ryan`) | 749.5 s | 548 |
| **Taylor Swift** | `taylor-original.wav` | 7 conversions (`taylor-to-*`) | 7 conversions (`*-to-taylor`) | 4,801.4 s | 1,055 |
| **Donald Trump** | `trump-original.wav` | 7 conversions (`trump-to-*`) | 7 conversions (`*-to-trump`) | 4,803.4 s | 1,055 |
| **Total** | **9 exhibits** | **57 exhibits** | **56 conversions** | **30,687.1 s** | **7,502 slices** |

### 2.2 Atomic Source Clustering Rule
Because every fake recording was created by taking a real recording and applying Retrieval-based Voice Conversion (RVC), both the genuine recording and its conversions share identical linguistic content, word phrasing, background noise, and pauses.
- **Rule:** All recordings and slices originating from the same acoustic source speaker belong to a single atomic `group_id`!
- For example, `linus-original.wav`, `linus-original-DEMO.mp3`, and all `linus-to-*` conversions are bound to `group_id = 'linus'`.

---

## 3. Leakage-Free Partitioning Implementation

Using `src/group_split.py`, the dataset of 7,502 slices was partitioned strictly by speaker groups:

### 3.1 Partition Composition

| Partition | Designated Speakers | Slice Count | Genuine (0) | Synthetic (1) | Balance Ratio | Percentage of Data |
|---|---|---|---|---|---|---|
| **Training** | `biden`, `linus`, `margot`, `musk`, `obama` (5 subjects) | **4,844** | 2,509 | 2,335 | 51.8% / 48.2% | **64.57%** |
| **Validation** | `ryan` (1 subject) | **548** | 93 | 455 | 17.0% / 83.0% | **7.30%** |
| **Testing** | `taylor`, `trump` (2 subjects) | **2,110** | 1,200 | 910 | 56.9% / 43.1% | **28.13%** |
| **Total** | **8 subjects** | **7,502** | **3,802** | **3,700** | **50.7% / 49.3%** | **100.00%** |

### 3.2 Automated Leakage Verification Audit
A mathematical disjointness check was executed across all partitions:
- $\text{Train Speakers} \cap \text{Val Speakers} = \emptyset$ (Disjoint: `['biden', 'linus', 'margot', 'musk', 'obama']` vs `['ryan']`)
- $\text{Train Speakers} \cap \text{Test Speakers} = \emptyset$ (Disjoint: `['biden', 'linus', 'margot', 'musk', 'obama']` vs `['taylor', 'trump']`)
- $\text{Val Speakers} \cap \text{Test Speakers} = \emptyset$ (Disjoint: `['ryan']` vs `['taylor', 'trump']`)
- $\text{Train Recordings} \cap \text{Test Recordings} = \emptyset$
- **Audit Verdict:** `PASSED — ZERO LEAKAGE`.

---

## 4. Leave-One-Speaker-Out (LOSO) Cross-Validation

To measure generalization across all 8 subjects without arbitrary split selection bias, an exhaustive 8-fold Leave-One-Speaker-Out (LOSO) evaluation was performed:

| Fold | Held-Out Subject | Test Slices ($N$) | Genuine / Synthetic | Accuracy | F1-Score | ROC-AUC |
|---|---|---|---|---|---|---|
| 1 | **Joe Biden** | 1,055 | 600 / 455 | 57.35% | 0.6501 | 0.7414 |
| 2 | **Linus Sebastian** | 1,145 | 630 / 515 | 44.89% | 0.5391 | 0.4798 |
| 3 | **Margot Robbie** | 534 | 79 / 455 | 81.84% | 0.8926 | 0.7996 |
| 4 | **Elon Musk** | 1,055 | 600 / 455 | 43.79% | 0.5779 | 0.6626 |
| 5 | **Barack Obama** | 1,055 | 600 / 455 | 63.51% | 0.6966 | 0.8684 |
| 6 | **Ryan Gosling** | 548 | 93 / 455 | 96.72% | 0.9802 | 0.9912 |
| 7 | **Taylor Swift** | 1,055 | 600 / 455 | 53.84% | 0.6406 | 0.7682 |
| 8 | **Donald Trump** | 1,055 | 600 / 455 | 65.97% | 0.7103 | 0.8952 |
| **Mean** | **All 8 Subjects** | **Total: 7,502** | **3,802 / 3,700** | **63.49%** | **0.7109** | **0.7758** |

### Critical Scientific Insights:
1. **Performance Variance Across Speakers:** Accuracy ranges from 43.79% (Elon Musk) to 96.72% (Ryan Gosling).
2. **Female vs Male Pitch Dynamics:** Margot Robbie (female pitch) showed high F1 (0.8926) when trained predominantly on male speakers, while Linus and Musk showed higher confusion.
3. **True Baseline for Forensic Voice Detection:** The mean LOSO accuracy of **63.49%** (with mean ROC-AUC of **0.7758**) represents the true uninflated forensic screening capability of classical spectral/cepstral classifiers on unseen subjects.

---

## 5. Acceptance Status

- **Status:** **ACCEPTED & RESOLVED**
- Leakage has been mathematically eliminated.
- Recording-level manifest saved at `dataset/recording_level_manifest.json`.
- Group-aware slice dataset saved at `dataset/preprocessed_slices_dataset.csv`.
- LOSO evaluation saved at `results/loso_evaluation_summary.csv`.
