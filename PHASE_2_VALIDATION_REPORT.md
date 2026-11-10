# Phase 2 Validation Report: Dataset Integration and Audio Preprocessing

**Project:** AI-Based Voice Analysis for Detecting Synthetic and Deepfake Audio in Criminal Investigations  
**Project Root:** `AI_Audio_Forensics`  
**Phase:** Phase 2 — Dataset Integration and Audio Preprocessing  
**Validation Date:** 2026-11-10  
**Status:** Completed and Verified  

---

## 1. Executive Summary

Phase 2 of the AI Audio Forensics project has been completed without breaking the existing architecture established in Phase 1. The audio preprocessing and feature extraction modules were implemented to meet stringent scientific and forensic standards:

1. **Evidence Preservation:** Original evidence files remain unmodified and intact.
2. **Chain of Custody:** Cryptographic SHA-256 hashes are computed and recorded for every original exhibit and its preprocessed derivative in `dataset/evidence_manifest.json`.
3. **Resilience & Safe Handling:** Empty, corrupted, and unsupported audio files are safely rejected without crashing the pipeline.
4. **Standardisation:** Multi-channel, multi-sample-rate audio files are converted to mono floating-point representations and standardized to 16,000 Hz, with peak normalization applied while preserving silence.
5. **Comprehensive Feature Extraction:** Extracted all 9 required forensic feature sets (87 feature values per recording, 92 columns total including metadata), completely implemented in deterministic NumPy transforms.
6. **Testing & Verification:** The unit test suite expanded from 5 to 24 tests, with all 24 passing in 0.63 seconds. End-to-end execution on the actual 66 dataset recordings generated `results/reports/extracted_features.csv` and `dataset/evidence_manifest.json` across 8.35 hours of audio.

---

## 2. Dataset Inspection & Architectural Provenance

### 2.1 Downloaded Archives Inspection

Two dataset archives were identified in the host environment:

1. **Primary Dataset: `archive (1).zip` (DEEP-VOICE: DeepFake Voice Recognition)**
   - **Compressed Size:** 3.95 GB
   - **Total Audio Files:** 66 audio recordings (64 WAV, 2 MP3)
   - **Benchmark Feature Matrix:** `KAGGLE/DATASET-balanced.csv` containing 11,778 precomputed 1-second segment feature rows (5,889 FAKE, 5,889 REAL).
   - **Internal Structure:**
     - `KAGGLE/AUDIO/REAL/`: 8 original bona fide speaker recordings (Biden, Linus, Margot Robbie, Musk, Obama, Ryan Reynolds, Taylor Swift, Trump).
     - `KAGGLE/AUDIO/FAKE/`: 56 synthetic voice conversion deepfakes (pairwise speaker conversions).
     - `DEMONSTRATION/DEMONSTRATION/`: 2 demonstration MP3 files (`linus-original-DEMO.mp3` and `linus-to-musk-DEMO.mp3`).

2. **Secondary Dataset: `archive.zip` (The Fake-or-Real / FoR Dataset)**
   - **Compressed Size:** 17.2 GB
   - **Total Audio Files:** 169,754 recordings
   - **Subsets:** `for-original` (raw files), `for-norm` (normalized), `for-2sec` (2-second slices), `for-rerec` (re-recorded through acoustic and telephony channels) partitioned into `training`, `validation`, and `testing`.

### 2.2 Forensic Folder Label Verification (Objective 3 Compliance)

In adherence to Objective 3 ("Never assume that all recordings in a particular folder are genuine or synthetic without checking the dataset documentation"):
- The directory `DEMONSTRATION/DEMONSTRATION/` contains both `linus-original-DEMO.mp3` (genuine) and `linus-to-musk-DEMO.mp3` (synthetic deepfake).
- Naive directory-level labeling would have misclassified the original recording as synthetic or the deepfake as genuine.
- The pipeline inspected each file individually, verifying ground truth provenance and segregating genuine recordings into `dataset/real/` and synthetic voice conversions into `dataset/fake/`.

---

## 3. Dataset Summary Statistics

The actual evidence dataset was extracted and analyzed through `src/audio_preprocessing.py`:

| Metric | Value |
|---|---|
| **Total Evidence Audio Files** | 66 recordings |
| **Total Genuine Recordings** | 9 (8 WAV, 1 MP3) |
| **Total Synthetic Recordings** | 57 (56 WAV, 1 MP3) |
| **Class Distribution (Genuine)** | 13.64% (9 / 66) |
| **Class Distribution (Synthetic)** | 86.36% (57 / 66) |
| **Total Audio Duration** | 30,069.47 seconds (8.35 hours) |
| **Minimum Audio Duration** | 60.00 seconds |
| **Maximum Audio Duration** | 600.45 seconds (~10.01 minutes) |
| **Mean Audio Duration** | 455.60 seconds |
| **Median Audio Duration** | 600.00 seconds |
| **Duration Standard Deviation** | 228.04 seconds |
| **Original Sample-Rate Distribution** | 40,000 Hz: 4 files<br>44,100 Hz: 59 files<br>48,000 Hz: 3 files |
| **Missing or Corrupted Files** | 0 files |
| **Target Standardised Format** | Mono, 16,000 Hz, 16-bit PCM WAV |

---

## 4. Implementation Details

### 4.1 Audio Preprocessing (`src/audio_preprocessing.py`)

- **Evidence Integrity & Hashing:** Implemented `compute_file_hash` using streaming 64 KB binary chunks to calculate SHA-256 digests of audio files of any size without memory exhaustion.
- **Metadata Inspection:** Implemented `get_audio_metadata` using `soundfile.info` to inspect file container, channels, sampling rate, duration, and encoding without decoding raw audio.
- **Robust Exception Hierarchy:**
  - `AudioProcessingError(ValueError)`: Base class ensuring backward compatibility.
  - `EmptyAudioError`: Rejects 0-byte files and zero-sample buffers.
  - `CorruptedAudioError`: Catches unparseable headers, non-finite values (NaN/Inf), and decoding errors.
  - `UnsupportedAudioFormatError`: Rejects non-audio extensions safely.
- **Audio Loading & Resampling:** Implemented `load_audio` with `soundfile.read` backend and `soxr.resample` / linear interpolation fallback, supporting WAV, MP3, FLAC, and OGG.
- **Channel Downmixing:** Implemented `convert_to_mono` via channel averaging across 2D multichannel signals.
- **Peak Normalization:** Implemented `normalize_audio` scaling peak amplitude to 1.0 while preserving pure silence.
- **Derivative Generation:** Implemented `preprocess_and_save` and `preprocess_dataset` to write standardised 16 kHz mono WAV derivatives into `dataset/preprocessed/real` and `dataset/preprocessed/fake`, logging both original and derivative SHA-256 hashes to `dataset/evidence_manifest.json`.

### 4.2 Feature Extraction (`src/feature_extraction.py`)

All 9 required forensic feature types are extracted via `extract_basic_features`, `extract_features`, and `extract_features_from_file`:

1. **20 MFCC Coefficients:** Mel filterbank with 40 bands, log-power compression, and Type-II Discrete Cosine Transform (mean and std for coefficients 1 to 20).
2. **Spectral Centroid:** Center of spectral mass $\frac{\sum f \cdot |X(f)|}{\sum |X(f)|}$ (mean and std).
3. **Spectral Bandwidth:** Spectral spread $\sqrt{\frac{\sum (f - \mu)^2 |X(f)|}{\sum |X(f)|}}$ (mean and std).
4. **Spectral Rolloff:** Frequency below which 85% of total spectral power resides (mean and std).
5. **Zero-Crossing Rate (ZCR):** Rate of sign changes in time domain (mean and std).
6. **RMS Energy:** Frame-level root-mean-square amplitude (mean and std).
7. **Chroma STFT:** 12 semitone pitch classes mapped from frequency bins above 60 Hz to MIDI pitch classes, normalized (overall mean and std, plus per-pitch-class means and stds).
8. **Spectral Contrast:** Energy difference between 85th percentile peaks and 15th percentile valleys across 6 octave sub-bands (200 Hz to Nyquist) in dB (overall mean and std, plus per-band means).
9. **Spectral Flatness:** Wiener entropy ratio of geometric mean to arithmetic mean of spectral power (mean and std).
- **Tabular Export:** Implemented `extract_dataset_features` which exported all 66 dataset recordings into `results/reports/extracted_features.csv` (92 columns total).

---

## 5. Test Suite & Verification Results

### 5.1 Unit Tests (`pytest`)

The test suite was executed using `python -m pytest -q`:

```text
........................                                                 [100%]
24 passed in 0.63s
```

#### Test Inventory:
1. `tests/test_preprocessing.py`:
   - `test_normalize_audio_scales_peak_to_one`: Peak amplitude scaled to 1.0.
   - `test_normalize_audio_preserves_silence`: Silence (zeros) preserved without divide-by-zero.
   - `test_normalize_audio_rejects_non_mono_input`: Rejects 2D inputs.
   - `test_convert_to_mono_averages_stereo`: Verifies channel averaging.
   - `test_resample_waveform_changes_sample_count`: Resampling 32 kHz to 16 kHz.
   - `test_compute_file_hash_deterministic`: Verifies SHA-256 reproducibility.
   - `test_compute_file_hash_missing_file`: Verifies `FileNotFoundError`.
   - `test_get_audio_metadata_valid_wav`: Inspects valid synthetic WAV.
   - `test_get_audio_metadata_empty_file`: Flags 0-byte file status as "empty".
   - `test_get_audio_metadata_corrupted_file`: Flags damaged file status as "corrupted".
   - `test_get_audio_metadata_unsupported_format`: Flags `.txt` as "unsupported".
   - `test_load_audio_resampling_and_mono`: Decodes stereo 48 kHz WAV to 16 kHz mono.
   - `test_load_audio_rejects_empty_file`: Raises `EmptyAudioError`.
   - `test_load_audio_rejects_corrupted_file`: Raises `CorruptedAudioError`.
   - `test_load_audio_rejects_unsupported_extension`: Raises `UnsupportedAudioFormatError`.
   - `test_preprocess_and_save_preserves_original`: Confirms original evidence file hash is unchanged after derivative creation.

2. `tests/test_features.py`:
   - `test_extract_basic_features_returns_finite_summary`: Verifies baseline features.
   - `test_extract_basic_features_rejects_empty_signal`: Rejects empty arrays.
   - `test_extract_features_all_nine_required_types_present`: Verifies all 9 required forensic feature representations are present and finite.
   - `test_extract_features_deterministic_across_runs`: Confirms bit-exact determinism across repeated executions.
   - `test_extract_features_rejects_non_finite_values`: Rejects NaN and Inf signals.
   - `test_extract_features_rejects_invalid_mfcc`: Rejects invalid `n_mfcc` counts.
   - `test_extract_features_spectral_centroid_tracks_frequency`: Verifies centroid scales with pitch (3000 Hz > 300 Hz).
   - `test_extract_features_from_file_integration`: End-to-end extraction from a generated WAV file.

### 5.2 Python Compilation Check

`python -m compileall src tests main.py` executed successfully with 0 syntax or bytecode errors.

### 5.3 CLI Execution Verification

- `python main.py` reports directory status without side effects:
  ```text
  real audio directory: 9 file(s)
  fake audio directory: 57 file(s)
  Project scaffold is ready; model training and prediction are not enabled.
  ```
- `python main.py --preprocess` successfully generated 66 derivative WAVs and `dataset/evidence_manifest.json`.
- `python main.py --extract-features` successfully extracted 92-column feature representations for all 66 files and wrote `results/reports/extracted_features.csv`.

---

## 6. Scientific & Forensic Standards Compliance

1. **Anti-Fabrication:** No audio samples, labels, or features were synthesized or falsified. Audio files originate from the verified DEEP-VOICE dataset.
2. **Evidence Immutability:** Original files in `dataset/real` and `dataset/fake` were opened in read-only binary mode (`"rb"`). Preprocessed derivatives are stored separately in `dataset/preprocessed/`.
3. **Forensic Traceability:** Every preprocessed derivative contains an audit record linking its SHA-256 hash to the original exhibit's SHA-256 hash in `dataset/evidence_manifest.json`.
4. **Reproducibility:** All mathematical transformations rely on deterministic NumPy vector operations and a fixed random seed (`42`).
5. **Data Leakage Mitigation:** 
   - Feature extraction processes files independently.
   - In subsequent modeling phases, train/val/test splits must be stratified at the *speaker level* rather than random file or slice splits, preventing deepfake artifacts from leaking across partitions.

---

## 7. Modified and Created Files

| File | Status | Description |
|---|---|---|
| `src/audio_preprocessing.py` | Modified | Complete audio loading, hashing, validation, mono downmixing, resampling, and preprocessing pipeline. |
| `src/feature_extraction.py` | Modified | Pure NumPy feature extraction for all 9 acoustic feature sets (87 statistics), file extraction, and dataset CSV exporter. |
| `src/utils.py` | Modified | Added preprocessed directory paths and JSON configuration loader. |
| `config/config.json` | Modified | Updated sample rate to 16 kHz and `n_mfcc` to 20 with full path configurations. |
| `main.py` | Modified | Enhanced CLI with `--preprocess`, `--extract-features`, and `--summary` flags while preserving baseline layout checks. |
| `tests/test_preprocessing.py` | Modified | Expanded to 16 unit tests covering evidence hashing, empty/corrupted/unsupported files, resampling, and derivative preservation. |
| `tests/test_features.py` | Modified | Expanded to 8 unit tests covering all 9 feature representations, frequency response, determinism, and integration. |
| `notebooks/exploratory_analysis.ipynb` | Modified | Added interactive exploratory cells for manifest loading, evidence verification, and feature extraction. |
| `dataset/evidence_manifest.json` | Created | Forensic audit manifest tracking 66 exhibits, source/derivative SHA-256 hashes, durations, and sample rates. |
| `results/reports/extracted_features.csv` | Created | Feature matrix of 66 recordings across 92 acoustic and metadata columns. |
| `PHASE_2_VALIDATION_REPORT.md` | Created | Comprehensive validation report for Phase 2. |

---

## 8. Remaining Issues & Next Phase Recommendations

1. **Class Imbalance in Raw Recordings:** The original DEEP-VOICE full-length recordings consist of 9 genuine and 57 synthetic files. The companion slice manifest `dataset/metadata/DATASET-balanced.csv` provides 11,778 balanced 1-second slices (5,889 genuine / 5,889 synthetic). Phase 3 should explore whether training on balanced sliced segments or full-length aggregate features provides superior generalization.
2. **Speaker Partitioning in Phase 3:** In Phase 3 (Model Training), data splitting must be grouped by speaker identity (Biden, Linus, Margot, Musk, Obama, Ryan, Taylor, Trump) so that evaluation metrics measure cross-speaker deepfake generalization rather than speaker voice memorization.
3. **No Model Training Performed:** As required by the Phase 2 specification, no model training, model checkpointing, or prediction was performed.

