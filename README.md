# AI-Based Voice Analysis for Detecting Synthetic and Deepfake Audio in Criminal Investigations

This repository is a modular Python starter project for research into voice
analysis and synthetic/deepfake audio. It establishes a maintainable structure
for future preprocessing, feature extraction, model evaluation, and reporting.
It does **not** currently implement an audio authenticity detector or train a
model.

## Important research and forensic limitations

Audio model outputs are probabilistic research results, not proof of identity,
authenticity, intent, or criminal activity. Any future investigative use
requires documented data provenance, validated methods, calibrated uncertainty,
independent human review, and compliance with applicable law and dataset
licenses. Do not rely on this scaffold for evidentiary conclusions.

## Project layout

| Path | Purpose |
| --- | --- |
| `dataset/real/` | Empty destination for lawfully sourced, documented real-audio samples. |
| `dataset/fake/` | Empty destination for lawfully sourced, documented synthetic-audio samples. |
| `dataset/README.md` | Dataset handling, provenance, and labeling guidance. |
| `models/` | Destination for locally generated model artifacts; ignored by Git. |
| `results/reports/` | Destination for generated research reports. |
| `results/graphs/` | Destination for generated visualizations. |
| `src/audio_preprocessing.py` | Audio loading and normalization helpers. |
| `src/feature_extraction.py` | Basic MFCC, RMS, and spectral-centroid summary features. |
| `src/model_training.py` | Reserved training entry point; intentionally unimplemented. |
| `src/audio_prediction.py` | Reserved prediction entry point; intentionally unimplemented. |
| `src/forensic_report.py` | Reserved reporting entry point; intentionally unimplemented. |
| `src/utils.py` | Shared project paths and logging. |
| `tests/` | Unit tests for preprocessing and basic feature extraction. |
| `notebooks/` | Exploratory analysis notebook starter. |
| `config/config.json` | Initial audio, feature, and relative-path settings. |
| `main.py` | Dataset-folder readiness check; does not train or modify data. |

The modules provide clear extension points for spectrogram-based analysis,
CNN architectures, and pretrained deep-learning models. These should be added
only with explicit data, evaluation, and reproducibility plans.

## Setup

Use Python 3.9 or later. From the project root, create and activate a virtual
environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the scaffold readiness check and tests:

```powershell
python main.py
python -m pytest
```

In VS Code, open this folder as the workspace and select the `.venv` Python
interpreter. The `notebooks/exploratory_analysis.ipynb` notebook can be opened
after installing the requirements.

## Dataset policy

`dataset/real/` and `dataset/fake/` are intentionally empty. Add genuine and
synthetic audio manually only after reviewing the source's terms and documenting
labels, provenance, and transformations. Audio recordings and generated model
artifacts are excluded from version control by `.gitignore`.

No dataset download, dataset modification, model training, or prediction is
performed by this starter project.
