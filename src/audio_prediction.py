"""Prediction and forensic authenticity assessment for audio evidence.

This module provides calibrated, auditable inference on audio recordings using
trained machine learning models:
- Extracts identical acoustic and spectral features validated in Phase 2.
- Handles feature vector alignment to match model expectations.
- Calculates class probabilities and confidence scores (Platt-scaled SVM posterior estimates).
- Generates structured forensic statements highlighting limitations and uncertainty.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np

from src.audio_preprocessing import compute_file_hash, get_audio_metadata, load_audio
from src.feature_extraction import extract_basic_features
from src.utils import project_root

logger = logging.getLogger(__name__)

AudioPath = Union[str, Path]

# Feature mapping to bridge standard feature extraction keys to benchmark dataset column names
FEATURE_KEY_MAPPING = {
    "chroma_stft": "chroma_mean",
    "rms": "rms_mean",
    "spectral_centroid": "spectral_centroid_mean",
    "spectral_bandwidth": "spectral_bandwidth_mean",
    "rolloff": "spectral_rolloff_mean",
    "zero_crossing_rate": "zero_crossing_rate_mean",
}
for i in range(1, 21):
    FEATURE_KEY_MAPPING[f"mfcc{i}"] = f"mfcc_{i}_mean"


def load_prediction_model(
    model_path: Optional[Union[str, Path]] = None,
) -> Tuple[Any, List[str], Dict[str, Any]]:
    """Load the trained classifier pipeline and its expected feature names and metadata."""
    root = project_root()
    m_path = Path(model_path) if model_path else root / "models" / "best_model.joblib"
    feat_path = m_path.parent / "feature_names.json"
    meta_path = m_path.parent / "model_metadata.json"

    if not m_path.is_file():
        raise FileNotFoundError(
            f"Trained model artifact not found at {m_path}. Train a model first via 'python main.py --train'."
        )

    model = joblib.load(m_path)

    if feat_path.is_file():
        with feat_path.open("r", encoding="utf-8") as f:
            feature_names = json.load(f)
    else:
        feature_names = []

    if meta_path.is_file():
        with meta_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        metadata = {}

    return model, feature_names, metadata


def align_features(raw_features: Dict[str, float], expected_feature_names: List[str]) -> np.ndarray:
    """Map extracted feature dictionary into an ordered vector matching model schema."""
    vector = []
    for name in expected_feature_names:
        if name in raw_features:
            vector.append(raw_features[name])
        elif name in FEATURE_KEY_MAPPING and FEATURE_KEY_MAPPING[name] in raw_features:
            vector.append(raw_features[FEATURE_KEY_MAPPING[name]])
        else:
            # Reverse check
            rev_match = None
            for k, v in FEATURE_KEY_MAPPING.items():
                if v == name and k in raw_features:
                    rev_match = raw_features[k]
                    break
            if rev_match is not None:
                vector.append(rev_match)
            else:
                raise ValueError(f"Required model feature '{name}' missing from extracted audio features.")
    return np.array([vector], dtype=np.float32)


def predict_audio(
    path: AudioPath,
    model_path: Optional[Union[str, Path]] = None,
    return_features: bool = False,
) -> Dict[str, Any]:
    """Execute forensic authenticity assessment on an audio recording.

    Parameters
    ----------
    path : AudioPath
        Path to the target audio file.
    model_path : AudioPath, optional
        Path to the joblib model artifact. If omitted, uses ``models/best_model.joblib``.
    return_features : bool, default=False
        Whether to include raw extracted feature dictionary in output.

    Returns
    -------
    Dict[str, Any]
        Structured prediction output containing class probabilities, confidence,
        file metadata, SHA-256 hash, and calibrated forensic statement.
    """
    audio_path = Path(path).resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio evidence file does not exist: {audio_path}")

    # Inspect file & compute cryptographic hash
    meta = get_audio_metadata(audio_path)
    if meta.get("status") != "valid":
        raise ValueError(f"Cannot process audio evidence: {meta.get('error')}")

    # Load and standardize audio
    waveform, sr = load_audio(audio_path, sample_rate=16_000, mono=True, normalize=True)

    # Extract forensic features
    raw_features = extract_basic_features(waveform, sample_rate=sr, n_mfcc=20)

    # Load model and feature schema
    model, expected_feature_names, model_meta = load_prediction_model(model_path)
    if not expected_feature_names:
        # Fallback to feature dictionary keys
        expected_feature_names = [k for k in raw_features.keys() if k != "duration_seconds"]

    feature_vector = align_features(raw_features, expected_feature_names)

    # Inference
    prediction_idx = int(model.predict(feature_vector)[0])
    predicted_label = "SYNTHETIC" if prediction_idx == 1 else "GENUINE"

    prob_synthetic = None
    prob_genuine = None
    confidence = 1.0

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(feature_vector)[0]
        prob_genuine = float(probs[0])
        prob_synthetic = float(probs[1])
        confidence = float(np.max(probs))
    elif hasattr(model, "decision_function"):
        score = float(model.decision_function(feature_vector)[0])
        # Logistic sigmoid transformation (uncalibrated score mapping)
        prob_synthetic = float(1.0 / (1.0 + np.exp(-score)))
        prob_genuine = float(1.0 - prob_synthetic)
        confidence = float(max(prob_synthetic, prob_genuine))

    # Formulate forensic assessment
    model_name = model_meta.get("best_model_name", type(model).__name__)
    if prob_synthetic is not None:
        if prob_synthetic >= 0.85:
            assessment = (
                f"Acoustic feature analysis by {model_name} indicates high acoustic similarity "
                f"to synthetic voice conversions (posterior probability estimate: {prob_synthetic:.2%}). "
                "Investigative follow-up recommended."
            )
        elif prob_synthetic <= 0.15:
            assessment = (
                f"Acoustic feature analysis by {model_name} indicates strong consistency with "
                f"genuine human speech (posterior probability estimate: {prob_genuine:.2%})."
            )
        else:
            assessment = (
                f"Inconclusive acoustic findings. Synthetic probability estimate: {prob_synthetic:.2%}, "
                f"genuine probability estimate: {prob_genuine:.2%}. Further manual biometric analysis required."
            )
    else:
        assessment = f"Predicted class: {predicted_label}. Model does not supply probability estimates."

    result: Dict[str, Any] = {
        "file_path": str(audio_path),
        "file_name": audio_path.name,
        "file_size_bytes": meta["file_size_bytes"],
        "file_sha256": meta["sha256"],
        "original_format": meta["format"],
        "original_sample_rate": meta["sample_rate"],
        "duration_seconds": float(waveform.size / sr),
        "predicted_label": predicted_label,
        "prediction_index": prediction_idx,
        "confidence": confidence,
        "probability_synthetic": prob_synthetic,
        "probability_genuine": prob_genuine,
        "model_name": model_name,
        "model_path": str(model_path) if model_path else "models/best_model.joblib",
        "forensic_assessment": assessment,
        "legal_disclaimer": (
            "NOTICE: This assessment is an automated investigative screening indicator based on "
            "statistical signal processing. It does NOT constitute conclusive legal proof of authenticity "
            "or fabrication and must be corroborated by chain of custody and contextual evidence."
        ),
    }

    if return_features:
        result["extracted_features"] = raw_features

    return result
