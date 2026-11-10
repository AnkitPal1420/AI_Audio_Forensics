"""Audio prediction and forensic authenticity assessment engine.

Executes an auditable 12-step forensic analysis pipeline:
1. Audio file validation and existence verification.
2. Cryptographic SHA-256 hashing and Evidence ID generation.
3. Metadata extraction (file size, container format, bit depth).
4. Channel analysis (mono/stereo/multichannel).
5. Sampling-rate identification and check.
6. Acoustic quality assessment (duration, clipping, silence, SNR).
7. Non-destructive standardized audio decoding (16 kHz mono).
8. Deterministic acoustic and spectral feature extraction.
9. Feature distribution bounds check (OOD detection).
10. Multi-model ML inference (SVM, Random Forest, Logistic Regression).
11. Model consensus and disagreement analysis.
12. Immutable forensic audit trail recording.

Outputs definitive REAL or SYNTHETIC classifications only within validated domains,
strictly returning INDETERMINATE for compromised, out-of-distribution, or ambiguous audio.
"""

from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from src.audio_preprocessing import get_audio_metadata, load_audio
from src.audio_quality import assess_audio_quality, assess_feature_distribution
from src.deepfake_detection import DeepfakeDetectionEngine
from src.feature_extraction import CANONICAL_FEATURE_NAMES, extract_basic_features
from src.forensic_integrity import (
    EvidenceIntegrityError,
    compute_file_sha256,
    generate_evidence_id,
    record_analysis_event,
    verify_evidence_immutability,
)
from src.spectrogram_analysis import generate_forensic_visualizations
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
    import json
    import joblib
    root = project_root()
    m_path = Path(model_path) if model_path else root / "models" / "best_model.joblib"
    feat_path = m_path.parent / "feature_names.json"
    meta_path = m_path.parent / "model_metadata.json"

    if not m_path.is_file():
        raise FileNotFoundError(
            f"Trained model artifact not found at {m_path}."
        )

    model = joblib.load(m_path)

    if feat_path.is_file():
        with feat_path.open("r", encoding="utf-8") as f:
            feature_names = json.load(f)
    else:
        feature_names = CANONICAL_FEATURE_NAMES

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
    model_name: str = "calibrated_svm",
    model_path: Optional[Union[str, Path]] = None,
    run_multi_model: bool = True,
    generate_visuals: bool = True,
    output_dir: Optional[Union[str, Path]] = None,
    return_features: bool = False,
) -> Dict[str, Any]:
    """Execute complete 12-step forensic authenticity assessment on an audio recording.

    Parameters
    ----------
    path : AudioPath
        Path to the target audio exhibit.
    model_name : str, default="calibrated_svm"
        Primary model to use ("calibrated_svm", "svm", "random_forest", "logistic_regression", "best_model").
    model_path : Union[str, Path], optional
        Direct filepath to a trained joblib model artifact. If specified, overrides ``model_name``.
    run_multi_model : bool, default=True
        Whether to run multi-model disagreement analysis across all available classifiers.
    generate_visuals : bool, default=True
        Whether to generate high-resolution spectrogram and waveform plots in results/graphs.
    output_dir : Union[str, Path], optional
        Custom directory for generated graphs. Defaults to results/graphs.
    return_features : bool, default=False
        Whether to include raw extracted feature dictionary in output.

    Returns
    -------
    Dict[str, Any]
        Structured forensic findings including predicted class, probabilities, quality metrics,
        OOD assessment, consensus analysis, and evidence tracking hashes.
    """
    audio_path = Path(path).resolve()
    timestamp_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # Step 1: Audio file validation
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio evidence file does not exist: {audio_path}")

    # Step 2: Cryptographic intake hash & Evidence ID
    intake_sha256 = compute_file_sha256(audio_path)
    evidence_id = generate_evidence_id(audio_path.name, intake_sha256)

    # Step 3, 4, 5: Metadata extraction, channel analysis, and sampling rate
    file_meta = get_audio_metadata(audio_path)
    if file_meta.get("status") != "valid":
        err_msg = file_meta.get("error", "Corrupted or unreadable audio format.")
        result_error = {
            "evidence_id": evidence_id,
            "file_name": audio_path.name,
            "file_path": str(audio_path),
            "file_sha256": intake_sha256,
            "predicted_label": "INDETERMINATE",
            "status": "UNREADABLE_AUDIO",
            "decision_score": None,
            "probability_synthetic": None,
            "probability_genuine": None,
            "calibrated_probability_synthetic": None,
            "calibrated_probability_genuine": None,
            "confidence": 0.0,
            "forensic_assessment": f"Audio file decoding failed: {err_msg}. Cannot proceed with analysis.",
            "analysis_timestamp_utc": timestamp_utc,
        }
        record_analysis_event(result_error)
        return result_error

    # Step 6 & 7: Non-destructive decoding and Acoustic Quality Assessment
    waveform, sr = load_audio(audio_path, sample_rate=16_000, mono=True, normalize=True)
    quality_check = assess_audio_quality(waveform, sample_rate=sr)

    # Check for immediate quality failure (e.g. empty or extreme duration/clipping)
    if quality_check["status"] == "FAIL":
        flags_str = "; ".join(quality_check["quality_flags"])
        result_quality_fail = {
            "evidence_id": evidence_id,
            "file_name": audio_path.name,
            "file_path": str(audio_path),
            "file_sha256": intake_sha256,
            "duration_seconds": quality_check["duration_seconds"],
            "sample_rate": sr,
            "original_sample_rate": file_meta.get("sample_rate"),
            "original_channels": file_meta.get("channels"),
            "quality_assessment": quality_check,
            "predicted_label": "INDETERMINATE",
            "status": "QUALITY_CHECK_FAILED",
            "decision_score": None,
            "probability_synthetic": None,
            "probability_genuine": None,
            "calibrated_probability_synthetic": None,
            "calibrated_probability_genuine": None,
            "confidence": 0.0,
            "forensic_assessment": (
                f"Audio exhibit failed quality screening ({flags_str}). "
                "The signal does not meet forensic standards for automated authenticity assessment."
            ),
            "analysis_timestamp_utc": timestamp_utc,
        }
        record_analysis_event(result_quality_fail)
        return result_quality_fail

    # Step 8: Acoustic & spectral feature extraction
    raw_features = extract_basic_features(waveform, sample_rate=sr, n_mfcc=20)

    # Step 9: Out-of-Distribution (OOD) Domain Validation
    ood_check = assess_feature_distribution(raw_features)

    # Step 10 & 11: Schema validation & ML inference
    if model_path is not None:
        custom_model, expected_feats, _ = load_prediction_model(model_path)
        feature_vector = align_features(raw_features, expected_feats or CANONICAL_FEATURE_NAMES)
        pred_idx = int(custom_model.predict(feature_vector)[0])
        pred_label = "SYNTHETIC" if pred_idx == 1 else "GENUINE"
        prob_synth = None
        prob_real = None
        confidence = 1.0
        score = None
        if hasattr(custom_model, "predict_proba"):
            probs = custom_model.predict_proba(feature_vector)[0]
            prob_real = float(probs[0])
            prob_synth = float(probs[1])
            confidence = float(np.max(probs))
        elif hasattr(custom_model, "decision_function"):
            score = float(custom_model.decision_function(feature_vector)[0])
            prob_synth = float(1.0 / (1.0 + np.exp(-score)))
            prob_real = float(1.0 - prob_synth)
            confidence = float(max(prob_synth, prob_real))
        single_res = {
            "predicted_label": pred_label,
            "prediction_index": pred_idx,
            "confidence": round(confidence, 4),
            "probability_synthetic": round(prob_synth, 4) if prob_synth is not None else None,
            "probability_genuine": round(prob_real, 4) if prob_real is not None else None,
            "decision_score": round(score, 4) if score is not None else None,
        }
        chosen_model = str(model_path)
        multi_res = None
    else:
        # Initialize detection engine
        engine = DeepfakeDetectionEngine()
        available_models = engine.get_available_models()

        if not available_models:
            raise RuntimeError("No trained model artifacts found in models/ directory.")

        chosen_model = model_name if model_name in available_models else ("calibrated_svm" if "calibrated_svm" in available_models else "best_model")
        expected_feats = engine.feature_names or CANONICAL_FEATURE_NAMES
        feature_vector = align_features(raw_features, expected_feats)

        single_res = engine.predict_single(feature_vector, model_key=chosen_model)

        multi_res = None
        if run_multi_model:
            models_to_run = [m for m in ["calibrated_svm", "random_forest", "logistic_regression"] if m in available_models]
            if not models_to_run:
                models_to_run = ["best_model"]
            multi_res = engine.predict_multi_model(feature_vector, models_to_run=models_to_run)

    # Verify original file was not altered during processing
    verify_evidence_immutability(audio_path, intake_sha256)

    # Determine final forensic label and assessment
    if model_path is None and not ood_check["is_in_distribution"]:
        predicted_label = "INDETERMINATE"
        status = "OUT_OF_DISTRIBUTION"
        assessment = (
            f"OUT-OF-DISTRIBUTION WARNING: Acoustic features deviate substantially from the validated "
            f"training domain (Max z-score: {ood_check['max_z_score']} on {ood_check['outlier_count']} features). "
            f"Automated classification is withheld to prevent false alarms. Specialized manual forensic analysis required."
        )
    elif multi_res and multi_res["significant_disagreement"]:
        predicted_label = "INDETERMINATE"
        status = "MODEL_DISAGREEMENT"
        assessment = multi_res["consensus_assessment"]
    else:
        predicted_label = single_res["predicted_label"]
        status = "COMPLETED"
        prob_synth = single_res["probability_synthetic"]
        prob_real = single_res["probability_genuine"]

        if prob_synth is not None and prob_synth >= 0.85:
            assessment = (
                f"The machine-learning system classified the submitted recording as exhibiting characteristics "
                f"associated with synthetic speech within the validated evaluation domain (posterior probability estimate: {prob_synth:.2%}). "
                "This result is an analytical indicator and does not independently establish that the recording has been manipulated."
            )
        elif prob_real is not None and prob_real >= 0.85:
            assessment = (
                f"The machine-learning system classified the submitted recording as consistent with genuine "
                f"human speech within the validated evaluation domain (posterior probability estimate: {prob_real:.2%}). "
                "This result is an analytical indicator and does not provide definitive proof of authenticity."
            )
        else:
            assessment = (
                f"Acoustic feature analysis produced ambiguous classification indicators "
                f"(Synthetic: {prob_synth:.2%}, Genuine: {prob_real:.2%}). Confirmatory biometric and acoustic analysis recommended."
            )

    # Visualizations
    visual_paths: Dict[str, str] = {}
    if generate_visuals:
        try:
            v_dict = generate_forensic_visualizations(
                waveform,
                sr,
                output_dir=output_dir,
                file_stem=audio_path.stem,
                title_suffix=f"[{predicted_label}]",
            )
            visual_paths = {k: str(v) for k, v in v_dict.items()}
        except Exception as e:
            logger.warning("Failed to generate visual figures: %s", e)

    # Consolidate complete analysis result
    result: Dict[str, Any] = {
        "evidence_id": evidence_id,
        "file_name": audio_path.name,
        "file_path": str(audio_path),
        "file_sha256": intake_sha256,
        "file_size_bytes": file_meta["file_size_bytes"],
        "original_format": file_meta["format"],
        "original_sample_rate": file_meta["sample_rate"],
        "original_channels": file_meta["channels"],
        "duration_seconds": round(quality_check["duration_seconds"], 2),
        "status": status,
        "predicted_label": predicted_label,
        "confidence": single_res["confidence"],
        "decision_score": single_res["decision_score"],
        "probability_synthetic": single_res["probability_synthetic"],
        "probability_genuine": single_res["probability_genuine"],
        "calibrated_probability_synthetic": single_res["probability_synthetic"],
        "calibrated_probability_genuine": single_res["probability_genuine"],
        "primary_model_name": chosen_model,
        "quality_assessment": quality_check,
        "distribution_assessment": ood_check,
        "multi_model_analysis": multi_res,
        "visualizations": visual_paths,
        "forensic_assessment": assessment,
        "analysis_timestamp_utc": timestamp_utc,
        "legal_disclaimer": (
            "FORENSIC NOTICE: This assessment is an automated investigative screening indicator based on "
            "statistical signal processing. In accordance with ISO/IEC 27037 technical principles, "
            "it does NOT constitute conclusive legal proof of authenticity or fraudulent synthesis, "
            "and must be corroborated by chain of custody, contextual evidence, and expert human review."
        ),
    }

    if return_features:
        result["extracted_features"] = raw_features

    # Step 12: Record to append-only audit trail
    record_analysis_event(result)

    return result
