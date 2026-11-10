"""Command-line entry point for project layout, preprocessing, model training, and prediction."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.audio_prediction import predict_audio
from src.audio_preprocessing import preprocess_dataset
from src.feature_extraction import extract_dataset_features
from src.forensic_report import create_forensic_report
from src.model_training import train_classifier
from src.utils import configure_logging, dataset_directories


def main() -> None:
    """Report dataset folder readiness or execute preprocessing, training, and prediction."""
    parser = argparse.ArgumentParser(
        description="AI Audio Forensics: Phase 3 Machine Learning Development & Prediction Pipeline"
    )
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Execute audio standardisation (mono, 16 kHz) and build evidence manifest.",
    )
    parser.add_argument(
        "--extract-features",
        action="store_true",
        help="Extract all 9 forensic feature types and write to CSV.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print comprehensive dataset summary and statistics.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Execute Phase 3 model training pipeline (SVM, Random Forest, Logistic Regression).",
    )
    parser.add_argument(
        "--predict",
        type=str,
        metavar="AUDIO_FILE",
        help="Execute forensic authenticity assessment on an audio file.",
    )
    args = parser.parse_args()

    configure_logging()

    # Directory status report
    for label, directory in dataset_directories().items():
        if directory.is_dir():
            audio_files = [
                path
                for path in directory.iterdir()
                if path.is_file()
                and path.suffix.lower() in {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
            ]
            print(f"{label} audio directory: {len(audio_files)} file(s)")
        else:
            print(f"{label} audio directory: 0 file(s)")

    if args.preprocess or args.summary:
        summary = preprocess_dataset()
        print("\n--- Dataset Preprocessing Summary ---")
        print(f"Genuine recordings: {summary['total_genuine_recordings']}")
        print(f"Synthetic recordings: {summary['total_synthetic_recordings']}")
        print(f"Sample rates: {summary['sample_rate_distribution']}")
        print(f"Total duration: {summary['audio_duration_statistics']['total_hours']:.2f} hours")
        print(f"Corrupted/missing: {summary['missing_or_corrupted_count']}")

    if args.extract_features:
        print("\nExtracting forensic audio features...")
        features = extract_dataset_features()
        print(f"Extracted features for {len(features)} recordings.")

    if args.train:
        print("\n--- Initiating Phase 3 Model Training Pipeline ---")
        results = train_classifier()
        print(f"Selected Best Model: {results['best_model_name']}")
        print(f"Test Accuracy: {results['test_metrics']['accuracy']:.4f}")
        print(f"Test F1-Score: {results['test_metrics']['f1_score']:.4f}")
        print(f"Test ROC-AUC:  {results['test_metrics']['roc_auc']:.4f}")
        print(f"Saved Model:   {results['saved_artifacts']['best_model']}")

    if args.predict:
        target = Path(args.predict)
        print(f"\n--- Running Forensic Authenticity Assessment on: {target.name} ---")
        pred = predict_audio(target)
        print(f"Predicted Class:   {pred['predicted_label']}")
        print(f"Confidence:        {pred['confidence']:.2%}")
        if pred["probability_synthetic"] is not None:
            print(f"Synthetic Prob:    {pred['probability_synthetic']:.2%}")
            print(f"Genuine Prob:      {pred['probability_genuine']:.2%}")
        print(f"Assessment:        {pred['forensic_assessment']}")
        rep = create_forensic_report(pred)
        print(f"Generated Forensic Report in results/reports/")

    if not any([args.preprocess, args.extract_features, args.summary, args.train, args.predict]):
        print("Project scaffold is ready; model training and prediction are not enabled.")


if __name__ == "__main__":
    main()
