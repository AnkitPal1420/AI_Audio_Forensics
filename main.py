"""Command-line entry point for project layout, preprocessing, model training, prediction, and UI."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from src.audio_prediction import predict_audio
from src.audio_preprocessing import load_audio, preprocess_dataset
from src.deepfake_detection import DeepfakeDetectionEngine
from src.feature_extraction import extract_dataset_features
from src.forensic_report import create_forensic_report
from src.model_training import train_classifier
from src.spectrogram_analysis import generate_forensic_visualizations
from src.utils import configure_logging, dataset_directories, project_root


def main() -> None:
    """Report dataset folder readiness or execute preprocessing, training, prediction, and UI."""
    parser = argparse.ArgumentParser(
        description="AI Audio Forensics: Phase 4 Audio Prediction & Forensic Examination Platform"
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
        help="Execute Phase 4 forensic authenticity assessment on an audio file.",
    )
    parser.add_argument(
        "--analyze-audio",
        type=str,
        metavar="AUDIO_FILE",
        help="Generate high-resolution spectrogram and waveform visual figures.",
    )
    parser.add_argument(
        "--models",
        action="store_true",
        help="List available trained model artifacts, feature schemas, and validation status.",
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Launch the interactive Streamlit forensic investigation web interface.",
    )
    args = parser.parse_args()

    configure_logging()

    if args.ui:
        root = project_root()
        app_file = root / "app.py"
        print(f"\nLaunching AI Audio Forensics Streamlit Web Interface ({app_file})...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_file)])
        return

    if args.models:
        engine = DeepfakeDetectionEngine()
        available = engine.get_available_models()
        print("\n--- Available Forensic Detection Models ---")
        for m in available:
            print(f"  • {m.upper()} -> models/{m}_model.joblib (Loaded successfully)")
        print(f"Expected Feature Count: {len(engine.feature_names)}")
        return

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
        print("\n--- Initiating Model Training Pipeline ---")
        results = train_classifier()
        print(f"Selected Best Model: {results['best_model_name']}")
        print(f"Test Accuracy: {results['test_metrics']['accuracy']:.4f}")
        print(f"Test F1-Score: {results['test_metrics']['f1_score']:.4f}")
        print(f"Test ROC-AUC:  {results['test_metrics']['roc_auc']:.4f}")
        print(f"Saved Model:   {results['saved_artifacts']['best_model']}")

    if args.analyze_audio:
        target = Path(args.analyze_audio).resolve()
        print(f"\n--- Generating Forensic Visualizations for: {target.name} ---")
        wave, sr = load_audio(target, sample_rate=16_000, mono=True, normalize=True)
        figures = generate_forensic_visualizations(wave, sr, file_stem=target.stem)
        for fig_name, fig_path in figures.items():
            print(f"  • {fig_name.title()} Figure: {fig_path}")

    if args.predict:
        target = Path(args.predict).resolve()
        print(f"\n--- Running Phase 4 Forensic Authenticity Assessment: {target.name} ---")
        pred = predict_audio(target, run_multi_model=True, generate_visuals=True)
        print(f"Evidence ID:       {pred['evidence_id']}")
        print(f"SHA-256 Digest:    {pred['file_sha256']}")
        print(f"Status:            {pred['status']}")
        print(f"Predicted Class:   {pred['predicted_label']}")
        print(f"Confidence:        {pred['confidence']:.2%}")
        if pred["calibrated_probability_synthetic"] is not None:
            print(f"Synthetic Prob:    {pred['calibrated_probability_synthetic']:.2%}")
            print(f"Genuine Prob:      {pred['calibrated_probability_genuine']:.2%}")
        if pred.get("decision_score") is not None:
            print(f"Decision Score:    {pred['decision_score']:.4f}")

        # Quality metrics
        q = pred.get("quality_assessment", {})
        print(f"Acoustic Quality:  {q.get('status')} (SNR: {q.get('estimated_snr_db', 0):.1f} dB, Clipping: {q.get('clipping_ratio', 0):.1%})")

        # Domain
        ood = pred.get("distribution_assessment", {})
        print(f"Domain Bounds:     {ood.get('status')} (Max z-score: {ood.get('max_z_score', 0)})")

        # Consensus
        multi = pred.get("multi_model_analysis", {})
        if multi:
            print(f"Model Consensus:   {multi.get('consensus_label')} (Spread: {multi.get('probability_spread', 0):.1%})")

        print(f"\nForensic Assessment:\n{pred['forensic_assessment']}")
        rep = create_forensic_report(pred, generate_pdf=True)
        print(f"\nGenerated Forensic Report: results/reports/FORENSIC_EXAMINATION_REPORT_{target.stem}.md")
        print(f"Generated Companion PDF:   results/reports/FORENSIC_EXAMINATION_REPORT_{target.stem}.pdf")

    if not any([args.preprocess, args.extract_features, args.summary, args.train, args.predict, args.analyze_audio, args.models, args.ui]):
        print("Project scaffold is ready. Use '--predict <audio_file>', '--ui', or '--help'.")


if __name__ == "__main__":
    main()
