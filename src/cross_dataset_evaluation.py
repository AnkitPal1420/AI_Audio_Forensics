"""Independent cross-dataset generalization evaluation module.

Evaluates trained models on external, out-of-distribution deepfake datasets
from the Fake-or-Real (FoR) benchmark archive (C:\\Users\\ANKIT\\Downloads\\archive.zip),
testing on completely unseen speakers, text, and acoustic synthesis conditions.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import zipfile

import numpy as np
import soundfile as sf
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from src.audio_preprocessing import compute_file_hash, load_audio
from src.feature_extraction import CANONICAL_FEATURE_NAMES, extract_basic_features, validate_feature_vector
from src.utils import project_root

logger = logging.getLogger(__name__)


def ingest_external_for_subset(
    zip_path: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    n_clean_per_class: int = 100,
    n_rerec_per_class: int = 50,
) -> Dict[str, Any]:
    """Extract a controlled, balanced evaluation subset from the 17.2 GB FoR archive."""
    root = project_root()
    archive_file = Path(zip_path) if zip_path else Path(r"C:\Users\ANKIT\Downloads\archive.zip")
    target_dir = Path(output_dir) if output_dir else root / "dataset" / "external_evaluation" / "for_test"

    if not archive_file.is_file():
        raise FileNotFoundError(f"External FoR archive not found at {archive_file}")

    target_dir.mkdir(parents=True, exist_ok=True)
    manifest_entries: List[Dict[str, Any]] = []

    with zipfile.ZipFile(archive_file, "r") as z:
        all_names = z.namelist()

        # 1. Clean 2-second testing files
        clean_fake = sorted([n for n in all_names if n.startswith("for-2sec/for-2seconds/testing/fake/") and n.endswith(".wav")])[:n_clean_per_class]
        clean_real = sorted([n for n in all_names if n.startswith("for-2sec/for-2seconds/testing/real/") and n.endswith(".wav")])[:n_clean_per_class]

        # 2. Re-recorded testing files
        rerec_fake = sorted([n for n in all_names if n.startswith("for-rerec/for-rerecorded/testing/fake/") and n.endswith(".wav")])[:n_rerec_per_class]
        rerec_real = sorted([n for n in all_names if n.startswith("for-rerec/for-rerecorded/testing/real/") and n.endswith(".wav")])[:n_rerec_per_class]

        batches = [
            ("clean_fake", clean_fake, "synthetic", 1, "FoR_2sec_Clean"),
            ("clean_real", clean_real, "genuine", 0, "FoR_2sec_Clean"),
            ("rerec_fake", rerec_fake, "synthetic", 1, "FoR_Rerecorded_Acoustic"),
            ("rerec_real", rerec_real, "genuine", 0, "FoR_Rerecorded_Acoustic"),
        ]

        for batch_name, file_list, label_str, label_idx, subset_type in batches:
            sub_folder = target_dir / batch_name
            sub_folder.mkdir(parents=True, exist_ok=True)

            for member in file_list:
                fname = Path(member).name
                dest_path = sub_folder / fname
                if not dest_path.is_file():
                    with z.open(member) as src, dest_path.open("wb") as dst:
                        dst.write(src.read())

                fhash = compute_file_hash(dest_path)
                manifest_entries.append({
                    "file_path": str(dest_path.resolve()),
                    "file_name": fname,
                    "file_sha256": fhash,
                    "label": label_str,
                    "label_idx": label_idx,
                    "subset_type": subset_type,
                    "original_archive_member": member,
                    "dataset_provenance": "Fake-or-Real (FoR) External Benchmark",
                })

    manifest_file = target_dir / "for_manifest.json"
    with manifest_file.open("w", encoding="utf-8") as f:
        json.dump(manifest_entries, f, indent=2)

    logger.info("Ingested %d external FoR evaluation files into %s", len(manifest_entries), target_dir)
    return {
        "total_files": len(manifest_entries),
        "manifest_path": str(manifest_file),
        "entries": manifest_entries,
    }


def evaluate_cross_dataset_performance(
    model: Any,
    manifest_entries: List[Dict[str, Any]],
    output_csv: Optional[Union[str, Path]] = None,
    sample_rate: int = 16_000,
) -> Dict[str, Any]:
    """Execute cross-dataset generalization evaluation across all ingested external exhibits."""
    root = project_root()
    out_file = Path(output_csv) if output_csv else root / "results" / "cross_dataset_evaluation_summary.csv"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    subset_results: Dict[str, Dict[str, Any]] = {}
    subsets = sorted(list(set(e["subset_type"] for e in manifest_entries)))

    for sub in subsets:
        entries = [e for e in manifest_entries if e["subset_type"] == sub]
        y_true, y_pred, y_prob = [], [], []

        for e in entries:
            fpath = Path(e["file_path"])
            try:
                waveform, sr = load_audio(fpath, sample_rate=sample_rate, mono=True, normalize=True)
                feats = extract_basic_features(waveform, sample_rate=sr)
                vec = validate_feature_vector(feats, CANONICAL_FEATURE_NAMES)
                pred = int(model.predict(vec)[0])
                prob = float(model.predict_proba(vec)[0, 1]) if hasattr(model, "predict_proba") else 0.5
                y_true.append(e["label_idx"])
                y_pred.append(pred)
                y_prob.append(prob)
            except Exception as exc:
                logger.warning("Error processing %s: %s", fpath, exc)

        yt = np.array(y_true, dtype=np.int32)
        yp = np.array(y_pred, dtype=np.int32)

        acc = float(accuracy_score(yt, yp))
        prec = float(precision_score(yt, yp, zero_division=0))
        rec = float(recall_score(yt, yp, zero_division=0))
        f1 = float(f1_score(yt, yp, zero_division=0))
        cm = confusion_matrix(yt, yp)
        tn, fp, fn, tp = cm.ravel()
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        subset_results[sub] = {
            "total_samples": len(yt),
            "genuine_count": int(np.sum(yt == 0)),
            "synthetic_count": int(np.sum(yt == 1)),
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "specificity": spec,
            "f1_score": f1,
            "confusion_matrix": cm.tolist(),
        }

    # Write summary table
    with out_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Subset", "Total", "Genuine", "Synthetic", "Accuracy", "Precision", "Recall", "Specificity", "F1_Score", "TN", "FP", "FN", "TP"])
        for sub, m in subset_results.items():
            cm = m["confusion_matrix"]
            writer.writerow([
                sub,
                m["total_samples"],
                m["genuine_count"],
                m["synthetic_count"],
                f"{m['accuracy']:.4f}",
                f"{m['precision']:.4f}",
                f"{m['recall']:.4f}",
                f"{m['specificity']:.4f}",
                f"{m['f1_score']:.4f}",
                cm[0][0], cm[0][1], cm[1][0], cm[1][1],
            ])

    logger.info("Saved cross-dataset evaluation summary to %s", out_file)
    return {
        "subset_results": subset_results,
        "summary_csv": str(out_file),
    }
