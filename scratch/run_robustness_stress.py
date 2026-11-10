import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import joblib
import numpy as np
import soundfile as sf
from src.robustness_testing import (
    evaluate_model_robustness,
    plot_robustness_performance_curve,
    add_gaussian_noise,
    apply_telephony_bandpass,
    apply_lossy_codec_simulation,
    apply_reverberation,
    apply_clipping,
    apply_resampling_stress,
)
from src.feature_extraction import extract_basic_features, validate_feature_vector, CANONICAL_FEATURE_NAMES

root = Path(".")
model_path = root / "models" / "calibrated_svm_model.joblib"
model = joblib.load(model_path)

# Prepare a balanced set of 20 test excerpts (10 genuine, 10 synthetic) from test files
real_files = sorted(list((root / "dataset" / "preprocessed" / "real").glob("*taylor*.wav")) + list((root / "dataset" / "preprocessed" / "real").glob("*trump*.wav")))
fake_files = sorted(list((root / "dataset" / "preprocessed" / "fake").glob("*taylor*.wav")) + list((root / "dataset" / "preprocessed" / "fake").glob("*trump*.wav")))[:10]

# Create temporary slices folder
slice_dir = root / "scratch" / "robustness_slices"
slice_dir.mkdir(parents=True, exist_ok=True)

test_paths = []
test_labels = []

# 10 genuine 2-second slices
slice_idx = 0
for rf in real_files:
    w, sr = sf.read(str(rf))
    for sec in range(0, min(10, int(len(w) // sr)), 2):
        chunk = w[sec*sr : (sec+2)*sr]
        p = slice_dir / f"real_slice_{slice_idx:02d}.wav"
        sf.write(str(p), chunk, sr)
        test_paths.append(p)
        test_labels.append(0)
        slice_idx += 1
        if slice_idx >= 10:
            break
    if slice_idx >= 10:
        break

# 10 synthetic 2-second slices
fake_idx = 0
for ff in fake_files:
    w, sr = sf.read(str(ff))
    chunk = w[:2*sr] if len(w) >= 2*sr else w
    p = slice_dir / f"fake_slice_{fake_idx:02d}.wav"
    sf.write(str(p), chunk, sr)
    test_paths.append(p)
    test_labels.append(1)
    fake_idx += 1
    if fake_idx >= 10:
        break

print(f"Prepared {len(test_paths)} robustness evaluation exhibits (10 genuine, 10 synthetic).")

res = evaluate_model_robustness(
    model=model,
    test_audio_paths=test_paths,
    test_labels=test_labels,
    output_dir=root / "results" / "robustness",
    sample_rate=16000,
)

print("\n=== ROBUSTNESS STRESS-TEST RESULTS ===")
for cond, m in res["results"].items():
    print(f"{cond:36s} -> Acc: {m['accuracy']:.4f}, Prec: {m['precision']:.4f}, Rec: {m['recall']:.4f}, F1: {m['f1_score']:.4f} (TP={m['true_positives']}, FP={m['false_positives']}, TN={m['true_negatives']}, FN={m['false_negatives']})")

chart_path = plot_robustness_performance_curve(res["results"])
print(f"\nSaved diagnostic robustness curve to {chart_path}")
