import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
from src.cross_dataset_evaluation import evaluate_cross_dataset_performance

root = Path(".")
manifest_path = root / "dataset" / "external_evaluation" / "for_test" / "for_manifest.json"
with open(manifest_path, "r", encoding="utf-8") as f:
    entries = json.load(f)

print(f"Loaded {len(entries)} external FoR manifest entries.")

models_to_test = [
    ("Calibrated SVM", root / "models" / "calibrated_svm_model.joblib"),
    ("Random Forest", root / "models" / "rf_model.joblib"),
    ("Logistic Regression", root / "models" / "logistic_regression_model.joblib"),
]

all_res = {}
for m_name, m_path in models_to_test:
    if m_path.is_file():
        m = joblib.load(m_path)
        print(f"\n=== Evaluating {m_name} ===")
        res = evaluate_cross_dataset_performance(m, entries)
        all_res[m_name] = res["subset_results"]
        for sub, metrics in res["subset_results"].items():
            cm = metrics["confusion_matrix"]
            print(f"  [{sub}] Total: {metrics['total_samples']} -> "
                  f"Acc: {metrics['accuracy']:.4f}, Prec: {metrics['precision']:.4f}, "
                  f"Rec: {metrics['recall']:.4f}, Spec: {metrics['specificity']:.4f}, "
                  f"F1: {metrics['f1_score']:.4f} (TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]})")

with open(root / "results" / "cross_dataset_results.json", "w", encoding="utf-8") as f:
    json.dump(all_res, f, indent=2)
print("\nSaved full cross-dataset evaluation to results/cross_dataset_results.json")
