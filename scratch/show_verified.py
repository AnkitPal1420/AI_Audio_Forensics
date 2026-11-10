import json
d = json.load(open("results/phase6_verified_metrics.json", encoding="utf-8"))
f = lambda x: f"{x:.4f}"
print("SAVED MODELS (val | test)")
for n, v in d["saved_models"].items():
    print(n, "VAL brier/ece/slope/int", [f(v['val'][k]) for k in ['brier','ece','slope','intercept']],
          "TEST brier/ece/slope/auc/pr/eer", [f(v['test'][k]) for k in ['brier','ece','slope','roc_auc','pr_auc','eer']])
print("\nval-selected thr", d["threshold_selected_on_val"])
for k in ["test_at_0.50", "test_at_0.80", "test_at_val_selected_threshold"]:
    r = d[k]; print(k, {x: (round(r[x], 4) if isinstance(r[x], float) else r[x]) for x in r})
print("\nCI95 grouped:", {k: [round(a, 3) for a in v] for k, v in d["bootstrap_ci95_grouped_by_recording"].items()})
print("\nLOSO")
for r in d["loso"]:
    print(r["speaker"], r["n"], "acc", f(r["accuracy"]), "bacc", f(r["balanced_accuracy"]), "auc", f(r["roc_auc"]), "spec", f(r["specificity"]), "rec", f(r["recall"]), "eer", f(r["eer"]))
print("macro", {k: round(v, 4) for k, v in d["loso_macro"].items()})
print("\nPARITY", d["feature_parity_training_vs_inference"])
print("\nFoR")
for s, v in d["for_external"].items():
    print(s, v["n"], "auc", f(v["roc_auc"]), "eer", f(v["eer"]), "@0.5 spec/rec", f(v["at_0.50"]["specificity"]), f(v["at_0.50"]["recall"]),
          "@0.8 spec/rec", f(v["at_0.80"]["specificity"]), f(v["at_0.80"]["recall"]))
print("\nOOD thr", d["ood"]["maha_threshold_p99_train"])
for k, v in d["ood"].items():
    if isinstance(v, dict): print(k, {a: round(b, 4) for a, b in v.items()})

