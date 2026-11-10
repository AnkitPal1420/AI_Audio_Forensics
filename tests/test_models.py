"""Unit tests for model training, dataset splitting, leakage prevention, and metrics."""

import json
from pathlib import Path

import joblib
import numpy as np
import pytest

from src.model_training import (
    compute_metrics,
    generate_evaluation_graphs,
    save_artifacts,
    split_dataset,
    train_logistic_regression,
    train_random_forest,
    train_svm,
)


@pytest.fixture
def synthetic_classification_data():
    """Deterministic 2-class synthetic dataset with 200 samples and 10 features."""
    rng = np.random.RandomState(42)
    # Class 0: centered around -1.0
    X0 = rng.randn(100, 10).astype(np.float32) - 1.0
    y0 = np.zeros(100, dtype=np.int32)
    # Class 1: centered around +1.0
    X1 = rng.randn(100, 10).astype(np.float32) + 1.0
    y1 = np.ones(100, dtype=np.int32)

    X = np.vstack([X0, X1])
    y = np.concatenate([y0, y1])
    feature_names = [f"feat_{i}" for i in range(10)]
    return X, y, feature_names


def test_split_dataset_stratified_proportions(synthetic_classification_data):
    X, y, _ = synthetic_classification_data
    splits = split_dataset(X, y, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_seed=42)

    X_train, y_train, train_idx = splits["train"]
    X_val, y_val, val_idx = splits["val"]
    X_test, y_test, test_idx = splits["test"]

    # Check sizes (200 total samples)
    assert len(X_train) == 140
    assert len(X_val) == 30
    assert len(X_test) == 30

    # Check zero overlap
    assert len(set(train_idx) & set(val_idx)) == 0
    assert len(set(train_idx) & set(test_idx)) == 0
    assert len(set(val_idx) & set(test_idx)) == 0

    # Check stratification: 50% in each class
    assert np.sum(y_train == 0) == 70
    assert np.sum(y_train == 1) == 70
    assert np.sum(y_val == 0) == 15
    assert np.sum(y_val == 1) == 15


def test_split_dataset_group_leakage_prevention(synthetic_classification_data):
    X, y, _ = synthetic_classification_data
    # 8 distinct speaker groups
    speaker_names = ["spk_A", "spk_B", "spk_C", "spk_D", "spk_E", "spk_F", "spk_G", "spk_H"]
    groups = [speaker_names[i % 8] for i in range(len(X))]

    splits = split_dataset(X, y, groups=groups, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_seed=42)

    _, _, train_idx = splits["train"]
    _, _, val_idx = splits["val"]
    _, _, test_idx = splits["test"]

    train_speakers = set(groups[i] for i in train_idx)
    val_speakers = set(groups[i] for i in val_idx)
    test_speakers = set(groups[i] for i in test_idx)

    # Strictly disjoint group verification
    assert train_speakers.isdisjoint(val_speakers), "Train and Val speakers overlap!"
    assert train_speakers.isdisjoint(test_speakers), "Train and Test speakers overlap!"
    assert val_speakers.isdisjoint(test_speakers), "Val and Test speakers overlap!"


def test_train_all_three_models_synthetic(synthetic_classification_data):
    X, y, _ = synthetic_classification_data
    splits = split_dataset(X, y, random_seed=42)
    X_train, y_train, _ = splits["train"]
    X_val, y_val, _ = splits["val"]

    # Model A: SVM
    svm_pipe, svm_params, svm_metrics = train_svm(X_train, y_train, X_val, y_val, random_seed=42)
    assert svm_pipe is not None
    assert "C" in svm_params
    assert svm_metrics["accuracy"] >= 0.85
    assert svm_metrics["f1_score"] >= 0.85
    assert svm_metrics["roc_auc"] >= 0.90

    # Model B: Random Forest
    rf_model, rf_params, rf_metrics, rf_importances = train_random_forest(X_train, y_train, X_val, y_val, random_seed=42)
    assert rf_model is not None
    assert "n_estimators" in rf_params
    assert rf_metrics["accuracy"] >= 0.85
    assert len(rf_importances) == 10
    assert np.isclose(np.sum(rf_importances), 1.0, atol=1e-4)

    # Model C: Logistic Regression
    lr_pipe, lr_params, lr_metrics = train_logistic_regression(X_train, y_train, X_val, y_val, random_seed=42)
    assert lr_pipe is not None
    assert "C" in lr_params
    assert lr_metrics["accuracy"] >= 0.85


def test_compute_metrics_structure():
    y_true = np.array([0, 0, 1, 1], dtype=np.int32)
    y_pred = np.array([0, 0, 1, 1], dtype=np.int32)
    y_prob = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float32)

    metrics = compute_metrics(y_true, y_pred, y_prob)
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1_score"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["equal_error_rate"] == 0.0
    assert metrics["confusion_matrix"] == [[2, 0], [0, 2]]


def test_save_and_load_artifacts(tmp_path: Path, synthetic_classification_data):
    X, y, feature_names = synthetic_classification_data
    splits = split_dataset(X, y, random_seed=42)
    X_train, y_train, _ = splits["train"]
    X_val, y_val, _ = splits["val"]
    X_test, y_test, _ = splits["test"]

    rf_model, rf_params, rf_metrics, rf_importances = train_random_forest(X_train, y_train, X_val, y_val, random_seed=42)

    all_models = {
        "Random Forest": {"model": rf_model, "params": rf_params, "val_metrics": rf_metrics}
    }
    test_metrics = compute_metrics(y_test, rf_model.predict(X_test), rf_model.predict_proba(X_test)[:, 1])

    models_dir = tmp_path / "models"
    results_dir = tmp_path / "results"
    graphs_dir = tmp_path / "graphs"

    graphs = generate_evaluation_graphs(
        models_dict=all_models,
        X_val=X_val,
        y_val=y_val,
        feature_names=feature_names,
        rf_importances=rf_importances,
        output_dir=graphs_dir,
    )
    assert len(graphs) >= 4

    artifacts = save_artifacts(
        best_model_name="Random Forest",
        best_model=rf_model,
        all_models=all_models,
        feature_names=feature_names,
        split_info=splits,
        test_metrics=test_metrics,
        graphs=graphs,
        models_dir=models_dir,
        results_dir=results_dir,
    )

    # Check joblib load
    loaded_model = joblib.load(artifacts["best_model"])
    pred = loaded_model.predict(X_test)
    assert np.array_equal(pred, rf_model.predict(X_test))

    # Check metadata json
    with open(artifacts["metadata"], encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["best_model_name"] == "Random Forest"
    assert meta["feature_count"] == 10


def test_reproducibility_across_identical_seeds(synthetic_classification_data):
    X, y, _ = synthetic_classification_data
    splits1 = split_dataset(X, y, random_seed=42)
    splits2 = split_dataset(X, y, random_seed=42)

    assert np.array_equal(splits1["train"][2], splits2["train"][2])
    assert np.array_equal(splits1["val"][2], splits2["val"][2])
    assert np.array_equal(splits1["test"][2], splits2["test"][2])

