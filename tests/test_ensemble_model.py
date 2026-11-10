"""Unit tests for the soft voting ensemble classifier."""

from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pytest

from src.deepfake_detection import DeepfakeDetectionEngine
from src.ensemble_model import SoftVotingEnsembleClassifier, build_and_save_ensemble
from src.utils import project_root


def test_soft_voting_ensemble_initialization():
    root = project_root()
    ensemble_path = root / "models" / "ensemble_voting_model.joblib"
    if not ensemble_path.is_file():
        build_and_save_ensemble()

    ensemble = joblib.load(ensemble_path)
    assert isinstance(ensemble, SoftVotingEnsembleClassifier)
    assert len(ensemble.estimators) == 3
    assert len(ensemble.classes_) == 2


def test_soft_voting_ensemble_predict_proba():
    root = project_root()
    ensemble_path = root / "models" / "ensemble_voting_model.joblib"
    ensemble = joblib.load(ensemble_path)

    # 26-dim feature vector
    X = np.zeros((3, 26), dtype=np.float32)
    # Sample 0: Typical speech center values
    X[0, 6] = -200.0
    # Sample 1: Extreme synthesis values
    X[1, 6] = 50.0

    probs = ensemble.predict_proba(X)
    assert probs.shape == (3, 2)
    # Probabilities must sum to 1.0 for each sample
    assert np.allclose(np.sum(probs, axis=1), 1.0)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)

    preds = ensemble.predict(X)
    assert preds.shape == (3,)
    assert set(preds).issubset({0, 1})


def test_soft_voting_ensemble_decision_function():
    root = project_root()
    ensemble_path = root / "models" / "ensemble_voting_model.joblib"
    ensemble = joblib.load(ensemble_path)

    X = np.zeros((2, 26), dtype=np.float32)
    scores = ensemble.decision_function(X)
    assert scores.shape == (2,)
    assert not np.isnan(scores).any()


def test_ensemble_registered_in_engine():
    engine = DeepfakeDetectionEngine()
    models = engine.get_available_models()
    assert "ensemble" in models

    model = engine.load_model("ensemble")
    assert model is not None

