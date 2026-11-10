"""Group-aware dataset partitioning and leakage verification.

Enforces speaker-disjoint and recording-disjoint partitioning to prevent
temporal, speaker, and acoustic channel leakage across evaluation splits.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

from src.utils import project_root

logger = logging.getLogger(__name__)


def load_slice_dataset(
    csv_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Load group-aware slice dataset containing features, labels, and speaker metadata."""
    root = project_root()
    path = Path(csv_path) if csv_path else root / "dataset" / "preprocessed_slices_dataset.csv"

    if not path.is_file():
        raise FileNotFoundError(f"Slice dataset not found at {path}")

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"Slice dataset is empty: {path}")

    meta_keys = [
        "slice_id",
        "recording_id",
        "group_id",
        "speaker_id",
        "acoustic_source_speaker",
        "label",
        "slice_start_sec",
        "slice_end_sec",
    ]
    feature_names = [k for k in rows[0].keys() if k not in meta_keys]

    X = np.array([[float(r[k]) for k in feature_names] for r in rows], dtype=np.float32)
    y = np.array([int(r["label"]) for r in rows], dtype=np.int32)
    groups = np.array([r["group_id"] for r in rows])
    recording_ids = np.array([r["recording_id"] for r in rows])

    return {
        "X": X,
        "y": y,
        "groups": groups,
        "recording_ids": recording_ids,
        "feature_names": feature_names,
        "rows": rows,
    }


def split_by_speaker_group(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    recording_ids: np.ndarray,
    train_speakers: Optional[Sequence[str]] = None,
    val_speakers: Optional[Sequence[str]] = None,
    test_speakers: Optional[Sequence[str]] = None,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Partition dataset by acoustic source speaker groups to eliminate data leakage.

    Default split partitions the 8 subjects:
    - Train (5 subjects): biden, linus, margot, musk, obama (~62.5% of subjects)
    - Val (1 subject): ryan (~12.5% of subjects)
    - Test (2 subjects): taylor, trump (~25% of subjects)
    """
    unique_groups = sorted(list(set(groups)))

    if train_speakers is None or val_speakers is None or test_speakers is None:
        if len(unique_groups) >= 8:
            # Deterministic, balanced subject grouping
            tr_spk = {"biden", "linus", "margot", "musk", "obama"}
            va_spk = {"ryan"}
            te_spk = {"taylor", "trump"}
        else:
            rng = np.random.RandomState(random_seed)
            shuffled = list(unique_groups)
            rng.shuffle(shuffled)
            n = len(shuffled)
            n_te = max(1, int(round(n * 0.20)))
            n_va = max(1, int(round(n * 0.15)))
            te_spk = set(shuffled[:n_te])
            va_spk = set(shuffled[n_te : n_te + n_va])
            tr_spk = set(shuffled[n_te + n_va :])
    else:
        tr_spk = set(train_speakers)
        va_spk = set(val_speakers)
        te_spk = set(test_speakers)

    # Enforce strict disjointness
    assert tr_spk.isdisjoint(va_spk), f"Train and Val speakers overlap: {tr_spk & va_spk}"
    assert tr_spk.isdisjoint(te_spk), f"Train and Test speakers overlap: {tr_spk & te_spk}"
    assert va_spk.isdisjoint(te_spk), f"Val and Test speakers overlap: {va_spk & te_spk}"

    train_idx = np.where(np.isin(groups, list(tr_spk)))[0]
    val_idx = np.where(np.isin(groups, list(va_spk)))[0]
    test_idx = np.where(np.isin(groups, list(te_spk)))[0]

    # Verify no index overlap
    assert len(set(train_idx) & set(val_idx)) == 0, "Train and Val indices overlap!"
    assert len(set(train_idx) & set(test_idx)) == 0, "Train and Test indices overlap!"
    assert len(set(val_idx) & set(test_idx)) == 0, "Val and Test indices overlap!"

    return {
        "train": (X[train_idx], y[train_idx], train_idx),
        "val": (X[val_idx], y[val_idx], val_idx),
        "test": (X[test_idx], y[test_idx], test_idx),
        "train_speakers": sorted(list(tr_spk)),
        "val_speakers": sorted(list(va_spk)),
        "test_speakers": sorted(list(te_spk)),
        "split_counts": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "class_distributions": {
            "train": {str(c): int(np.sum(y[train_idx] == c)) for c in np.unique(y)},
            "val": {str(c): int(np.sum(y[val_idx] == c)) for c in np.unique(y)},
            "test": {str(c): int(np.sum(y[test_idx] == c)) for c in np.unique(y)},
        },
        "random_seed": random_seed,
    }


def leave_one_speaker_out_folds(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> List[Dict[str, Any]]:
    """Generate 8 Leave-One-Speaker-Out (LOSO) cross-validation folds."""
    unique_speakers = sorted(list(set(groups)))
    folds: List[Dict[str, Any]] = []

    for held_out_speaker in unique_speakers:
        test_mask = groups == held_out_speaker
        train_mask = ~test_mask

        test_idx = np.where(test_mask)[0]
        train_idx = np.where(train_mask)[0]

        folds.append({
            "held_out_speaker": held_out_speaker,
            "train_speakers": [s for s in unique_speakers if s != held_out_speaker],
            "train_idx": train_idx,
            "test_idx": test_idx,
            "train_count": len(train_idx),
            "test_count": len(test_idx),
            "train_y_dist": {str(c): int(np.sum(y[train_idx] == c)) for c in np.unique(y)},
            "test_y_dist": {str(c): int(np.sum(y[test_idx] == c)) for c in np.unique(y)},
        })

    return folds


def verify_no_leakage(
    train_speakers: Sequence[str],
    val_speakers: Sequence[str],
    test_speakers: Sequence[str],
    train_recordings: Sequence[str],
    val_recordings: Sequence[str],
    test_recordings: Sequence[str],
) -> Dict[str, Any]:
    """Execute rigorous mathematical leakage audit between evaluation partitions."""
    tr_s = set(train_speakers)
    va_s = set(val_speakers)
    te_s = set(test_speakers)

    tr_r = set(train_recordings)
    va_r = set(val_recordings)
    te_r = set(test_recordings)

    speaker_overlap_tr_va = tr_s & va_s
    speaker_overlap_tr_te = tr_s & te_s
    speaker_overlap_va_te = va_s & te_s

    recording_overlap_tr_va = tr_r & va_r
    recording_overlap_tr_te = tr_r & te_r
    recording_overlap_va_te = va_r & te_r

    has_leakage = bool(
        speaker_overlap_tr_va
        or speaker_overlap_tr_te
        or speaker_overlap_va_te
        or recording_overlap_tr_va
        or recording_overlap_tr_te
        or recording_overlap_va_te
    )

    return {
        "has_leakage": has_leakage,
        "speaker_overlap_train_val": sorted(list(speaker_overlap_tr_va)),
        "speaker_overlap_train_test": sorted(list(speaker_overlap_tr_te)),
        "speaker_overlap_val_test": sorted(list(speaker_overlap_va_te)),
        "recording_overlap_train_val": sorted(list(recording_overlap_tr_va)),
        "recording_overlap_train_test": sorted(list(recording_overlap_tr_te)),
        "recording_overlap_val_test": sorted(list(recording_overlap_va_te)),
        "status": "PASSED — ZERO LEAKAGE" if not has_leakage else "FAILED — LEAKAGE DETECTED",
    }
