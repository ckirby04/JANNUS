"""
Subject-level 5-fold cross-validation splitter for stacking classifier training.

Guarantees zero patient leakage across folds by splitting on subject IDs
rather than individual slices or voxels.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold


def validate_no_patient_leakage(
    folds: list[tuple[np.ndarray, np.ndarray]],
    subject_ids: Sequence[str],
) -> None:
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        train_subjects = set(subject_ids[i] for i in train_idx)
        val_subjects = set(subject_ids[i] for i in val_idx)
        overlap = train_subjects & val_subjects
        assert len(overlap) == 0, (
            f"PATIENT LEAKAGE DETECTED in fold {fold_idx}: "
            f"{len(overlap)} subjects appear in both train and val: {overlap}"
        )
    print("CV validation passed: zero patient leakage across all folds.")


def build_subject_folds(
    subject_ids: Sequence[str],
    has_lesion: Sequence[bool],
    n_splits: int = 5,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build stratified k-fold splits at subject level.

    Args:
        subject_ids: list of subject ID strings
        has_lesion: binary label per subject (True if any foreground voxels)
        n_splits: number of folds
        seed: random state for reproducibility

    Returns:
        List of (train_indices, val_indices) numpy arrays, one per fold.
    """
    subject_ids = list(subject_ids)
    labels = np.array([int(h) for h in has_lesion])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for train_idx, val_idx in skf.split(subject_ids, labels):
        folds.append((train_idx, val_idx))

    validate_no_patient_leakage(folds, subject_ids)

    _validate_full_coverage(folds, len(subject_ids), n_splits)

    return folds


def _validate_full_coverage(
    folds: list[tuple[np.ndarray, np.ndarray]],
    n_subjects: int,
    n_splits: int,
) -> None:
    val_indices = set()
    for _, val_idx in folds:
        val_indices.update(val_idx.tolist())
    assert len(val_indices) == n_subjects, (
        f"Not all subjects covered: {len(val_indices)} / {n_subjects}"
    )


def folds_from_cache_dir(
    cache_dir: str | Path,
    n_splits: int = 5,
    seed: int = 42,
) -> tuple[list[str], list[tuple[np.ndarray, np.ndarray]]]:
    """Build folds from a stacking cache directory.

    Returns:
        (subject_ids, folds) where subject_ids is the list of case IDs
        and folds is the list of (train_idx, val_idx) arrays.
    """
    cache_dir = Path(cache_dir)
    cache_files = sorted(cache_dir.glob("*.npz"))
    if not cache_files:
        raise RuntimeError(f"No .npz files found in {cache_dir}")

    subject_ids = [f.stem for f in cache_files]

    has_lesion = []
    for f in cache_files:
        data = np.load(f)
        mask = data["mask"]
        has_lesion.append(bool(mask.sum() > 0))

    folds = build_subject_folds(subject_ids, has_lesion, n_splits=n_splits, seed=seed)
    return subject_ids, folds
