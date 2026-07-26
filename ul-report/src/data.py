"""
Data loading, sampling, and preprocessing for the UL Report.

Two entry points, matching the two leakage regimes in the FAQ (§5 "Do I need
train, validation, and test splits for unsupervised learning?"):

  get_working_sample()  -- EXPLORATORY regime (Parts 1-3: clustering + DR).
      Fit scaling on the whole ~20k working sample. This is allowed because
      labels are used ONLY after fitting, for interpretation/evaluation.

  get_split()           -- DOWNSTREAM regime (Part 4: NN after DR).
      Stratified train/val/test split with the continuous-feature scaler fit on
      TRAIN ONLY, then applied to val/test. Every learned transformation (scaler,
      PCA, ICA, RP) in Part 4 is fit on train only -- the DR objects themselves
      are fit inside nn_dr.py, also on train only.

Preprocessing choice (config.SCALE_ALL): scale continuous cartographic features
only (default), leaving binary wilderness/soil indicators as 0/1.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from . import config as C


# ── Load / sample ─────────────────────────────────────────────────────────────
def load_full() -> pd.DataFrame:
    if not os.path.exists(C.DATA_PATH):
        raise FileNotFoundError(
            f"Covertype data not found at {C.DATA_PATH}. Copy the SL Report data here."
        )
    return pd.read_csv(C.DATA_PATH, header=None, names=C.FEATURE_COLS + [C.TARGET_COL])


def load_sample() -> pd.DataFrame:
    """Load the SL stratified sample if present, else rebuild it (and warn)."""
    if os.path.exists(C.SL_SAMPLE_PATH):
        return pd.read_csv(C.SL_SAMPLE_PATH)

    warnings.warn(
        f"SL sample not found at {C.SL_SAMPLE_PATH}. Rebuilding a stratified "
        f"n={C.SAMPLE_SIZE} sample with seed={C.SAMPLE_SEED}. CONFIRM this matches "
        "the SL Report sample or mark the SL/OL comparison as approximate.",
        stacklevel=2,
    )
    full = load_full()
    _, sample = train_test_split(
        full, test_size=C.SAMPLE_SIZE, random_state=C.SAMPLE_SEED,
        stratify=full[C.TARGET_COL],
    )
    return sample.reset_index(drop=True)


def _cont_idx() -> np.ndarray:
    return np.array([C.FEATURE_COLS.index(c) for c in C.CONTINUOUS_COLS])


def _scale_cols(scaler_fit_on: np.ndarray) -> np.ndarray:
    """Which column indices the scaler applies to (all, or continuous only)."""
    if C.SCALE_ALL:
        return np.arange(scaler_fit_on.shape[1])
    return _cont_idx()


# ── Exploratory working sample (Parts 1-3) ────────────────────────────────────
@dataclass
class WorkingSample:
    X: np.ndarray            # preprocessed feature matrix (n, 54)
    y: np.ndarray            # Cover_Type mapped to 0..6 (external label only)
    feature_names: list[str]
    scaler: StandardScaler
    scaled_idx: np.ndarray

    def label_1to7(self) -> np.ndarray:
        return self.y + 1


def get_working_sample() -> WorkingSample:
    """~20k Covertype rows, preprocessed for exploratory clustering/DR.

    Scaling is fit on the whole sample (unsupervised); labels are attached only
    for post-hoc evaluation and are never used to fit anything here.
    """
    df = load_sample()
    X = df[C.FEATURE_COLS].to_numpy(dtype=np.float64)
    y = (df[C.TARGET_COL].to_numpy() - 1).astype(np.int64)  # 1..7 -> 0..6

    idx = _scale_cols(X)
    scaler = StandardScaler().fit(X[:, idx])
    X[:, idx] = scaler.transform(X[:, idx])
    return WorkingSample(X, y, list(C.FEATURE_COLS), scaler, idx)


# ── Leakage-safe split for the downstream NN (Part 4) ─────────────────────────
@dataclass
class DataSplit:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler
    scaled_idx: np.ndarray

    def sizes(self) -> dict:
        return {"train": len(self.y_train), "val": len(self.y_val), "test": len(self.y_test)}


def get_split() -> DataSplit:
    """Stratified train/val/test split with scaler fit on TRAIN ONLY.

    Identical strategy to the OL Report so the NN-after-DR comparison stays on the
    same footing as prior reports.
    """
    df = load_sample()
    X = df[C.FEATURE_COLS].to_numpy(dtype=np.float32)
    y = (df[C.TARGET_COL].to_numpy() - 1).astype(np.int64)

    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=C.TEST_FRAC, random_state=C.SPLIT_SEED, stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=C.VAL_FRAC, random_state=C.SPLIT_SEED, stratify=y_tmp,
    )

    idx = _scale_cols(X_train)
    scaler = StandardScaler().fit(X_train[:, idx])
    for arr in (X_train, X_val, X_test):
        arr[:, idx] = scaler.transform(arr[:, idx])

    return DataSplit(X_train, y_train, X_val, y_val, X_test, y_test, scaler, idx)


def class_weights(y: np.ndarray):
    """Inverse-frequency weights for the imbalanced Covertype classes (for NN)."""
    import torch

    counts = np.bincount(y, minlength=C.N_CLASSES).astype(np.float64)
    w = counts.sum() / (C.N_CLASSES * np.maximum(counts, 1))
    return torch.tensor(w, dtype=torch.float32)
