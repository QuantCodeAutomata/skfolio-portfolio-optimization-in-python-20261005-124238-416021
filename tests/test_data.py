"""Unit tests for src/data.py — shared data loading and preprocessing."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import (
    chronological_split,
    get_dataset_metadata,
    load_factor_returns,
    load_sp500_returns,
)


# ---------------------------------------------------------------------------
# SP500 dataset
# ---------------------------------------------------------------------------

def test_load_sp500_returns_shape():
    """Returns DataFrame must have positive observations and multiple assets."""
    X, prices = load_sp500_returns()
    assert isinstance(X, pd.DataFrame)
    assert X.shape[0] > 100, "Too few observations"
    assert X.shape[1] > 1, "Too few assets"


def test_load_sp500_returns_no_nans_edge():
    """Check that the first and last returns rows are non-null."""
    X, _ = load_sp500_returns()
    assert not X.iloc[0].isnull().all(), "First row is all NaN"
    assert not X.iloc[-1].isnull().all(), "Last row is all NaN"


def test_load_sp500_returns_contains_aapl():
    """AAPL must be present in the asset universe (required by exp_1 constraint)."""
    X, _ = load_sp500_returns()
    assert "AAPL" in X.columns, "AAPL not found in SP500 dataset"


def test_load_sp500_returns_index_sorted():
    """Date index must be sorted in ascending order."""
    X, _ = load_sp500_returns()
    assert X.index.is_monotonic_increasing, "Returns index is not sorted"


# ---------------------------------------------------------------------------
# Factor dataset
# ---------------------------------------------------------------------------

def test_load_factor_returns_alignment():
    """Asset and factor return DataFrames must have the same index after loading."""
    X, y, _, _ = load_factor_returns()
    assert (X.index == y.index).all(), "X and y indices are not aligned"


def test_load_factor_returns_shapes():
    """Both X and y must be non-empty with multiple columns."""
    X, y, _, _ = load_factor_returns()
    assert X.shape[0] > 0 and y.shape[0] > 0
    assert X.shape[1] > 1 and y.shape[1] >= 1


# ---------------------------------------------------------------------------
# Chronological split
# ---------------------------------------------------------------------------

def test_chronological_split_no_shuffle():
    """Train indices must strictly precede test indices."""
    X, _ = load_sp500_returns()
    X_train, X_test = chronological_split(X, test_size=0.2)
    assert X_train.index[-1] < X_test.index[0], "Temporal leakage in split"


def test_chronological_split_sizes():
    """Train+test size must equal original; test fraction must be ~0.2."""
    X, _ = load_sp500_returns()
    X_train, X_test = chronological_split(X, test_size=0.2)
    assert len(X_train) + len(X_test) == len(X)
    assert abs(len(X_test) / len(X) - 0.2) < 0.02, "Test fraction far from 20%"


def test_chronological_split_with_factors():
    """Joint (X, y) split must preserve alignment."""
    X, y, _, _ = load_factor_returns()
    X_train, X_test, y_train, y_test = chronological_split(X, y=y, test_size=0.2)
    assert (X_train.index == y_train.index).all()
    assert (X_test.index == y_test.index).all()
    assert X_train.index[-1] < X_test.index[0]


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def test_get_dataset_metadata_keys():
    """Metadata dict must contain the expected keys."""
    X, prices = load_sp500_returns()
    meta = get_dataset_metadata(X, prices)
    for key in ("n_obs", "n_assets", "date_start", "date_end", "assets"):
        assert key in meta, f"Missing key: {key}"


def test_get_dataset_metadata_values():
    """Metadata values must be consistent with the data."""
    X, prices = load_sp500_returns()
    meta = get_dataset_metadata(X, prices)
    assert meta["n_obs"] == len(X)
    assert meta["n_assets"] == X.shape[1]
    assert len(meta["assets"]) == X.shape[1]
