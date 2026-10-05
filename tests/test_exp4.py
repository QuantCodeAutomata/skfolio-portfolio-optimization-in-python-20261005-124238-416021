"""Tests for Experiment 4 — walk-forward stacking portfolio optimization."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from skfolio import Population
from skfolio.measures import RiskMeasure
from skfolio.model_selection import WalkForward
from skfolio.optimization import (
    EqualWeighted,
    HierarchicalRiskParity,
    InverseVolatility,
    MeanRisk,
    StackingOptimization,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import chronological_split, load_sp500_returns


@pytest.fixture(scope="module")
def sp500_split():
    X, prices = load_sp500_returns()
    X_train, X_test = chronological_split(X, test_size=0.2)
    return X_train, X_test


def test_walkforward_n_splits(sp500_split):
    """WalkForward(252, 60) on X_test must produce at least 1 split."""
    _, X_test = sp500_split
    cv = WalkForward(train_size=252, test_size=60)
    n_splits = cv.get_n_splits(X_test)
    assert n_splits >= 1, (
        f"WalkForward produces {n_splits} splits on {len(X_test)}-row X_test; "
        "need >= 1"
    )


def test_walkforward_temporal_ordering(sp500_split):
    """Every walk-forward fold: all train indices < all test indices."""
    _, X_test = sp500_split
    cv = WalkForward(train_size=252, test_size=60)
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_test)):
        assert train_idx[-1] < test_idx[0], (
            f"Fold {fold_idx}: temporal leakage train_max={train_idx[-1]} "
            f">= test_min={test_idx[0]}"
        )


def test_stacking_optimization_construction():
    """StackingOptimization must accept the three named base estimators."""
    base = [
        ("IV", InverseVolatility()),
        ("CVAR", MeanRisk(risk_measure=RiskMeasure.CVAR)),
        ("HRP", HierarchicalRiskParity()),
    ]
    stack = StackingOptimization(
        estimators=base,
        final_estimator=MeanRisk(risk_measure=RiskMeasure.CVAR),
    )
    names = [n for n, _ in stack.estimators]
    assert "IV" in names and "CVAR" in names and "HRP" in names


def test_equal_weighted_fit(sp500_split):
    """EqualWeighted must fit and produce weights summing to 1."""
    X_train, _ = sp500_split
    ew = EqualWeighted()
    ew.fit(X_train)
    w_sum = float(np.sum(ew.weights_))
    assert abs(w_sum - 1.0) < 1e-6, f"EW weights sum={w_sum}"


def test_equal_weighted_uniform(sp500_split):
    """EqualWeighted weights should all be equal (1/n)."""
    X_train, _ = sp500_split
    ew = EqualWeighted()
    ew.fit(X_train)
    n = X_train.shape[1]
    expected = 1.0 / n
    tol = 1e-6
    assert np.all(np.abs(ew.weights_ - expected) < tol), (
        "EqualWeighted weights are not uniform"
    )


def test_population_summary_has_both_strategies(sp500_split):
    """Population summary must contain both 'Stacking' and 'EW' entries."""
    from skfolio.model_selection import cross_val_predict

    _, X_test = sp500_split
    X_train, _ = sp500_split

    base = [
        ("IV", InverseVolatility()),
        ("CVAR", MeanRisk(risk_measure=RiskMeasure.CVAR)),
        ("HRP", HierarchicalRiskParity()),
    ]
    stack = StackingOptimization(
        estimators=base,
        final_estimator=MeanRisk(risk_measure=RiskMeasure.CVAR),
    )
    ew = EqualWeighted()
    ew.fit(X_train)

    cv = WalkForward(train_size=252, test_size=60)
    pred_stack = cross_val_predict(stack, X_test, cv=cv, portfolio_params={"name": "Stacking"})
    pred_ew = cross_val_predict(ew, X_test, cv=cv, portfolio_params={"name": "EW"})

    pop = Population([pred_stack, pred_ew])
    summary = pop.summary()

    assert pred_stack.name == "Stacking"
    assert pred_ew.name == "EW"
    assert summary is not None
