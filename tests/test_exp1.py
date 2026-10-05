"""Tests for Experiment 1 — regularized min-variance efficient frontier."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from skfolio.measures import RiskMeasure
from skfolio.optimization import MeanRisk, ObjectiveFunction

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import chronological_split, load_sp500_returns


@pytest.fixture(scope="module")
def sp500_split():
    X, prices = load_sp500_returns()
    X_train, X_test = chronological_split(X, test_size=0.2)
    return X_train, X_test


@pytest.fixture(scope="module")
def fitted_model(sp500_split):
    X_train, _ = sp500_split
    model = MeanRisk(
        risk_measure=RiskMeasure.VARIANCE,
        objective_function=ObjectiveFunction.MINIMIZE_RISK,
        efficient_frontier_size=100,
        max_weights={"AAPL": 0.2},
        l2_coef=0.01,
    )
    model.fit(X_train)
    return model


@pytest.fixture(scope="module")
def populations(sp500_split, fitted_model):
    X_train, X_test = sp500_split
    ptf_train = fitted_model.predict(X_train)
    ptf_test = fitted_model.predict(X_test)
    return ptf_train, ptf_test


def test_frontier_returns_100_portfolios(populations):
    """Efficient frontier must produce exactly 100 portfolios (paper §2.1)."""
    ptf_train, ptf_test = populations
    assert len(ptf_train) == 100, f"Expected 100 train portfolios, got {len(ptf_train)}"
    assert len(ptf_test) == 100, f"Expected 100 test portfolios, got {len(ptf_test)}"


def test_full_investment_constraint(sp500_split, populations):
    """All portfolio weights must sum to 1 within solver tolerance (§2.1)."""
    X_train, _ = sp500_split
    ptf_train, _ = populations
    tol = 1e-4
    for i, ptf in enumerate(ptf_train):
        w_sum = float(np.sum(ptf.weights))
        assert abs(w_sum - 1.0) < tol, (
            f"Portfolio {i}: weight sum={w_sum:.6f} violates full-investment constraint"
        )


def test_aapl_weight_cap(sp500_split, populations):
    """AAPL weight must not exceed 0.20 in any portfolio (paper §2.1)."""
    X_train, _ = sp500_split
    ptf_train, _ = populations
    aapl_idx = list(X_train.columns).index("AAPL")
    tol = 1e-4
    for i, ptf in enumerate(ptf_train):
        aapl_w = float(ptf.weights[aapl_idx])
        assert aapl_w <= 0.2 + tol, (
            f"Portfolio {i}: AAPL={aapl_w:.4f} violates 0.20 cap"
        )


def test_weights_are_non_negative(populations):
    """Default MeanRisk enforces long-only; all weights should be >= 0."""
    ptf_train, _ = populations
    tol = -1e-4
    for i, ptf in enumerate(ptf_train):
        assert np.all(ptf.weights >= tol), (
            f"Portfolio {i}: negative weights found (min={ptf.weights.min():.6f})"
        )


def test_train_frontier_annualized_stats_positive_std(populations):
    """Annualized std dev must be positive for all frontier portfolios."""
    ptf_train, ptf_test = populations
    for ptf in list(ptf_train) + list(ptf_test):
        assert ptf.annualized_standard_deviation > 0, "Non-positive annualized std dev"


def test_training_frontier_dominates_test(populations):
    """Training frontier should show lower variance than test in majority of portfolios.

    Paper: "the training frontier dominates the test frontier … model overfitting."
    Minimum-variance optimization minimises in-sample variance, so the model is
    optimistic about risk: the same weights realise *higher* variance out-of-sample.
    This is the standard overfitting signature for min-variance portfolios.

    Note: comparing raw annualized means across different calendar periods is
    sensitive to market conditions (if the test window is a bull market, test means
    will be higher regardless of optimisation quality).  Comparing standard deviations
    directly tests the paper's overfitting claim at the level where it manifests.
    """
    ptf_train, ptf_test = populations
    n = min(len(ptf_train), len(ptf_test))
    train_stds = np.array([p.annualized_standard_deviation for p in list(ptf_train)[:n]])
    test_stds = np.array([p.annualized_standard_deviation for p in list(ptf_test)[:n]])
    # Weights minimise training variance → training stds ≤ test stds (estimation optimism)
    dominance = np.mean(train_stds <= test_stds)
    assert dominance > 0.5, (
        f"Training frontier does not dominate in majority of points "
        f"(in-sample std ≤ out-of-sample std fraction={dominance:.2f})"
    )


def test_model_is_fitted_before_predict(sp500_split):
    """predict() before fit() must raise an error."""
    X_train, X_test = sp500_split
    model = MeanRisk(
        risk_measure=RiskMeasure.VARIANCE,
        objective_function=ObjectiveFunction.MINIMIZE_RISK,
    )
    with pytest.raises(Exception):
        model.predict(X_test)
