"""Tests for Experiment 2 — factor-model prior with ridge regression."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from skfolio.measures import RiskMeasure
from skfolio.optimization import MeanRisk, ObjectiveFunction
from skfolio.prior import FactorModel, LoadingMatrixRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import chronological_split, load_factor_returns


@pytest.fixture(scope="module")
def factor_split():
    X, y, prices, factor_prices = load_factor_returns()
    X_train, X_test, y_train, y_test = chronological_split(X, y=y, test_size=0.2)
    return X_train, X_test, y_train, y_test


@pytest.fixture(scope="module")
def fitted_factor_model(factor_split):
    X_train, X_test, y_train, y_test = factor_split
    model = MeanRisk(
        risk_measure=RiskMeasure.VARIANCE,
        objective_function=ObjectiveFunction.MINIMIZE_RISK,
        max_weights={"AAPL": 0.2},
        l2_coef=0.01,
        prior_estimator=FactorModel(
            loading_matrix_estimator=LoadingMatrixRegression(
                linear_regressor=Ridge(alpha=0.1)
            )
        ),
    )
    model.fit(X_train, y_train)
    return model, X_train, X_test


def test_factor_model_fit_successful(fitted_factor_model):
    """Model must fit without error and have non-None weights."""
    model, X_train, _ = fitted_factor_model
    assert model.weights_ is not None


def test_factor_model_weights_full_investment(fitted_factor_model, factor_split):
    """Weights must sum to 1 within solver tolerance."""
    model, X_train, _ = fitted_factor_model
    w_sum = float(np.sum(model.weights_))
    assert abs(w_sum - 1.0) < 1e-4, f"Weight sum={w_sum:.6f}"


def test_factor_model_aapl_cap(fitted_factor_model, factor_split):
    """AAPL weight must not exceed 0.20."""
    model, X_train, _ = fitted_factor_model
    aapl_idx = list(X_train.columns).index("AAPL")
    aapl_w = float(model.weights_[aapl_idx])
    assert aapl_w <= 0.2 + 1e-4, f"AAPL weight={aapl_w:.4f}"


def test_factor_model_weight_vector_length(fitted_factor_model, factor_split):
    """Weight vector length must equal number of assets."""
    model, X_train, _ = fitted_factor_model
    assert len(model.weights_) == X_train.shape[1]


def test_temporal_alignment(factor_split):
    """Training must end before test begins; X and y must share same indices."""
    X_train, X_test, y_train, y_test = factor_split
    assert X_train.index[-1] < X_test.index[0], "Temporal leakage"
    assert (X_train.index == y_train.index).all(), "X_train / y_train index mismatch"
    assert (X_test.index == y_test.index).all(), "X_test / y_test index mismatch"


def test_ridge_alpha_parameter():
    """LoadingMatrixRegression must expose the Ridge alpha=0.1."""
    lm = LoadingMatrixRegression(linear_regressor=Ridge(alpha=0.1))
    assert lm.linear_regressor.alpha == 0.1


def test_factor_model_predict_out_of_sample(fitted_factor_model):
    """predict on X_test must succeed and return a Portfolio or Population."""
    model, _, X_test = fitted_factor_model
    result = model.predict(X_test)
    assert result is not None
