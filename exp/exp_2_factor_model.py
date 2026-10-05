"""Experiment 2: Minimum-variance optimization with a ridge-regression factor-model prior.

Paper: skfolio: Portfolio Optimization in Python (Nicolini, Manzi, Delatte)
Section: §2.2 "Prior Estimation Methods".

Methodology (confirmed from paper §2.2):
1. Load SP500 prices and factor prices from skfolio bundled datasets.
2. Convert and time-align: X, y = prices_to_returns(prices, factor_prices).
3. Chronological 80/20 split → X_train, X_test, y_train, y_test.
4. Ridge(alpha=0.1) → LoadingMatrixRegression → FactorModel → MeanRisk.
5. model.fit(X_train, y_train).
6. Validate fit and record diagnostics.

Expected outcome: successful fit using factor-model prior; no numeric target.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

# Using skfolio — Context7 confirmed
from skfolio.measures import RiskMeasure
from skfolio.optimization import MeanRisk, ObjectiveFunction
from skfolio.prior import FactorModel, LoadingMatrixRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import chronological_split, get_dataset_metadata, load_factor_returns

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def run_exp2() -> dict:
    """Run Experiment 2 and return result dict."""
    # -----------------------------------------------------------------------
    # 1. Load aligned asset and factor returns
    # -----------------------------------------------------------------------
    X, y, prices, factor_prices = load_factor_returns()
    meta = get_dataset_metadata(X, prices)
    logger.info("Dataset metadata: %s", meta)
    logger.info("Factor columns: %s", list(y.columns))

    # -----------------------------------------------------------------------
    # 2. Chronological 80/20 split (joint split preserving alignment)
    # -----------------------------------------------------------------------
    X_train, X_test, y_train, y_test = chronological_split(X, y=y, test_size=0.2)

    # Verify alignment — no test data entered training
    assert X_train.index[-1] < X_test.index[0], "Temporal leakage detected!"
    assert (X_train.index == y_train.index).all(), "X and y training indices misaligned!"
    assert (X_test.index == y_test.index).all(), "X and y test indices misaligned!"
    logger.info("Temporal alignment verified: train ends %s, test starts %s",
                X_train.index[-1], X_test.index[0])

    # -----------------------------------------------------------------------
    # 3. Build the estimator hierarchy:
    #    Ridge(alpha=0.1) → LoadingMatrixRegression → FactorModel → MeanRisk
    # -----------------------------------------------------------------------
    ridge_regressor = Ridge(alpha=0.1)
    loading_matrix_estimator = LoadingMatrixRegression(linear_regressor=ridge_regressor)
    factor_prior = FactorModel(loading_matrix_estimator=loading_matrix_estimator)

    model = MeanRisk(
        risk_measure=RiskMeasure.VARIANCE,
        objective_function=ObjectiveFunction.MINIMIZE_RISK,
        max_weights={"AAPL": 0.2},
        l2_coef=0.01,
        prior_estimator=factor_prior,
    )

    # -----------------------------------------------------------------------
    # 4. Fit on training data only (y_train = factor returns aligned to X_train)
    # -----------------------------------------------------------------------
    model.fit(X_train, y_train)
    logger.info("Model fitted successfully.")

    # -----------------------------------------------------------------------
    # 5. Inspect fitted weights and prior attributes
    # -----------------------------------------------------------------------
    weights = model.weights_
    assert weights is not None, "model.weights_ is None after fit"
    assert len(weights) == X_train.shape[1], "Weight vector length mismatch"
    weight_sum = float(np.sum(weights))
    aapl_weight = float(weights[list(X_train.columns).index("AAPL")])

    tol = 1e-4
    assert abs(weight_sum - 1.0) < tol, f"Full-investment violated: sum={weight_sum}"
    assert aapl_weight <= 0.2 + tol, f"AAPL cap violated: {aapl_weight:.4f}"

    logger.info("Weights sum: %.6f (should be 1.0)", weight_sum)
    logger.info("AAPL weight: %.4f (cap: 0.20)", aapl_weight)

    # Top 5 assets by weight
    weight_series = pd.Series(weights, index=X_train.columns).sort_values(ascending=False)
    logger.info("Top-5 weights:\n%s", weight_series.head(5).to_string())

    # Inspect prior (loading matrix)
    fitted_prior = model.prior_estimator_
    loading_matrix = None
    if hasattr(fitted_prior, "loading_matrix_estimator_"):
        lm_est = fitted_prior.loading_matrix_estimator_
        if hasattr(lm_est, "loading_matrix_"):
            loading_matrix = lm_est.loading_matrix_
            logger.info("Loading matrix shape: %s", loading_matrix.shape)
            logger.info("Loading matrix (first 3 rows):\n%s", loading_matrix[:3])

    # -----------------------------------------------------------------------
    # 6. Optional out-of-sample prediction (implementation check only)
    # -----------------------------------------------------------------------
    ptf_test = model.predict(X_test)
    logger.info("Out-of-sample portfolio predicted (implementation check): %s", ptf_test)

    # -----------------------------------------------------------------------
    # 7. Collect results
    # -----------------------------------------------------------------------
    results = {
        "experiment": "exp_2",
        "dataset_metadata": meta,
        "factor_columns": list(y.columns),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "n_assets": X_train.shape[1],
        "n_factors": y_train.shape[1],
        "ridge_alpha": 0.1,
        "l2_coef": 0.01,
        "aapl_cap": 0.2,
        "fit_successful": True,
        "weights_sum": weight_sum,
        "aapl_weight": aapl_weight,
        "constraint_full_investment": bool(abs(weight_sum - 1.0) < tol),
        "constraint_aapl_cap_ok": bool(aapl_weight <= 0.2 + tol),
        "loading_matrix_shape": list(loading_matrix.shape) if loading_matrix is not None else None,
        "top5_weights": weight_series.head(5).to_dict(),
    }

    out_json = RESULTS_DIR / "exp2_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", out_json)

    weights_csv = RESULTS_DIR / "exp2_weights.csv"
    weight_series.to_csv(weights_csv, header=["weight"])
    logger.info("Weights saved to %s", weights_csv)

    return results


if __name__ == "__main__":
    results = run_exp2()
    print("\n=== Experiment 2 Summary ===")
    print(f"Fit successful:              {results['fit_successful']}")
    print(f"Weights sum:                 {results['weights_sum']:.6f}")
    print(f"AAPL weight:                 {results['aapl_weight']:.4f}")
    print(f"Constraint full-investment:  {results['constraint_full_investment']}")
    print(f"Constraint AAPL cap:         {results['constraint_aapl_cap_ok']}")
    print(f"Loading matrix shape:        {results['loading_matrix_shape']}")
    print("Top-5 weights:")
    for ticker, w in results["top5_weights"].items():
        print(f"  {ticker}: {w:.4f}")
