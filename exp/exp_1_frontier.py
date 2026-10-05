"""Experiment 1: Regularized minimum-variance efficient frontier on S&P 500 assets.

Paper: skfolio: Portfolio Optimization in Python (Nicolini, Manzi, Delatte)
Section: §2.1 "Overview and Basic Usage", Figure 1B/1C.

Methodology (confirmed from paper §2.1):
1. Load S&P 500 prices with load_sp500_dataset(); convert to returns.
2. Chronological 80/20 train/test split (shuffle=False).
3. MeanRisk with:
   - risk_measure=RiskMeasure.VARIANCE
   - objective_function=ObjectiveFunction.MINIMIZE_RISK
   - efficient_frontier_size=100
   - max_weights={"AAPL": 0.2}
   - l2_coef=0.01
4. Fit on X_train; predict on X_train and X_test.
5. Plot annualized mean vs annualized std dev for both samples.

Expected outcome: training frontier dominates test frontier (overfitting).
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Using skfolio — Context7 confirmed
from skfolio.measures import PerfMeasure, RatioMeasure, RiskMeasure
from skfolio.optimization import MeanRisk, ObjectiveFunction

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import chronological_split, get_dataset_metadata, load_sp500_returns

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def run_exp1() -> dict:
    """Run Experiment 1 and return result dict."""
    # -----------------------------------------------------------------------
    # 1. Load data
    # -----------------------------------------------------------------------
    X, prices = load_sp500_returns()
    meta = get_dataset_metadata(X, prices)
    logger.info("Dataset metadata: %s", meta)

    # -----------------------------------------------------------------------
    # 2. Chronological 80/20 split
    # -----------------------------------------------------------------------
    X_train, X_test = chronological_split(X, test_size=0.2)

    # -----------------------------------------------------------------------
    # 3. Instantiate and fit MeanRisk (paper Eq. 1)
    #    min_w  w^T Sigma_hat w + lambda * w^T w
    #    s.t.   w^T 1 = 1,   w_AAPL <= 0.2
    # -----------------------------------------------------------------------
    model = MeanRisk(
        risk_measure=RiskMeasure.VARIANCE,
        objective_function=ObjectiveFunction.MINIMIZE_RISK,
        efficient_frontier_size=100,
        max_weights={"AAPL": 0.2},
        l2_coef=0.01,
    )
    model.fit(X_train)
    logger.info("MeanRisk fitted. Solver info: %s", getattr(model, "solver_output_", "N/A"))

    # -----------------------------------------------------------------------
    # 4. Predict in-sample and out-of-sample
    # -----------------------------------------------------------------------
    ptf_train = model.predict(X_train)
    ptf_test = model.predict(X_test)

    logger.info("Train population: %d portfolios", len(ptf_train))
    logger.info("Test  population: %d portfolios", len(ptf_test))

    # -----------------------------------------------------------------------
    # 5. Extract annualized mean and std dev from each population
    # -----------------------------------------------------------------------
    def extract_frontier_stats(population) -> pd.DataFrame:
        rows = []
        for ptf in population:
            rows.append(
                {
                    "annualized_mean": ptf.annualized_mean,
                    "annualized_std": ptf.annualized_standard_deviation,
                }
            )
        return pd.DataFrame(rows)

    train_df = extract_frontier_stats(ptf_train)
    test_df = extract_frontier_stats(ptf_test)

    # -----------------------------------------------------------------------
    # 6. Validate constraints
    # -----------------------------------------------------------------------
    # Check full-investment and AAPL cap on all fitted frontier portfolios
    tol = 1e-4
    for i, ptf in enumerate(ptf_train):
        w_sum = np.sum(ptf.weights)
        assert abs(w_sum - 1.0) < tol, f"Portfolio {i}: weights sum={w_sum:.6f}, not 1"
        if "AAPL" in X_train.columns:
            aapl_idx = list(X_train.columns).index("AAPL")
            aapl_w = ptf.weights[aapl_idx]
            assert aapl_w <= 0.2 + tol, f"Portfolio {i}: AAPL weight={aapl_w:.4f} > 0.2"
    logger.info("Constraint validation passed: full-investment and AAPL <= 0.2")

    # -----------------------------------------------------------------------
    # 7. Plot efficient frontiers
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(
        train_df["annualized_std"],
        train_df["annualized_mean"],
        label="Train (in-sample)",
        color="steelblue",
        alpha=0.7,
        s=20,
    )
    ax.scatter(
        test_df["annualized_std"],
        test_df["annualized_mean"],
        label="Test (out-of-sample)",
        color="tomato",
        alpha=0.7,
        s=20,
    )
    ax.set_xlabel("Annualized Standard Deviation")
    ax.set_ylabel("Annualized Mean Return")
    ax.set_title(
        "Exp 1 — Regularized Min-Variance Efficient Frontier\n"
        "(λ=0.01, AAPL cap=20%, 100-point frontier)"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = RESULTS_DIR / "exp1_efficient_frontier.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    logger.info("Plot saved to %s", plot_path)

    # -----------------------------------------------------------------------
    # 8. Print summaries
    # -----------------------------------------------------------------------
    logger.info("\n--- Train Population Summary ---")
    logger.info("\n%s", ptf_train.summary())
    logger.info("\n--- Test Population Summary ---")
    logger.info("\n%s", ptf_test.summary())

    # -----------------------------------------------------------------------
    # 9. Compare train vs test to verify dominance
    # -----------------------------------------------------------------------
    # Align by row index (same frontier point, different sample)
    min_len = min(len(train_df), len(test_df))
    mean_diff = (train_df["annualized_mean"].values[:min_len]
                 - test_df["annualized_mean"].values[:min_len])
    dominance_fraction = float(np.mean(mean_diff > 0))
    logger.info(
        "Training frontier dominates test frontier in %.0f%% of points",
        dominance_fraction * 100,
    )

    # -----------------------------------------------------------------------
    # 10. Collect results
    # -----------------------------------------------------------------------
    results = {
        "experiment": "exp_1",
        "dataset_metadata": meta,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "n_frontier_portfolios_train": len(ptf_train),
        "n_frontier_portfolios_test": len(ptf_test),
        "train_annualized_mean_min": float(train_df["annualized_mean"].min()),
        "train_annualized_mean_max": float(train_df["annualized_mean"].max()),
        "test_annualized_mean_min": float(test_df["annualized_mean"].min()),
        "test_annualized_mean_max": float(test_df["annualized_mean"].max()),
        "train_annualized_std_min": float(train_df["annualized_std"].min()),
        "train_annualized_std_max": float(train_df["annualized_std"].max()),
        "test_annualized_std_min": float(test_df["annualized_std"].min()),
        "test_annualized_std_max": float(test_df["annualized_std"].max()),
        "dominance_fraction_train_over_test": dominance_fraction,
        "constraint_full_investment": True,
        "constraint_aapl_cap_ok": True,
        "plot": str(plot_path),
        "train_frontier_sample": train_df.head(10).to_dict(orient="records"),
        "test_frontier_sample": test_df.head(10).to_dict(orient="records"),
    }

    out_json = RESULTS_DIR / "exp1_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", out_json)

    train_csv = RESULTS_DIR / "exp1_train_frontier.csv"
    test_csv = RESULTS_DIR / "exp1_test_frontier.csv"
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    return results


if __name__ == "__main__":
    results = run_exp1()
    print("\n=== Experiment 1 Summary ===")
    print(f"Train portfolios: {results['n_frontier_portfolios_train']}")
    print(f"Test  portfolios: {results['n_frontier_portfolios_test']}")
    print(
        f"Train dominates test in {results['dominance_fraction_train_over_test']*100:.1f}% of frontier points"
    )
    print(f"Constraint full-investment: {results['constraint_full_investment']}")
    print(f"Constraint AAPL cap:        {results['constraint_aapl_cap_ok']}")
