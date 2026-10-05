"""Experiment 4: Walk-forward evaluation of stacked portfolio optimization vs equal weighting.

Paper: skfolio: Portfolio Optimization in Python (Nicolini, Manzi, Delatte)
Section: §2.3 "Ensemble and Clustering", §3 "Model Selection".

Methodology (confirmed from paper §2.3/§3):
1. Load S&P 500 returns; chronological 80/20 split.
2. Base estimators: InverseVolatility, MeanRisk(CVaR), HierarchicalRiskParity.
3. StackingOptimization with final_estimator=MeanRisk(CVaR).
4. EqualWeighted benchmark (pre-fitted on X_train, then used on X_test).
5. WalkForward(train_size=252, test_size=60).
6. cross_val_predict on X_test for both stacking and benchmark.
7. Combine in Population and print summary.

Expected: two out-of-sample prediction objects; no numeric superiority claim.
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
from skfolio import Population
from skfolio.measures import RiskMeasure
from skfolio.model_selection import WalkForward, cross_val_predict
from skfolio.optimization import (
    EqualWeighted,
    HierarchicalRiskParity,
    InverseVolatility,
    MeanRisk,
    ObjectiveFunction,
    StackingOptimization,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import chronological_split, get_dataset_metadata, load_sp500_returns

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def run_exp4() -> dict:
    """Run Experiment 4 and return result dict."""
    # -----------------------------------------------------------------------
    # 1. Load data and split
    # -----------------------------------------------------------------------
    X, prices = load_sp500_returns()
    meta = get_dataset_metadata(X, prices)
    logger.info("Dataset metadata: %s", meta)

    X_train, X_test = chronological_split(X, test_size=0.2)

    # Verify sufficient observations for walk-forward (needs train_size + test_size <= n)
    min_required = 252 + 60
    if len(X_test) < min_required:
        raise ValueError(
            f"X_test has only {len(X_test)} rows; walk-forward needs >= {min_required} "
            f"(train_size=252 + test_size=60)."
        )
    logger.info(
        "Train: %d obs | Test: %d obs (need >= %d for at least one WF split)",
        len(X_train), len(X_test), min_required,
    )

    # -----------------------------------------------------------------------
    # 2. Define base estimators with paper-specified names
    # -----------------------------------------------------------------------
    base_estimators = [
        ("IV", InverseVolatility()),
        ("CVAR", MeanRisk(risk_measure=RiskMeasure.CVAR)),
        ("HRP", HierarchicalRiskParity()),
    ]

    # -----------------------------------------------------------------------
    # 3. Stacking optimizer: final estimator = MeanRisk(CVaR)
    # -----------------------------------------------------------------------
    model_stacking = StackingOptimization(
        estimators=base_estimators,
        final_estimator=MeanRisk(risk_measure=RiskMeasure.CVAR),
    )

    # -----------------------------------------------------------------------
    # 4. Equal-weighted benchmark (pre-fitted on X_train per paper §3)
    # -----------------------------------------------------------------------
    benchmark = EqualWeighted()
    benchmark.fit(X_train)
    logger.info("Equal-weighted benchmark fitted on X_train.")

    # -----------------------------------------------------------------------
    # 5. Walk-forward cross-validator
    # -----------------------------------------------------------------------
    cv = WalkForward(train_size=252, test_size=60)
    n_splits = cv.get_n_splits(X_test)
    logger.info("WalkForward produces %d splits on X_test", n_splits)

    # -----------------------------------------------------------------------
    # 6. cross_val_predict on X_test for stacking and benchmark
    # -----------------------------------------------------------------------
    logger.info("Running cross_val_predict for stacking model …")
    pred_stacking = cross_val_predict(
        model_stacking,
        X_test,
        cv=cv,
        portfolio_params={"name": "Stacking"},
    )

    logger.info("Running cross_val_predict for equal-weighted benchmark …")
    pred_benchmark = cross_val_predict(
        benchmark,
        X_test,
        cv=cv,
        portfolio_params={"name": "EW"},
    )

    # -----------------------------------------------------------------------
    # 7. Combine and summarise
    # -----------------------------------------------------------------------
    population = Population([pred_stacking, pred_benchmark])
    summary = population.summary()
    logger.info("\nPopulation summary:\n%s", summary.to_string())

    # -----------------------------------------------------------------------
    # 8. Verify temporal ordering of each walk-forward split
    # -----------------------------------------------------------------------
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_test)):
        assert train_idx[-1] < test_idx[0], (
            f"Fold {fold_idx}: temporal leakage — train max={train_idx[-1]}, "
            f"test min={test_idx[0]}"
        )
    logger.info("Temporal ordering verified for all %d walk-forward folds.", n_splits)

    # -----------------------------------------------------------------------
    # 9. Optional cumulative-return plot
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(11, 8))

    # Cumulative returns
    for ptf, color, lbl in [
        (pred_stacking, "steelblue", "Stacking"),
        (pred_benchmark, "tomato", "EW"),
    ]:
        cumret = (1 + pd.Series(ptf.returns)).cumprod() - 1
        axes[0].plot(cumret.values, label=lbl, color=color)
    axes[0].set_title("Exp 4 — Cumulative Out-of-Sample Returns")
    axes[0].set_ylabel("Cumulative Return")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Drawdown
    for ptf, color, lbl in [
        (pred_stacking, "steelblue", "Stacking"),
        (pred_benchmark, "tomato", "EW"),
    ]:
        cum = (1 + pd.Series(ptf.returns)).cumprod()
        drawdown = cum / cum.cummax() - 1
        axes[1].fill_between(
            range(len(drawdown)), drawdown.values, 0, alpha=0.35, color=color, label=lbl
        )
    axes[1].set_title("Drawdown")
    axes[1].set_ylabel("Drawdown")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = RESULTS_DIR / "exp4_stacking_vs_ew.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    logger.info("Plot saved to %s", plot_path)

    # -----------------------------------------------------------------------
    # 10. Collect results
    # -----------------------------------------------------------------------
    summary_dict = summary.to_dict()

    results = {
        "experiment": "exp_4",
        "dataset_metadata": meta,
        "train_size_obs": len(X_train),
        "test_size_obs": len(X_test),
        "walkforward_train_size": 252,
        "walkforward_test_size": 60,
        "n_walkforward_splits": n_splits,
        "stacking_portfolio_name": pred_stacking.name,
        "benchmark_portfolio_name": pred_benchmark.name,
        "population_summary": {k: {str(kk): vv for kk, vv in v.items()} for k, v in summary_dict.items()},
        "temporal_ordering_verified": True,
        "plot": str(plot_path),
    }

    out_json = RESULTS_DIR / "exp4_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Results saved to %s", out_json)

    # Save summary CSV
    summary_csv = RESULTS_DIR / "exp4_population_summary.csv"
    summary.to_csv(summary_csv)
    logger.info("Summary saved to %s", summary_csv)

    return results


if __name__ == "__main__":
    results = run_exp4()
    print("\n=== Experiment 4 Summary ===")
    print(f"Walk-forward splits: {results['n_walkforward_splits']}")
    print(f"Stacking portfolio:  {results['stacking_portfolio_name']}")
    print(f"Benchmark portfolio: {results['benchmark_portfolio_name']}")
    print(f"Temporal ordering:   {results['temporal_ordering_verified']}")
