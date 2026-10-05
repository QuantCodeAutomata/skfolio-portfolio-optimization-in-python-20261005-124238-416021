"""Experiment 3: Copula-based synthetic return generation and dependence visualization.

Paper: skfolio: Portfolio Optimization in Python (Nicolini, Manzi, Delatte)
Section: §2.2.2 "Copula-based Synthetic Priors", Figure 2.

Methodology:
1. Load historical returns for AAPL, JPM, LLY from S&P 500 dataset.
2. Fit a 3-dimensional vine copula (pyvinecopulib).
3. Generate synthetic returns with the same size as the historical sample.
4. Plot pairwise scatter: historical vs synthetic for AAPL-JPM, JPM-LLY, LLY-AAPL.
5. Fit bivariate Student-t copula for AAPL-JPM; visualize density (scale 0 to 2.0).

DEVIATION: skfolio 0.6.0 has no copula module. pyvinecopulib is used as a substitute.
           Reported in results/DEVIATIONS.md.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.copula_utils import (
    bivariate_student_t_density_grid,
    fit_bivariate_student_t_copula,
    fit_vine_copula,
    pseudo_obs_to_returns,
    sample_vine_copula,
)
from src.data import get_dataset_metadata, load_sp500_returns

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(exist_ok=True)

RANDOM_SEED = 42
ASSETS = ["AAPL", "JPM", "LLY"]
ASSET_PAIRS = [("AAPL", "JPM"), ("JPM", "LLY"), ("LLY", "AAPL")]


def run_exp3() -> dict:
    """Run Experiment 3 and return result dict."""
    # -----------------------------------------------------------------------
    # 1. Load SP500 returns and select assets
    # -----------------------------------------------------------------------
    X, prices = load_sp500_returns()
    meta = get_dataset_metadata(X, prices)

    # Validate assets are present
    missing = [a for a in ASSETS if a not in X.columns]
    if missing:
        raise ValueError(f"Assets not found in dataset: {missing}. Available: {list(X.columns)}")

    X_sub = X[ASSETS].dropna()
    n_hist = len(X_sub)
    logger.info(
        "Using %d historical observations for assets: %s", n_hist, ASSETS
    )

    # -----------------------------------------------------------------------
    # 2. Fit 3-dimensional vine copula
    # -----------------------------------------------------------------------
    vine, U_hist = fit_vine_copula(X_sub, ASSETS, random_state=RANDOM_SEED)

    vine_structure = str(vine.structure)
    vine_families = [str(p.family) for p in vine.pair_copulas[0]]
    logger.info("Vine structure: %s", vine_structure)
    logger.info("Pair copula families (tree 1): %s", vine_families)

    # -----------------------------------------------------------------------
    # 3. Generate synthetic returns (same size as historical)
    # -----------------------------------------------------------------------
    U_sim = sample_vine_copula(vine, n_samples=n_hist, random_state=RANDOM_SEED)
    X_hist_arr = X_sub.values.astype(float)
    X_sim_arr = pseudo_obs_to_returns(U_sim, X_hist_arr)

    X_sim = pd.DataFrame(X_sim_arr, columns=ASSETS)

    logger.info(
        "Synthetic returns generated: %d obs × %d assets", *X_sim.shape
    )

    # -----------------------------------------------------------------------
    # 4. Pairwise scatter plots: historical vs synthetic
    #    Figure 2 axis ranges (paper): AAPL-JPM: ±0.2; JPM-LLY: ±0.1; LLY-AAPL: ±0.1/0.15
    # -----------------------------------------------------------------------
    axis_ranges = {
        ("AAPL", "JPM"): ((-0.25, 0.25), (-0.25, 0.25)),
        ("JPM", "LLY"): ((-0.15, 0.15), (-0.15, 0.15)),
        ("LLY", "AAPL"): ((-0.15, 0.15), (-0.20, 0.20)),
    }

    fig_scatter, axes = plt.subplots(2, 3, figsize=(15, 10))
    row_labels = ["Historical", "Synthetic"]
    pair_labels = [f"{a1}-{a2}" for a1, a2 in ASSET_PAIRS]

    for col, (a1, a2) in enumerate(ASSET_PAIRS):
        xlim, ylim = axis_ranges[(a1, a2)]

        # Historical
        axes[0, col].scatter(
            X_sub[a1].values,
            X_sub[a2].values,
            alpha=0.25,
            s=4,
            color="steelblue",
        )
        axes[0, col].set_xlim(xlim)
        axes[0, col].set_ylim(ylim)
        axes[0, col].set_xlabel(a1)
        axes[0, col].set_ylabel(a2)
        axes[0, col].set_title(f"Historical: {a1}–{a2}")
        axes[0, col].grid(True, alpha=0.3)

        # Synthetic
        axes[1, col].scatter(
            X_sim[a1].values,
            X_sim[a2].values,
            alpha=0.25,
            s=4,
            color="tomato",
        )
        axes[1, col].set_xlim(xlim)
        axes[1, col].set_ylim(ylim)
        axes[1, col].set_xlabel(a1)
        axes[1, col].set_ylabel(a2)
        axes[1, col].set_title(f"Synthetic: {a1}–{a2}")
        axes[1, col].grid(True, alpha=0.3)

    fig_scatter.suptitle(
        "Exp 3 — Vine Copula: Historical vs Synthetic Pairwise Returns\n"
        "(AAPL-JPM | JPM-LLY | LLY-AAPL)",
        fontsize=13,
    )
    plt.tight_layout()
    scatter_path = RESULTS_DIR / "exp3_pairwise_scatter.png"
    fig_scatter.savefig(scatter_path, dpi=150)
    plt.close(fig_scatter)
    logger.info("Pairwise scatter saved to %s", scatter_path)

    # -----------------------------------------------------------------------
    # 5. Bivariate Student-t copula for AAPL-JPM
    # -----------------------------------------------------------------------
    rho, nu, U_bivariate, params = fit_bivariate_student_t_copula(X_sub, "AAPL", "JPM")

    uu1, uu2, density = bivariate_student_t_density_grid(rho, nu, grid_size=100)

    # Density plot (paper: scale 0.0 to 2.0, ticks at 0.4, 0.8, 1.2, 1.6)
    fig_density, ax_d = plt.subplots(figsize=(7, 6))
    # Clip density at 2.0 to match paper figure
    density_clipped = np.clip(density, 0, 2.0)
    pcm = ax_d.pcolormesh(
        uu1, uu2, density_clipped,
        shading="auto",
        cmap="viridis",
        vmin=0.0,
        vmax=2.0,
    )
    cbar = fig_density.colorbar(pcm, ax=ax_d, ticks=[0.0, 0.4, 0.8, 1.2, 1.6, 2.0])
    cbar.set_label("Density")
    ax_d.contour(uu1, uu2, density_clipped,
                 levels=[0.4, 0.8, 1.2, 1.6, 2.0],
                 colors="white", linewidths=0.6, alpha=0.5)
    ax_d.set_xlabel("AAPL (quantile)")
    ax_d.set_ylabel("JPM (quantile)")
    ax_d.set_title(
        f"Exp 3 — Bivariate Student-t Copula Density\n"
        f"AAPL–JPM  (ρ={rho:.3f}, ν={nu:.1f})"
    )
    density_path = RESULTS_DIR / "exp3_student_t_copula_density.png"
    fig_density.savefig(density_path, dpi=150)
    plt.close(fig_density)
    logger.info("Density plot saved to %s", density_path)

    # -----------------------------------------------------------------------
    # 6. Collect results
    # -----------------------------------------------------------------------
    results = {
        "experiment": "exp_3",
        "dataset_metadata": meta,
        "assets": ASSETS,
        "n_hist_obs": n_hist,
        "n_synthetic_obs": int(X_sim.shape[0]),
        "random_seed": RANDOM_SEED,
        "vine_structure": vine_structure,
        "vine_pair_copula_families_tree1": vine_families,
        "bivariate_student_t_rho": rho,
        "bivariate_student_t_nu": nu,
        "plots": {
            "pairwise_scatter": str(scatter_path),
            "student_t_density": str(density_path),
        },
        "deviation": "skfolio 0.6.0 lacks copula module; pyvinecopulib used instead.",
    }

    out_json = RESULTS_DIR / "exp3_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", out_json)

    # Save synthetic returns sample
    X_sim.head(50).to_csv(RESULTS_DIR / "exp3_synthetic_returns_sample.csv", index=False)

    return results


if __name__ == "__main__":
    results = run_exp3()
    print("\n=== Experiment 3 Summary ===")
    print(f"Assets:              {results['assets']}")
    print(f"Historical obs:      {results['n_hist_obs']}")
    print(f"Synthetic obs:       {results['n_synthetic_obs']}")
    print(f"Vine families (T1):  {results['vine_pair_copula_families_tree1']}")
    print(f"Student-t  rho:      {results['bivariate_student_t_rho']:.4f}")
    print(f"Student-t  nu:       {results['bivariate_student_t_nu']:.2f}")
