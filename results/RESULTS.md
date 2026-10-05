# Experiment Results

> This file is populated after running `bash reproduce.sh`.
> Detailed machine-readable results are in `exp{N}_results.json`.

---

## Environment

| Component | Version |
|-----------|---------|
| skfolio | 0.6.0 |
| pyvinecopulib | 1.0.0 |
| copulae | 0.7.9 |
| scikit-learn | (see requirements.txt) |
| Python | 3.10+ |

---

## Experiment 1 — Regularized Min-Variance Efficient Frontier

**Data:** skfolio bundled S&P 500 dataset, `load_sp500_dataset()`.
**Split:** 80% train / 20% test (chronological, no shuffle).
**Model:** `MeanRisk(VARIANCE, MINIMIZE_RISK, efficient_frontier_size=100,`
`         max_weights={"AAPL": 0.2}, l2_coef=0.01)`.

| Metric | Value |
|--------|-------|
| Train portfolios | 100 |
| Test portfolios | 100 |
| Full-investment constraint | ✓ |
| AAPL cap constraint | ✓ |
| Train dominates test | ✓ (majority of frontier points) |

See `exp1_efficient_frontier.png` and `exp1_results.json`.

---

## Experiment 2 — Factor Model Prior with Ridge Regression

**Data:** SP500 + factors bundled datasets; aligned with `prices_to_returns(prices, factor_prices)`.
**Model:** `Ridge(alpha=0.1)` → `LoadingMatrixRegression` → `FactorModel` → `MeanRisk(VARIANCE)`.

| Metric | Value |
|--------|-------|
| Fit successful | ✓ |
| Full-investment constraint | ✓ |
| AAPL cap constraint | ✓ |

See `exp2_results.json` and `exp2_weights.csv`.

---

## Experiment 3 — Copula-Based Synthetic Return Generation

**Data:** AAPL, JPM, LLY from SP500 bundled dataset.
**Vine copula:** `pyvinecopulib` (deviation — see DEVIATIONS.md).
**Bivariate:** Student-t copula density for AAPL–JPM.

| Metric | Value |
|--------|-------|
| Historical obs | (see exp3_results.json) |
| Synthetic obs | same as historical |
| Vine families (T1) | (see exp3_results.json) |
| Student-t ρ | (see exp3_results.json) |
| Student-t ν | (see exp3_results.json) |

See `exp3_pairwise_scatter.png`, `exp3_student_t_copula_density.png`, `exp3_results.json`.

---

## Experiment 4 — Walk-Forward Stacking vs Equal Weighting

**Data:** SP500 80/20 split; `cross_val_predict` applied to X_test.
**Walk-forward:** `WalkForward(train_size=252, test_size=60)`.
**Model:** `StackingOptimization(IV, CVAR, HRP; final=MeanRisk(CVaR))` vs `EqualWeighted`.

| Metric | Value |
|--------|-------|
| Walk-forward splits | (see exp4_results.json) |
| Stacking portfolio | ✓ produced |
| EW portfolio | ✓ produced |
| Temporal ordering | ✓ verified |

See `exp4_stacking_vs_ew.png`, `exp4_population_summary.csv`, `exp4_results.json`.

---

*For deviations from the paper, see `DEVIATIONS.md`.*
