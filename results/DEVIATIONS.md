# Methodology Deviations

This file records every deviation from the paper's prescribed methodology.

---

## Experiment 3 — Copula Module

**Paper claim:** "skfolio provides copula-based synthetic prior estimation."

**Deviation:** skfolio 0.6.0 (installed version) does not expose a public copula
module (no `skfolio.copula` namespace). The paper's Figure 2 copula functionality
is implemented using `pyvinecopulib` (v1.0.0) for vine copula fitting and simulation,
and `scipy.stats` for the bivariate Student-t copula density evaluation.

**Impact:** The vine structure selection, pair-copula family set, fitting algorithm,
and marginal transform may differ from skfolio's internal implementation. Results are
qualitatively comparable but are not an exact reproduction of the paper's Figure 2.

**Evidence:** Paper §2.2.2 attributes the capability to skfolio; no class names or
method signatures are provided. `dir(skfolio)` in v0.6.0 reveals no copula module.

---

## Experiment 1 — Annualization Convention

**Uncertainty:** The paper does not specify the annualization factor (e.g., 252, 260).
The value used by `skfolio.portfolio.Portfolio.annualized_mean` and
`annualized_std` in v0.6.0 was accepted as-is. Inspected via
`help(Portfolio)` — defaults to 252 trading days per year.

**Impact:** Annualized metrics may differ if a different convention was used in the
original paper experiment. No paper-specified numeric target to compare against.

---

## Experiment 4 — Dataset for Walk-Forward

**Uncertainty:** Paper §3 does not explicitly restate the data source and 80/20 split.
We interpret it as reusing the same S&P 500 80/20 split from §2.1 (X_train, X_test),
with `cross_val_predict` applied to X_test. Recorded as an interpretation.

---

## All Experiments — Solver and Tolerance

The paper mentions Clarabel as the default interior-point solver but gives no explicit
solver tolerance. Default skfolio/cvxpy solver tolerances were used.

---

## Paper Citation Information

The paper refers to itself as "skfolio: Portfolio Optimization in Python" but does not
state authors, journal, year, or DOI in the excerpts retrieved from the index.
All methodology evidence was retrieved from the indexed paper document.
