# skfolio: Portfolio Optimization in Python — Experiment Replication

Replication of selected experiments from the paper **"skfolio: Portfolio Optimization in Python"**.
All methodology details are taken from the indexed paper document (retrieved via LlamaCloud).

---

## Experiments

| ID | Title | Section |
|----|-------|---------|
| exp_1 | Regularized min-variance efficient frontier on S&P 500 assets | §2.1 |
| exp_2 | Minimum-variance optimization with ridge-regression factor-model prior | §2.2 |
| exp_3 | Copula-based synthetic return generation and dependence visualization | §2.2.2 |
| exp_4 | Walk-forward evaluation of stacked portfolio optimization vs equal weighting | §2.3, §3 |

---

## Repository Structure

```
.
├── data/               # Data directory (no raw downloaded files committed)
├── exp/                # Experiment scripts (call src/ API)
│   ├── exp_1_frontier.py
│   ├── exp_2_factor_model.py
│   ├── exp_3_copula.py
│   └── exp_4_stacking.py
├── results/            # Saved plots, metrics, and CSVs
│   ├── RESULTS.md      # Summary of all experiment findings
│   └── DEVIATIONS.md   # Methodology deviations from paper
├── src/                # Core shared implementation
│   ├── __init__.py
│   ├── data.py         # Shared data loader and preprocessing
│   └── copula_utils.py # Copula utilities (pyvinecopulib + scipy)
├── tests/              # pytest test suite
│   ├── test_data.py
│   ├── test_exp1.py
│   ├── test_exp2.py
│   ├── test_exp3.py
│   └── test_exp4.py
├── reproduce.sh        # End-to-end reproduction script
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run all experiments and tests
bash reproduce.sh
```

Or run individual components:

```bash
# Tests only
python -m pytest tests/ -v

# Individual experiments
python exp/exp_1_frontier.py
python exp/exp_2_factor_model.py
python exp/exp_3_copula.py
python exp/exp_4_stacking.py
```

---

## Dependencies

- **skfolio** 0.6.0 — portfolio optimization (scikit-learn compatible)
- **pyvinecopulib** — vine copula fitting and simulation (exp_3 deviation; see `results/DEVIATIONS.md`)
- **scikit-learn** — Ridge regression, train_test_split
- **scipy**, **numpy**, **pandas**, **matplotlib**

---

## Data

All experiments use the **bundled S&P 500 and factor datasets** shipped with skfolio
(`load_sp500_dataset()`, `load_factors_dataset()`). No external market data API is required.

---

## Deviations

See `results/DEVIATIONS.md` for all known deviations from the paper methodology.

The principal deviation is in **exp_3**: skfolio 0.6.0 does not expose a public copula
module. `pyvinecopulib` is used instead for vine copula fitting and simulation.

---

## Results

See `results/RESULTS.md` for a summary of all experiment outputs and
`results/exp{N}_results.json` for machine-readable metrics.
