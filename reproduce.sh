#!/usr/bin/env bash
# reproduce.sh — end-to-end reproduction script
# Usage: bash reproduce.sh
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "================================================================"
echo "  skfolio: Portfolio Optimization in Python — Replication"
echo "================================================================"
echo ""

# ---------------------------------------------------------------------------
# 1. Check / install dependencies
# ---------------------------------------------------------------------------
echo "[1/6] Checking Python environment …"
python -c "import skfolio; print(f'  skfolio  {skfolio.__version__}')"
python -c "import pyvinecopulib as pv; print(f'  pyvinecopulib  {pv.__version__}')"
python -c "import copulae; print(f'  copulae  {copulae.__version__}')"
python -c "import sklearn; print(f'  scikit-learn  {sklearn.__version__}')"
echo ""

# ---------------------------------------------------------------------------
# 2. Run test suite
# ---------------------------------------------------------------------------
echo "[2/6] Running pytest …"
python -m pytest tests/ -v --tb=short 2>&1
echo ""

# ---------------------------------------------------------------------------
# 3. Experiment 1 — Efficient Frontier
# ---------------------------------------------------------------------------
echo "[3/6] Experiment 1: Regularized min-variance efficient frontier …"
python exp/exp_1_frontier.py
echo ""

# ---------------------------------------------------------------------------
# 4. Experiment 2 — Factor Model Prior
# ---------------------------------------------------------------------------
echo "[4/6] Experiment 2: Factor-model prior with ridge regression …"
python exp/exp_2_factor_model.py
echo ""

# ---------------------------------------------------------------------------
# 5. Experiment 3 — Copula Synthetic Returns
# ---------------------------------------------------------------------------
echo "[5/6] Experiment 3: Copula-based synthetic return generation …"
python exp/exp_3_copula.py
echo ""

# ---------------------------------------------------------------------------
# 6. Experiment 4 — Walk-Forward Stacking
# ---------------------------------------------------------------------------
echo "[6/6] Experiment 4: Walk-forward stacking vs equal-weight …"
python exp/exp_4_stacking.py
echo ""

echo "================================================================"
echo "  All experiments completed. Results saved in results/"
echo "================================================================"
ls -lh results/
