"""Tests for Experiment 3 — copula-based synthetic return generation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.copula_utils import (
    _bivariate_t_pdf,
    bivariate_student_t_density_grid,
    fit_bivariate_student_t_copula,
    fit_vine_copula,
    pseudo_obs_to_returns,
    sample_vine_copula,
    to_pseudo_obs,
)
from src.data import load_sp500_returns


@pytest.fixture(scope="module")
def sp500_sub():
    X, _ = load_sp500_returns()
    assets = ["AAPL", "JPM", "LLY"]
    return X[assets].dropna()


# ---------------------------------------------------------------------------
# to_pseudo_obs
# ---------------------------------------------------------------------------

def test_pseudo_obs_range():
    """Pseudo-observations must lie strictly in (0, 1)."""
    rng = np.random.default_rng(0)
    data = rng.normal(size=(200, 3))
    U = to_pseudo_obs(data)
    assert np.all(U > 0) and np.all(U < 1), "Pseudo-obs outside (0, 1)"


def test_pseudo_obs_shape():
    rng = np.random.default_rng(1)
    data = rng.normal(size=(150, 4))
    U = to_pseudo_obs(data)
    assert U.shape == data.shape


# ---------------------------------------------------------------------------
# Vine copula fit + sample
# ---------------------------------------------------------------------------

def test_vine_copula_fit(sp500_sub):
    """Vine copula must fit without error and return correct types."""
    vine, U_hist = fit_vine_copula(sp500_sub, list(sp500_sub.columns), random_state=0)
    assert vine is not None
    assert U_hist.shape[1] == 3


def test_vine_copula_sample(sp500_sub):
    """Sampled pseudo-observations must lie in (0, 1)^d."""
    vine, _ = fit_vine_copula(sp500_sub, list(sp500_sub.columns), random_state=0)
    U_sim = sample_vine_copula(vine, n_samples=200, random_state=0)
    assert U_sim.shape == (200, 3)
    assert np.all(U_sim > 0) and np.all(U_sim < 1)


def test_pseudo_obs_to_returns_shape(sp500_sub):
    """Back-transformed returns must have same sample count as U_sim."""
    vine, _ = fit_vine_copula(sp500_sub, list(sp500_sub.columns), random_state=0)
    U_sim = sample_vine_copula(vine, n_samples=100, random_state=0)
    X_sim = pseudo_obs_to_returns(U_sim, sp500_sub.values.astype(float))
    assert X_sim.shape == (100, 3)


def test_pseudo_obs_to_returns_range(sp500_sub):
    """Back-transformed returns must lie within historical min/max."""
    vine, _ = fit_vine_copula(sp500_sub, list(sp500_sub.columns), random_state=0)
    U_sim = sample_vine_copula(vine, n_samples=500, random_state=42)
    X_hist = sp500_sub.values.astype(float)
    X_sim = pseudo_obs_to_returns(U_sim, X_hist)
    for j in range(X_hist.shape[1]):
        assert X_sim[:, j].min() >= X_hist[:, j].min() - 1e-10
        assert X_sim[:, j].max() <= X_hist[:, j].max() + 1e-10


# ---------------------------------------------------------------------------
# Bivariate Student-t copula
# ---------------------------------------------------------------------------

def test_bivariate_student_t_copula_fit(sp500_sub):
    """Student-t copula fit must return rho in (-1, 1) and nu > 0."""
    rho, nu, U, params = fit_bivariate_student_t_copula(sp500_sub, "AAPL", "JPM")
    assert -1 < rho < 1, f"rho={rho} not in (-1, 1)"
    assert nu > 0, f"nu={nu} not positive"
    assert U.shape[1] == 2


def test_bivariate_density_grid_shape():
    """Density grid must have the requested grid_size × grid_size shape."""
    uu1, uu2, density = bivariate_student_t_density_grid(rho=0.3, nu=5.0, grid_size=20)
    assert uu1.shape == (20, 20)
    assert uu2.shape == (20, 20)
    assert density.shape == (20, 20)


def test_bivariate_density_non_negative():
    """Copula density must be non-negative everywhere on the grid."""
    _, _, density = bivariate_student_t_density_grid(rho=0.3, nu=5.0, grid_size=30)
    assert np.all(density >= 0), "Negative density values found"


def test_bivariate_density_zero_correlation():
    """For rho=0 the density grid must be roughly uniform (close to 1.0)."""
    _, _, density = bivariate_student_t_density_grid(rho=0.0, nu=30.0, grid_size=20)
    interior = density[5:15, 5:15]
    mean_density = float(np.mean(interior))
    assert 0.5 < mean_density < 2.0, f"Unexpected mean density={mean_density:.4f}"


def test_bivariate_t_pdf_positive():
    """_bivariate_t_pdf must return positive values for non-degenerate inputs."""
    rng = np.random.default_rng(0)
    z = rng.standard_normal((50, 2))
    cov = np.array([[1.0, 0.3], [0.3, 1.0]])
    pdf_vals = _bivariate_t_pdf(z, cov=cov, df=5.0)
    assert np.all(pdf_vals > 0)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_vine_copula_small_sample():
    """Vine fit should succeed on a small synthetic sample (n=50, d=3)."""
    rng = np.random.default_rng(99)
    X_small = rng.standard_normal((50, 3))
    X_df = __import__("pandas").DataFrame(X_small, columns=["A", "B", "C"])
    vine, U = fit_vine_copula(X_df, ["A", "B", "C"], random_state=0)
    assert U.shape == (50, 3)
