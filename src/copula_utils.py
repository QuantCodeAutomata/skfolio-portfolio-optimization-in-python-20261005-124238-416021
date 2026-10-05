"""Copula utilities for synthetic return generation (Experiment 3).

skfolio 0.6.0 does not include a copula module; this module uses:
  - pyvinecopulib  for multivariate vine copula fitting and sampling
  - scipy.stats    for bivariate Student-t copula density evaluation

Deviation from paper: the paper attributes the copula capability to skfolio,
but skfolio 0.6.0 lacks this module. pyvinecopulib is used as a substitute.
See results/DEVIATIONS.md for the full record.

Paper reference: §2.2.2 "Copula-based Synthetic Priors", Figure 2.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import pyvinecopulib as pv
from scipy import special, stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pseudo-observation transform
# ---------------------------------------------------------------------------

def to_pseudo_obs(X: np.ndarray) -> np.ndarray:
    """Convert observations to pseudo-observations on (0, 1)^d.

    Uses the empirical CDF (rank-based transform) as required by copula models.
    Ranks are scaled to (i / (n + 1)) to avoid boundary issues.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Raw return observations.

    Returns
    -------
    U : np.ndarray, shape (n, d)
        Pseudo-observations in (0, 1)^d.
    """
    n, d = X.shape
    U = np.zeros_like(X, dtype=float)
    for j in range(d):
        ranks = stats.rankdata(X[:, j])
        U[:, j] = ranks / (n + 1)
    return U


# ---------------------------------------------------------------------------
# Vine copula
# ---------------------------------------------------------------------------

def fit_vine_copula(
    X: pd.DataFrame,
    assets: list[str],
    random_state: int = 42,
) -> tuple[pv.Vinecop, np.ndarray]:
    """Fit a multivariate vine copula to historical asset returns.

    Parameters
    ----------
    X : pd.DataFrame
        Asset returns (all assets present as columns).
    assets : list[str]
        Column names (in order) to use for fitting.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    vine : pv.Vinecop
        Fitted vine copula model.
    U_hist : np.ndarray, shape (n, len(assets))
        Pseudo-observations used for fitting.
    """
    # Custom — Context7 found no library equivalent in skfolio (§2.2.2)
    data = X[assets].dropna().values.astype(float)
    U_hist = to_pseudo_obs(data)

    controls = pv.FitControlsVinecop(
        family_set=[pv.BicopFamily.gaussian, pv.BicopFamily.student, pv.BicopFamily.clayton,
                    pv.BicopFamily.gumbel, pv.BicopFamily.frank, pv.BicopFamily.joe],
        selection_criterion="aic",
        num_threads=1,
        seeds=[random_state],
    )
    # pyvinecopulib ≥1.0.0: use factory method; data must be Fortran-contiguous
    vine = pv.Vinecop.from_data(np.asfortranarray(U_hist), controls=controls)
    logger.info("Vine copula fitted on %d obs × %d assets", *U_hist.shape)
    return vine, U_hist


def sample_vine_copula(
    vine: pv.Vinecop,
    n_samples: int,
    random_state: int = 42,
) -> np.ndarray:
    """Draw samples from the fitted vine copula (as pseudo-observations).

    Parameters
    ----------
    vine : pv.Vinecop
        Fitted vine copula.
    n_samples : int
        Number of samples to draw.
    random_state : int
        Random seed.

    Returns
    -------
    U_sim : np.ndarray, shape (n_samples, d)
        Simulated pseudo-observations on (0, 1)^d.
    """
    # pyvinecopulib ≥1.0.0: sample() is the stable method; simulate() is deprecated
    U_sim = vine.sample(n=n_samples, seeds=[random_state])
    return np.array(U_sim)


def pseudo_obs_to_returns(
    U_sim: np.ndarray,
    X_hist: np.ndarray,
) -> np.ndarray:
    """Map simulated pseudo-observations back to return space via quantile transform.

    Uses the empirical quantile function (inverse CDF) of the historical returns.

    Parameters
    ----------
    U_sim : np.ndarray, shape (n_samples, d)
        Simulated pseudo-observations.
    X_hist : np.ndarray, shape (n_hist, d)
        Historical returns used to invert the empirical CDF.

    Returns
    -------
    X_sim : np.ndarray, shape (n_samples, d)
        Simulated returns in the original return space.
    """
    n_hist, d = X_hist.shape
    n_sim = U_sim.shape[0]
    X_sim = np.zeros((n_sim, d))
    for j in range(d):
        sorted_hist = np.sort(X_hist[:, j])
        # Empirical quantile via interpolation
        q_levels = np.linspace(0, 1, n_hist)
        X_sim[:, j] = np.interp(U_sim[:, j], q_levels, sorted_hist)
    return X_sim


# ---------------------------------------------------------------------------
# Bivariate Student-t copula density
# ---------------------------------------------------------------------------

def fit_bivariate_student_t_copula(
    X: pd.DataFrame,
    asset1: str,
    asset2: str,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Fit a bivariate Student-t copula to two asset return series.

    Uses pyvinecopulib with Student-t family only.

    Parameters
    ----------
    X : pd.DataFrame
        Asset returns DataFrame.
    asset1 : str
        First asset column name.
    asset2 : str
        Second asset column name.

    Returns
    -------
    rho : float
        Estimated correlation parameter.
    nu : float
        Estimated degrees of freedom.
    U : np.ndarray, shape (n, 2)
        Pseudo-observations used for fitting.
    params : np.ndarray
        Raw bicop parameters [rho, nu].
    """
    data = X[[asset1, asset2]].dropna().values.astype(float)
    U = to_pseudo_obs(data)

    controls = pv.FitControlsBicop(
        family_set=[pv.BicopFamily.student],
        num_threads=1,
    )
    # pyvinecopulib ≥1.0.0: use factory method; data must be Fortran-contiguous
    bicop = pv.Bicop.from_data(np.asfortranarray(U), controls=controls)
    params = bicop.parameters
    rho = float(np.asarray(params[0]).flat[0])
    nu = float(np.asarray(params[1]).flat[0])
    logger.info(
        "Bivariate Student-t copula (%s, %s): rho=%.4f, nu=%.4f",
        asset1, asset2, rho, nu,
    )
    return rho, nu, U, params


def bivariate_student_t_density_grid(
    rho: float,
    nu: float,
    grid_size: int = 80,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the bivariate Student-t copula density over a 2D grid.

    The copula density is c(u1, u2) = f(t_nu^{-1}(u1), t_nu^{-1}(u2)) /
    (f_nu(t_nu^{-1}(u1)) * f_nu(t_nu^{-1}(u2))), where f is the bivariate
    t pdf and f_nu is the univariate t pdf.

    Parameters
    ----------
    rho : float
        Correlation parameter.
    nu : float
        Degrees of freedom (must be > 0).
    grid_size : int
        Number of grid points per axis.

    Returns
    -------
    uu1, uu2 : np.ndarray, shape (grid_size, grid_size)
        Grid coordinates on (0, 1).
    density : np.ndarray, shape (grid_size, grid_size)
        Copula density values.
    """
    # Custom — Context7 found no library equivalent in skfolio (paper §2.2.2 Figure 2)
    eps = 1e-3
    u_vals = np.linspace(eps, 1 - eps, grid_size)
    uu1, uu2 = np.meshgrid(u_vals, u_vals, indexing="ij")

    t_dist = stats.t(df=nu)
    z1 = t_dist.ppf(uu1)
    z2 = t_dist.ppf(uu2)

    cov = np.array([[1.0, rho], [rho, 1.0]])
    z = np.stack([z1.ravel(), z2.ravel()], axis=1)

    # Bivariate t pdf
    mvt_pdf = _bivariate_t_pdf(z, cov=cov, df=nu)

    # Univariate t pdfs
    f1 = t_dist.pdf(z1.ravel())
    f2 = t_dist.pdf(z2.ravel())

    with np.errstate(divide="ignore", invalid="ignore"):
        cop_density = mvt_pdf / (f1 * f2)

    density = cop_density.reshape(grid_size, grid_size)
    return uu1, uu2, density


def _bivariate_t_pdf(z: np.ndarray, cov: np.ndarray, df: float) -> np.ndarray:
    """Bivariate Student-t pdf for n × 2 array z.

    Parameters
    ----------
    z : np.ndarray, shape (n, 2)
    cov : np.ndarray, shape (2, 2)
    df : float

    Returns
    -------
    pdf_vals : np.ndarray, shape (n,)
    """
    n, d = z.shape
    cov_inv = np.linalg.inv(cov)
    det = np.linalg.det(cov)
    # Mahalanobis distance
    quad = np.einsum("ni,ij,nj->n", z, cov_inv, z)
    # Custom — Context7 found no library equivalent (paper §2.2.2)
    # gammaln lives in scipy.special, not scipy.stats
    coeff = (
        np.exp(special.gammaln((df + d) / 2) - special.gammaln(df / 2))
        / (np.sqrt((np.pi * df) ** d * det))
    )
    pdf_vals = coeff * (1 + quad / df) ** (-(df + d) / 2)
    return pdf_vals
