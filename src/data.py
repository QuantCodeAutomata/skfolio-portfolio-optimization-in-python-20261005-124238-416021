"""Shared data-loading and preprocessing pipeline.

All experiments must import from this module rather than implement their own loaders.
Provides:
  - load_sp500_returns()  → (X, prices)
  - load_factor_returns() → (X, y, prices, factor_prices)
  - chronological_split() → (X_train, X_test)  /  (X_train, X_test, y_train, y_test)

Uses skfolio's bundled datasets (no external API required).
Source: paper §2.1 "Overview and Basic Usage".
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Using skfolio datasets and preprocessing — Context7 confirmed
from skfolio.datasets import load_factors_dataset, load_sp500_dataset
from skfolio.preprocessing import prices_to_returns

logger = logging.getLogger(__name__)


def load_sp500_returns() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the bundled S&P 500 price dataset and convert to returns.

    Returns
    -------
    X : pd.DataFrame
        Asset log/simple returns (row = date, column = ticker).
    prices : pd.DataFrame
        Raw price data as loaded from skfolio.
    """
    prices = load_sp500_dataset()
    X = prices_to_returns(prices)

    logger.info(
        "SP500 dataset loaded: %d observations, %d assets, "
        "date range %s to %s",
        len(X),
        X.shape[1],
        X.index[0],
        X.index[-1],
    )
    return X, prices


def load_factor_returns(
    prices: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load asset and factor returns aligned in time.

    Parameters
    ----------
    prices : pd.DataFrame, optional
        Pre-loaded S&P 500 price data.  If None, calls load_sp500_dataset().

    Returns
    -------
    X : pd.DataFrame
        Asset returns.
    y : pd.DataFrame
        Factor returns, time-aligned with X.
    prices : pd.DataFrame
        Raw S&P 500 prices.
    factor_prices : pd.DataFrame
        Raw factor prices.
    """
    if prices is None:
        prices = load_sp500_dataset()
    factor_prices = load_factors_dataset()

    # prices_to_returns with factor_prices produces aligned (X, y)
    X, y = prices_to_returns(prices, factor_prices)

    logger.info(
        "Factor dataset loaded: %d asset observations, %d assets, "
        "%d factor observations, %d factors",
        len(X),
        X.shape[1],
        len(y),
        y.shape[1],
    )
    return X, y, prices, factor_prices


def chronological_split(
    X: pd.DataFrame,
    y: Optional[pd.DataFrame] = None,
    test_size: float = 0.2,
) -> tuple:
    """Chronological train/test split with no shuffling.

    Parameters
    ----------
    X : pd.DataFrame
        Returns matrix (observations × assets).
    y : pd.DataFrame, optional
        Factor returns matrix aligned with X.
    test_size : float
        Fraction of observations held out as test.

    Returns
    -------
    If y is None:
        (X_train, X_test)
    If y is provided:
        (X_train, X_test, y_train, y_test)
    """
    if y is None:
        X_train, X_test = train_test_split(X, test_size=test_size, shuffle=False)
        logger.info(
            "Train/test split: train=%d, test=%d (test_size=%.0f%%)",
            len(X_train),
            len(X_test),
            test_size * 100,
        )
        return X_train, X_test
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )
        logger.info(
            "Train/test split (with factors): train=%d, test=%d (test_size=%.0f%%)",
            len(X_train),
            len(X_test),
            test_size * 100,
        )
        return X_train, X_test, y_train, y_test


def get_dataset_metadata(
    X: pd.DataFrame,
    prices: Optional[pd.DataFrame] = None,
) -> dict:
    """Return a metadata dict describing the dataset for logging and results.

    Parameters
    ----------
    X : pd.DataFrame
        Returns DataFrame.
    prices : pd.DataFrame, optional
        Raw price DataFrame for additional metadata.

    Returns
    -------
    dict with keys: n_obs, n_assets, date_start, date_end, frequency.
    """
    meta: dict = {
        "n_obs": int(len(X)),
        "n_assets": int(X.shape[1]),
        "date_start": str(X.index[0].date()) if hasattr(X.index[0], "date") else str(X.index[0]),
        "date_end": str(X.index[-1].date()) if hasattr(X.index[-1], "date") else str(X.index[-1]),
        "frequency": "inferred_daily",
        "assets": list(X.columns),
    }
    if prices is not None:
        meta["price_date_start"] = (
            str(prices.index[0].date())
            if hasattr(prices.index[0], "date")
            else str(prices.index[0])
        )
        meta["price_date_end"] = (
            str(prices.index[-1].date())
            if hasattr(prices.index[-1], "date")
            else str(prices.index[-1])
        )
    return meta
