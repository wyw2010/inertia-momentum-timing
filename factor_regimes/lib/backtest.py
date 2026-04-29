"""
Inertia v2 backtest helpers: convert favorable probabilities into
factor-timing weights, and evaluate the resulting timed strategies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def prob_to_weight(probs: pd.Series, mode: str = "linear",
                   cap: tuple[float, float] = (-1.0, 1.0)) -> pd.Series:
    """
    Convert a P(favorable) series to a portfolio weight on the underlying
    factor. Mode controls the mapping.

    'linear'   : w = 2*p - 1                  (long when p>0.5, short when p<0.5)
    'longflat' : w = 1 if p>0.5 else 0        (long-only crash shield, binary)
    'longonly' : w = max(2*p - 1, 0)          (long-only with confidence,
                                               clipped to [0, 1])
    'soft'     : w = 4*(p - 0.5), clipped     (more aggressive linear scaling)
    """
    if mode == "linear":
        w = 2 * probs - 1.0
    elif mode == "longflat":
        w = (probs > 0.5).astype(float)
    elif mode == "longonly":
        w = (2 * probs - 1.0).clip(lower=0.0, upper=1.0)
    elif mode == "soft":
        w = ((probs - 0.5) * 4.0).clip(-1, 1)
    else:
        raise ValueError(f"unknown mode: {mode}")
    return w.clip(lower=cap[0], upper=cap[1])


def apply_weights(weights: pd.Series, factor_returns: pd.Series,
                  cost_bps_oneway: float = 5.0) -> pd.DataFrame:
    """
    Apply a weight series to NEXT-month factor returns and deduct a
    simple linear transaction cost based on weight changes.

    The cost assumption is intentionally lower than v1 (5 bp vs 20 bp)
    because trading FF5 factor portfolios does not require the same
    short-leg infrastructure as UMD.
    """
    idx = weights.index
    r_next = factor_returns.shift(-1).reindex(idx)
    r_gross = weights * r_next
    turnover = weights.diff().abs().fillna(weights.abs())
    cost = cost_bps_oneway * turnover / 1e4
    r_net = r_gross - cost
    return pd.DataFrame({
        "weight": weights, "r_next": r_next, "r_gross": r_gross,
        "turnover": turnover, "cost": cost, "r_net": r_net,
    }).dropna()


def static_factor_returns(factor_returns: pd.Series,
                          dates: pd.DatetimeIndex) -> pd.Series:
    """Buy-and-hold factor returns over the same dates as the timed series."""
    next_r = factor_returns.shift(-1).reindex(dates).dropna()
    return next_r
