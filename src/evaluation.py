"""
Evaluation helpers for Inertia approaches.

- Summary stats (annualized mean, vol, Sharpe, skew, kurtosis, drawdown)
- Sharpe ratio with block-bootstrap 95% CI
- Factor alphas: CAPM, FF3, FF5+UMD, with HAC (Newey-West) SEs
- Subsample tables

All functions work on monthly decimal returns.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm


# ---------------------------------------------------------------------------
# Headline performance stats
# ---------------------------------------------------------------------------
def perf_stats(r: pd.Series) -> dict:
    """Annualized performance stats for a monthly return series (decimal)."""
    r = r.dropna()
    mean_a = r.mean() * 12
    vol_a  = r.std(ddof=1) * np.sqrt(12)
    cum = (1 + r).cumprod()
    dd  = cum / cum.cummax() - 1
    return {
        "n_months":     len(r),
        "start":        r.index.min().strftime("%Y-%m"),
        "end":          r.index.max().strftime("%Y-%m"),
        "mean_ann":     mean_a,
        "vol_ann":      vol_a,
        "sharpe_ann":   mean_a / vol_a if vol_a > 0 else np.nan,
        "skew":         r.skew(),
        "excess_kurt":  r.kurt(),
        "max_drawdown": dd.min(),
    }


def perf_table(returns_dict: dict[str, pd.Series]) -> pd.DataFrame:
    """Stack perf_stats for a {label -> return series} dict."""
    return pd.DataFrame(
        {name: perf_stats(r) for name, r in returns_dict.items()}
    ).T


# ---------------------------------------------------------------------------
# Sharpe with block-bootstrap CI
# ---------------------------------------------------------------------------
def sharpe_bootstrap_ci(
    r: pd.Series,
    n_boot: int = 2000,
    block_size: int = 12,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict:
    """
    Annualized Sharpe ratio with a moving-block bootstrap CI.

    Block size defaults to 12 months (captures autocorrelation without
    destroying the signal). Returns point estimate plus (1-alpha) CI.
    """
    r = r.dropna().values
    n = len(r)
    if n < block_size * 2:
        return {"sharpe": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n_boot": 0}

    def _sr(x):
        m, s = x.mean(), x.std(ddof=1)
        return np.nan if s <= 0 else (m * 12) / (s * np.sqrt(12))

    point = _sr(r)

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_size))
    starts = np.arange(n - block_size + 1)

    boot = np.empty(n_boot)
    for b in range(n_boot):
        chosen = rng.choice(starts, size=n_blocks, replace=True)
        resample = np.concatenate([r[s : s + block_size] for s in chosen])[:n]
        boot[b] = _sr(resample)

    lo, hi = np.nanpercentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"sharpe": point, "ci_low": lo, "ci_high": hi, "n_boot": n_boot}


# ---------------------------------------------------------------------------
# Factor alphas
# ---------------------------------------------------------------------------
def alpha_regression(
    r: pd.Series,
    factors: pd.DataFrame,
    factor_cols: Sequence[str],
    rf_col: str = "RF",
    hac_lags: int = 6,
) -> dict:
    """
    Regress excess returns on factor excess returns.

        (r - rf) = alpha + beta * F + eps

    Returns alpha (monthly + annualized), t-stat, p-value, factor loadings.
    HAC (Newey-West) standard errors.
    """
    factors = factors.reindex(r.index).dropna()
    r = r.loc[factors.index]
    excess = r - factors[rf_col]

    X = sm.add_constant(factors[list(factor_cols)])
    y = excess

    ols = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})

    out = {
        "alpha_monthly":  ols.params["const"],
        "alpha_annual":   ols.params["const"] * 12,
        "alpha_t":        ols.tvalues["const"],
        "alpha_p":        ols.pvalues["const"],
        "r2":             ols.rsquared,
        "n_obs":          int(ols.nobs),
    }
    for f in factor_cols:
        out[f"{f}_beta"] = ols.params[f]
        out[f"{f}_t"]    = ols.tvalues[f]
    return out


def alpha_table(
    returns_dict: dict[str, pd.Series],
    factor_panel: pd.DataFrame,
    spec: str = "FF5_UMD",
    hac_lags: int = 6,
) -> pd.DataFrame:
    """Alpha regression table for multiple strategies against a spec."""
    specs = {
        "CAPM":    ["MKT_RF"],
        "FF3":     ["MKT_RF", "SMB", "HML"],
        "FF5":     ["MKT_RF", "SMB", "HML", "RMW", "CMA"],
        "FF5_UMD": ["MKT_RF", "SMB", "HML", "RMW", "CMA", "UMD"],
    }
    factor_cols = specs[spec]
    rows = {
        name: alpha_regression(r, factor_panel, factor_cols, hac_lags=hac_lags)
        for name, r in returns_dict.items()
    }
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------------------
# Subsample tables
# ---------------------------------------------------------------------------
def subsample_table(
    returns_dict: dict[str, pd.Series],
    splits: dict[str, tuple[str | None, str | None]],
) -> pd.DataFrame:
    """
    Subsample Sharpe table.

    splits = {"2000-2010": ("2000-01", "2010-12"), ...}
    """
    rows = {}
    for name, r in returns_dict.items():
        stats = {}
        for label, (lo, hi) in splits.items():
            r_sub = r.loc[lo:hi].dropna()
            stats[label] = (
                np.nan if len(r_sub) < 12
                else (r_sub.mean() * 12) / (r_sub.std(ddof=1) * np.sqrt(12))
            )
        rows[name] = stats
    return pd.DataFrame(rows).T
