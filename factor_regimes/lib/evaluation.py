"""
Inertia v2 evaluation helpers, ported from v1 with minor cleanups.

  perf_stats             - annualized headline statistics
  sharpe_bootstrap_ci    - block-bootstrap 95% CI on Sharpe
  sharpe_diff_ci         - paired block-bootstrap CI on Sharpe difference
  alpha_regression       - factor-model regression with HAC standard errors
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm


def perf_stats(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) == 0:
        return {}
    mean_a = r.mean() * 12
    vol_a  = r.std(ddof=1) * np.sqrt(12)
    cum    = (1 + r).cumprod()
    dd     = cum / cum.cummax() - 1
    return {
        "n_months":     len(r),
        "mean_ann":     mean_a,
        "vol_ann":      vol_a,
        "sharpe_ann":   mean_a / vol_a if vol_a > 0 else np.nan,
        "skew":         r.skew(),
        "excess_kurt":  r.kurt(),
        "max_drawdown": dd.min(),
    }


def perf_table(returns_dict: dict[str, pd.Series]) -> pd.DataFrame:
    return pd.DataFrame({k: perf_stats(v) for k, v in returns_dict.items()}).T


def sharpe_bootstrap_ci(r: pd.Series, n_boot: int = 2000,
                       block_size: int = 12, seed: int = 42,
                       alpha: float = 0.05) -> dict:
    r = r.dropna().values
    n = len(r)
    if n < block_size * 2:
        return {"sharpe": np.nan, "ci_low": np.nan, "ci_high": np.nan}

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
        resample = np.concatenate([r[s:s + block_size] for s in chosen])[:n]
        boot[b] = _sr(resample)

    lo, hi = np.nanpercentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"sharpe": point, "ci_low": lo, "ci_high": hi}


def sharpe_diff_ci(r_strat: pd.Series, r_bench: pd.Series,
                  n_boot: int = 2000, block_size: int = 12,
                  seed: int = 42, alpha: float = 0.05) -> dict:
    """Paired bootstrap on Sharpe difference (strategy - benchmark)."""
    j = pd.concat([r_strat, r_bench], axis=1).dropna()
    j.columns = ["s", "b"]
    arr = j.values
    n = len(arr)
    if n < block_size * 2:
        return {"diff": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan}

    def _sr(x):
        m, s = x.mean(), x.std(ddof=1)
        return np.nan if s <= 0 else (m * 12) / (s * np.sqrt(12))

    point = _sr(arr[:, 0]) - _sr(arr[:, 1])

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_size))
    starts = np.arange(n - block_size + 1)

    diffs = np.empty(n_boot)
    for b in range(n_boot):
        chosen = rng.choice(starts, size=n_blocks, replace=True)
        rows = np.concatenate([arr[s:s + block_size] for s in chosen])[:n]
        diffs[b] = _sr(rows[:, 0]) - _sr(rows[:, 1])

    lo, hi = np.nanpercentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p_one = float((diffs <= 0).mean())
    return {"diff": point, "ci_low": lo, "ci_high": hi,
            "p_value": min(2 * p_one, 2 * (1 - p_one))}


def alpha_regression(r: pd.Series, factors: pd.DataFrame,
                    factor_cols: Sequence[str],
                    rf_col: str = "RF", hac_lags: int = 6) -> dict:
    factors = factors.reindex(r.index).dropna()
    r = r.loc[factors.index]
    excess = r - factors[rf_col]

    X = sm.add_constant(factors[list(factor_cols)])
    ols = sm.OLS(excess, X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})

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
