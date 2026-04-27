"""
Inertia v2 data layer.

Reuses the Ken French and FRED fetchers from the v1 `src/` module
without copying code, by inserting the repo root onto sys.path.
Adds two v2-specific helpers:
  * `get_ff5_monthly()`   --- five Fama-French factor returns + RF
  * `build_factor_panel()` --- FF5 + macro features + targets,
                              ready for regime-detection backtests.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Bring v1 src onto path so we can reuse data fetchers and caching.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data import get_ff5, get_fred_panel  # noqa: E402


FF5_FACTORS = ["MKT_RF", "SMB", "HML", "RMW", "CMA"]


def get_ff5_monthly() -> pd.DataFrame:
    """
    Return the Fama-French 5-factor monthly panel from Ken French.
    Columns: MKT_RF, SMB, HML, RMW, CMA, RF (decimal monthly).
    Index:   month-end Timestamps.
    Sample:  1963-07 to most recent month.
    """
    return get_ff5()


def build_factor_panel(include_macro: bool = True) -> pd.DataFrame:
    """
    Build the unified factor panel used by Methods A, B, C.

    Returns a monthly DataFrame with:
      Targets:  MKT_RF, SMB, HML, RMW, CMA  (next-month versions also added)
      Macro:    vix, term_spread, credit_spread (FRED, post-1990 only)
      Lagged:   factor_lag1_*    (own one-month lag of each factor)
                factor_vol6_*    (six-month rolling vol)
      Risk-free: RF
      Helper:   factor_drawdown_*  (current drawdown of each factor)
    """
    ff = get_ff5_monthly()

    # Lagged values and rolling features for each factor (lag-by-1 to avoid leakage)
    for f in FF5_FACTORS:
        ff[f"lag1_{f}"]   = ff[f].shift(1)
        ff[f"vol6_{f}"]   = ff[f].rolling(6).std() * np.sqrt(12)
        cum = (1 + ff[f]).cumprod()
        ff[f"dd_{f}"]     = cum / cum.cummax() - 1

    # Next-month targets (for supervised methods)
    for f in FF5_FACTORS:
        ff[f"next_{f}"] = ff[f].shift(-1)

    if include_macro:
        try:
            macro = get_fred_panel()
            m = pd.DataFrame(index=macro.index)
            m["vix"]            = macro.get("VIX")
            m["vix_chg"]        = m["vix"].diff()
            m["term_spread"]    = macro.get("TERM")
            m["credit_spread"]  = macro.get("BAA_AAA")
            ff = ff.join(m, how="left")
        except Exception as e:  # pragma: no cover
            print(f"WARNING: FRED join failed ({e}); panel returned without macro.")

    return ff


def factor_static_stats(ff: pd.DataFrame, factors: list[str] | None = None) -> pd.DataFrame:
    """
    Annualized static performance stats for each factor in `factors`,
    over the full panel sample. Intended as the first table in
    Notebook 01.
    """
    if factors is None:
        factors = FF5_FACTORS
    rows = {}
    for f in factors:
        r = ff[f].dropna()
        mean_a = r.mean() * 12
        vol_a  = r.std(ddof=1) * np.sqrt(12)
        cum    = (1 + r).cumprod()
        dd     = cum / cum.cummax() - 1
        rows[f] = {
            "n_months":      len(r),
            "start":         r.index.min().strftime("%Y-%m"),
            "end":           r.index.max().strftime("%Y-%m"),
            "mean_ann":      mean_a,
            "vol_ann":       vol_a,
            "sharpe_ann":    mean_a / vol_a if vol_a > 0 else np.nan,
            "skew":          r.skew(),
            "excess_kurt":   r.kurt(),
            "max_drawdown":  dd.min(),
        }
    return pd.DataFrame(rows).T


if __name__ == "__main__":
    ff = build_factor_panel()
    print(f"Panel shape: {ff.shape}")
    print(f"Range: {ff.index.min().date()} -> {ff.index.max().date()}")
    print()
    print("Static FF5 stats:")
    print(factor_static_stats(ff).round(4))
