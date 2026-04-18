"""
Unified feature panel for all three Inertia approaches.

Design principle: one `build_features()` returns the same panel every
approach consumes. No per-approach feature cherry-picking -> no p-hacking
across approaches.

Feature groups
--------------
DM (Daniel-Moskowitz) originals --- back to 1928:
  * bear           : 1 if cumulative 24-month market excess return <= 0
  * mom_var6       : 6-month UMD realized variance (annualized)
  * bear_x_var     : interaction

Market context --- back to 1928:
  * mkt_vol6       : 6-month market realized vol (annualized)
  * mkt_dd         : current market drawdown from trailing peak
  * umd_dd         : current UMD drawdown from trailing peak
  * mkt_ret_1m     : lagged 1-month market return
  * mkt_ret_12m    : trailing 12-month market return

Macro (FRED) --- limited by earliest FRED date, typically 1990+:
  * vix            : VIX level
  * vix_chg        : 1-month VIX change
  * term_spread    : 10y - 3m
  * credit_spread  : BAA - 10y

Target
------
  * UMD_next       : UMD return in month t+1 (for supervised models)
  * UMD            : current-month UMD (for labelling crash months)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import get_ff_momentum, get_ff3, get_fred_panel


DM_FEATURES      = ["bear", "mom_var6", "bear_x_var"]
MARKET_FEATURES  = ["mkt_vol6", "mkt_dd", "umd_dd", "mkt_ret_1m", "mkt_ret_12m"]
MACRO_FEATURES   = ["vix", "vix_chg", "term_spread", "credit_spread"]

ALL_FEATURES = DM_FEATURES + MARKET_FEATURES + MACRO_FEATURES


def build_features(include_macro: bool = True) -> pd.DataFrame:
    """
    Build the full monthly feature panel.

    Parameters
    ----------
    include_macro : bool
        If True, merge in FRED features (restricts sample to ~1990+).
        If False, return UMD+market features only (back to 1928).

    Returns
    -------
    DataFrame indexed by month-end with all feature columns plus:
        UMD        : current-month UMD return
        UMD_next   : next-month UMD return (target)
        MKT_RF     : market excess return
        RF         : risk-free rate
    """
    umd = get_ff_momentum()["UMD"]
    ff3 = get_ff3()
    df = pd.concat(
        [umd, ff3["MKT_RF"], ff3["RF"]],
        axis=1, sort=True,
    ).dropna()
    df["MKT"] = df["MKT_RF"] + df["RF"]

    # --- DM originals ---
    df["mkt_24m_cum"] = (
        (1 + df["MKT_RF"]).rolling(24).apply(lambda x: x.prod() - 1, raw=True)
    )
    df["bear"] = (df["mkt_24m_cum"] <= 0).astype(int)
    df["mom_var6"] = df["UMD"].rolling(6).var() * 12
    df["bear_x_var"] = df["bear"] * df["mom_var6"]

    # --- market context ---
    df["mkt_vol6"] = df["MKT_RF"].rolling(6).std(ddof=1) * np.sqrt(12)
    df["mkt_ret_1m"] = df["MKT_RF"]
    df["mkt_ret_12m"] = (
        (1 + df["MKT_RF"]).rolling(12).apply(lambda x: x.prod() - 1, raw=True)
    )
    mkt_cum = (1 + df["MKT"]).cumprod()
    df["mkt_dd"] = mkt_cum / mkt_cum.cummax() - 1
    umd_cum = (1 + df["UMD"]).cumprod()
    df["umd_dd"] = umd_cum / umd_cum.cummax() - 1

    # --- macro (optional) ---
    if include_macro:
        try:
            fred = get_fred_panel()
            macro = pd.DataFrame(index=fred.index)
            macro["vix"] = fred.get("VIX")
            macro["vix_chg"] = macro["vix"].diff()
            macro["term_spread"] = fred.get("TERM")
            macro["credit_spread"] = fred.get("BAA_AAA")
            # Align index; month-end
            df = df.join(macro, how="left")
        except Exception as e:  # pragma: no cover
            print(f"WARNING: FRED features unavailable ({e}); returning without macro.")

    # --- target ---
    df["UMD_next"] = df["UMD"].shift(-1)

    return df.dropna(subset=DM_FEATURES + MARKET_FEATURES).copy()


def feature_sets(include_macro: bool = True) -> dict:
    """Return the ordered feature-name lists each approach uses."""
    dm_only = DM_FEATURES
    expanded = DM_FEATURES + MARKET_FEATURES + (MACRO_FEATURES if include_macro else [])
    hmm_obs = ["MKT_RF", "mkt_vol6", "mom_var6"]  # raw observations for unsupervised HMM
    return {
        "dm_only":  dm_only,
        "expanded": expanded,
        "hmm":      hmm_obs,
    }


if __name__ == "__main__":
    df = build_features(include_macro=True)
    print(f"Shape: {df.shape}")
    print(f"Range: {df.index.min().date()} -> {df.index.max().date()}")
    print()
    print("Missingness per feature (full panel):")
    print(df[ALL_FEATURES + ["UMD_next"]].isna().mean().sort_values())
