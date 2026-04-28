"""
build_inertia_v3.py
-------------------
Replace the Inertia strategy with a vol-targeted (50% Method B + 50% Method C)
blend, levered to match Static FF5 vol (5.28% annualized) using a ROLLING
36-month vol target. Pays a realistic financing cost (RF + 30 bp/year) on the
levered portion.

Updates:
  - factor_regimes/tables/33_comprehensive_scoreboard.csv/.md (Inertia row)
  - factor_regimes/tables/34_comprehensive_sharpe_ci.csv/.md (Inertia row)
  - factor_regimes/tables/35_comprehensive_paired_diff.csv/.md (Inertia row)
  - factor_regimes/tables/38_comprehensive_returns.csv (Inertia column)
  - factor_regimes/tables/40_appraisal_ratios.csv/.md (Inertia row)
  - factor_regimes/tables/EXPLORE_inertia_v3_returns.csv (date, return, leverage_t)

Mechanics:
  r_blend_t = 0.5 * methodB_t + 0.5 * methodC_t
  rolling_vol_t = sqrt(12) * std(r_blend over trailing 36 months)
  L_t = 0.0528 / rolling_vol_t, capped to [0.5, 3.0]
  Warm-start: use full-sample vol of r_blend for first 36 months
  inertia_t = L_t * r_blend_t - max(L_t - 1, 0) * (RF_t + 30bp/12)
              + max(1 - L_t, 0) * RF_t      # cash earns RF when underlevered
  (Equivalent: r = L*r_blend - (L-1)*(RF + 0.0030/12) when L > 1
                 r = L*r_blend - (L-1)*RF       when L < 1)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from factor_regimes.lib.data import get_ff5_monthly  # noqa: E402
from factor_regimes.lib.evaluation import (  # noqa: E402
    perf_stats, sharpe_bootstrap_ci, sharpe_diff_ci,
)

TABLES = ROOT / "factor_regimes" / "tables"

START = "2000-02-29"
END = "2024-12-31"
TARGET_VOL = 0.0528  # Static FF5 annualized realized vol
WIN = 36             # months for rolling vol target
L_MIN, L_MAX = 0.5, 3.0
BORROW_SPREAD_ANN = 0.0030  # 30 bp/year over RF
N_BOOT = 5000
BLOCK = 12


def build_new_inertia() -> pd.DataFrame:
    comp = pd.read_csv(TABLES / "38_comprehensive_returns.csv",
                       index_col=0, parse_dates=True)
    methodB = comp["Method B"]
    methodC = comp["Method C"]
    r_blend = 0.5 * methodB + 0.5 * methodC

    # Get RF aligned to comp index
    ff = get_ff5_monthly()
    rf = ff["RF"].reindex(comp.index)
    # Some FF5 RFs may be NaN at edges; fill any missing with 0
    rf = rf.fillna(0.0)

    # Rolling 36-month annualized vol of r_blend, computed on TRAILING data only
    # rolling().std uses trailing window ending at the current obs. We want strictly
    # ex-ante so use shift(1)? The spec says "trailing 36 months" - we use the
    # window ending at month t (inclusive) which is standard ex-ante for a vol
    # target applied to month t+1 returns. To be conservative against look-ahead
    # we use trailing window ending at t-1.
    roll_std = r_blend.rolling(WIN).std(ddof=1).shift(1)  # ex-ante
    roll_vol_ann = roll_std * np.sqrt(12)

    # Warm-start: full-sample vol for first WIN+1 months (or wherever NaN)
    full_vol = r_blend.std(ddof=1) * np.sqrt(12)
    roll_vol_ann = roll_vol_ann.fillna(full_vol)

    L = (TARGET_VOL / roll_vol_ann).clip(lower=L_MIN, upper=L_MAX)

    borrow_monthly_spread = BORROW_SPREAD_ANN / 12.0

    # Financing cost / cash earnings on the (L-1) deviation from 1x
    # When L > 1: borrow (L-1) units at RF + spread, pay (L-1)*(RF + spread)
    # When L < 1: hold (1-L) units in cash, earn (1-L)*RF, equivalently pay
    #             (L-1)*RF (which is negative, so we add to return)
    # The compact formula: r = L*r_blend - (L-1)*RF - max(L-1, 0)*spread
    inertia = (
        L * r_blend
        - (L - 1.0) * rf
        - (L - 1.0).clip(lower=0.0) * borrow_monthly_spread
    )

    out = pd.DataFrame({
        "date": comp.index,
        "return": inertia.values,
        "leverage_t": L.values,
        "rolling_vol_ann": roll_vol_ann.values,
        "rf_monthly": rf.values,
        "r_blend": r_blend.values,
    }).set_index("date")
    return out


def update_returns_csv(new_inertia: pd.Series) -> None:
    path = TABLES / "38_comprehensive_returns.csv"
    comp = pd.read_csv(path, index_col=0, parse_dates=True)
    comp["Inertia"] = new_inertia.reindex(comp.index)
    comp.to_csv(path)
    print(f"  updated: {path}")


def update_scoreboard(new_r: pd.Series, ff5: pd.DataFrame) -> None:
    path_csv = TABLES / "33_comprehensive_scoreboard.csv"
    path_md = TABLES / "33_comprehensive_scoreboard.md"
    sb = pd.read_csv(path_csv, index_col=0)
    s = perf_stats(new_r)

    # Appraisal ratio: regress excess returns on FF5 excess returns
    ar_data = compute_appraisal(new_r, ff5)
    appraisal = ar_data["appraisal_ratio"]

    sb.loc["Inertia"] = {
        "n_months": s["n_months"],
        "mean_ann": s["mean_ann"],
        "vol_ann": s["vol_ann"],
        "sharpe_ann": s["sharpe_ann"],
        "skew": s["skew"],
        "excess_kurt": s["excess_kurt"],
        "max_drawdown": s["max_drawdown"],
        "appraisal_ratio": appraisal,
    }
    sb.to_csv(path_csv, float_format="%.4f")
    with open(path_md, "w") as f:
        f.write(sb.to_markdown(floatfmt=".4f"))
    print(f"  updated: {path_csv} and .md")


def update_sharpe_ci(new_r: pd.Series) -> None:
    path_csv = TABLES / "34_comprehensive_sharpe_ci.csv"
    path_md = TABLES / "34_comprehensive_sharpe_ci.md"
    df = pd.read_csv(path_csv, index_col=0)
    res = sharpe_bootstrap_ci(new_r, n_boot=N_BOOT, block_size=BLOCK, seed=42)
    df.loc["Inertia"] = {
        "sharpe": res["sharpe"],
        "ci_low": res["ci_low"],
        "ci_high": res["ci_high"],
    }
    df.to_csv(path_csv, float_format="%.4f")
    with open(path_md, "w") as f:
        f.write(df.to_markdown(floatfmt=".4f"))
    print(f"  updated: {path_csv} and .md")
    return res


def update_paired_diff(new_r: pd.Series, static_r: pd.Series) -> dict:
    path_csv = TABLES / "35_comprehensive_paired_diff.csv"
    path_md = TABLES / "35_comprehensive_paired_diff.md"
    df = pd.read_csv(path_csv, index_col=0)
    res = sharpe_diff_ci(new_r, static_r, n_boot=N_BOOT, block_size=BLOCK, seed=42)
    df.loc["Inertia"] = {
        "diff": res["diff"],
        "ci_low": res["ci_low"],
        "ci_high": res["ci_high"],
        "p_value": res["p_value"],
        "beats_FF5_5pct": (res["p_value"] < 0.05) and (res["diff"] > 0),
    }
    df.to_csv(path_csv, float_format="%.4f")
    with open(path_md, "w") as f:
        f.write(df.to_markdown(floatfmt=".4f"))
    print(f"  updated: {path_csv} and .md")
    return res


def compute_appraisal(r: pd.Series, ff5: pd.DataFrame) -> dict:
    """Regress excess returns on FF5; return alpha, residual std, appraisal."""
    factors = ff5.reindex(r.index).dropna()
    r2 = r.loc[factors.index]
    excess = r2 - factors["RF"]
    X = sm.add_constant(factors[["MKT_RF", "SMB", "HML", "RMW", "CMA"]])
    ols = sm.OLS(excess, X).fit()
    alpha_m = float(ols.params["const"])
    resid_std_m = float(np.std(ols.resid, ddof=1))
    alpha_ann = alpha_m * 12
    resid_std_ann = resid_std_m * np.sqrt(12)
    return {
        "alpha_monthly": alpha_m,
        "beta": float(ols.params["MKT_RF"]),
        "resid_std_monthly": resid_std_m,
        "alpha_ann": alpha_ann,
        "resid_std_ann": resid_std_ann,
        "appraisal_ratio": alpha_ann / resid_std_ann if resid_std_ann > 0 else np.nan,
    }


def update_appraisal(new_r: pd.Series, ff5: pd.DataFrame) -> dict:
    path_csv = TABLES / "40_appraisal_ratios.csv"
    path_md = TABLES / "40_appraisal_ratios.md"
    df = pd.read_csv(path_csv, index_col=0)
    res = compute_appraisal(new_r, ff5)
    df.loc["Inertia"] = res
    df.to_csv(path_csv, float_format="%.4f")
    with open(path_md, "w") as f:
        f.write(df.to_markdown(floatfmt=".4f"))
    print(f"  updated: {path_csv} and .md")
    return res


def main():
    out = build_new_inertia()
    inertia_r = out["return"]
    L = out["leverage_t"]

    # Trim to [START, END] for stats consistency
    inertia_eval = inertia_r.loc[START:END].dropna()

    print("=== New Inertia (vol-targeted B+C blend) ===")
    s = perf_stats(inertia_eval)
    cum = (1 + inertia_eval).prod()
    print(f"  n={s['n_months']}  Sharpe={s['sharpe_ann']:.4f}  "
          f"mean={s['mean_ann']*100:.3f}%  vol={s['vol_ann']*100:.3f}%  "
          f"DD={s['max_drawdown']*100:.3f}%  cumret=$1->${cum:.4f}")
    print(f"  L_t: min={L.min():.3f}  max={L.max():.3f}  "
          f"mean={L.mean():.3f}  median={L.median():.3f}")

    # Static FF5 reference
    comp = pd.read_csv(TABLES / "38_comprehensive_returns.csv",
                       index_col=0, parse_dates=True)
    static_r = comp["Static FF5"].loc[START:END].dropna()
    static_cum = (1 + static_r).prod()
    print(f"\n  Static FF5: cumret=$1->${static_cum:.4f}  "
          f"Sharpe={(static_r.mean()*12)/(static_r.std(ddof=1)*np.sqrt(12)):.4f}")

    # Save explore CSV
    explore_path = TABLES / "EXPLORE_inertia_v3_returns.csv"
    out_save = out[["return", "leverage_t"]].copy()
    out_save.loc[START:END].to_csv(explore_path)
    print(f"\n  saved: {explore_path}")

    # Update tables (use eval window)
    update_returns_csv(inertia_eval)

    ff5 = get_ff5_monthly()
    update_scoreboard(inertia_eval, ff5)
    sci = update_sharpe_ci(inertia_eval)
    pd_res = update_paired_diff(inertia_eval, static_r)
    apr = update_appraisal(inertia_eval, ff5)

    print("\n=== Final stats ===")
    print(f"  Sharpe        : {s['sharpe_ann']:.4f}  "
          f"95% CI [{sci['ci_low']:.4f}, {sci['ci_high']:.4f}]")
    print(f"  Mean (ann)    : {s['mean_ann']*100:.3f}%")
    print(f"  Vol  (ann)    : {s['vol_ann']*100:.3f}%")
    print(f"  Max DD        : {s['max_drawdown']*100:.3f}%")
    print(f"  Appraisal     : {apr['appraisal_ratio']:.4f} "
          f"(alpha_ann={apr['alpha_ann']*100:.3f}%, "
          f"resid_std_ann={apr['resid_std_ann']*100:.3f}%)")
    print(f"  Diff vs FF5   : ΔSharpe={pd_res['diff']:.4f}  "
          f"95% CI [{pd_res['ci_low']:.4f}, {pd_res['ci_high']:.4f}]  "
          f"p={pd_res['p_value']:.4f}")
    print(f"  cumret Inertia: ${cum:.4f}")
    print(f"  cumret FF5    : ${static_cum:.4f}")


if __name__ == "__main__":
    main()
