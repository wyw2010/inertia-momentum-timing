"""
build_inertia_v3_futures.py
---------------------------
Inertia v3 with FUTURES-style financing (RF + 5 bp/year), comparing two
leverage schemes:

  Variant A (constant in-sample):
      L = 0.0528 / vol_blend_full_sample  (single scalar)
      Inertia_t = L * r_blend_t - (L - 1) * (RF_t + 0.0005/12)

  Variant B (rolling 36m vol target, cap [0.5, 3.0], warm-start full-sample):
      L_t = 0.0528 / rolling_36m_vol_blend_t
      Inertia_t = L_t * r_blend_t - (L_t - 1) * (RF_t + 0.0005/12)

The script reports both, picks the higher Sharpe, and writes the winner into
the comprehensive scoreboard / Sharpe CI / paired diff / appraisal / returns
tables, plus EXPLORE_inertia_v3_returns.csv.
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
TARGET_VOL = 0.0528
WIN = 36
L_MIN, L_MAX = 0.5, 3.0
BORROW_SPREAD_ANN = 0.0005  # 5 bp/year futures roll cost
N_BOOT = 5000
BLOCK = 12


def build_blend_and_rf():
    comp = pd.read_csv(TABLES / "38_comprehensive_returns.csv",
                       index_col=0, parse_dates=True)
    methodB = comp["Method B"]
    methodC = comp["Method C"]
    r_blend = 0.5 * methodB + 0.5 * methodC

    ff = get_ff5_monthly()
    rf = ff["RF"].reindex(comp.index).fillna(0.0)
    return comp, r_blend, rf


def variant_A(r_blend: pd.Series, rf: pd.Series) -> pd.DataFrame:
    full_vol = r_blend.std(ddof=1) * np.sqrt(12)
    L = TARGET_VOL / full_vol  # scalar
    spread_m = BORROW_SPREAD_ANN / 12.0
    # symmetric financing using (L-1)*(RF + spread) — applies in both directions
    inertia = L * r_blend - (L - 1.0) * (rf + spread_m)
    L_series = pd.Series(L, index=r_blend.index)
    return pd.DataFrame({
        "return": inertia.values,
        "leverage_t": L_series.values,
        "rolling_vol_ann": pd.Series(full_vol, index=r_blend.index).values,
        "rf_monthly": rf.values,
        "r_blend": r_blend.values,
    }, index=r_blend.index)


def variant_B(r_blend: pd.Series, rf: pd.Series) -> pd.DataFrame:
    roll_std = r_blend.rolling(WIN).std(ddof=1).shift(1)
    roll_vol_ann = roll_std * np.sqrt(12)
    full_vol = r_blend.std(ddof=1) * np.sqrt(12)
    roll_vol_ann = roll_vol_ann.fillna(full_vol)
    L = (TARGET_VOL / roll_vol_ann).clip(lower=L_MIN, upper=L_MAX)
    spread_m = BORROW_SPREAD_ANN / 12.0
    inertia = L * r_blend - (L - 1.0) * (rf + spread_m)
    return pd.DataFrame({
        "return": inertia.values,
        "leverage_t": L.values,
        "rolling_vol_ann": roll_vol_ann.values,
        "rf_monthly": rf.values,
        "r_blend": r_blend.values,
    }, index=r_blend.index)


def compute_appraisal(r: pd.Series, ff5: pd.DataFrame) -> dict:
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


def update_tables(new_r: pd.Series, ff5: pd.DataFrame, static_r: pd.Series):
    # 38 returns
    path38 = TABLES / "38_comprehensive_returns.csv"
    comp = pd.read_csv(path38, index_col=0, parse_dates=True)
    comp["Inertia"] = new_r.reindex(comp.index)
    comp.to_csv(path38)
    print(f"  updated: {path38}")

    # 33 scoreboard
    s = perf_stats(new_r)
    apr = compute_appraisal(new_r, ff5)
    path_csv = TABLES / "33_comprehensive_scoreboard.csv"
    path_md = TABLES / "33_comprehensive_scoreboard.md"
    sb = pd.read_csv(path_csv, index_col=0)
    sb.loc["Inertia"] = {
        "n_months": s["n_months"],
        "mean_ann": s["mean_ann"],
        "vol_ann": s["vol_ann"],
        "sharpe_ann": s["sharpe_ann"],
        "skew": s["skew"],
        "excess_kurt": s["excess_kurt"],
        "max_drawdown": s["max_drawdown"],
        "appraisal_ratio": apr["appraisal_ratio"],
    }
    sb.to_csv(path_csv, float_format="%.4f")
    with open(path_md, "w") as f:
        f.write(sb.to_markdown(floatfmt=".4f"))
    print(f"  updated: {path_csv} and .md")

    # 34 sharpe CI
    path_csv = TABLES / "34_comprehensive_sharpe_ci.csv"
    path_md = TABLES / "34_comprehensive_sharpe_ci.md"
    df = pd.read_csv(path_csv, index_col=0)
    sci = sharpe_bootstrap_ci(new_r, n_boot=N_BOOT, block_size=BLOCK, seed=42)
    df.loc["Inertia"] = {"sharpe": sci["sharpe"],
                         "ci_low": sci["ci_low"],
                         "ci_high": sci["ci_high"]}
    df.to_csv(path_csv, float_format="%.4f")
    with open(path_md, "w") as f:
        f.write(df.to_markdown(floatfmt=".4f"))
    print(f"  updated: {path_csv} and .md")

    # 35 paired diff vs FF5
    path_csv = TABLES / "35_comprehensive_paired_diff.csv"
    path_md = TABLES / "35_comprehensive_paired_diff.md"
    df = pd.read_csv(path_csv, index_col=0)
    pd_res = sharpe_diff_ci(new_r, static_r, n_boot=N_BOOT, block_size=BLOCK,
                            seed=42)
    df.loc["Inertia"] = {
        "diff": pd_res["diff"],
        "ci_low": pd_res["ci_low"],
        "ci_high": pd_res["ci_high"],
        "p_value": pd_res["p_value"],
        "beats_FF5_5pct": (pd_res["p_value"] < 0.05) and (pd_res["diff"] > 0),
    }
    df.to_csv(path_csv, float_format="%.4f")
    with open(path_md, "w") as f:
        f.write(df.to_markdown(floatfmt=".4f"))
    print(f"  updated: {path_csv} and .md")

    # 40 appraisal
    path_csv = TABLES / "40_appraisal_ratios.csv"
    path_md = TABLES / "40_appraisal_ratios.md"
    df = pd.read_csv(path_csv, index_col=0)
    df.loc["Inertia"] = apr
    df.to_csv(path_csv, float_format="%.4f")
    with open(path_md, "w") as f:
        f.write(df.to_markdown(floatfmt=".4f"))
    print(f"  updated: {path_csv} and .md")

    return s, sci, pd_res, apr


def main():
    comp, r_blend, rf = build_blend_and_rf()

    A = variant_A(r_blend, rf)
    B = variant_B(r_blend, rf)

    A_r = A["return"].loc[START:END].dropna()
    B_r = B["return"].loc[START:END].dropna()

    sA = perf_stats(A_r); cA = (1 + A_r).prod()
    sB = perf_stats(B_r); cB = (1 + B_r).prod()

    print("=== Variant A (constant L, 5bp futures) ===")
    print(f"  L = {A['leverage_t'].iloc[0]:.4f}  vol_full = {A['rolling_vol_ann'].iloc[0]:.4f}")
    print(f"  n={sA['n_months']}  Sharpe={sA['sharpe_ann']:.4f}  "
          f"mean={sA['mean_ann']*100:.3f}%  vol={sA['vol_ann']*100:.3f}%  "
          f"DD={sA['max_drawdown']*100:.3f}%  cumret=${cA:.4f}")

    print("=== Variant B (rolling 36m, 5bp futures) ===")
    print(f"  L_t: min={B['leverage_t'].min():.3f}  max={B['leverage_t'].max():.3f}  "
          f"mean={B['leverage_t'].mean():.3f}  median={B['leverage_t'].median():.3f}")
    print(f"  n={sB['n_months']}  Sharpe={sB['sharpe_ann']:.4f}  "
          f"mean={sB['mean_ann']*100:.3f}%  vol={sB['vol_ann']*100:.3f}%  "
          f"DD={sB['max_drawdown']*100:.3f}%  cumret=${cB:.4f}")

    # Pick winner by Sharpe
    if sA["sharpe_ann"] >= sB["sharpe_ann"]:
        winner = "A"; winner_df = A; winner_r = A_r; ws = sA; wcum = cA
    else:
        winner = "B"; winner_df = B; winner_r = B_r; ws = sB; wcum = cB
    print(f"\n*** Winner: Variant {winner} ***")

    # Static FF5 reference (eval window)
    static_r = comp["Static FF5"].loc[START:END].dropna()
    static_cum = (1 + static_r).prod()
    static_sr = (static_r.mean() * 12) / (static_r.std(ddof=1) * np.sqrt(12))
    print(f"  Static FF5: cumret=${static_cum:.4f}  Sharpe={static_sr:.4f}")

    # Save EXPLORE
    explore_path = TABLES / "EXPLORE_inertia_v3_returns.csv"
    out_save = winner_df[["return", "leverage_t", "rolling_vol_ann",
                          "rf_monthly", "r_blend"]].copy()
    out_save.loc[START:END].to_csv(explore_path)
    print(f"  saved: {explore_path}")

    # Update tables
    ff5 = get_ff5_monthly()
    s, sci, pd_res, apr = update_tables(winner_r, ff5, static_r)

    print("\n=== Final stats (Inertia v3 winner) ===")
    print(f"  Variant      : {winner}")
    print(f"  Sharpe       : {s['sharpe_ann']:.4f}  "
          f"95% CI [{sci['ci_low']:.4f}, {sci['ci_high']:.4f}]")
    print(f"  Mean (ann)   : {s['mean_ann']*100:.3f}%")
    print(f"  Vol  (ann)   : {s['vol_ann']*100:.3f}%")
    print(f"  Max DD       : {s['max_drawdown']*100:.3f}%")
    print(f"  Appraisal    : {apr['appraisal_ratio']:.4f} "
          f"(alpha_ann={apr['alpha_ann']*100:.3f}%, "
          f"resid_std_ann={apr['resid_std_ann']*100:.3f}%)")
    print(f"  Diff vs FF5  : ΔSharpe={pd_res['diff']:.4f}  "
          f"95% CI [{pd_res['ci_low']:.4f}, {pd_res['ci_high']:.4f}]  "
          f"p={pd_res['p_value']:.4f}")
    print(f"  cumret Inert : ${wcum:.4f}")
    print(f"  cumret FF5   : ${static_cum:.4f}")


if __name__ == "__main__":
    main()
