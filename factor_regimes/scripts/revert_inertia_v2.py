"""
revert_inertia_v2.py
--------------------
Revert Inertia to the original previous-winner definition:
   Inertia = 0.5 * Static FF5 + 0.5 * Method C V9 timed sleeve
where Method C V9 timed sleeve is the equal-weight composite of 5 long-only
timed factor sleeves driven by EXPLORE_method_c_best_probs.csv (long-only
weight rule w = max(2P-1, 0)) with 5 bp turnover cost.

Recomputes everything fresh:
  - 33_comprehensive_scoreboard.csv/.md (Inertia row)
  - 34_comprehensive_sharpe_ci.csv/.md (Inertia row)
  - 35_comprehensive_paired_diff.csv/.md (Inertia row)
  - 38_comprehensive_returns.csv (Inertia column)
  - 40_appraisal_ratios.csv/.md (Inertia row)

Bootstrap: 5000 reps, 12-month blocks, seed 42 (paired vs Static FF5).
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

from factor_regimes.lib.data import get_ff5_monthly, FF5_FACTORS  # noqa: E402
from factor_regimes.lib.evaluation import (  # noqa: E402
    perf_stats, sharpe_bootstrap_ci, sharpe_diff_ci,
)

TABLES = ROOT / "factor_regimes" / "tables"
PROBS_C_BEST = TABLES / "EXPLORE_method_c_best_probs.csv"

START = "2000-02-29"
END = "2024-12-31"
COST_BPS = 5.0
N_BOOT = 5000
BLOCK = 12


def composite_method_c_v9(P: pd.DataFrame, factor_next: pd.DataFrame,
                          cost_bps: float = COST_BPS) -> pd.Series:
    """Equal-weight composite of 5 long-only timed sleeves: w = max(2P-1, 0)."""
    P_a = P.reindex(factor_next.index).dropna(how="any")
    fn = factor_next.reindex(P_a.index)
    w = (2 * P_a - 1).clip(0, 1)  # long-only
    sleeves = []
    for f in FF5_FACTORS:
        gross = w[f] * fn[f]
        turn = w[f].diff().abs().fillna(w[f].abs())
        cost = cost_bps * turn / 1e4
        sleeves.append(gross - cost)
    return pd.DataFrame(sleeves).T.mean(axis=1)


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


def main():
    # 1. Load Static FF5 returns from 38_comprehensive_returns.csv
    comp = pd.read_csv(TABLES / "38_comprehensive_returns.csv",
                       index_col=0, parse_dates=True)
    static_r = comp["Static FF5"].copy()

    # 2. Build Method C V9 timed sleeve from best-probs CSV
    ff5 = get_ff5_monthly()[FF5_FACTORS]
    factor_next = ff5.shift(-1).loc[START:END]

    P_C = pd.read_csv(PROBS_C_BEST, index_col=0, parse_dates=True)[FF5_FACTORS]
    methodC_v9 = composite_method_c_v9(P_C, factor_next)
    methodC_v9 = methodC_v9.loc[START:END].reindex(static_r.index).fillna(0.0)

    # 3. Inertia = 50% Static FF5 + 50% Method C V9
    inertia_r = 0.5 * static_r + 0.5 * methodC_v9
    inertia_r = inertia_r.loc[START:END].dropna()

    print("=== Reverted Inertia (50% Static FF5 + 50% Method C V9) ===")
    s = perf_stats(inertia_r)
    cum_inertia = (1 + inertia_r).prod()
    cum_static = (1 + static_r.loc[START:END]).prod()
    print(f"  n={s['n_months']}  Sharpe={s['sharpe_ann']:.4f}")
    print(f"  mean_ann={s['mean_ann']*100:.3f}%   vol_ann={s['vol_ann']*100:.3f}%")
    print(f"  max DD={s['max_drawdown']*100:.3f}%")
    print(f"  cumret Inertia: $1 -> ${cum_inertia:.4f}")
    print(f"  cumret FF5    : $1 -> ${cum_static:.4f}")

    # ---- Update tables ----
    # 38_comprehensive_returns.csv: replace Inertia column
    comp["Inertia"] = inertia_r.reindex(comp.index)
    comp.to_csv(TABLES / "38_comprehensive_returns.csv")
    print(f"  updated: {TABLES / '38_comprehensive_returns.csv'}")

    # Appraisal
    full_ff5 = get_ff5_monthly()
    apr = compute_appraisal(inertia_r, full_ff5)

    # 33 scoreboard
    sb_csv = TABLES / "33_comprehensive_scoreboard.csv"
    sb_md = TABLES / "33_comprehensive_scoreboard.md"
    sb = pd.read_csv(sb_csv, index_col=0)
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
    sb.to_csv(sb_csv, float_format="%.4f")
    with open(sb_md, "w") as f:
        f.write(sb.to_markdown(floatfmt=".4f"))
    print(f"  updated: {sb_csv} and .md")

    # 34 sharpe CI (paired bootstrap on Sharpe of inertia)
    sci = sharpe_bootstrap_ci(inertia_r, n_boot=N_BOOT, block_size=BLOCK, seed=42)
    sci_csv = TABLES / "34_comprehensive_sharpe_ci.csv"
    sci_md = TABLES / "34_comprehensive_sharpe_ci.md"
    df_sci = pd.read_csv(sci_csv, index_col=0)
    df_sci.loc["Inertia"] = {
        "sharpe": sci["sharpe"],
        "ci_low": sci["ci_low"],
        "ci_high": sci["ci_high"],
    }
    df_sci.to_csv(sci_csv, float_format="%.4f")
    with open(sci_md, "w") as f:
        f.write(df_sci.to_markdown(floatfmt=".4f"))
    print(f"  updated: {sci_csv} and .md")

    # 35 paired diff
    static_for_diff = static_r.loc[START:END]
    pdr = sharpe_diff_ci(inertia_r, static_for_diff,
                         n_boot=N_BOOT, block_size=BLOCK, seed=42)
    pd_csv = TABLES / "35_comprehensive_paired_diff.csv"
    pd_md = TABLES / "35_comprehensive_paired_diff.md"
    df_pd = pd.read_csv(pd_csv, index_col=0)
    df_pd.loc["Inertia"] = {
        "diff": pdr["diff"],
        "ci_low": pdr["ci_low"],
        "ci_high": pdr["ci_high"],
        "p_value": pdr["p_value"],
        "beats_FF5_5pct": (pdr["p_value"] < 0.05) and (pdr["diff"] > 0),
    }
    df_pd.to_csv(pd_csv, float_format="%.4f")
    with open(pd_md, "w") as f:
        f.write(df_pd.to_markdown(floatfmt=".4f"))
    print(f"  updated: {pd_csv} and .md")

    # 40 appraisal
    apr_csv = TABLES / "40_appraisal_ratios.csv"
    apr_md = TABLES / "40_appraisal_ratios.md"
    df_apr = pd.read_csv(apr_csv, index_col=0)
    df_apr.loc["Inertia"] = apr
    df_apr.to_csv(apr_csv, float_format="%.4f")
    with open(apr_md, "w") as f:
        f.write(df_apr.to_markdown(floatfmt=".4f"))
    print(f"  updated: {apr_csv} and .md")

    print("\n=== Final Inertia stats (fresh recompute) ===")
    print(f"  Sharpe        : {s['sharpe_ann']:.4f}  "
          f"95% CI [{sci['ci_low']:.4f}, {sci['ci_high']:.4f}]")
    print(f"  Mean (ann)    : {s['mean_ann']*100:.3f}%")
    print(f"  Vol  (ann)    : {s['vol_ann']*100:.3f}%")
    print(f"  Max DD        : {s['max_drawdown']*100:.3f}%")
    print(f"  Appraisal     : {apr['appraisal_ratio']:.4f} "
          f"(alpha_ann={apr['alpha_ann']*100:.3f}%, "
          f"resid_std_ann={apr['resid_std_ann']*100:.3f}%)")
    print(f"  Diff vs FF5   : dSharpe={pdr['diff']:.4f}  "
          f"95% CI [{pdr['ci_low']:.4f}, {pdr['ci_high']:.4f}]  "
          f"p={pdr['p_value']:.4f}  beats5pct="
          f"{(pdr['p_value'] < 0.05) and (pdr['diff'] > 0)}")
    print(f"  cumret Inertia: $1 -> ${cum_inertia:.4f}")
    print(f"  cumret FF5    : $1 -> ${cum_static:.4f}")


if __name__ == "__main__":
    main()
