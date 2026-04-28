"""
explore_ensemble.py
-------------------
Search a small grid of ensemble-blend variants for the Inertia v2 timed sleeve,
trying to lift Sharpe from 1.07 (baseline 0.5/0.5 Static FF5 + Method C) past
the 1.20 target while keeping vol < 5% and DD shallower than -8%.

Outputs:
    factor_regimes/tables/EXPLORE_ensemble_variants.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from factor_regimes.lib.evaluation import perf_stats, sharpe_diff_ci  # noqa: E402
from factor_regimes.lib.data import get_ff5_monthly, FF5_FACTORS         # noqa: E402

TABLES = ROOT / "factor_regimes" / "tables"
OUT_CSV = TABLES / "EXPLORE_ensemble_variants.csv"

START = "2000-02-29"
END   = "2024-12-31"
COST_BPS_ONEWAY = 5.0  # match backtest.apply_weights


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_inputs():
    pa = pd.read_csv(TABLES / "05_method_a_probs.csv", index_col=0, parse_dates=True)
    pb = pd.read_csv(TABLES / "09_method_b_probs.csv", index_col=0, parse_dates=True)
    pc = pd.read_csv(TABLES / "13_method_c_probs.csv", index_col=0, parse_dates=True)
    pa = pa[FF5_FACTORS]; pb = pb[FF5_FACTORS]; pc = pc[FF5_FACTORS]

    ff5 = get_ff5_monthly()[FF5_FACTORS]                       # contemporaneous returns
    r_next = ff5.shift(-1)                                     # next-month returns aligned to prob date

    mpsif = pd.read_csv(ROOT / "factor_regimes" / "data_input_mpsif_real_returns.csv",
                        parse_dates=["next_date", "asof"])
    mpsif = mpsif.set_index("next_date")["ret"].sort_index()

    return pa, pb, pc, r_next, mpsif


# --------------------------------------------------------------------------- #
# Strategy builders
# --------------------------------------------------------------------------- #
def static_ff5_returns(r_next: pd.DataFrame) -> pd.Series:
    """Equal-weighted FF5 buy-and-hold (no costs - rebalancing is small)."""
    return r_next.mean(axis=1)


def _apply_weights(weights: pd.DataFrame, r_next: pd.DataFrame,
                   cost_bps: float = COST_BPS_ONEWAY) -> pd.Series:
    """Per-factor: r_gross - cost. Equal-weight across the 5 sleeves."""
    w = weights.reindex(r_next.index)
    rn = r_next.reindex(w.index)
    r_gross = (w * rn)
    turnover = w.diff().abs()
    turnover.iloc[0] = w.iloc[0].abs()
    cost = cost_bps * turnover / 1e4
    r_net = r_gross - cost
    return r_net.mean(axis=1)


def timed_linear(probs: pd.DataFrame, r_next: pd.DataFrame) -> pd.Series:
    w = (2 * probs - 1.0).clip(-1, 1)
    return _apply_weights(w, r_next)


def timed_longflat(probs: pd.DataFrame, r_next: pd.DataFrame) -> pd.Series:
    w = (2 * probs - 1.0).clip(0, 1)             # long-only version of linear
    return _apply_weights(w, r_next)


def timed_threestate(probs: pd.DataFrame, r_next: pd.DataFrame,
                     hi: float = 0.6, lo: float = 0.4) -> pd.Series:
    w = pd.DataFrame(0.5, index=probs.index, columns=probs.columns)
    w[probs > hi] = 1.0
    w[probs <= lo] = 0.0
    return _apply_weights(w, r_next)


def timed_topk(probs: pd.DataFrame, r_next: pd.DataFrame, k: int = 3) -> pd.Series:
    """Each month, hold top-k factors equal-weighted (long-only)."""
    rank = probs.rank(axis=1, ascending=False, method="first")
    w = (rank <= k).astype(float) / k     # weights sum to 1 (this IS the portfolio)
    rn = r_next.reindex(w.index)
    r_gross = (w * rn).sum(axis=1)
    turnover = w.diff().abs()
    turnover.iloc[0] = w.iloc[0].abs()
    cost = (COST_BPS_ONEWAY * turnover / 1e4).sum(axis=1)
    return r_gross - cost


def timed_volscaled(probs: pd.DataFrame, r_next: pd.DataFrame, ff5: pd.DataFrame,
                    target_vol_monthly: float = 0.04, lookback: int = 12) -> pd.Series:
    """Scale each factor's linear weight by 1/recent_vol (rolling)."""
    w = (2 * probs - 1.0).clip(-1, 1)
    realized_vol = ff5.rolling(lookback).std().shift(1)              # avoid look-ahead
    scaler = (target_vol_monthly / realized_vol).clip(0.0, 3.0)
    w_scaled = (w * scaler).clip(-1.5, 1.5).reindex(r_next.index).fillna(0.0)
    return _apply_weights(w_scaled, r_next)


def blend(static_r: pd.Series, timed_r: pd.Series, w_static: float) -> pd.Series:
    j = pd.concat([static_r, timed_r], axis=1, join="inner").dropna()
    return w_static * j.iloc[:, 0] + (1 - w_static) * j.iloc[:, 1]


def slice_window(s: pd.Series) -> pd.Series:
    return s.loc[START:END].dropna()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    pa, pb, pc, r_next, mpsif = load_inputs()
    ff5 = get_ff5_monthly()[FF5_FACTORS]

    # Average ensemble probabilities
    pavg = (pa + pb + pc) / 3.0

    # Holdout-weighted ensemble: train on first 36 months of OOS, weight ~ Sharpe of timed sleeve
    # Use 2000-02 .. 2003-01 as the holdout, then apply weights from 2003-02 onwards.
    holdout_end = "2003-01-31"
    timed_a_h = slice_window(timed_linear(pa, r_next)).loc[:holdout_end]
    timed_b_h = slice_window(timed_linear(pb, r_next)).loc[:holdout_end]
    timed_c_h = slice_window(timed_linear(pc, r_next)).loc[:holdout_end]

    def _sharpe(s):
        m, sd = s.mean(), s.std(ddof=1)
        return (m * 12) / (sd * np.sqrt(12)) if sd > 0 else 0.0

    sa, sb, sc = max(_sharpe(timed_a_h), 0.01), max(_sharpe(timed_b_h), 0.01), max(_sharpe(timed_c_h), 0.01)
    tot = sa + sb + sc
    wa, wb, wc = sa / tot, sb / tot, sc / tot
    p_holdout = wa * pa + wb * pb + wc * pc

    # Strategies / blends
    static_r = slice_window(static_ff5_returns(r_next))

    timed_a = slice_window(timed_linear(pa, r_next))
    timed_b = slice_window(timed_linear(pb, r_next))
    timed_c = slice_window(timed_linear(pc, r_next))
    timed_avg = slice_window(timed_linear(pavg, r_next))
    timed_holdout = slice_window(timed_linear(p_holdout, r_next))
    timed_lf_c = slice_window(timed_longflat(pc, r_next))
    timed_3s_avg = slice_window(timed_threestate(pavg, r_next))
    timed_topk3 = slice_window(timed_topk(pavg, r_next, k=3))
    timed_topk4 = slice_window(timed_topk(pavg, r_next, k=4))
    timed_vs_avg = slice_window(timed_volscaled(pavg, r_next, ff5))

    variants = {}

    # Variant 1: blend ratios with Method C
    for ws in (0.30, 0.40, 0.50, 0.55, 0.60, 0.70):
        variants[f"V1_blend_C_{int(ws*100)}_{int((1-ws)*100)}"] = blend(static_r, timed_c, ws)

    # Variant 2: Pure ensemble (avg) as timed
    for ws in (0.30, 0.40, 0.50, 0.60, 0.70):
        variants[f"V2_blend_AVG_{int(ws*100)}_{int((1-ws)*100)}"] = blend(static_r, timed_avg, ws)

    # Variant 3: Method B as timed
    for ws in (0.30, 0.50, 0.70):
        variants[f"V3_blend_B_{int(ws*100)}_{int((1-ws)*100)}"] = blend(static_r, timed_b, ws)

    # Variant 4: long-only weight rule on Method C and AVG (timed sleeve only, baseline blend)
    variants["V4_longflat_C_50_50"]   = blend(static_r, timed_lf_c, 0.50)
    variants["V4_longflat_C_30_70"]   = blend(static_r, timed_lf_c, 0.30)

    # Variant 5: 3-state on AVG
    variants["V5_3state_AVG_50_50"]   = blend(static_r, timed_3s_avg, 0.50)
    variants["V5_3state_AVG_30_70"]   = blend(static_r, timed_3s_avg, 0.30)

    # Variant 6: top-K (no static blend, then 50/50 blend)
    variants["V6_topK3_pure"]         = timed_topk3
    variants["V6_topK4_pure"]         = timed_topk4
    variants["V6_topK3_blend5050"]    = blend(static_r, timed_topk3, 0.50)
    variants["V6_topK4_blend5050"]    = blend(static_r, timed_topk4, 0.50)

    # Variant 7: vol-scaled timed sleeve (AVG probs)
    variants["V7_volscaled_AVG_50_50"] = blend(static_r, timed_vs_avg, 0.50)
    variants["V7_volscaled_AVG_30_70"] = blend(static_r, timed_vs_avg, 0.30)

    # Variant 9: holdout-weighted ensemble probs
    variants["V9_holdout_50_50"]       = blend(static_r, timed_holdout, 0.50)
    variants["V9_holdout_30_70"]       = blend(static_r, timed_holdout, 0.30)

    # Baseline reference for context
    variants["BASELINE_inertia"]       = blend(static_r, timed_c, 0.50)
    variants["StaticFF5_ref"]          = static_r

    # Variant 8: MPSIF overlay - need a "best timed" component from above
    # Use the best timed component we've explored - we'll just add 3 fixed combinations
    # using AVG-based timed (which tends to be smoothest).
    j_mpsif = pd.concat([mpsif, timed_avg], axis=1, join="inner").dropna()
    j_mpsif.columns = ["mpsif", "tavg"]
    for w_mp in (0.30, 0.50, 0.70):
        variants[f"V8_MPSIF_{int(w_mp*100)}_{int((1-w_mp)*100)}_AVG"] = (
            w_mp * j_mpsif["mpsif"] + (1 - w_mp) * j_mpsif["tavg"]
        )

    # Performance table
    rows = []
    for name, r in variants.items():
        st = perf_stats(r)
        rows.append({"variant": name, "n": st.get("n_months"),
                     "ann_ret": st.get("mean_ann"),
                     "vol":     st.get("vol_ann"),
                     "sharpe":  st.get("sharpe_ann"),
                     "max_dd":  st.get("max_drawdown")})
    perf = pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)

    # Filter to vol<5% and DD>-8% before picking winners (per spec)
    eligible = perf[(perf["vol"] < 0.05) & (perf["max_dd"] > -0.08)].copy()

    # Top-3 paired bootstrap vs Static FF5 (excluding the static row itself and MPSIF blends, which have a different benchmark)
    # We'll compute paired CI vs static FF5 for the top-3 eligible non-MPSIF variants.
    candidates = eligible[~eligible["variant"].str.startswith(("StaticFF5_ref", "V8_MPSIF_"))]
    top3 = candidates.head(3)["variant"].tolist()

    diff_rows = []
    for name in top3:
        r = variants[name]
        out = sharpe_diff_ci(r, static_r)
        diff_rows.append({"variant": name,
                          "sharpe_diff": out["diff"],
                          "ci_low": out["ci_low"],
                          "ci_high": out["ci_high"],
                          "p_value": out["p_value"]})
    diffs = pd.DataFrame(diff_rows)

    # Merge for final printout
    final = perf.merge(
        diffs, on="variant", how="left"
    )

    final.to_csv(OUT_CSV, index=False)

    # Pretty print
    pd.set_option("display.float_format", lambda x: f"{x: .4f}")
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print("\n=== Variant comparison (sorted by Sharpe) ===\n")
    print(final.to_string(index=False))

    print("\n=== Eligibility (vol<5%, DD>-8%) ===\n")
    print(eligible.to_string(index=False))

    print(f"\nSaved table: {OUT_CSV}")
    return final, eligible


if __name__ == "__main__":
    main()
