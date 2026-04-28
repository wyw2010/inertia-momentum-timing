"""
explore_inertia_returns.py
--------------------------
Explore HIGHER-RETURN variants of the Inertia strategy.

Baselines (from 38_comprehensive_returns.csv):
  - Static FF5:  Sharpe 0.99, vol 5.3%, mean 5.25%
  - Inertia v2:  Sharpe 1.07, vol 2.8%, mean 3.02%

Goal: find a variant whose CUMULATIVE return beats Static FF5 AND Sharpe > 0.99,
ideally simply by vol-targeting Inertia.

Outputs:
  factor_regimes/tables/EXPLORE_inertia_returns_variants.csv
  factor_regimes/tables/EXPLORE_inertia_high_return_best.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from factor_regimes.lib.data import get_ff5_monthly, FF5_FACTORS  # noqa: E402
from factor_regimes.lib.evaluation import perf_stats               # noqa: E402

TABLES = ROOT / "factor_regimes" / "tables"
COMP_CSV = TABLES / "38_comprehensive_returns.csv"
PROBS_B_BEST = TABLES / "EXPLORE_method_b_best_probs.csv"
PROBS_C_BEST = TABLES / "EXPLORE_method_c_best_probs.csv"

START = "2000-02-29"
END   = "2024-12-31"
COST_BPS = 5.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def composite_return(P: pd.DataFrame, factor_next: pd.DataFrame,
                      mode: str = "linear", cost_bps: float = COST_BPS) -> pd.Series:
    """Equal-weight composite of timed sleeves (5 factors)."""
    P_a = P.reindex(factor_next.index).dropna(how="any")
    fn = factor_next.reindex(P_a.index)
    if mode == "linear":
        w = (2 * P_a - 1).clip(-1, 1)
    elif mode == "soft":
        w = ((P_a - 0.5) * 4).clip(-1, 1)
    elif mode == "longflat":
        w = (P_a > 0.5).astype(float)
    elif mode == "longonly":
        w = (2 * P_a - 1).clip(0, 1)
    else:
        raise ValueError(mode)
    sleeves = []
    for f in FF5_FACTORS:
        gross = w[f] * fn[f]
        turn = w[f].diff().abs().fillna(w[f].abs())
        cost = cost_bps * turn / 1e4
        sleeves.append(gross - cost)
    return pd.DataFrame(sleeves).T.mean(axis=1)


def perf_with_cumret(r: pd.Series) -> dict:
    s = perf_stats(r)
    cum = (1 + r.dropna()).prod()
    s["cumret_1to_X"] = cum
    return s


def slice_window(s: pd.Series) -> pd.Series:
    return s.loc[START:END].dropna()


# ---------------------------------------------------------------------------
# Build variants
# ---------------------------------------------------------------------------
def main():
    # Load comprehensive returns (already aligned 2000-02 -> 2024-12)
    comp = pd.read_csv(COMP_CSV, index_col=0, parse_dates=True)
    static_r   = comp["Static FF5"]
    methodA_r  = comp["Method A"]
    methodB_r  = comp["Method B"]
    methodC_r  = comp["Method C"]
    inertia_r  = comp["Inertia v2"] if "Inertia v2" in comp.columns else comp["Inertia"]

    # Build "best-probs" timed series for B (V3 longflat) and C (V9 long-only)
    ff5 = get_ff5_monthly()[FF5_FACTORS]
    factor_next = ff5.shift(-1)
    factor_next = factor_next.loc[START:END]

    P_B = pd.read_csv(PROBS_B_BEST, index_col=0, parse_dates=True)[FF5_FACTORS]
    P_C = pd.read_csv(PROBS_C_BEST, index_col=0, parse_dates=True)[FF5_FACTORS]

    # V3 method B winner uses longflat weight rule; V9 method C uses long-only
    methodB_best_r = composite_return(P_B, factor_next, mode="longflat")
    methodC_best_r = composite_return(P_C, factor_next, mode="longonly")

    methodB_best_r = slice_window(methodB_best_r).reindex(static_r.index).fillna(0.0)
    methodC_best_r = slice_window(methodC_best_r).reindex(static_r.index).fillna(0.0)

    # Sanity-check baseline numbers
    print("=== Reference baselines ===")
    for name, r in [("Static FF5", static_r), ("Method B (csv)", methodB_r),
                    ("Method C (csv)", methodC_r), ("Inertia v2 (csv)", inertia_r),
                    ("Method B best-probs", methodB_best_r),
                    ("Method C best-probs", methodC_best_r)]:
        s = perf_with_cumret(r)
        print(f"  {name:25s}  Sharpe={s['sharpe_ann']:.3f}  "
              f"mean={s['mean_ann']*100:.2f}%  vol={s['vol_ann']*100:.2f}%  "
              f"DD={s['max_drawdown']*100:.2f}%  cumret={s['cumret_1to_X']:.3f}")

    target_vol = static_r.std(ddof=1) * np.sqrt(12)
    print(f"\nStatic FF5 realized annualized vol = {target_vol:.4f}")

    variants: dict[str, pd.Series] = {}

    # 0. References
    variants["Static_FF5"]              = static_r
    variants["Inertia_v2_baseline"]     = inertia_r
    variants["Method_B_baseline"]       = methodB_r
    variants["Method_C_baseline"]       = methodC_r
    variants["Method_B_best_V3"]        = methodB_best_r
    variants["Method_C_best_V9"]        = methodC_best_r

    # 1. Vol-targeted Inertia (frictionless leverage)
    inertia_vol = inertia_r.std(ddof=1) * np.sqrt(12)
    scale = target_vol / inertia_vol
    variants[f"V1_inertia_voltarget_x{scale:.2f}"] = scale * inertia_r

    # 2. Higher Method C allocation (using baseline Method C composite)
    for ws in (0.30, 0.20, 0.10, 0.0):
        variants[f"V2_static{int(ws*100)}_C{int((1-ws)*100)}"] = ws * static_r + (1 - ws) * methodC_r

    # 2b. Same but using Method C best-probs V9 (long-only)
    for ws in (0.50, 0.30, 0.20, 0.0):
        variants[f"V2b_static{int(ws*100)}_Cbest{int((1-ws)*100)}"] = (
            ws * static_r + (1 - ws) * methodC_best_r
        )

    # 3. Method B + Method C blends (both baseline composites)
    for wb in (0.50, 0.40, 0.30):
        variants[f"V3_B{int(wb*100)}_C{int((1-wb)*100)}"] = wb * methodB_r + (1 - wb) * methodC_r

    # 3b. Method B best + Method C best blend
    for wb in (0.50, 0.30):
        variants[f"V3b_Bbest{int(wb*100)}_Cbest{int((1-wb)*100)}"] = (
            wb * methodB_best_r + (1 - wb) * methodC_best_r
        )

    # 3c. Static + Bbest + Cbest three-way mixes (favor higher-mean B)
    for ws, wb, wc in [(0.34, 0.33, 0.33), (0.20, 0.40, 0.40),
                        (0.50, 0.25, 0.25), (0.40, 0.30, 0.30),
                        (0.30, 0.50, 0.20)]:
        variants[f"V3c_S{int(ws*100)}_Bb{int(wb*100)}_Cb{int(wc*100)}"] = (
            ws * static_r + wb * methodB_best_r + wc * methodC_best_r
        )

    # 4. Pure Method B (best V3) standalone
    variants["V4_methodB_best_only"] = methodB_best_r
    variants["V4_methodB_baseline_only"] = methodB_r

    # 5. Vol-targeted Method B variants
    for src_name, src_r in [("Bbest", methodB_best_r), ("Bcsv", methodB_r)]:
        v = src_r.std(ddof=1) * np.sqrt(12)
        if v > 1e-6:
            sc = target_vol / v
            variants[f"V5_{src_name}_voltarget_x{sc:.2f}"] = sc * src_r

    # 5b. Vol-target Method-B+Method-C blends and Static+Bbest+Cbest blend
    for label, r in [
        ("V3_B50_C50", 0.5 * methodB_r + 0.5 * methodC_r),
        ("V3b_Bb50_Cb50", 0.5 * methodB_best_r + 0.5 * methodC_best_r),
        ("V3c_S34_Bb33_Cb33",
         0.34 * static_r + 0.33 * methodB_best_r + 0.33 * methodC_best_r),
    ]:
        v = r.std(ddof=1) * np.sqrt(12)
        if v > 1e-6:
            sc = target_vol / v
            variants[f"V5b_{label}_voltarget_x{sc:.2f}"] = sc * r

    # 6. Dynamic blend - tilt to Method B in low-vol regimes, C in high-vol
    # Use 12-month rolling vol of FF5 average (proxy for market regime)
    market_proxy = static_r
    rolling_vol = market_proxy.rolling(12, min_periods=6).std().shift(1)
    median_vol = rolling_vol.median()
    # in low-vol regime: 70% Bbest, 30% Cbest; in high-vol: 30% Bbest, 70% Cbest
    low_vol_mask = (rolling_vol <= median_vol).astype(float).fillna(0.5)
    wB = 0.30 + 0.40 * low_vol_mask     # 0.30 in high vol, 0.70 in low vol
    wC = 1.0 - wB
    dyn_BC = wB * methodB_best_r + wC * methodC_best_r
    variants["V6_dyn_lowvol_Bbest_highvol_Cbest"] = dyn_BC

    # 6b. Dynamic with static blend
    for ws in (0.50, 0.30):
        variants[f"V6b_static{int(ws*100)}_dyn{int((1-ws)*100)}"] = (
            ws * static_r + (1 - ws) * dyn_BC
        )

    # 6c. Vol-target dynamic
    v = dyn_BC.std(ddof=1) * np.sqrt(12)
    sc = target_vol / v
    variants[f"V6c_dyn_voltarget_x{sc:.2f}"] = sc * dyn_BC

    # 7. Best-of-N: pick the timing method with highest trailing 12m Sharpe
    candidates = {"B": methodB_best_r, "C": methodC_best_r, "Static": static_r}
    win = 12
    chosen = []
    idx = static_r.index
    bestN = pd.Series(index=idx, dtype=float)
    for i, dt in enumerate(idx):
        if i < win:
            # warm-up - use static
            bestN.iloc[i] = static_r.iloc[i]
            chosen.append("Static")
            continue
        sharpes = {}
        for k, r in candidates.items():
            window_r = r.iloc[i - win:i]
            mu = window_r.mean() * 12
            sd = window_r.std(ddof=1) * np.sqrt(12)
            sharpes[k] = mu / sd if sd > 0 else -np.inf
        pick = max(sharpes, key=sharpes.get)
        chosen.append(pick)
        bestN.iloc[i] = candidates[pick].iloc[i]
    variants["V7_bestof3_trail12mSharpe"] = bestN
    print(f"\nV7 chose Static {chosen.count('Static')}, B {chosen.count('B')}, "
          f"C {chosen.count('C')} times")

    # 8. Stacking: average probs first, then long-only or long-flat weight
    # Need probs for A too
    P_A = pd.read_csv(TABLES / "05_method_a_probs.csv", index_col=0, parse_dates=True)[FF5_FACTORS]
    P_A = P_A.reindex(factor_next.index)
    P_B_orig = pd.read_csv(TABLES / "09_method_b_probs.csv", index_col=0,
                            parse_dates=True)[FF5_FACTORS].reindex(factor_next.index)
    P_C_orig = pd.read_csv(TABLES / "13_method_c_probs.csv", index_col=0,
                            parse_dates=True)[FF5_FACTORS].reindex(factor_next.index)

    # Stack with original probs (linear)
    P_avg = (P_A + P_B_orig + P_C_orig) / 3.0
    r_stack_lin  = composite_return(P_avg, factor_next, mode="linear")
    r_stack_lf   = composite_return(P_avg, factor_next, mode="longflat")
    r_stack_lo   = composite_return(P_avg, factor_next, mode="longonly")
    # Stack with best probs (B, C only) since those are the winners
    P_avg_best = (P_B + P_C) / 2.0
    r_stack_best_lo = composite_return(P_avg_best, factor_next, mode="longonly")
    r_stack_best_lf = composite_return(P_avg_best, factor_next, mode="longflat")

    for label, r in [("V8_stack_avg_linear", r_stack_lin),
                      ("V8_stack_avg_longflat", r_stack_lf),
                      ("V8_stack_avg_longonly", r_stack_lo),
                      ("V8_stack_BCbest_longonly", r_stack_best_lo),
                      ("V8_stack_BCbest_longflat", r_stack_best_lf)]:
        r2 = slice_window(r).reindex(static_r.index).fillna(0.0)
        variants[label] = r2
        # also a 50/50 with static
        variants[f"{label}_blend50static"] = 0.5 * static_r + 0.5 * r2
        # vol-target
        v = r2.std(ddof=1) * np.sqrt(12)
        if v > 1e-6:
            sc = target_vol / v
            variants[f"{label}_voltarget_x{sc:.2f}"] = sc * r2

    # ---------------- summarize ----------------
    rows = []
    for name, r in variants.items():
        r2 = r.loc[START:END].dropna()
        s = perf_with_cumret(r2)
        rows.append({
            "variant": name,
            "n_months": s.get("n_months"),
            "sharpe_ann": s.get("sharpe_ann"),
            "mean_ann": s.get("mean_ann"),
            "vol_ann": s.get("vol_ann"),
            "max_drawdown": s.get("max_drawdown"),
            "cumret_1to_X": s.get("cumret_1to_X"),
        })

    df = pd.DataFrame(rows).sort_values("cumret_1to_X", ascending=False).reset_index(drop=True)

    # Reference baselines for filtering
    static_cumret = (1 + static_r).prod()
    static_sharpe = static_r.mean() * 12 / (static_r.std(ddof=1) * np.sqrt(12))

    print("\n=== All variants (sorted by cumret descending) ===")
    pd.set_option("display.float_format", lambda x: f"{x: .4f}")
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 200)
    print(df.to_string(index=False))

    # Eligibility: cumret > Static FF5 AND Sharpe > 0.99
    eligible = df[(df["cumret_1to_X"] > static_cumret) &
                   (df["sharpe_ann"] > static_sharpe) &
                   (~df["variant"].isin(["Static_FF5"]))].copy()

    print(f"\nStatic FF5 reference: cumret={static_cumret:.4f}, Sharpe={static_sharpe:.4f}")
    print(f"\n=== Eligible (cumret > Static AND Sharpe > Static) ===")
    print(eligible.to_string(index=False))

    # Save summary table
    df.to_csv(TABLES / "EXPLORE_inertia_returns_variants.csv", index=False)
    print(f"\nSaved: {TABLES / 'EXPLORE_inertia_returns_variants.csv'}")

    # Pick winner: among eligible, highest Sharpe; tie-break on cumret
    if len(eligible) > 0:
        winner = eligible.sort_values(["sharpe_ann", "cumret_1to_X"], ascending=False).iloc[0]
        winner_name = winner["variant"]
        winner_r = variants[winner_name].loc[START:END].dropna()
        winner_r.to_csv(TABLES / "EXPLORE_inertia_high_return_best.csv",
                         header=["return"])
        print(f"\n*** WINNER: {winner_name} ***")
        print(f"  Sharpe={winner['sharpe_ann']:.4f}  "
              f"mean={winner['mean_ann']*100:.2f}%  "
              f"vol={winner['vol_ann']*100:.2f}%  "
              f"DD={winner['max_drawdown']*100:.2f}%  "
              f"cumret=$1 -> ${winner['cumret_1to_X']:.4f}")
        print(f"Saved winner returns: {TABLES / 'EXPLORE_inertia_high_return_best.csv'}")
    else:
        print("\nNo variant beat Static FF5 cumret AND Sharpe!")


if __name__ == "__main__":
    main()
