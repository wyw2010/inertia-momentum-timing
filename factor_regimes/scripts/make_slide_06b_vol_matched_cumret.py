"""
Generate SLIDE_06b: vol-matched cumulative-return scoreboard chart.

For each scoreboard strategy, apply CONSTANT in-sample leverage to match
Static FF5's full-sample volatility (5.28% ann). Pay realistic futures
financing: RF + 5 bp/year on the levered (L-1) notional.

Formula:
    L = 0.0528 / vol_strategy_fullsample
    vol_matched_t = L * r_strategy_t - (L - 1) * (RF_t + 0.0005/12)

Then plot cumulative product (1+r) over 2000-02 to 2024-12.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))
from lib.style import apply_style, C  # noqa: E402
from factor_regimes.lib.data import get_ff5_monthly  # noqa: E402

TABLES = ROOT / "tables"
FIG_DIR = ROOT / "slides" / "figures"

TARGET_VOL = 0.0528           # Static FF5 full-sample annualized realized vol
SPREAD_ANN = 0.0005           # 5 bp/year financing on the levered notional


def main() -> None:
    apply_style()

    comp = pd.read_csv(TABLES / "38_comprehensive_returns.csv",
                       index_col=0, parse_dates=True)

    cols = ["Static FF5", "Method A", "Method B", "Method C",
            "Pure ensemble", "Inertia"]

    # RF aligned to comp index
    ff5 = get_ff5_monthly()
    rf = ff5["RF"].reindex(comp.index).fillna(0.0)
    spread_m = SPREAD_ANN / 12.0

    # Compute vol-matched returns for each strategy
    vm = pd.DataFrame(index=comp.index, columns=cols, dtype=float)
    leverages = {}
    for col in cols:
        r = comp[col]
        v = r.std(ddof=1) * np.sqrt(12)
        L = TARGET_VOL / v
        leverages[col] = L
        # vol_matched_t = L * r_t - (L - 1) * (RF_t + spread_m)
        vm[col] = L * r - (L - 1.0) * (rf + spread_m)

    cum = (1 + vm).cumprod()

    print("Constant in-sample leverages and final $:")
    for col in cols:
        print(f"  {col:18s}  vol={comp[col].std(ddof=1)*np.sqrt(12)*100:.2f}%  "
              f"L={leverages[col]:.3f}  cum_$={cum[col].iloc[-1]:.4f}")

    color_map = {
        "Static FF5":    C["blue"],
        "Method A":      C["muted"],
        "Method B":      C["muted"],
        "Method C":      C["muted"],
        "Pure ensemble": C["muted"],
        "Inertia":       C["purple"],
    }
    lw_map = {
        "Static FF5":    2.4,
        "Method A":      1.0,
        "Method B":      1.0,
        "Method C":      1.0,
        "Pure ensemble": 1.0,
        "Inertia":       2.6,
    }
    alpha_map = {
        "Static FF5":    1.0,
        "Method A":      0.55,
        "Method B":      0.55,
        "Method C":      0.55,
        "Pure ensemble": 0.55,
        "Inertia":       1.0,
    }
    z_map = {
        "Static FF5":    4,
        "Inertia":       5,
        "Method A":      2,
        "Method B":      2,
        "Method C":      2,
        "Pure ensemble": 2,
    }

    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    for col in cols:
        ax.plot(cum.index, cum[col],
                color=color_map[col], linewidth=lw_map[col],
                alpha=alpha_map[col], zorder=z_map[col],
                label=col)

    # End-value labels for highlighted lines
    last = cum.iloc[-1]
    for col in ["Static FF5", "Inertia"]:
        ax.annotate(f"{col}: ${last[col]:.2f}",
                    xy=(cum.index[-1], last[col]),
                    xytext=(8, 0),
                    textcoords="offset points",
                    fontsize=9, color=color_map[col],
                    fontweight="bold", va="center")

    ax.set_xlabel("")
    ax.set_ylabel("Cumulative wealth ($1 invested 2000-02)")
    ax.set_title("Risk-adjusted equity curves (vol-matched to FF5)",
                 loc="left")
    # Subtitle / footnote
    fig.text(0.01, -0.02,
             "Each strategy levered to Static FF5's volatility "
             "(5.28% ann) via index futures (RF + 5 bp financing).",
             fontsize=8.5, style="italic", color=C["muted"])

    ax.grid(True, alpha=0.35)
    ax.set_xlim(cum.index[0], cum.index[-1] + pd.Timedelta(days=720))

    handles, labels = ax.get_legend_handles_labels()
    order = ["Inertia", "Static FF5", "Method B", "Method C", "Method A",
             "Pure ensemble"]
    h_by_l = dict(zip(labels, handles))
    ax.legend([h_by_l[l] for l in order if l in h_by_l],
              [l for l in order if l in h_by_l],
              loc="upper left", frameon=False, fontsize=9, ncol=2)

    out = FIG_DIR / "SLIDE_06b_vol_matched_cumret.png"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  saved: {out}")


if __name__ == "__main__":
    main()
