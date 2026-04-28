"""
Generate SLIDE_06 cumulative-return scoreboard chart.

Shows $1 invested at 2000-02 growing through 2024-12 across the FF5-strategy
scoreboard (Static FF5, Method A/B/C, Pure ensemble, Inertia). Inertia is the
THICK PURPLE line, Static FF5 is the BLUE line, other methods are muted gray.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.style import apply_style, C, line_end_labels  # noqa: E402

TABLES = ROOT / "tables"
FIG_DIR = ROOT / "slides" / "figures"


def main() -> None:
    apply_style()

    comp = pd.read_csv(TABLES / "38_comprehensive_returns.csv",
                       index_col=0, parse_dates=True)

    cols = ["Static FF5", "Method A", "Method B", "Method C",
            "Pure ensemble", "Inertia"]
    cum = (1 + comp[cols]).cumprod()

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

    # Final-value labels at right edge for the highlighted lines
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
    ax.set_title("Strategy scoreboard: cumulative wealth, 2000-02 to 2024-12",
                 loc="left")
    ax.grid(True, alpha=0.35)
    # Make room for the right-edge labels
    ax.set_xlim(cum.index[0], cum.index[-1] + pd.Timedelta(days=720))

    # Legend with only the named methods, in the desired order
    handles, labels = ax.get_legend_handles_labels()
    order = ["Inertia", "Static FF5", "Method B", "Method C", "Method A",
            "Pure ensemble"]
    h_by_l = dict(zip(labels, handles))
    ax.legend([h_by_l[l] for l in order if l in h_by_l],
              [l for l in order if l in h_by_l],
              loc="upper left", frameon=False, fontsize=9, ncol=2)

    out = FIG_DIR / "SLIDE_06_scoreboard_cumret.png"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  saved: {out}")


if __name__ == "__main__":
    main()
