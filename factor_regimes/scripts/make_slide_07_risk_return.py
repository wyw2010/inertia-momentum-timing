"""
Generate SLIDE_07 risk-return scatter for the FF5-strategy comprehensive scoreboard.

Plots annualized vol on x and annualized Sharpe on y, one bubble per strategy,
with labels positioned adjacent to each bubble (no collisions).
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.style import apply_style, C  # noqa: E402

TABLES = ROOT / "tables"
FIG_DIR = ROOT / "slides" / "figures"


def main() -> None:
    apply_style()

    sb = pd.read_csv(TABLES / "33_comprehensive_scoreboard.csv", index_col=0)

    # Plot only FF5 strategies (exclude MPSIF rows, which live on a different scale).
    ff5_strats = ["Static FF5", "Method A", "Method B", "Method C",
                  "Pure ensemble", "Inertia"]
    sb = sb.loc[ff5_strats]

    # Color and emphasis per strategy
    color_map = {
        "Static FF5":    C["blue"],
        "Method A":      C["muted"],
        "Method B":      C["green"],
        "Method C":      C["red"],
        "Pure ensemble": C["muted"],
        "Inertia":       C["purple"],
    }
    # Highlight winners with bigger size + black edge
    highlight = {"Method B", "Inertia"}

    # Per-label offset (in points) chosen so each label sits ~6-10pt from
    # its bubble and no two labels collide. Tune these by eye.
    #   "right":  ( 10,  0), va=center,  ha=left
    #   "left":   (-10,  0), va=center,  ha=right
    #   "above":  (  0,  9), va=bottom, ha=center
    #   "below":  (  0, -9), va=top,    ha=center
    label_pos = {
        "Static FF5":    {"xytext": ( 10,   0), "ha": "left",   "va": "center"},
        "Method A":      {"xytext": (  0,  -9), "ha": "center", "va": "top"},
        "Method B":      {"xytext": (  0,   9), "ha": "center", "va": "bottom"},
        "Method C":      {"xytext": (-10,   0), "ha": "right",  "va": "center"},
        "Pure ensemble": {"xytext": (  0,  -9), "ha": "center", "va": "top"},
        "Inertia":       {"xytext": (  0,   9), "ha": "center", "va": "bottom"},
    }

    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    for name, row in sb.iterrows():
        x = float(row["vol_ann"]) * 100  # percent
        y = float(row["sharpe_ann"])
        is_top = name in highlight
        size = 220 if is_top else 90
        edge = "#1F1F1F" if is_top else "none"
        ax.scatter(x, y, s=size, color=color_map[name],
                   edgecolor=edge, linewidth=1.0, zorder=3)

        pos = label_pos[name]
        ax.annotate(name,
                    xy=(x, y),
                    xytext=pos["xytext"],
                    textcoords="offset points",
                    ha=pos["ha"], va=pos["va"],
                    fontsize=10, color=color_map[name])

    ax.set_xlabel("Annualized volatility (%)")
    ax.set_ylabel("Annualized Sharpe ratio")
    ax.set_title("Risk-return: Method B and Inertia at the top", loc="left")
    ax.set_xlim(0, 7)
    ax.set_ylim(-0.05, 1.30)
    ax.grid(True, alpha=0.35)

    out = FIG_DIR / "SLIDE_07_risk_return_scatter.png"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  saved: {out}")


if __name__ == "__main__":
    main()
