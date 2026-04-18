"""
Inertia Capital chart + table style.

Adapted from the Agentic Learning ICML figure template ---
Computer Modern serif, muted palette, 1200 DPI output, minimal spines.

Usage:
    from src.inertia_style import (
        apply_style, C, SINGLE_COL, FULL_COL,
        save_fig, save_table, legend_below,
    )
    apply_style()
    fig, ax = plt.subplots(figsize=(FULL_COL, 3.2))
    ...
    legend_below(ax)
    save_fig(fig, "baseline_cumret")
    save_table(perf_df, "baseline_performance")
"""

from __future__ import annotations

import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd


# -------------------------------------------------------------------
# Palette --- Inertia brand (blue primary, purple accent, green/red P&L)
# -------------------------------------------------------------------
C = {
    "blue":       "#4A7BF7",   # primary
    "dark":       "#1E2A3A",   # headline text / near-black
    "purple":     "#7B5EA7",   # secondary
    "green":      "#2E9E6E",   # positive / gain
    "red":        "#D94F4F",   # negative / crash
    "muted":      "#A0A8B4",   # benchmark / baseline line
    "light_blue": "#E8EEFB",
    "light_purp": "#EEEAF4",
    "bg_empty":   "#F5F6F8",
}

# Academic column widths (inches)
SINGLE_COL = 3.25
FULL_COL   = 6.75


def apply_style() -> None:
    """Apply the Inertia matplotlib style globally."""
    matplotlib.rcParams.update({
        # Font: Computer Modern with DejaVu Serif / Times fallback so
        # unicode glyphs (en-dash, etc.) that cmr10 lacks still render.
        "font.family": "serif",
        "font.serif": [
            "cmr10", "Computer Modern Roman",
            "DejaVu Serif", "Times New Roman",
        ],
        "mathtext.fontset": "cm",
        "mathtext.fallback": "cm",
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,

        # Sizes --- generous for readability
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7,
        "legend.title_fontsize": 7,

        # Resolution --- fast preview, high-res save
        "figure.dpi": 300,
        "savefig.dpi": 1200,
        "savefig.pad_inches": 0.08,

        # Layout --- constrained_layout handles spacing for outside legends
        "figure.constrained_layout.use": True,

        # Grid --- very subtle
        "axes.grid": True,
        "grid.alpha": 0.10,
        "grid.linewidth": 0.3,
        "grid.color": "#d0d0d0",

        # Axes --- minimal
        "axes.linewidth": 0.4,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#999999",
        "xtick.major.width": 0.4,
        "ytick.major.width": 0.4,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.color": "#555555",
        "ytick.color": "#555555",
        "axes.labelcolor": "#222222",
        "text.color": "#222222",

        # Lines
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
    })


def save_fig(fig, name: str, out_dir: str = "figures") -> None:
    """Save figure as both PDF (vector) and PNG (high-DPI raster)."""
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(f"{out_dir}/{name}.pdf")
    fig.savefig(f"{out_dir}/{name}.png")
    plt.close(fig)
    print(f"  saved: {out_dir}/{name}.{{pdf,png}}")


def legend_below(ax, ncol: int | None = None, pad: float = -0.18, **kwargs):
    """
    Place a legend below the axes, outside the plot area.

    `pad` is the y-offset in axes fraction (negative = below axes).
    `ncol` defaults to one column per handle (horizontal strip).
    """
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return None
    if ncol is None:
        ncol = len(handles)
    return ax.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, pad),
        ncol=ncol,
        frameon=False,
        **kwargs,
    )


def save_table(
    df: pd.DataFrame,
    name: str,
    out_dir: str = "tables",
    float_format: str = "%.4f",
    index: bool = True,
) -> None:
    """
    Persist a DataFrame as CSV (reproducibility), Markdown (prospectus draft),
    and LaTeX (prospectus typeset). Silently skips formats that error.
    """
    os.makedirs(out_dir, exist_ok=True)

    csv_path = f"{out_dir}/{name}.csv"
    df.to_csv(csv_path, float_format=float_format, index=index)
    formats_written = ["csv"]

    try:
        with open(f"{out_dir}/{name}.md", "w") as f:
            f.write(df.to_markdown(floatfmt=".4f", index=index))
        formats_written.append("md")
    except Exception as e:
        print(f"  (skipped md for {name}: {e})")

    try:
        df.to_latex(
            f"{out_dir}/{name}.tex",
            float_format=float_format,
            index=index,
            escape=False,
        )
        formats_written.append("tex")
    except Exception as e:
        print(f"  (skipped tex for {name}: {e})")

    exts = ",".join(formats_written)
    print(f"  saved: {out_dir}/{name}.{{{exts}}}")
