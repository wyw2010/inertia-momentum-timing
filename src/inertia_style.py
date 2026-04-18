"""
Inertia Capital chart style.

Adapted from the Agentic Learning ICML figure template —
Computer Modern serif, muted palette, 600 DPI, minimal spines.

Usage:
    from src.inertia_style import apply_style, C, SINGLE_COL, FULL_COL, save_fig
    apply_style()
    fig, ax = plt.subplots(figsize=(FULL_COL, 3.2))
    ...
    save_fig(fig, "baseline_cumret")
"""

from __future__ import annotations

import os
import matplotlib
import matplotlib.pyplot as plt


# -------------------------------------------------------------------
# Palette — Inertia brand (blue primary, purple accent, green/red P&L)
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
        # Font: Computer Modern to match LaTeX body text
        "font.family": "serif",
        "font.serif": ["cmr10", "Computer Modern Roman", "Times New Roman"],
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,

        # Sizes — generous for readability
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7,
        "legend.title_fontsize": 7,

        # Resolution + layout
        "figure.dpi": 600,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,

        # Grid — very subtle
        "axes.grid": True,
        "grid.alpha": 0.10,
        "grid.linewidth": 0.3,
        "grid.color": "#d0d0d0",

        # Axes — minimal
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
    """Save figure as both PDF (vector) and PNG (raster)."""
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(f"{out_dir}/{name}.pdf")
    fig.savefig(f"{out_dir}/{name}.png")
    plt.close(fig)
    print(f"  saved: {out_dir}/{name}.{{pdf,png}}")
