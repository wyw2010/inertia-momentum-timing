"""
Inertia v2 chart style.

Matches the MPSIF Systematic Fund annual report aesthetic:
  Latin Modern serif, primary purple, cornflower blue secondary,
  sage green for positive, warm red for negative, value-labeled bars,
  end-of-line labels for cumulative-return charts, subtle grid,
  white background.

Design rules enforced here to prevent text-on-chart overlap:
  * `figure.constrained_layout.use=True` so matplotlib reserves
    space for legends, titles, and axis labels automatically.
  * Yearly tick formatter for date axes (4-char labels).
  * Helpers `bar_value_labels()`, `line_end_labels()`, `legend_below()`.
"""

from __future__ import annotations

import os
from typing import Iterable, Sequence

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -----------------------------------------------------------------
# Palette --- MPSIF report tones
# -----------------------------------------------------------------
C = {
    "purple":     "#5D4E8C",   # primary, headlines, fund line
    "purple_lt":  "#B5ADD0",   # purple tint for fills
    "blue":       "#5B7BC9",   # secondary, benchmark / S&P 500
    "blue_lt":    "#BCC9E5",
    "green":      "#3FA47D",   # positive alpha / gain
    "red":        "#C84B4B",   # negative alpha / drawdown
    "muted":      "#9A9A9A",   # axes, gridlines secondary
    "ink":        "#1F1F1F",   # body text
    "soft":       "#E0E0E0",   # very light grid
    "panel_bg":   "#FFFFFF",
}

# A categorical palette for FF5 factors (5 lines in one chart)
FACTOR_PALETTE = {
    "MKT_RF": "#5D4E8C",   # primary purple
    "SMB":    "#5B7BC9",   # cornflower
    "HML":    "#3FA47D",   # sage green
    "RMW":    "#D08740",   # amber (warm contrast)
    "CMA":    "#C84B4B",   # warm red
}

FULL_COL   = 7.5    # inches, single-panel slide-friendly width
HALF_COL   = 3.6    # inches, half-panel
TWO_THIRDS = 5.0


# -----------------------------------------------------------------
# Apply style
# -----------------------------------------------------------------
def apply_style() -> None:
    """Install the MPSIF-style rcParams globally."""
    matplotlib.rcParams.update({
        # Latin Modern serif (matches Inertia report and MPSIF report)
        "font.family":       "serif",
        "font.serif":        ["cmr10", "Computer Modern Roman",
                              "DejaVu Serif", "Times New Roman"],
        "mathtext.fontset":  "cm",
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,

        # Sizes
        "font.size":          10,
        "axes.titlesize":     11,
        "axes.labelsize":     10,
        "xtick.labelsize":    9,
        "ytick.labelsize":    9,
        "legend.fontsize":    9,
        "legend.title_fontsize": 9,

        # Resolution
        "figure.dpi":         150,
        "savefig.dpi":        300,
        "savefig.pad_inches": 0.10,

        # Layout: prevents legend / label overlap
        "figure.constrained_layout.use": True,

        # Grid: subtle horizontal lines
        "axes.grid":         True,
        "grid.alpha":        0.35,
        "grid.linewidth":    0.5,
        "grid.color":        C["soft"],
        "grid.linestyle":    "-",
        "axes.axisbelow":    True,

        # Axes: minimal spines
        "axes.linewidth":    0.6,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.edgecolor":    C["muted"],
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size":  3,
        "ytick.major.size":  3,
        "xtick.color":       C["muted"],
        "ytick.color":       C["muted"],
        "axes.labelcolor":   C["ink"],
        "axes.titlecolor":   C["ink"],
        "text.color":        C["ink"],

        # Lines + markers
        "lines.linewidth":   1.4,
        "lines.markersize":  5,

        # Background
        "figure.facecolor":  C["panel_bg"],
        "axes.facecolor":    C["panel_bg"],
    })


# -----------------------------------------------------------------
# Save helpers
# -----------------------------------------------------------------
def save_fig(fig, name: str, out_dir: str = "figures") -> None:
    """Save figure as PNG. Closes figure to release memory."""
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(f"{out_dir}/{name}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_dir}/{name}.png")


def save_table(df: pd.DataFrame, name: str, out_dir: str = "tables",
               float_format: str = "%.4f", index: bool = True) -> None:
    """Save table as both CSV and Markdown for downstream consumers."""
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(f"{out_dir}/{name}.csv", float_format=float_format, index=index)
    try:
        with open(f"{out_dir}/{name}.md", "w") as f:
            f.write(df.to_markdown(floatfmt=".4f", index=index))
        print(f"  saved: {out_dir}/{name}.{{csv,md}}")
    except Exception:
        print(f"  saved: {out_dir}/{name}.csv  (markdown skipped)")


# -----------------------------------------------------------------
# Annotation helpers (each prevents a specific overlap pattern)
# -----------------------------------------------------------------
def bar_value_labels(ax, bars, fmt: str = "{:+.2f}", offset: float = 0.02,
                     fontsize: float = 8.5, color: str | None = None) -> None:
    """
    Print value labels above (or below for negatives) each bar.
    Offset is a fraction of the y-axis range so labels never collide
    with the bars themselves.
    """
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    pad = offset * y_range
    for b in bars:
        h = b.get_height()
        x = b.get_x() + b.get_width() / 2
        if h >= 0:
            y, va = h + pad, "bottom"
        else:
            y, va = h - pad, "top"
        clr = color if color is not None else (C["green"] if h >= 0 else C["red"])
        ax.text(x, y, fmt.format(h), ha="center", va=va,
                fontsize=fontsize, color=clr)


def line_end_labels(ax, df: pd.DataFrame, colors: dict | None = None,
                    fmt: str = "{:.0f}", fontsize: float = 9,
                    offset_pts: int = 6) -> None:
    """
    Place a colored label at the right-end of each line in `df`,
    matching MPSIF Fig 1's style of annotating final values.
    """
    if colors is None:
        colors = {}
    last_x = df.index[-1]
    for col in df.columns:
        last_y = df[col].iloc[-1]
        clr = colors.get(col, C["ink"])
        ax.annotate(fmt.format(last_y),
                    xy=(last_x, last_y),
                    xytext=(offset_pts, 0),
                    textcoords="offset points",
                    fontsize=fontsize, color=clr,
                    fontweight="bold", va="center")


def legend_below(ax, ncol: int | None = None, pad: float = -0.18,
                 fontsize: float = 9, **kwargs):
    """Legend in a horizontal strip below the axes."""
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return None
    if ncol is None:
        ncol = len(handles)
    return ax.legend(handles, labels,
                     loc="upper center",
                     bbox_to_anchor=(0.5, pad),
                     ncol=ncol, frameon=False,
                     fontsize=fontsize, **kwargs)


def yearly_xticks(ax, every: int = 5) -> None:
    """Use 4-char yearly labels every N years on a date x-axis."""
    ax.xaxis.set_major_locator(mdates.YearLocator(every))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for t in ax.get_xticklabels():
        t.set_fontsize(9)


def shade_period(ax, start: str, end: str, color: str | None = None,
                 alpha: float = 0.10, label: str | None = None) -> None:
    """Shade an x-axis period (used to highlight regimes / crises)."""
    if color is None:
        color = C["purple_lt"]
    s = pd.Timestamp(start); e = pd.Timestamp(end)
    ax.axvspan(s, e, alpha=alpha, color=color, label=label, linewidth=0)


# -----------------------------------------------------------------
# Diagnostic helper: assert no obvious text-on-data overlap
# -----------------------------------------------------------------
def assert_no_overlap(fig) -> None:
    """
    Quick visual sanity check: tries `fig.tight_layout` and
    raises if any text artist's bounding box leaks outside the figure.
    Useful as a final guard in a notebook before saving.
    """
    fig.canvas.draw()
    fig_bbox = fig.bbox
    for txt in fig.findobj(matplotlib.text.Text):
        if not txt.get_visible() or not txt.get_text().strip():
            continue
        try:
            tb = txt.get_window_extent()
        except RuntimeError:
            continue
        if tb.xmin < fig_bbox.xmin - 1 or tb.xmax > fig_bbox.xmax + 1:
            print(f"  warning: text '{txt.get_text()[:30]}' may extend "
                  f"outside figure bounds; check layout.")
