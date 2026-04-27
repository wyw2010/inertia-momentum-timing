# Figure and Table Index

Index of all figures and tables produced by `factor_regimes/notebooks/`. Slide-builder agent should consult this to know what visuals are available.

## Sprint 1 (Notebook 01: factor panel)

### Figures

| File | What it shows | Use on slide... |
|---|---|---|
| `figures/01_ff5_cumret.png` | Cumulative growth of $1 in each FF5 factor since 1963, log scale, 5 colored lines, end-of-line value labels | "Factor returns are different across decades" |
| `figures/02_ff5_correlation_heatmap.png` | 5x5 pairwise correlation heatmap for FF5 monthly returns | "Factors are diverse: each needs its own model" |
| `figures/03_ff5_decade_sharpe_bars.png` | Grouped bar chart of Sharpe ratio per factor per decade (1963-79, 1980-99, 2000-09, 2010-19, 2020-26) | "Factor premia are time-varying" (motivation for regime detection) |
| `figures/04_ff5_drawdown_smallmultiples.png` | 5-panel small-multiples chart of each factor's drawdown path, with max-DD annotation per panel | "Each factor has its own drawdown signature" |

### Tables

| File | What it shows |
|---|---|
| `tables/01_ff5_static_stats.{csv,md}` | Per-factor n_months, start, end, mean_ann, vol_ann, sharpe_ann, skew, excess_kurt, max_drawdown |
| `tables/01_ff5_correlation.{csv,md}` | 5x5 pairwise correlation matrix |
| `tables/02_ff5_decade_sharpe.{csv,md}` | Decade-by-factor Sharpe ratios in tabular form |

## Sprint 2 (coming next)

Method-A, Method-B, Method-C notebooks will add:

- Per-factor regime probability time series (line plots with shaded stressed regions)
- Per-factor predictive R² and information-coefficient tables
- Per-factor Sharpe of timed strategy vs static factor (paired bar chart)
- Confusion matrices for regime classification (Methods B, C)

## Sprint 3 (coming after)

- Ensemble scoreboard (the headline table)
- Paired Sharpe-difference vs static benchmark for each factor
- Risk-return scatter (timed vs static)
- Subsample Sharpe bar chart (decade by decade)

## Sprint 4 (final)

- MPSIF overlay demo: factor exposures of May Rebalance portfolio with and without the regime overlay
- Variance decomposition donut (matching MPSIF Figure 6)
- Factor-beta bar chart with one-sigma whiskers and significance stars (matching MPSIF Figure 5)
