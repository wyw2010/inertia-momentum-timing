# Inertia v2: Factor Regime Detection

Project pivot from Inertia v1 (momentum timing). Same fund name. New mandate: detect regime changes in major Fama-French factors and use the signals to time factor exposures.

## Folder layout

```
factor_regimes/
├── lib/                       Self-contained Python modules for v2
│   ├── data.py                FF5 + macro panel (reuses v1's Ken French and FRED fetchers)
│   ├── style.py               Chart style matching the MPSIF annual report aesthetic
│   ├── methods.py             [coming Sprint 2] Methods A, B, C and ensemble
│   ├── backtest.py            [coming Sprint 2] OOS expanding-window harness
│   └── evaluation.py          [coming Sprint 2] Sharpe bootstrap, paired tests, alpha regressions
├── notebooks/                 End-to-end analysis, run in order
│   ├── 01_factor_panel.ipynb              Static FF5 stats + visual exploration
│   ├── 02_method_a_markov.ipynb           [coming] Markov regime-switching regression (Hamilton 1989)
│   ├── 03_method_b_ridge.ipynb            [coming] Predictive regression with macro features
│   ├── 04_method_c_gbm.ipynb              [coming] Gradient-boosted regime classifier
│   ├── 05_ensemble_scoreboard.ipynb       [coming] Ensemble + paired Sharpe test
│   └── 06_mpsif_overlay.ipynb             [coming] Apply signals to May Rebalance portfolio
├── figures/                   PNG charts saved at 300 DPI in MPSIF style
└── tables/                    CSV + Markdown tables
```

## Universe and OOS sample

| Setting | Value |
|---|---|
| Factors | FF5 (`MKT_RF`, `SMB`, `HML`, `RMW`, `CMA`) |
| Frequency | Monthly |
| Sample start | 1963-07 |
| Sample end | most recent month |
| OOS start | 1990-01 |
| Refit cadence | Annual, expanding window |

## Reusing v1 infrastructure

`lib/data.py` adds the repo root to `sys.path` and imports v1's `src.data.get_ff5` and `src.data.get_fred_panel`. No code is duplicated. Cached files in `data/raw/` (Ken French zips, FRED CSVs) are shared between v1 and v2.

## Style

Charts in `figures/` follow the MPSIF annual report aesthetic:

- Latin Modern serif throughout
- Primary purple `#5D4E8C`, secondary cornflower blue `#5B7BC9`
- Sage green `#3FA47D` for gains, warm red `#C84B4B` for losses
- Per-factor palette in `style.FACTOR_PALETTE`
- Yearly tick formatter on date axes (4-character labels, no overlap)
- End-of-line value labels for cumulative-return charts (no overlap with chart body)
- `figure.constrained_layout=True` globally to prevent legend or label collision

## Running the notebooks

```bash
cd factor_regimes/notebooks
jupyter nbconvert --to notebook --execute 01_factor_panel.ipynb --output 01_factor_panel.ipynb
```

Each notebook saves its figures and tables to the shared `figures/` and `tables/` folders, ready for downstream consumption (the slide-builder agent reads these directly without needing to re-execute).
