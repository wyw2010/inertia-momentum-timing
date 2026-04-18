# Inertia — Momentum Timing

A research project on **crash-resistant momentum**: detecting regime changes that predict momentum-strategy crashes, and dynamically cutting exposure.

Final project for "Data Driven Investing" (UG54), Spring 2026.

## Thesis

Equity momentum (Jegadeesh-Titman 1993) is one of the most reliable cross-sectional anomalies in history — but it is punctuated by rare, catastrophic crashes (Daniel-Moskowitz 2015) that can erase a decade of compounding in weeks. Inertia identifies the conditions preceding these crashes in real time and dynamically scales momentum exposure, turning static momentum into a conditionally-managed strategy with materially better risk-adjusted returns.

## Repo layout

```
inertia-momentum-timing/
├── src/                   # reusable modules
│   ├── data.py            # Ken French + FRED fetchers
│   ├── portfolio.py       # vol-targeting, weight construction
│   ├── regime.py          # regime classifier + OOS harness
│   ├── evaluation.py      # Sharpe bootstrap, alpha regressions
│   └── inertia_style.py   # chart styling
├── notebooks/
│   ├── 01_baseline_momentum.ipynb
│   ├── 02_daniel_moskowitz.ipynb
│   ├── 03_regime_detector.ipynb
│   ├── 04_portfolio_construction.ipynb
│   └── 99_final_report.ipynb
├── data/                  # cached raw + processed data
├── figures/               # exported charts (PDF + PNG)
├── prospectus/            # PDF write-up
└── slides/                # presentation
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/01_baseline_momentum.ipynb
```

## Principals

- TBD
- TBD

## Academic anchor

Daniel, K. and Moskowitz, T., 2016. *Momentum crashes.* Journal of Financial Economics, 122(2), pp.221-247.
