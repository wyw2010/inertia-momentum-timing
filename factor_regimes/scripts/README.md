# WRDS Backtest Scripts

Scripts for backtesting the MPSIF rebalance strategy using point-in-time WRDS data.

## `backtest_mpsif_wrds.py`

Replays the rebalance code's monthly stock-selection logic from January 2000 through March 2026 using:

- **CRSP** monthly stock file (returns, prices, market cap)
- **CRSP** S&P 500 historical membership (no survivorship bias)
- **Compustat** quarterly fundamentals (revenue, EPS, book equity, operating income)
- **CRSP/Compustat link table** (`ccmxpf_linktable`) to join permno and gvkey

Each month it:

1. Filters the universe to S&P 500 members at that date
2. Computes 12-1 momentum from monthly returns
3. Fetches the most recent Compustat fundamentals released **before** the month-end (`rdq <= asof`) — strict point-in-time, no look-ahead
4. Computes z-scores per factor sleeve (momentum, growth, value), then a composite
5. Picks the top 50 stocks
6. Weights by `mktcap^alpha * (1 + gamma * z_combo)`, capped at 10% per name with redistribution
7. Holds for one month, then rebalances

This mirrors the logic in `mpsif_rebalance_code.py` exactly, except for QoQ growth which is computed from quarterly data here (vs the live code which also uses QoQ from the same source).

## How to run

### Option A: Locally with Python
```bash
pip install wrds pandas numpy scipy pyarrow
cd factor_regimes/scripts
python backtest_mpsif_wrds.py
```

The first run will prompt for your WRDS username, then send a Duo Mobile push to your phone. Approve it. The first run takes 5 to 10 minutes (downloads CRSP + Compustat + link table). Subsequent runs use cached parquet files in `wrds_cache/` and take about 1 minute.

### Option B: WRDS Cloud Jupyter
1. Log into [wrds-cloud.wharton.upenn.edu](https://wrds-cloud.wharton.upenn.edu) (Duo prompt on phone)
2. Upload `backtest_mpsif_wrds.py`
3. Open a terminal, `pip install pyarrow` if needed (other deps are pre-installed)
4. `python backtest_mpsif_wrds.py`
5. Download the resulting CSV from `wrds_output/`

## Outputs

Two CSVs in `wrds_output/`:

- `mpsif_backtest_returns.csv` — monthly portfolio return time series. Columns: `next_date`, `asof`, `ret`, `n_holdings`. Roughly 312 rows (2000-01 through 2026-03).
- `mpsif_backtest_weights.csv` — long-format table of every (date, permno, weight) plus per-stock z-scores and composite. Roughly 15,000 rows.

The console will print summary stats: annualized mean, vol, Sharpe, max drawdown.

## What to do with the output

After running, **commit the returns CSV to the repo**:

```bash
cp wrds_output/mpsif_backtest_returns.csv \
   /path/to/repo/factor_regimes/data_input_mpsif_real_returns.csv
git add factor_regimes/data_input_mpsif_real_returns.csv
git commit -m "Add 26-year MPSIF strategy backtest results from WRDS"
git push
```

Once the CSV is in the repo, I can plug the real returns into the overlay analysis and produce a proper apples-to-apples comparison: actual MPSIF rebalanced strategy with vs without the Inertia v2 overlay.

## Tunable parameters

All at the top of `backtest_mpsif_wrds.py`:

| Parameter | Default | Description |
|---|---|---|
| `START_DATE` | `1998-01-01` | Earliest data pulled (need 2y prior to first rebalance) |
| `FIRST_REBALANCE` | `2000-01-31` | First month-end traded |
| `LAST_REBALANCE` | `2026-03-31` | Last month traded |
| `TOP_N` | `50` | Number of stocks held |
| `ALPHA` | `0.0` | Market-cap weighting exponent (0 = equal weight) |
| `GAMMA` | `0.25` | Tilt strength toward higher composite scores |
| `TILT_FLOOR` | `0.20` | Minimum tilt multiplier |
| `WEIGHT_CAP` | `0.10` | Max weight per stock; excess is redistributed |

These match the live rebalance code.

## Caveats

1. **No transaction costs**: the backtest holds for one month and then fully reweights. No bid-ask spread or commission deducted. Realistic friction would reduce Sharpe by maybe 0.5 to 1.5 percentage points per year.
2. **Compustat fundq vs the live rebalance's `Ticker.info`**: small differences in how trailing-twelve-month metrics are computed. The Compustat path is more rigorous and point-in-time.
3. **FCF yield approximation**: Compustat fundq does not have a direct free cash flow column at the quarterly level. We use trailing 4-quarter operating income before depreciation (`oibdpq`) divided by market cap as a proxy. This differs slightly from the live code's `freeCashflow / marketCap`. The score's value sleeve is robust enough that this proxy should not materially change rankings.
4. **Delisted positions**: if a holding delists between rebalances we assume zero return for that month (conservative).
5. **First-run download time**: pulling 26 years of CRSP monthly data for 1500+ historical S&P 500 names plus Compustat fundq is roughly 50 to 100 MB of data and takes 5 to 10 minutes on the first run.

## Troubleshooting

**"No module named wrds"**: run `pip install wrds`.

**Duo Mobile timeout**: re-run; you have 60 seconds to approve the push.

**Slow query**: the first SQL pull is the longest. If it stalls, check that you have CRSP and Compustat subscriptions active in WRDS. Stern provides both by default.

**Memory issues**: the CRSP monthly file is ~5M rows. If you hit OOM, the script can be modified to filter to a smaller universe upfront — let me know.
