"""
MPSIF Strategy Backtest using WRDS (CRSP + Compustat)
=====================================================

Replays the MPSIF rebalance code's logic over 2000 to 2026 using
point-in-time data: CRSP monthly returns, Compustat quarterly
fundamentals, and CRSP S&P 500 historical membership.

The output is a monthly portfolio return time series that can be
compared directly against an Inertia v2 overlay.

PREREQUISITES
-------------
WRDS account with CRSP + Compustat subscriptions (NYU Stern has these).
Python 3.9+ with: wrds, pandas, numpy, scipy.

USAGE
-----
1. pip install wrds pandas numpy scipy
2. python backtest_mpsif_wrds.py
   (a Duo Mobile push will arrive on first login; approve on phone)
3. Output CSV will be saved to OUT_PATH (configurable below).

Re-runs after the first successful execution use cached pulls and
take roughly one minute.

OUTPUTS
-------
Two CSVs in OUT_DIR:
  mpsif_backtest_returns.csv   monthly portfolio return time series
  mpsif_backtest_weights.csv   monthly weights for the top-50 holdings

The returns CSV is what we need for the overlay analysis. Commit it
back to the repo at factor_regimes/data_input_mpsif_real_returns.csv.
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================
START_DATE      = "1998-01-01"   # 2y prior to first rebalance for momentum window
END_DATE        = "2026-04-30"
FIRST_REBALANCE = "2000-01-31"   # first month-end where we trade
LAST_REBALANCE  = "2026-03-31"   # last month with full following-month return

TOP_N           = 50             # hold top-N stocks
ALPHA           = 0.0            # mktcap weighting exponent (matches rebalance code)
GAMMA           = 0.25           # tilt strength
TILT_FLOOR      = 0.20           # minimum tilt multiplier
WEIGHT_CAP      = 0.10           # max single-name weight, then redistribute

OUT_DIR    = Path("./wrds_output")
CACHE_DIR  = Path("./wrds_cache")
OUT_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)


# ============================================================
# WRDS connection + cached SQL pulls
# ============================================================
def connect():
    import wrds
    print("Connecting to WRDS (approve Duo prompt on your phone)...")
    return wrds.Connection()


def load_or_query(name: str, query: str, db) -> pd.DataFrame:
    cache_path = CACHE_DIR / f"{name}.parquet"
    if cache_path.exists():
        print(f"  loading {name} from cache: {cache_path}")
        return pd.read_parquet(cache_path)
    print(f"  querying {name} from WRDS...")
    t0 = time.time()
    df = db.raw_sql(query)
    print(f"    {len(df):,} rows in {time.time() - t0:.1f}s")
    df.to_parquet(cache_path)
    return df


def pull_data(db) -> dict:
    print("\n=== Pulling raw data from WRDS ===")

    sp500 = load_or_query("sp500_history", """
        SELECT permno, start::date AS sp500_start, ending::date AS sp500_end
        FROM crsp.msp500list
    """, db)

    msf = load_or_query("crsp_msf", f"""
        SELECT permno, date::date AS date,
               ret, abs(prc) AS prc, shrout,
               (abs(prc) * shrout) / 1000.0 AS mcap_mil
        FROM crsp.msf
        WHERE date BETWEEN '{START_DATE}' AND '{END_DATE}'
          AND permno IN (SELECT DISTINCT permno FROM crsp.msp500list)
    """, db)

    fundq = load_or_query("comp_fundq", f"""
        SELECT gvkey, datadate::date AS datadate, rdq::date AS rdq,
               revtq, epspxq, epsfxq, ceqq, niq, oibdpq,
               atq, ltq, oancfy
        FROM comp.fundq
        WHERE datadate BETWEEN '{START_DATE}' AND '{END_DATE}'
          AND fic = 'USA'
          AND consol = 'C' AND popsrc = 'D' AND indfmt = 'INDL' AND datafmt = 'STD'
    """, db)

    link = load_or_query("ccm_link", """
        SELECT gvkey, lpermno AS permno,
               linkdt::date AS linkdt,
               COALESCE(linkenddt::date, '2099-12-31'::date) AS linkenddt
        FROM crsp.ccmxpf_linktable
        WHERE linktype IN ('LU', 'LC')
          AND linkprim IN ('P', 'C')
          AND lpermno IS NOT NULL
    """, db)

    return {"sp500": sp500, "msf": msf, "fundq": fundq, "link": link}


# ============================================================
# Build the analysis panel
# ============================================================
def build_fundq_panel(fundq: pd.DataFrame, link: pd.DataFrame) -> pd.DataFrame:
    """
    Join Compustat fundq with CRSP permno via the link table, keep
    only the rows where datadate falls inside the link window. Add
    YoY and QoQ growth metrics for revenue and EPS, computed within
    each (permno) group so we never cross firms.
    """
    df = fundq.merge(link, on="gvkey", how="inner")
    df = df[(df["datadate"] >= df["linkdt"]) & (df["datadate"] <= df["linkenddt"])]

    # Some firms have multiple gvkey-permno mappings; keep earliest rdq per (permno, datadate)
    df = (df.sort_values(["permno", "datadate", "rdq"])
            .drop_duplicates(subset=["permno", "datadate"], keep="first")
            .reset_index(drop=True))

    # Use diluted EPS if available, else basic
    df["eps"] = df["epsfxq"].fillna(df["epspxq"])

    # Sort then compute lagged values within each permno
    df = df.sort_values(["permno", "datadate"]).reset_index(drop=True)
    g = df.groupby("permno", group_keys=False)

    # QoQ: divide by previous quarter
    df["rev_qoq"] = g["revtq"].apply(lambda s: s.pct_change(periods=1))
    df["eps_qoq"] = g["eps"  ].apply(lambda s: s.pct_change(periods=1))
    # YoY: divide by 4 quarters back
    df["rev_yoy"] = g["revtq"].apply(lambda s: s.pct_change(periods=4))
    df["eps_yoy"] = g["eps"  ].apply(lambda s: s.pct_change(periods=4))

    # rdq is the public release date. Some firms have NaN rdq; impute datadate + 60 days.
    df["rdq"] = df["rdq"].fillna(df["datadate"] + pd.Timedelta(days=60))

    return df


def universe_at(asof: pd.Timestamp, sp500: pd.DataFrame) -> set:
    """Set of permnos in the S&P 500 on `asof`."""
    m = sp500[(sp500["sp500_start"] <= asof) & (sp500["sp500_end"] >= asof)]
    return set(m["permno"].astype(int))


def momentum_12_1(msf: pd.DataFrame, asof: pd.Timestamp,
                  permnos: set) -> pd.Series:
    """
    12-1 momentum: cumulative return from t-12 to t-1 for each permno.
    Skip the most recent month (the standard academic definition).
    """
    t12 = asof - pd.DateOffset(months=12)
    t1  = asof - pd.DateOffset(months=1)
    win = msf[(msf["date"] >= t12) & (msf["date"] <= t1)
              & msf["permno"].isin(permnos)]
    return win.groupby("permno")["ret"].apply(
        lambda x: (1 + x).prod() - 1
    )


def latest_fundamentals(fundq: pd.DataFrame, asof: pd.Timestamp,
                        permnos: set) -> pd.DataFrame:
    """
    For each permno, return the most recent fundq row whose rdq is at
    or before `asof` (point-in-time). This avoids look-ahead bias.
    """
    candidates = fundq[(fundq["rdq"] <= asof)
                       & fundq["permno"].isin(permnos)]
    return (candidates.sort_values("rdq")
                      .groupby("permno")
                      .tail(1)
                      .set_index("permno"))


def winsorize(s: pd.Series, lo=0.01, hi=0.99) -> pd.Series:
    s = s.copy()
    a, b = s.quantile(lo), s.quantile(hi)
    return s.clip(a, b)


def zscore(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    mu, sd = np.nanmean(s), np.nanstd(s)
    if sd == 0 or not np.isfinite(sd):
        return pd.Series(np.nan, index=s.index)
    return (s - mu) / sd


def cap_and_redistribute(w: pd.Series, cap: float = WEIGHT_CAP,
                         tol: float = 1e-12, max_iter: int = 200) -> pd.Series:
    w = w.copy().astype(float)
    w = w / w.sum()
    for _ in range(max_iter):
        over = w > cap + tol
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        under = w < cap - tol
        if not under.any():
            return w / w.sum()
        under_sum = w[under].sum()
        if under_sum <= tol:
            w.loc[under] += excess / under.sum()
        else:
            w.loc[under] += w.loc[under] / under_sum * excess
        w = w / w.sum()
    return w


# ============================================================
# Single-month rebalance (mirrors the live rebalance code)
# ============================================================
def rebalance_month(asof: pd.Timestamp, sp500, msf, fundq) -> pd.DataFrame:
    """
    Replicates the rebalance code's logic exactly:
      * mktcap and momentum from CRSP at month-end
      * fundamentals from latest publicly-available Compustat row
      * compute z-scores per factor sleeve, then composite
      * pick top N, weight by mktcap^alpha * (1 + gamma*z_combo) clipped
      * cap each name at WEIGHT_CAP and redistribute
    """
    permnos = universe_at(asof, sp500)
    if len(permnos) < TOP_N + 5:
        return pd.DataFrame()

    # Price / mcap row at this month-end
    px = msf[(msf["date"] == asof) & msf["permno"].isin(permnos)
             & msf["mcap_mil"].notna() & (msf["mcap_mil"] > 0)].set_index("permno")
    if len(px) < TOP_N + 5:
        return pd.DataFrame()

    # Momentum
    mom = momentum_12_1(msf, asof, permnos)

    # Latest fundamentals as of asof
    fnd = latest_fundamentals(fundq, asof, permnos)

    # Build cross-section
    df = pd.DataFrame(index=px.index)
    df["mcap"]      = px["mcap_mil"]
    df["price"]     = px["prc"]
    df["mom_12_1"]  = mom

    df = df.join(fnd[["rev_qoq","rev_yoy","eps_qoq","eps_yoy",
                      "eps","ceqq","oibdpq"]], how="left")

    # Value metrics. epsq is per-share; price is per-share.
    df["eps_ttm"]        = 4 * df["eps"]
    df["earnings_yield"] = df["eps_ttm"] / df["price"]
    # book_to_price = book_value / market_cap. ceqq is total book equity (M).
    df["book_to_price"]  = df["ceqq"] / df["mcap"]
    # FCF proxy: trailing 4-quarter operating income (oibdpq) / mcap
    # (proper FCF requires capex; oibdp is a reasonable proxy in fundq)
    df["fcf_yield"]      = df["oibdpq"] * 4 / df["mcap"]

    # Winsorize raw factor inputs
    for c in ["rev_qoq","rev_yoy","eps_qoq","eps_yoy",
              "earnings_yield","book_to_price","fcf_yield","mom_12_1"]:
        df[c] = winsorize(df[c])

    # Z-scores
    df["z_mom"] = zscore(df["mom_12_1"])
    growth_cols = ["rev_qoq","rev_yoy","eps_qoq","eps_yoy"]
    for c in growth_cols:
        df[f"z_{c}"] = zscore(df[c])
    df["z_growth"] = df[[f"z_{c}" for c in growth_cols]].mean(axis=1, skipna=True)
    has_rev = df[["z_rev_qoq","z_rev_yoy"]].notna().any(axis=1)
    has_eps = df[["z_eps_qoq","z_eps_yoy"]].notna().any(axis=1)
    df.loc[~(has_rev & has_eps), "z_growth"] = np.nan

    value_cols = ["earnings_yield","book_to_price","fcf_yield"]
    for c in value_cols:
        df[f"z_{c}"] = zscore(df[c])
    df["z_value"] = df[[f"z_{c}" for c in value_cols]].mean(axis=1, skipna=True)

    # Composite (require all 3 sleeves non-null)
    df["combo"] = df[["z_mom","z_growth","z_value"]].mean(axis=1, skipna=False)

    df = df.dropna(subset=["combo","mcap"])
    df = df[df["mcap"] > 0]
    if len(df) < TOP_N:
        return pd.DataFrame()

    # Pick top N
    top = df.sort_values("combo", ascending=False).head(TOP_N).copy()
    top["z_combo"] = zscore(top["combo"])

    base = top["mcap"].astype(float) ** ALPHA
    tilt = (1.0 + GAMMA * top["z_combo"].astype(float)).clip(lower=TILT_FLOOR)
    raw  = (base * tilt).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if raw.sum() <= 0:
        w = pd.Series(np.ones(len(top)) / len(top), index=top.index)
    else:
        w = raw / raw.sum()
    top["weight"] = cap_and_redistribute(w)

    out = top[["mcap","mom_12_1","z_mom","z_growth","z_value","combo","weight"]].reset_index()
    out["asof"] = asof
    return out


# ============================================================
# Main backtest
# ============================================================
def run_backtest():
    db = connect()
    try:
        data = pull_data(db)
    finally:
        db.close()

    print("\n=== Building panel ===")
    fundq = build_fundq_panel(data["fundq"], data["link"])
    msf   = data["msf"]
    sp500 = data["sp500"]

    # Convert dtypes
    msf["permno"] = msf["permno"].astype(int)
    msf["date"]   = pd.to_datetime(msf["date"])
    fundq["permno"] = fundq["permno"].astype(int)
    fundq["datadate"] = pd.to_datetime(fundq["datadate"])
    fundq["rdq"] = pd.to_datetime(fundq["rdq"])
    sp500["permno"] = sp500["permno"].astype(int)
    sp500["sp500_start"] = pd.to_datetime(sp500["sp500_start"])
    sp500["sp500_end"]   = pd.to_datetime(sp500["sp500_end"])

    # Calendar of month-ends (use CRSP month-end dates so they align with msf)
    msf_dates = sorted(msf["date"].unique())
    rebalance_dates = [d for d in msf_dates
                       if pd.Timestamp(FIRST_REBALANCE) <= d <= pd.Timestamp(LAST_REBALANCE)]
    print(f"  rebalance dates: {len(rebalance_dates)}  "
          f"({rebalance_dates[0].date()} to {rebalance_dates[-1].date()})")

    print("\n=== Monthly rebalance loop ===")
    all_weights = []
    for i, asof in enumerate(rebalance_dates):
        wts = rebalance_month(asof, sp500, msf, fundq)
        if wts.empty:
            continue
        all_weights.append(wts)
        if i % 12 == 0:
            print(f"  {asof.date()}: {len(wts)} holdings, "
                  f"top weight {wts['weight'].max():.3f}")

    weights = pd.concat(all_weights, ignore_index=True)

    # Compute next-month return for each held position
    print("\n=== Computing portfolio returns ===")
    msf_idx = msf.set_index(["permno","date"])["ret"]

    rows = []
    for asof, grp in weights.groupby("asof"):
        # Find the next rebalance date in the calendar (strict >)
        future_idx = next((d for d in rebalance_dates if d > asof), None)
        if future_idx is None:
            continue
        # Sum of weight * realized monthly return between asof and future_idx
        # Returns held one month: from asof to future_idx
        rets = []
        for _, row in grp.iterrows():
            try:
                r = msf_idx.loc[(int(row["permno"]), future_idx)]
                rets.append(row["weight"] * float(r))
            except KeyError:
                rets.append(0.0)  # delisted or missing — assume zero (safest)
        rows.append({"asof": asof,
                     "next_date": future_idx,
                     "ret": float(np.sum(rets)),
                     "n_holdings": len(grp)})
    returns = pd.DataFrame(rows).set_index("next_date").sort_index()

    print(f"\n  {len(returns)} monthly returns, "
          f"{returns.index.min().date()} to {returns.index.max().date()}")
    print(f"  mean   ann: {returns['ret'].mean()*12*100:6.2f}%")
    print(f"  vol    ann: {returns['ret'].std(ddof=1)*np.sqrt(12)*100:6.2f}%")
    sharpe = (returns['ret'].mean()*12) / (returns['ret'].std(ddof=1)*np.sqrt(12))
    print(f"  Sharpe ann: {sharpe:6.3f}")
    cum = (1 + returns['ret']).cumprod()
    dd = cum / cum.cummax() - 1
    print(f"  max DD    : {dd.min()*100:6.2f}%")

    # Save
    ret_path = OUT_DIR / "mpsif_backtest_returns.csv"
    wts_path = OUT_DIR / "mpsif_backtest_weights.csv"
    returns.to_csv(ret_path)
    weights.to_csv(wts_path, index=False)
    print(f"\n  saved: {ret_path}")
    print(f"  saved: {wts_path}")

    print("\n=== Done. Commit mpsif_backtest_returns.csv to the repo at ===")
    print("  factor_regimes/data_input_mpsif_real_returns.csv")


if __name__ == "__main__":
    run_backtest()
