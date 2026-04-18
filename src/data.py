"""
Data fetchers for Inertia Capital.

Primary sources:
  - Ken French Data Library  (UMD, FF5 factors, risk-free rate)
  - FRED                      (VIX, yield spreads, credit spreads)

All series returned as monthly DataFrames indexed by month-end Timestamps,
in DECIMAL returns (not percent). Cached locally to data/raw/.
"""

from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import urllib.request as _urlreq
except ImportError:  # pragma: no cover
    import urllib2 as _urlreq  # type: ignore


REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR   = REPO_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Ken French data library
# ---------------------------------------------------------------------------
KF_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"

KF_FILES = {
    "ff3":  f"{KF_BASE}/F-F_Research_Data_Factors_CSV.zip",
    "ff5":  f"{KF_BASE}/F-F_Research_Data_5_Factors_2x3_CSV.zip",
    "umd":  f"{KF_BASE}/F-F_Momentum_Factor_CSV.zip",
}


def _download(url: str, cache_name: str, force: bool = False) -> bytes:
    """Download a URL (cached) and return raw bytes."""
    cache_path = RAW_DIR / cache_name
    if cache_path.exists() and not force:
        return cache_path.read_bytes()
    req = _urlreq.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with _urlreq.urlopen(req, timeout=60) as resp:
        data = resp.read()
    cache_path.write_bytes(data)
    return data


def _parse_kf_zip(raw_bytes: bytes) -> str:
    """Extract the single CSV from a Ken French ZIP file."""
    with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise ValueError("No CSV found in Ken French zip")
        return zf.read(names[0]).decode("latin-1")


def _parse_kf_monthly(csv_text: str) -> pd.DataFrame:
    """
    Parse a Ken French monthly CSV. The file has:
      - Some header/license lines
      - A monthly block (6-digit YYYYMM dates)
      - A blank line
      - An annual block (4-digit YYYY)
    We keep only the monthly block.
    """
    lines = csv_text.splitlines()

    # Find first line that starts with a 6-digit YYYYMM (first data row)
    start = None
    for i, line in enumerate(lines):
        token = line.split(",")[0].strip()
        if token.isdigit() and len(token) == 6:
            start = i
            break
    if start is None:
        raise ValueError("Could not locate monthly data block in Ken French CSV")

    # Header is the nearest non-empty line above start
    header_idx = start - 1
    while header_idx > 0 and not lines[header_idx].strip():
        header_idx -= 1
    header = ["date"] + [c.strip() for c in lines[header_idx].split(",")[1:]]

    # Read monthly rows until we hit a non-6-digit row (annual block)
    rows = []
    for line in lines[start:]:
        parts = [p.strip() for p in line.split(",")]
        if not parts or not parts[0].isdigit() or len(parts[0]) != 6:
            break
        rows.append(parts)

    df = pd.DataFrame(rows, columns=header)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m") + pd.offsets.MonthEnd(0)
    df = df.set_index("date")
    # Ken French returns are in percent; convert to decimal
    df = df.astype(float) / 100.0
    return df


def get_ff_momentum(force: bool = False) -> pd.DataFrame:
    """
    Fetch Ken French momentum factor (UMD).

    Returns a DataFrame indexed by month-end with columns:
      - UMD : monthly momentum long-short return (decimal)
    """
    raw = _download(KF_FILES["umd"], "kf_umd.zip", force=force)
    csv = _parse_kf_zip(raw)
    df = _parse_kf_monthly(csv)
    # Column name in file is "Mom   " with trailing spaces
    df.columns = [c.strip() for c in df.columns]
    mom_col = [c for c in df.columns if c.lower().startswith("mom")][0]
    return df[[mom_col]].rename(columns={mom_col: "UMD"})


def get_ff3(force: bool = False) -> pd.DataFrame:
    """
    Fetch Fama-French 3-factor file.

    Returns DataFrame indexed by month-end with columns:
      - MKT_RF, SMB, HML, RF  (all decimal)
    """
    raw = _download(KF_FILES["ff3"], "kf_ff3.zip", force=force)
    csv = _parse_kf_zip(raw)
    df = _parse_kf_monthly(csv)
    df.columns = [c.strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        low = c.lower()
        if low.startswith("mkt"):
            rename[c] = "MKT_RF"
        elif low == "smb":
            rename[c] = "SMB"
        elif low == "hml":
            rename[c] = "HML"
        elif low == "rf":
            rename[c] = "RF"
    return df.rename(columns=rename)[["MKT_RF", "SMB", "HML", "RF"]]


def get_ff5(force: bool = False) -> pd.DataFrame:
    """
    Fetch Fama-French 5-factor file.

    Columns: MKT_RF, SMB, HML, RMW, CMA, RF  (decimal)
    """
    raw = _download(KF_FILES["ff5"], "kf_ff5.zip", force=force)
    csv = _parse_kf_zip(raw)
    df = _parse_kf_monthly(csv)
    df.columns = [c.strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        low = c.lower()
        if low.startswith("mkt"):
            rename[c] = "MKT_RF"
        elif low == "smb":
            rename[c] = "SMB"
        elif low == "hml":
            rename[c] = "HML"
        elif low == "rmw":
            rename[c] = "RMW"
        elif low == "cma":
            rename[c] = "CMA"
        elif low == "rf":
            rename[c] = "RF"
    return df.rename(columns=rename)[["MKT_RF", "SMB", "HML", "RMW", "CMA", "RF"]]


def get_factor_panel(force: bool = False) -> pd.DataFrame:
    """
    Assemble a unified factor panel.

    Columns: MKT_RF, SMB, HML, RMW, CMA, UMD, RF   (monthly, decimal)
    """
    ff5 = get_ff5(force=force)
    umd = get_ff_momentum(force=force)
    panel = ff5.join(umd, how="inner")
    # Re-order
    return panel[["MKT_RF", "SMB", "HML", "RMW", "CMA", "UMD", "RF"]]


# ---------------------------------------------------------------------------
# FRED macro features (for regime detector in Sprint 3)
# ---------------------------------------------------------------------------
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"

FRED_SERIES = {
    "VIX":       "VIXCLS",       # CBOE Volatility Index (daily)
    "TERM":      "T10Y3M",       # 10y - 3m spread (daily)
    "BAA_AAA":   "BAA10Y",       # BAA corporate yield - 10y  (Moody's via FRED)
    "T10Y2Y":    "T10Y2Y",       # 10y - 2y
    "UNRATE":    "UNRATE",       # monthly
}


def get_fred_series(series_id: str, force: bool = False) -> pd.Series:
    """Fetch a single FRED series, cached."""
    raw = _download(
        FRED_CSV.format(sid=series_id),
        f"fred_{series_id}.csv",
        force=force,
    )
    df = pd.read_csv(io.BytesIO(raw))
    df.columns = [c.strip() for c in df.columns]
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col)
    s = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna()
    s.name = series_id
    return s


def get_fred_panel(force: bool = False) -> pd.DataFrame:
    """
    Fetch all FRED features and resample to month-end.
    Daily series take the last observation of the month.
    """
    out = {}
    for name, sid in FRED_SERIES.items():
        s = get_fred_series(sid, force=force)
        out[name] = s.resample("M").last()
    return pd.DataFrame(out)


if __name__ == "__main__":
    panel = get_factor_panel()
    print(panel.tail())
    print("Shape:", panel.shape)
    print("Date range:", panel.index.min(), "→", panel.index.max())
