"""
Shared OOS backtest harness for Inertia approaches A / B / C.

The harness refits a user-supplied model at a fixed cadence (default
annually) on an expanding training window, generates predictions for
the corresponding out-of-sample window, and returns the OOS prediction
series --- no look-ahead.

Each approach then maps its predictions to portfolio weights via
`weights_from_predictions()`, which implements the
Daniel-Moskowitz-style scaling + floor + cap.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd


OOS_START_DEFAULT = pd.Timestamp("2000-01-01")


def expanding_window_oos(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    fit_fn: Callable,
    oos_start: pd.Timestamp = OOS_START_DEFAULT,
    refit_months: int = 12,
    min_train_months: int = 60,
) -> pd.Series:
    """
    Run an expanding-window, OOS backtest of `fit_fn`.

    Parameters
    ----------
    df : DataFrame
        Panel with features and target, indexed by month-end.
    feature_cols : sequence of column names
        Feature columns passed to the model.
    target_col : str
        Target column (e.g. "UMD_next" for regression, or a crash-label
        column for classification).
    fit_fn : callable
        `fit_fn(X_train, y_train)` should return an object with `.predict(X)`.
        For classification, return an object with `.predict_proba(X)[:,1]`
        via a small wrapper (see note below).
    oos_start : Timestamp
        First date for which we need an OOS prediction.
    refit_months : int
        Refit the model every N months (default 12).
    min_train_months : int
        Require at least this many training rows before starting predictions.

    Returns
    -------
    Series indexed by month-end with OOS predictions (one value per month).
    Months where no model could be fit are omitted.

    Notes
    -----
    Classification: wrap your classifier so `model.predict(X)` returns the
    probability of the positive class. E.g.:

        class ProbWrapper:
            def __init__(self, clf): self.clf = clf
            def predict(self, X): return self.clf.predict_proba(X)[:, 1]

        def fit_fn(X, y):
            clf = GradientBoostingClassifier().fit(X, y)
            return ProbWrapper(clf)
    """
    df = df.sort_index().copy()
    feature_cols = list(feature_cols)

    # Keep only rows with complete features and non-null target
    usable = df[feature_cols + [target_col]].dropna()
    if usable.empty:
        return pd.Series(dtype=float)

    preds = pd.Series(index=usable.index, dtype=float)
    oos_start = pd.Timestamp(oos_start)

    # Refit schedule: pick month-starts of each year on or after oos_start
    oos_dates = usable.index[usable.index >= oos_start]
    if len(oos_dates) == 0:
        return pd.Series(dtype=float)

    # Determine refit boundaries every `refit_months` months
    refit_boundaries = []
    cursor = oos_dates[0]
    while cursor <= oos_dates[-1]:
        refit_boundaries.append(cursor)
        # Advance by refit_months months
        cursor = cursor + pd.DateOffset(months=refit_months)

    for i, boundary in enumerate(refit_boundaries):
        window_end = (
            refit_boundaries[i + 1]
            if i + 1 < len(refit_boundaries)
            else oos_dates[-1] + pd.DateOffset(days=1)
        )

        # Training set: all usable data strictly before this boundary
        train = usable.loc[usable.index < boundary]
        if len(train) < min_train_months:
            continue

        X_train = train[feature_cols].values
        y_train = train[target_col].values
        model = fit_fn(X_train, y_train)

        # Predict for every month in [boundary, window_end)
        pred_mask = (usable.index >= boundary) & (usable.index < window_end)
        pred_rows = usable.loc[pred_mask, feature_cols]
        if pred_rows.empty:
            continue
        preds.loc[pred_rows.index] = model.predict(pred_rows.values)

    return preds.dropna()


# ---------------------------------------------------------------------------
# Weight construction --- DM-style mean-variance scaling with floor + cap
# ---------------------------------------------------------------------------
def weights_from_predictions(
    expected_ret: pd.Series,
    umd: pd.Series,
    var_halflife: int = 6,
    var_floor_pct: float = 0.10,
    target_r_next: pd.Series | None = None,
    cap: tuple[float, float] = (-1.0, 3.0),
) -> tuple[pd.Series, float]:
    """
    Convert a series of E_t[r_{t+1}] predictions to portfolio weights.

    w_t = c * Ê[r_{t+1}] / Var̂[r_{t+1}],   clipped to `cap`.

    Var̂ is EWMA variance of UMD (annualized) with the given half-life,
    lagged one period (info through t, not t+1), floored at the
    `var_floor_pct` quantile of its full distribution to prevent blow-ups.

    Scaling constant `c` is chosen so the UNCAPPED strategy has the same
    unconditional volatility as static UMD (on `target_r_next` if given,
    else on `umd.shift(-1)` aligned to `expected_ret`).

    Parameters
    ----------
    expected_ret : Series
        Predicted next-month UMD returns, indexed by month-end.
    umd : Series
        Full monthly UMD return history (decimal).
    var_halflife : int
        EWMA half-life in months.
    var_floor_pct : float
        Quantile floor applied to the EWMA variance.
    target_r_next : Series, optional
        Next-month UMD returns to realize (defaults to `umd.shift(-1)`
        aligned to `expected_ret`).
    cap : (low, high)
        Weight clipping bounds.

    Returns
    -------
    (weights, scaling_c)
        weights : Series of post-cap weights aligned to `expected_ret`.
        scaling_c : the chosen scaling constant.
    """
    if target_r_next is None:
        target_r_next = umd.shift(-1).reindex(expected_ret.index)

    # EWMA variance of UMD, annualized, lagged one period
    ewma_var = (umd.ewm(halflife=var_halflife).var(bias=False) * 12).shift(1)
    ewma_var = ewma_var.reindex(expected_ret.index)
    floor = ewma_var.quantile(var_floor_pct)
    cond_var = ewma_var.clip(lower=floor)

    raw_w = expected_ret / cond_var

    # Uncapped scaling so dynamic.vol == static.vol
    r_next = target_r_next.loc[expected_ret.index]
    dyn_pre = raw_w * r_next
    vol_static = r_next.std(ddof=1)
    vol_pre = dyn_pre.std(ddof=1)
    c = vol_static / vol_pre if vol_pre > 0 else 1.0

    w_scaled = c * raw_w
    w = w_scaled.clip(lower=cap[0], upper=cap[1])
    return w, float(c)


def weights_from_crash_prob(
    p_crash: pd.Series,
    umd: pd.Series,
    target_vol_annual: float = 0.15,
    cap: tuple[float, float] = (-1.0, 3.0),
) -> tuple[pd.Series, float]:
    """
    Alternative weight construction for Approach B (classification).

    w_t = c * (1 - p_crash_t),  scaled so the unconditional vol of the
    strategy matches `target_vol_annual` (defaults to static-UMD-style 15%).

    Returns
    -------
    (weights, scaling_c)
    """
    base = (1.0 - p_crash.clip(0, 1))
    r_next = umd.shift(-1).reindex(base.index)
    dyn_pre = base * r_next
    vol_pre = dyn_pre.std(ddof=1) * np.sqrt(12)
    c = target_vol_annual / vol_pre if vol_pre > 0 else 1.0

    w = (c * base).clip(lower=cap[0], upper=cap[1])
    return w, float(c)


def apply_weights(
    weights: pd.Series,
    umd: pd.Series,
    cost_bps_oneway: float = 20.0,
) -> pd.DataFrame:
    """
    Apply a weight series to next-month UMD returns and deduct a simple
    linear transaction cost.

    Returns DataFrame with columns:
        weight       : weight at end of month t
        r_next       : UMD return in t+1
        r_gross      : weight * r_next
        turnover     : |w_t - w_{t-1}|
        cost         : cost_bps_oneway * turnover / 10000
        r_net        : r_gross - cost
    """
    idx = weights.index
    r_next = umd.shift(-1).reindex(idx)
    r_gross = weights * r_next
    turnover = weights.diff().abs().fillna(weights.abs())  # first period assumed from 0
    cost = cost_bps_oneway * turnover / 1e4
    r_net = r_gross - cost
    return pd.DataFrame({
        "weight": weights,
        "r_next": r_next,
        "r_gross": r_gross,
        "turnover": turnover,
        "cost": cost,
        "r_net": r_net,
    }).dropna()
