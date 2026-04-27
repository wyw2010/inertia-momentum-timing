"""
Inertia v2: three regime-detection methods, each producing a
per-factor "favorable probability" time series under expanding-window
out-of-sample discipline.

  Method A: Markov regime-switching regression (Hamilton 1989)
            via statsmodels MarkovRegression with switching variance.
  Method B: Predictive regression with macro features
            via sklearn RidgeCV in a Pipeline.
  Method C: Gradient-boosted classifier on regime labels
            via sklearn GradientBoostingClassifier.

Convention: each method outputs `p_favorable_t in [0, 1]` for each
out-of-sample month, where 1 means full-long the factor and 0 means
full-short.
"""

from __future__ import annotations

import warnings
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Method A: Markov regime-switching regression with offline fit + online filter
# ---------------------------------------------------------------------------
def _fit_markov_2state(returns: pd.Series):
    """
    Fit a 2-state Markov regime-switching mean+variance model.
    Returns dict with consts, sigmas2, trans, init_p, favorable_state_index.
    """
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

    mod = MarkovRegression(
        returns.values, k_regimes=2,
        trend="c", switching_variance=True,
    )
    res = mod.fit(disp=False)

    # Param vector layout for k_regimes=2, trend='c', switching_variance=True:
    #   [0:2]  transition probs (p[0->0], p[1->0])
    #   [2:4]  regime means (const[0], const[1])
    #   [4:6]  regime variances (sigma2[0], sigma2[1])
    p = np.asarray(res.params).ravel()
    consts  = [float(p[2]), float(p[3])]
    sigmas2 = [float(p[4]), float(p[5])]

    # Transition matrix from statsmodels regime_transition (shape can be (2,2) or (2,2,1))
    trans = np.asarray(res.regime_transition)
    if trans.ndim == 3:
        trans = trans[:, :, 0]
    if trans.shape != (2, 2):
        trans = np.array([[0.95, 0.05], [0.05, 0.95]])

    # Initial filtered probability: last smoothed prob from training
    try:
        smoothed = np.asarray(res.smoothed_marginal_probabilities)
        # shape can be (T, 2) or (2, T) depending on version
        if smoothed.shape[0] == 2:
            init_p = smoothed[:, -1]
        else:
            init_p = smoothed[-1, :]
    except Exception:
        init_p = np.array([0.5, 0.5])
    init_p = np.asarray(init_p, dtype=float)
    if init_p.sum() <= 0:
        init_p = np.array([0.5, 0.5])
    init_p = init_p / init_p.sum()

    favorable = int(np.argmax(consts))

    return {
        "consts":    consts,
        "sigmas2":   sigmas2,
        "trans":     trans,
        "init_p":    init_p,
        "favorable": favorable,
    }


def _forward_filter(obs: np.ndarray, params: dict) -> np.ndarray:
    """
    Run the forward (filter) algorithm: returns P(s_t | obs_{1..t}).
    Output shape (T, 2) of posterior probabilities.
    """
    consts  = params["consts"]
    sigmas  = [np.sqrt(s2) for s2 in params["sigmas2"]]
    trans   = params["trans"]
    p       = params["init_p"].copy()

    T = len(obs)
    out = np.zeros((T, 2))
    for t in range(T):
        # Predict step (transition)
        p = p @ trans
        # Update step (likelihood-weighted)
        like = np.array([norm.pdf(obs[t], loc=consts[k], scale=sigmas[k])
                         for k in range(2)])
        p = p * like
        s = p.sum()
        if s <= 0 or not np.isfinite(s):
            p = np.array([0.5, 0.5])
        else:
            p = p / s
        out[t] = p
    return out


def fit_predict_markov_oos(
    returns: pd.Series,
    oos_start: str = "1990-01-01",
    refit_months: int = 12,
    min_train: int = 120,
) -> pd.Series:
    """
    Expanding-window OOS Markov regime-switching forecast for one factor.
    Returns p_favorable indexed by month-end Timestamps over OOS.
    """
    returns = returns.dropna().astype(float)
    out = pd.Series(index=returns.index, dtype=float)
    oos_start = pd.Timestamp(oos_start)
    oos_dates = returns.index[returns.index >= oos_start]
    if len(oos_dates) == 0:
        return out

    # Build refit boundaries every refit_months months
    boundaries = [oos_dates[0]]
    while boundaries[-1] < oos_dates[-1]:
        boundaries.append(boundaries[-1] + pd.DateOffset(months=refit_months))

    for i, b in enumerate(boundaries):
        train = returns.loc[returns.index < b]
        if len(train) < min_train:
            continue
        try:
            params = _fit_markov_2state(train)
        except Exception as e:
            print(f"  Markov fit failed at {b.date()} ({e}); using uniform 0.5")
            params = None

        next_b = boundaries[i + 1] if i + 1 < len(boundaries) else returns.index[-1] + pd.DateOffset(days=1)
        oos_window = returns.loc[(returns.index >= b) & (returns.index < next_b)]

        if params is None:
            out.loc[oos_window.index] = 0.5
            continue

        post = _forward_filter(oos_window.values, params)
        out.loc[oos_window.index] = post[:, params["favorable"]]

    return out.dropna()


# ---------------------------------------------------------------------------
# Method B: Predictive regression with macro features (Ridge)
# ---------------------------------------------------------------------------
def fit_predict_ridge_oos(
    panel: pd.DataFrame,
    factor: str,
    feature_cols: list[str],
    oos_start: str = "1990-01-01",
    refit_months: int = 12,
    min_train: int = 120,
) -> pd.Series:
    """
    Predict next-month return of `factor`, then convert to a favorable
    probability via sigmoid(predicted / training_std).
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    target_col = f"next_{factor}"
    cols = list(feature_cols) + [target_col]
    df = panel[cols].dropna().copy()
    if df.empty:
        return pd.Series(dtype=float)

    out = pd.Series(index=df.index, dtype=float)
    oos_start = pd.Timestamp(oos_start)
    oos_dates = df.index[df.index >= oos_start]
    if len(oos_dates) == 0:
        return out

    boundaries = [oos_dates[0]]
    while boundaries[-1] < oos_dates[-1]:
        boundaries.append(boundaries[-1] + pd.DateOffset(months=refit_months))

    for i, b in enumerate(boundaries):
        train = df.loc[df.index < b]
        if len(train) < min_train:
            continue

        Xtr = train[feature_cols].values
        ytr = train[target_col].values

        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("ridge", RidgeCV(alphas=np.logspace(-2, 2, 13), cv=5)),
        ]).fit(Xtr, ytr)

        train_std = float(np.std(ytr, ddof=1)) or 1.0

        next_b = boundaries[i + 1] if i + 1 < len(boundaries) else df.index[-1] + pd.DateOffset(days=1)
        oos_rows = df.loc[(df.index >= b) & (df.index < next_b), feature_cols]
        if oos_rows.empty:
            continue

        preds = pipe.predict(oos_rows.values)
        # Sigmoid map: predicted return / training std --> probability
        scaled = preds / train_std
        prob = 1.0 / (1.0 + np.exp(-scaled))
        out.loc[oos_rows.index] = prob

    return out.dropna()


# ---------------------------------------------------------------------------
# Method C: Gradient-boosted regime classifier
# ---------------------------------------------------------------------------
def fit_predict_gbm_oos(
    panel: pd.DataFrame,
    factor: str,
    feature_cols: list[str],
    oos_start: str = "1990-01-01",
    refit_months: int = 12,
    min_train: int = 120,
    label_quantile: float = 0.50,
) -> pd.Series:
    """
    Label each training month "favorable" if next-month factor return
    exceeds the in-sample `label_quantile` of next-month returns.
    Predict P(favorable) for each OOS month with a gradient-boosted classifier.
    """
    from sklearn.ensemble import GradientBoostingClassifier

    target_col = f"next_{factor}"
    cols = list(feature_cols) + [target_col]
    df = panel[cols].dropna().copy()
    if df.empty:
        return pd.Series(dtype=float)

    out = pd.Series(index=df.index, dtype=float)
    oos_start = pd.Timestamp(oos_start)
    oos_dates = df.index[df.index >= oos_start]
    if len(oos_dates) == 0:
        return out

    boundaries = [oos_dates[0]]
    while boundaries[-1] < oos_dates[-1]:
        boundaries.append(boundaries[-1] + pd.DateOffset(months=refit_months))

    for i, b in enumerate(boundaries):
        train = df.loc[df.index < b]
        if len(train) < min_train:
            continue

        Xtr = train[feature_cols].values
        ytr_ret = train[target_col].values
        thresh = float(np.quantile(ytr_ret, label_quantile))
        ytr = (ytr_ret > thresh).astype(int)

        # Skip if degenerate (only one class)
        if len(np.unique(ytr)) < 2:
            continue

        clf = GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=42,
        ).fit(Xtr, ytr)

        next_b = boundaries[i + 1] if i + 1 < len(boundaries) else df.index[-1] + pd.DateOffset(days=1)
        oos_rows = df.loc[(df.index >= b) & (df.index < next_b), feature_cols]
        if oos_rows.empty:
            continue

        prob = clf.predict_proba(oos_rows.values)[:, 1]
        out.loc[oos_rows.index] = prob

    return out.dropna()


# ---------------------------------------------------------------------------
# Convenience: run a method across all FF5 factors at once
# ---------------------------------------------------------------------------
def run_method_all_factors(
    fit_predict_fn: Callable,
    panel: pd.DataFrame,
    factors: list[str],
    **kwargs,
) -> pd.DataFrame:
    """
    Run a fit_predict_*_oos function across multiple factors and stack the
    resulting probabilities into a wide DataFrame (rows=months, cols=factors).
    """
    out = {}
    for f in factors:
        if fit_predict_fn is fit_predict_markov_oos:
            out[f] = fit_predict_fn(panel[f], **kwargs)
        else:
            out[f] = fit_predict_fn(panel, f, **kwargs)
    return pd.DataFrame(out)
