"""
Method A (Markov regime-switching) variant exploration.

Goal: lift composite Sharpe from baseline ~0.04 to >= 0.30 over the
common evaluation window 2000-02 to 2024-12.

For each variant we:
  - fit per-factor MarkovRegression with the variant's hyperparameters
  - forward-filter OOS probabilities (expanding window, per refit cadence)
  - convert to weights via the variant's weight rule
  - apply to next-month factor returns, compute equal-weight composite
  - report Sharpe, vol, max drawdown over 2000-02 to 2024-12

Saves:
  factor_regimes/tables/EXPLORE_method_a_variants.csv
  factor_regimes/tables/EXPLORE_method_a_best_probs.csv
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings("ignore")

# Make `lib` importable
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent  # factor_regimes/
sys.path.insert(0, str(ROOT))

from lib.data import build_factor_panel, FF5_FACTORS  # noqa: E402
from lib.evaluation import perf_stats  # noqa: E402

OOS_START = "1990-01-01"
EVAL_START = "2000-02-01"
EVAL_END = "2024-12-31"
COST_BPS = 5.0


# ---------------------------------------------------------------------------
# Generalized Markov fit / forecast supporting k_regimes, switching_ar, switching_variance
# ---------------------------------------------------------------------------
def _fit_markov(returns: pd.Series, k_regimes: int = 2,
                switching_ar: int = 0, switching_variance: bool = True):
    """Fit a Markov regression and return a result object."""
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression

    if switching_ar > 0:
        mod = MarkovAutoregression(
            returns.values, k_regimes=k_regimes, order=switching_ar,
            switching_ar=True, switching_variance=switching_variance,
        )
    else:
        mod = MarkovRegression(
            returns.values, k_regimes=k_regimes, trend="c",
            switching_variance=switching_variance,
        )
    res = mod.fit(disp=False)
    return res


def _favorable_state(res, k_regimes: int):
    """Identify which regime has the highest mean (favorable)."""
    # Best-effort across statsmodels versions: pull regime constants.
    try:
        # MarkovRegression exposes 'const[i]' or with switching variance, names vary.
        params = res.params
        # Try named index first
        idx = list(params.index)
        consts = []
        for k in range(k_regimes):
            # Possible naming: f"const[{k}]" or f"const.{k}"
            candidates = [f"const[{k}]", f"const.{k}", f"const_{k}"]
            val = None
            for c in candidates:
                if c in idx:
                    val = float(params[c])
                    break
            consts.append(val)
        if any(v is None for v in consts):
            raise KeyError("can't find const params")
    except Exception:
        # Fallback: assume layout for k_regimes=2, trend='c', switching_variance=True
        p = np.asarray(res.params).ravel()
        # Transition probs come first (k*(k-1) free), then constants, then variances/AR
        # Heuristic: pick the last k*(1+switching_var) values for [const,var]
        # But simpler: just default to argmax of smoothed means
        return int(np.argmax(_smoothed_means(res, k_regimes)))
    return int(np.argmax(consts))


def _smoothed_means(res, k_regimes: int) -> np.ndarray:
    """Mean return in each regime, computed from smoothed weights."""
    smoothed = np.asarray(res.smoothed_marginal_probabilities)
    # shape (T, k) or (k, T)
    if smoothed.shape[0] == k_regimes and smoothed.shape[1] != k_regimes:
        smoothed = smoothed.T  # -> (T, k)
    obs = np.asarray(res.model.endog).ravel()
    T = min(len(obs), smoothed.shape[0])
    smoothed = smoothed[-T:]
    obs = obs[-T:]
    means = np.zeros(k_regimes)
    for k in range(k_regimes):
        w = smoothed[:, k]
        if w.sum() > 1e-9:
            means[k] = float(np.average(obs, weights=w))
    return means


def _extract_params(res, k_regimes: int, switching_variance: bool):
    """Extract regime means, sigmas, transition matrix, init prob from a fitted model."""
    # Means: weighted means under smoothed probabilities (robust across versions)
    means = _smoothed_means(res, k_regimes)

    # Variances: try to read from params; fallback to weighted residual variance
    smoothed = np.asarray(res.smoothed_marginal_probabilities)
    if smoothed.shape[0] == k_regimes and smoothed.shape[1] != k_regimes:
        smoothed = smoothed.T
    obs = np.asarray(res.model.endog).ravel()
    T = min(len(obs), smoothed.shape[0])
    smoothed = smoothed[-T:]
    obs = obs[-T:]
    sigmas2 = np.zeros(k_regimes)
    for k in range(k_regimes):
        w = smoothed[:, k]
        if w.sum() > 1e-9:
            resid = obs - means[k]
            sigmas2[k] = float(np.average(resid ** 2, weights=w))
        else:
            sigmas2[k] = float(np.var(obs))
    sigmas2 = np.maximum(sigmas2, 1e-10)

    # Transition matrix
    trans = np.asarray(res.regime_transition)
    if trans.ndim == 3:
        trans = trans[:, :, 0]
    if trans.shape != (k_regimes, k_regimes):
        # uniform
        trans = np.full((k_regimes, k_regimes), 1.0 / k_regimes)

    # Init: last smoothed prob
    init_p = smoothed[-1, :].astype(float)
    if init_p.sum() <= 0:
        init_p = np.full(k_regimes, 1.0 / k_regimes)
    init_p = init_p / init_p.sum()

    favorable = int(np.argmax(means))

    return {
        "means": means.tolist(),
        "sigmas2": sigmas2.tolist(),
        "trans": trans,
        "init_p": init_p,
        "favorable": favorable,
        "k": k_regimes,
    }


def _forward_filter(obs: np.ndarray, params: dict) -> np.ndarray:
    k = params["k"]
    means = params["means"]
    sigmas = [np.sqrt(s2) for s2 in params["sigmas2"]]
    trans = params["trans"]
    p = params["init_p"].copy()

    T = len(obs)
    out = np.zeros((T, k))
    for t in range(T):
        p = p @ trans
        like = np.array([norm.pdf(obs[t], loc=means[i], scale=sigmas[i])
                         for i in range(k)])
        p = p * like
        s = p.sum()
        if s <= 0 or not np.isfinite(s):
            p = np.full(k, 1.0 / k)
        else:
            p = p / s
        out[t] = p
    return out


def fit_predict_markov_variant(
    returns: pd.Series,
    oos_start: str = "1990-01-01",
    refit_months: int = 12,
    min_train: int = 120,
    k_regimes: int = 2,
    switching_ar: int = 0,
    switching_variance: bool = True,
) -> pd.Series:
    """Generalized Markov OOS forecaster. Returns p_favorable per OOS month.

    For k_regimes=3, p_favorable is interpreted as P(state with highest mean).
    Caller can post-process raw probabilities; for backward compatibility we
    return P(top-mean state).
    """
    returns = returns.dropna().astype(float)
    out = pd.Series(index=returns.index, dtype=float)
    oos_start = pd.Timestamp(oos_start)
    oos_dates = returns.index[returns.index >= oos_start]
    if len(oos_dates) == 0:
        return out

    boundaries = [oos_dates[0]]
    while boundaries[-1] < oos_dates[-1]:
        boundaries.append(boundaries[-1] + pd.DateOffset(months=refit_months))

    for i, b in enumerate(boundaries):
        train = returns.loc[returns.index < b]
        if len(train) < min_train:
            continue
        try:
            res = _fit_markov(train, k_regimes=k_regimes,
                              switching_ar=switching_ar,
                              switching_variance=switching_variance)
            params = _extract_params(res, k_regimes, switching_variance)
        except Exception:
            params = None

        next_b = boundaries[i + 1] if i + 1 < len(boundaries) else returns.index[-1] + pd.DateOffset(days=1)
        oos_window = returns.loc[(returns.index >= b) & (returns.index < next_b)]
        if oos_window.empty:
            continue

        if params is None:
            out.loc[oos_window.index] = 0.5
            continue

        post = _forward_filter(oos_window.values, params)
        # Return P(favorable state)
        out.loc[oos_window.index] = post[:, params["favorable"]]

    return out.dropna()


# ---------------------------------------------------------------------------
# Weight rules
# ---------------------------------------------------------------------------
def weight_linear(p: pd.Series) -> pd.Series:
    return (2 * p - 1).clip(-1, 1)


def weight_long_only(p: pd.Series) -> pd.Series:
    return (2 * p - 1).clip(lower=0, upper=1)


def weight_three_step(p: pd.Series) -> pd.Series:
    w = pd.Series(0.0, index=p.index)
    w[p > 0.4] = 0.5
    w[p > 0.6] = 1.0
    return w


def weight_vol_scaled(p: pd.Series, factor_returns: pd.Series,
                      target_vol_ann: float = 0.10) -> pd.Series:
    """w = (2P-1) * (target_vol / recent_vol). Recent vol = trailing 12-month std."""
    base = 2 * p - 1
    recent_vol = factor_returns.rolling(12).std() * np.sqrt(12)
    recent_vol = recent_vol.reindex(p.index).fillna(method="ffill")
    scale = (target_vol_ann / recent_vol).clip(0.25, 4.0)
    return (base * scale).clip(-1.5, 1.5)


# ---------------------------------------------------------------------------
# Apply weights -> returns, with simple linear t-cost
# ---------------------------------------------------------------------------
def apply_weights_simple(weights: pd.Series, factor_returns: pd.Series,
                         cost_bps: float = COST_BPS) -> pd.Series:
    idx = weights.index
    r_next = factor_returns.shift(-1).reindex(idx)
    turnover = weights.diff().abs().fillna(weights.abs())
    cost = cost_bps * turnover / 1e4
    return (weights * r_next - cost).dropna()


# ---------------------------------------------------------------------------
# Variant runner
# ---------------------------------------------------------------------------
def run_variant(name: str, panel: pd.DataFrame, *,
                k_regimes: int = 2,
                switching_ar: int = 0,
                switching_variance: bool = True,
                refit_months: int = 12,
                min_train: int = 120,
                weight_fn=weight_linear,
                weight_kwargs=None) -> dict:
    weight_kwargs = weight_kwargs or {}
    t0 = time.time()
    probs = {}
    for f in FF5_FACTORS:
        probs[f] = fit_predict_markov_variant(
            panel[f],
            oos_start=OOS_START,
            refit_months=refit_months,
            min_train=min_train,
            k_regimes=k_regimes,
            switching_ar=switching_ar,
            switching_variance=switching_variance,
        )
    P = pd.DataFrame(probs)

    # Build per-factor timed returns
    sleeve_returns = {}
    for f in FF5_FACTORS:
        p_f = P[f].dropna()
        if weight_fn is weight_vol_scaled:
            w = weight_vol_scaled(p_f, panel[f], **weight_kwargs)
        else:
            w = weight_fn(p_f)
        sleeve_returns[f] = apply_weights_simple(w, panel[f])
    R = pd.DataFrame(sleeve_returns)

    # Equal-weight composite
    composite = R.mean(axis=1).dropna()

    # Restrict to common evaluation window
    eval_mask = (composite.index >= pd.Timestamp(EVAL_START)) & \
                (composite.index <= pd.Timestamp(EVAL_END))
    composite_eval = composite.loc[eval_mask]
    stats = perf_stats(composite_eval)
    elapsed = time.time() - t0

    return {
        "name": name,
        "n_months": stats.get("n_months", 0),
        "sharpe": stats.get("sharpe_ann", np.nan),
        "vol_ann": stats.get("vol_ann", np.nan),
        "mean_ann": stats.get("mean_ann", np.nan),
        "max_dd": stats.get("max_drawdown", np.nan),
        "elapsed_s": elapsed,
        "_probs": P,
        "_returns": composite_eval,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"[{time.strftime('%H:%M:%S')}] Loading factor panel...")
    panel = build_factor_panel(include_macro=False)
    print(f"  panel: {panel.shape}, {panel.index.min().date()} -> {panel.index.max().date()}")

    variants_to_run = [
        # Baseline (sanity check vs reported 0.04)
        dict(name="V0_baseline_2state_lin",
             k_regimes=2, switching_ar=0, switching_variance=True,
             refit_months=12, min_train=120,
             weight_fn=weight_linear),

        # 1) Long-only weight rule (cheapest fix; addresses noisy short side)
        dict(name="V1_2state_long_only",
             k_regimes=2, switching_ar=0, switching_variance=True,
             refit_months=12, min_train=120,
             weight_fn=weight_long_only),

        # 2) Three-step weight rule (deadband around 0.5)
        dict(name="V2_2state_three_step",
             k_regimes=2, switching_ar=0, switching_variance=True,
             refit_months=12, min_train=120,
             weight_fn=weight_three_step),

        # 3) Mean-only switching (more stable fits)
        dict(name="V3_2state_mean_only_long",
             k_regimes=2, switching_ar=0, switching_variance=False,
             refit_months=12, min_train=120,
             weight_fn=weight_long_only),

        # 4) AR(1) component (captures serial corr)
        dict(name="V4_2state_ar1_long",
             k_regimes=2, switching_ar=1, switching_variance=True,
             refit_months=12, min_train=120,
             weight_fn=weight_long_only),

        # 5) Monthly refit + long-only
        dict(name="V5_2state_monthly_refit_long",
             k_regimes=2, switching_ar=0, switching_variance=True,
             refit_months=1, min_train=120,
             weight_fn=weight_long_only),

        # 6) 3-state model with three-step rule
        dict(name="V6_3state_three_step",
             k_regimes=3, switching_ar=0, switching_variance=True,
             refit_months=12, min_train=120,
             weight_fn=weight_three_step),
    ]

    results = []
    all_probs = {}
    all_returns = {}
    for cfg in variants_to_run:
        name = cfg["name"]
        print(f"\n[{time.strftime('%H:%M:%S')}] Running {name}...")
        try:
            res = run_variant(panel=panel, **cfg)
            print(f"  -> Sharpe={res['sharpe']:.3f} vol={res['vol_ann']:.3f} "
                  f"DD={res['max_dd']:.3f} n={res['n_months']} ({res['elapsed_s']:.1f}s)")
            results.append({k: v for k, v in res.items() if not k.startswith("_")})
            all_probs[name] = res["_probs"]
            all_returns[name] = res["_returns"]
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({"name": name, "sharpe": np.nan, "vol_ann": np.nan,
                            "max_dd": np.nan, "n_months": 0,
                            "mean_ann": np.nan, "elapsed_s": np.nan})

    summary = pd.DataFrame(results).set_index("name")
    summary = summary[["n_months", "mean_ann", "vol_ann", "sharpe", "max_dd", "elapsed_s"]]
    summary = summary.sort_values("sharpe", ascending=False)
    print("\n" + "=" * 70)
    print("VARIANT COMPARISON (eval window 2000-02 to 2024-12)")
    print("=" * 70)
    print(summary.round(4).to_string())

    # Save
    out_dir = ROOT / "tables"
    out_dir.mkdir(exist_ok=True)
    summary.to_csv(out_dir / "EXPLORE_method_a_variants.csv")
    print(f"\nSaved variant table -> {out_dir / 'EXPLORE_method_a_variants.csv'}")

    # Best variant probs
    valid = summary["sharpe"].dropna()
    if len(valid):
        best_name = valid.idxmax()
        best_probs = all_probs[best_name]
        best_probs.to_csv(out_dir / "EXPLORE_method_a_best_probs.csv")
        print(f"Saved best probs ({best_name}) -> {out_dir / 'EXPLORE_method_a_best_probs.csv'}")
        print(f"\nBEST: {best_name}")
        print(summary.loc[best_name].round(4).to_string())
    else:
        print("\nNo valid variants completed.")


if __name__ == "__main__":
    main()
