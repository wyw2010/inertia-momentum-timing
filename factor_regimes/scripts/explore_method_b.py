"""
Explore variants of Method B (Ridge predictive regression) to push composite
Sharpe from baseline 0.53 toward >= 0.65.

Each variant:
  - Builds OOS monthly favorable probabilities per FF5 factor
    over 2000-02 to 2024-12.
  - Maps probs to weights, computes timed returns, equal-weights across
    5 sleeves to form a composite, and reports Sharpe / vol / max DD.

Outputs:
  tables/EXPLORE_method_b_variants.csv
  tables/EXPLORE_method_b_best_probs.csv
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---- Path setup -----------------------------------------------------------
HERE = Path(__file__).resolve().parent
FACTOR_REGIMES = HERE.parent
REPO_ROOT = FACTOR_REGIMES.parent
sys.path.insert(0, str(FACTOR_REGIMES))
sys.path.insert(0, str(REPO_ROOT))

from lib.data import build_factor_panel, FF5_FACTORS  # noqa: E402
from lib.backtest import prob_to_weight, apply_weights  # noqa: E402
from lib.evaluation import perf_stats  # noqa: E402

OOS_START = "2000-02-01"
OOS_END = "2024-12-31"
REFIT_MONTHS = 12
MIN_TRAIN = 120
TABLES_DIR = FACTOR_REGIMES / "tables"


# ---------------------------------------------------------------------------
# Generic Ridge OOS with configurable feature transformations and target
# ---------------------------------------------------------------------------
def ridge_oos(
    panel: pd.DataFrame,
    factor: str,
    feature_cols: list[str],
    target_col: str | None = None,
    alphas: np.ndarray | None = None,
    rolling_norm_window: int | None = None,
    oos_start: str = OOS_START,
    refit_months: int = REFIT_MONTHS,
    min_train: int = MIN_TRAIN,
) -> pd.Series:
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    if target_col is None:
        target_col = f"next_{factor}"
    if alphas is None:
        alphas = np.logspace(-2, 2, 13)

    cols = list(feature_cols) + [target_col]
    df = panel[cols].dropna().copy()

    # Optional rolling normalization of features
    if rolling_norm_window:
        for c in feature_cols:
            roll_std = df[c].rolling(rolling_norm_window, min_periods=12).std()
            df[c] = df[c] / roll_std.replace(0, np.nan)
        df = df.dropna()

    if df.empty:
        return pd.Series(dtype=float)

    out = pd.Series(index=df.index, dtype=float)
    oos_ts = pd.Timestamp(oos_start)
    oos_dates = df.index[df.index >= oos_ts]
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
            ("ridge", RidgeCV(alphas=alphas, cv=5)),
        ]).fit(Xtr, ytr)

        train_std = float(np.std(ytr, ddof=1)) or 1.0

        next_b = (boundaries[i + 1] if i + 1 < len(boundaries)
                  else df.index[-1] + pd.DateOffset(days=1))
        oos_rows = df.loc[(df.index >= b) & (df.index < next_b), feature_cols]
        if oos_rows.empty:
            continue

        preds = pipe.predict(oos_rows.values)
        scaled = preds / train_std
        prob = 1.0 / (1.0 + np.exp(-scaled))
        out.loc[oos_rows.index] = prob

    return out.dropna()


# ---------------------------------------------------------------------------
# Composite return + perf for a probability frame
# ---------------------------------------------------------------------------
def composite_perf(
    probs: pd.DataFrame,
    panel: pd.DataFrame,
    weight_mode: str = "soft",
    horizon: int = 1,
) -> tuple[dict, pd.Series]:
    """
    probs: DataFrame indexed by month with one column per FF5 factor.
    horizon: 1 means weight applies to next-1m return (standard).
             3 means weights applied to the next-3m average return,
             rebalanced monthly (overlapping but standard practice).
    Returns (perf_dict, composite_returns_series).
    """
    sleeve_returns = {}
    for f in FF5_FACTORS:
        if f not in probs.columns:
            continue
        p = probs[f].dropna()
        if p.empty:
            continue
        w = prob_to_weight(p, mode=weight_mode)
        if horizon == 1:
            tr = apply_weights(w, panel[f], cost_bps_oneway=5.0)
            sleeve_returns[f] = tr["r_net"]
        else:
            # Average next-h months of factor returns; w is held flat.
            r = panel[f]
            fwd = pd.concat(
                [r.shift(-k) for k in range(1, horizon + 1)],
                axis=1
            ).mean(axis=1)
            r_next = fwd.reindex(w.index)
            r_gross = w * r_next
            turnover = w.diff().abs().fillna(w.abs())
            cost = 5.0 * turnover / 1e4
            sleeve_returns[f] = (r_gross - cost).dropna()

    sleeve_df = pd.DataFrame(sleeve_returns).dropna(how="all")
    # restrict to OOS window
    mask = (sleeve_df.index >= pd.Timestamp(OOS_START)) & (sleeve_df.index <= pd.Timestamp(OOS_END))
    sleeve_df = sleeve_df.loc[mask]
    composite = sleeve_df.mean(axis=1).dropna()
    return perf_stats(composite), composite


# ---------------------------------------------------------------------------
# Variant runners
# ---------------------------------------------------------------------------
def run_variant(name: str, panel: pd.DataFrame, **ridge_kwargs) -> tuple[pd.DataFrame, pd.Series, dict]:
    weight_mode = ridge_kwargs.pop("_weight_mode", "soft")
    horizon = ridge_kwargs.pop("_horizon", 1)
    target_template = ridge_kwargs.pop("_target_template", None)  # e.g. "next3_{f}"

    probs = {}
    for f in FF5_FACTORS:
        kw = dict(ridge_kwargs)
        # Per-factor feature spec may depend on f
        feat_fn = kw.pop("_feature_fn")
        kw["feature_cols"] = feat_fn(f)
        if target_template is not None:
            kw["target_col"] = target_template.format(f=f)
        probs[f] = ridge_oos(panel, f, **kw)
    probs_df = pd.DataFrame(probs)

    perf, comp = composite_perf(probs_df, panel, weight_mode=weight_mode, horizon=horizon)
    perf["variant"] = name
    return probs_df, comp, perf


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Building factor panel (with macro)...")
    panel = build_factor_panel(include_macro=True)
    print(f"  panel shape: {panel.shape}, range: {panel.index.min().date()} -> {panel.index.max().date()}")

    macro_cols = [c for c in ["vix", "vix_chg", "term_spread", "credit_spread"] if c in panel.columns]
    print(f"  macro cols available: {macro_cols}")

    # Add 3-month forward target for horizon variants
    for f in FF5_FACTORS:
        panel[f"next3_{f}"] = (
            pd.concat([panel[f].shift(-k) for k in range(1, 4)], axis=1).mean(axis=1)
        )

    # Feature constructors --------------------------------------------------
    def feat_baseline(f):
        # cross-factor lags + rolling vols (matches notebook 03 baseline)
        return [f"lag1_{x}" for x in FF5_FACTORS] + [f"vol6_{x}" for x in FF5_FACTORS]

    def feat_baseline_plus_macro(f):
        return feat_baseline(f) + macro_cols

    def feat_with_interactions(f):
        # add interaction terms lag * vol per factor
        base = feat_baseline(f)
        # We'll synthesize interaction columns into the panel below.
        return base + [f"interact_{x}" for x in FF5_FACTORS]

    def feat_baseline_plus_macro_interact(f):
        return feat_baseline_plus_macro(f) + [f"interact_{x}" for x in FF5_FACTORS]

    # Ensure interaction columns exist
    for x in FF5_FACTORS:
        panel[f"interact_{x}"] = panel[f"lag1_{x}"] * panel[f"vol6_{x}"]

    results = []
    all_probs = {}

    # Variant 0: replicate baseline (cross-factor, soft, default alphas)
    print("\n[V0] baseline (cross-factor lags + vols, soft weight)")
    p, c, perf = run_variant(
        "V0_baseline", panel,
        _feature_fn=feat_baseline,
        _weight_mode="soft",
    )
    print(f"  Sharpe={perf['sharpe_ann']:.3f}  vol={perf['vol_ann']:.3f}  maxDD={perf['max_drawdown']:.3f}")
    results.append(perf); all_probs["V0_baseline"] = p

    # Variant 1: baseline + macro (VIX, term, credit)
    print("\n[V1] baseline + macro features")
    p, c, perf = run_variant(
        "V1_macro", panel,
        _feature_fn=feat_baseline_plus_macro,
        _weight_mode="soft",
    )
    print(f"  Sharpe={perf['sharpe_ann']:.3f}  vol={perf['vol_ann']:.3f}  maxDD={perf['max_drawdown']:.3f}")
    results.append(perf); all_probs["V1_macro"] = p

    # Variant 2: long-only weight rule on baseline
    print("\n[V2] baseline + long-only (longflat) weight rule")
    p, c, perf = run_variant(
        "V2_longflat", panel,
        _feature_fn=feat_baseline,
        _weight_mode="longflat",
    )
    print(f"  Sharpe={perf['sharpe_ann']:.3f}  vol={perf['vol_ann']:.3f}  maxDD={perf['max_drawdown']:.3f}")
    results.append(perf); all_probs["V2_longflat"] = p

    # Variant 3: macro + long-only weight rule
    print("\n[V3] macro + long-only weight rule")
    p, c, perf = run_variant(
        "V3_macro_longflat", panel,
        _feature_fn=feat_baseline_plus_macro,
        _weight_mode="longflat",
    )
    print(f"  Sharpe={perf['sharpe_ann']:.3f}  vol={perf['vol_ann']:.3f}  maxDD={perf['max_drawdown']:.3f}")
    results.append(perf); all_probs["V3_macro_longflat"] = p

    # Variant 4: macro + interactions
    print("\n[V4] macro + interaction terms (lag*vol)")
    p, c, perf = run_variant(
        "V4_macro_interact", panel,
        _feature_fn=feat_baseline_plus_macro_interact,
        _weight_mode="soft",
    )
    print(f"  Sharpe={perf['sharpe_ann']:.3f}  vol={perf['vol_ann']:.3f}  maxDD={perf['max_drawdown']:.3f}")
    results.append(perf); all_probs["V4_macro_interact"] = p

    # Variant 5: macro + 3m horizon target (smoother)
    print("\n[V5] macro + 3-month forward target")
    p, c, perf = run_variant(
        "V5_macro_h3", panel,
        _feature_fn=feat_baseline_plus_macro,
        _weight_mode="soft",
        _target_template="next3_{f}",
    )
    print(f"  Sharpe={perf['sharpe_ann']:.3f}  vol={perf['vol_ann']:.3f}  maxDD={perf['max_drawdown']:.3f}")
    results.append(perf); all_probs["V5_macro_h3"] = p

    # ----- summarize -------------------------------------------------------
    df = pd.DataFrame(results)[
        ["variant", "n_months", "mean_ann", "vol_ann", "sharpe_ann",
         "max_drawdown", "skew", "excess_kurt"]
    ].set_index("variant").sort_values("sharpe_ann", ascending=False)
    print("\n=========== Method B variant comparison ===========")
    print(df.round(4).to_string())

    # ----- save ------------------------------------------------------------
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = TABLES_DIR / "EXPLORE_method_b_variants.csv"
    df.to_csv(out_csv)
    print(f"\nSaved comparison table to {out_csv}")

    best_name = df.index[0]
    best_probs = all_probs[best_name]
    best_csv = TABLES_DIR / "EXPLORE_method_b_best_probs.csv"
    best_probs.to_csv(best_csv)
    print(f"Best variant: {best_name} (Sharpe={df.loc[best_name, 'sharpe_ann']:.3f})")
    print(f"Saved best probabilities to {best_csv}")


if __name__ == "__main__":
    main()
