"""
Explore Method C (gradient-boosted regime classifier) variants for
Inertia v2 factor timing. Goal: lift composite Sharpe above 0.55.

OOS window: 2000-02 to 2024-12. Composite is equal-weight across the 5
FF5 factor sleeves.

Run from repo root:
    python -m factor_regimes.scripts.explore_method_c
or directly:
    python factor_regimes/scripts/explore_method_c.py
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ----------------------------- path setup -----------------------------
_THIS = Path(__file__).resolve()
_REPO_ROOT = _THIS.parent.parent.parent  # inertia-momentum-timing
_FR_ROOT = _THIS.parent.parent            # factor_regimes
for p in (_REPO_ROOT, _FR_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from lib.data import build_factor_panel, FF5_FACTORS  # noqa: E402
from lib.backtest import apply_weights                 # noqa: E402
from lib.evaluation import perf_stats                  # noqa: E402

OOS_START = "2000-02-01"
OOS_END   = "2024-12-31"
TABLE_DIR = _FR_ROOT / "tables"


# ------------------------- core fitter --------------------------------
def _fit_predict_oos(
    panel: pd.DataFrame,
    factor: str,
    feature_cols: list[str],
    *,
    model: str = "gbm",
    n_estimators: int = 200,
    max_depth: int = 3,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    label_quantile: float = 0.50,
    class_balance: bool = False,
    oos_start: str = OOS_START,
    refit_months: int = 12,
    min_train: int = 120,
) -> pd.Series:
    """
    Generalized OOS GBM/LightGBM classifier for one factor.
    """
    target_col = f"next_{factor}"
    cols = list(feature_cols) + [target_col]
    df = panel[cols].dropna().copy()
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
        ytr_ret = train[target_col].values
        thresh = float(np.quantile(ytr_ret, label_quantile))
        ytr = (ytr_ret > thresh).astype(int)

        if len(np.unique(ytr)) < 2:
            continue

        sw = None
        if class_balance:
            n0 = (ytr == 0).sum()
            n1 = (ytr == 1).sum()
            if n0 > 0 and n1 > 0:
                w0 = 0.5 / n0 * len(ytr)
                w1 = 0.5 / n1 * len(ytr)
                sw = np.where(ytr == 1, w1, w0)

        if model == "gbm":
            from sklearn.ensemble import GradientBoostingClassifier
            clf = GradientBoostingClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                random_state=42,
            )
            if sw is not None:
                clf.fit(Xtr, ytr, sample_weight=sw)
            else:
                clf.fit(Xtr, ytr)
        elif model == "lgbm":
            from lightgbm import LGBMClassifier
            clf = LGBMClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth if max_depth else -1,
                num_leaves=min(31, max(7, 2 ** max_depth - 1)) if max_depth else 31,
                learning_rate=learning_rate,
                subsample=subsample,
                subsample_freq=1,
                colsample_bytree=0.8,
                class_weight=("balanced" if class_balance else None),
                random_state=42,
                verbose=-1,
            )
            clf.fit(Xtr, ytr)
        else:
            raise ValueError(f"unknown model {model}")

        next_b = boundaries[i + 1] if i + 1 < len(boundaries) else df.index[-1] + pd.DateOffset(days=1)
        oos_rows = df.loc[(df.index >= b) & (df.index < next_b), feature_cols]
        if oos_rows.empty:
            continue

        prob = clf.predict_proba(oos_rows.values)[:, 1]
        out.loc[oos_rows.index] = prob

    return out.dropna()


# --------------------- weight rule + composite ------------------------
def _prob_to_weight(probs: pd.Series, mode: str) -> pd.Series:
    if mode == "linear":
        w = (2 * probs - 1.0).clip(-1, 1)
    elif mode == "longflat_soft":
        # long-only: w = max(2P-1, 0)
        w = (2 * probs - 1.0).clip(lower=0.0, upper=1.0)
    elif mode == "soft5":
        # aggressive sigmoid-like scaling: w = 5*(2P-1) clipped
        w = (5.0 * (2.0 * probs - 1.0)).clip(-1, 1)
    else:
        raise ValueError(mode)
    return w


def _composite_from_probs(panel: pd.DataFrame, probs: pd.DataFrame,
                          weight_mode: str) -> pd.Series:
    """
    For each factor, convert P_t to weight w_t and apply to NEXT-month
    return. Equal-weight across the 5 sleeves.
    """
    sleeves = {}
    for f in FF5_FACTORS:
        if f not in probs.columns:
            continue
        p = probs[f].dropna()
        if p.empty:
            continue
        w = _prob_to_weight(p, weight_mode)
        df = apply_weights(w, panel[f], cost_bps_oneway=5.0)
        sleeves[f] = df["r_net"]
    if not sleeves:
        return pd.Series(dtype=float)
    sl = pd.DataFrame(sleeves)
    composite = sl.mean(axis=1)
    return composite.loc[(composite.index >= pd.Timestamp(OOS_START)) &
                         (composite.index <= pd.Timestamp(OOS_END))]


# ------------------------ variant runner ------------------------------
def run_variant(name: str, panel: pd.DataFrame, *,
                feature_cols: list[str],
                weight_mode: str = "linear",
                **fit_kwargs) -> tuple[pd.DataFrame, pd.Series, dict]:
    print(f"\n[ {name} ] fitting 5 factors ...", flush=True)
    probs = {}
    for f in FF5_FACTORS:
        probs[f] = _fit_predict_oos(panel, f, feature_cols, **fit_kwargs)
    P = pd.DataFrame(probs)
    composite = _composite_from_probs(panel, P, weight_mode)
    stats = perf_stats(composite)
    sharpe = stats.get("sharpe_ann", np.nan)
    vol    = stats.get("vol_ann",    np.nan)
    mdd    = stats.get("max_drawdown", np.nan)
    print(f"    Sharpe={sharpe:.3f}  Vol={vol:.3f}  MaxDD={mdd:.3f}  N={stats.get('n_months')}",
          flush=True)
    return P, composite, stats


def main():
    print("Loading panel ...", flush=True)
    panel = build_factor_panel(include_macro=True)
    print(f"Panel shape: {panel.shape}, range {panel.index.min().date()} -> {panel.index.max().date()}",
          flush=True)

    # ----- feature sets -----
    # cross-factor features (all factors' lag1 + vol6); this is what
    # notebook 04 uses, mirroring "current implementation"
    cross_feats = [f"lag1_{f}" for f in FF5_FACTORS] + [f"vol6_{f}" for f in FF5_FACTORS]

    macro_cols = [c for c in ("vix", "vix_chg", "term_spread", "credit_spread") if c in panel.columns]
    cross_macro = cross_feats + macro_cols

    # baseline params
    base_kw = dict(
        model="gbm", n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, label_quantile=0.50,
    )

    variants: list[tuple[str, dict]] = []

    # V0 baseline (current implementation)
    variants.append(("V0_baseline_gbm200_d3_lr05",
                     dict(feature_cols=cross_feats, weight_mode="linear", **base_kw)))

    # V1: add macro features (cross + macro)
    variants.append(("V1_gbm_cross_plus_macro",
                     dict(feature_cols=cross_macro, weight_mode="linear", **base_kw)))

    # V2: hyperparameter (300, depth 5, lr 0.03) on cross+macro
    variants.append(("V2_gbm300_d5_lr03_cross_macro",
                     dict(feature_cols=cross_macro, weight_mode="linear",
                          model="gbm", n_estimators=300, max_depth=5,
                          learning_rate=0.03, subsample=0.8, label_quantile=0.50)))

    # V3: shallower + slower, more trees (500,3,0.02) on cross+macro
    variants.append(("V3_gbm500_d3_lr02_cross_macro",
                     dict(feature_cols=cross_macro, weight_mode="linear",
                          model="gbm", n_estimators=500, max_depth=3,
                          learning_rate=0.02, subsample=0.8, label_quantile=0.50)))

    # V4: LightGBM cross + macro, default-ish
    variants.append(("V4_lgbm_cross_macro",
                     dict(feature_cols=cross_macro, weight_mode="linear",
                          model="lgbm", n_estimators=300, max_depth=4,
                          learning_rate=0.03, subsample=0.8, label_quantile=0.50)))

    # V5: long-only weight rule on best-shape baseline+macro
    variants.append(("V5_gbm_cross_macro_longonly",
                     dict(feature_cols=cross_macro, weight_mode="longflat_soft",
                          **base_kw)))

    # V6: 25th percentile label (more positives) + macro
    variants.append(("V6_gbm_cross_macro_q25",
                     dict(feature_cols=cross_macro, weight_mode="linear",
                          model="gbm", n_estimators=200, max_depth=3,
                          learning_rate=0.05, subsample=0.8, label_quantile=0.25)))

    # V7: long-only weight + macro + q=0.25 (combine V5 and V6)
    variants.append(("V7_gbm_cross_macro_longonly_q25",
                     dict(feature_cols=cross_macro, weight_mode="longflat_soft",
                          model="gbm", n_estimators=200, max_depth=3,
                          learning_rate=0.05, subsample=0.8, label_quantile=0.25)))

    # V8: long-only weight, NO macro (cross only, q=0.25)
    variants.append(("V8_gbm_cross_only_longonly_q25",
                     dict(feature_cols=cross_feats, weight_mode="longflat_soft",
                          model="gbm", n_estimators=200, max_depth=3,
                          learning_rate=0.05, subsample=0.8, label_quantile=0.25)))

    # V9: long-only, baseline cross-only (no macro), q=0.50
    variants.append(("V9_gbm_cross_only_longonly_q50",
                     dict(feature_cols=cross_feats, weight_mode="longflat_soft",
                          model="gbm", n_estimators=200, max_depth=3,
                          learning_rate=0.05, subsample=0.8, label_quantile=0.50)))

    results = []
    probs_store = {}

    for name, kw in variants:
        try:
            P, comp, stats = run_variant(name, panel, **kw)
            row = {
                "variant": name,
                "sharpe_ann":   stats.get("sharpe_ann", np.nan),
                "mean_ann":     stats.get("mean_ann",   np.nan),
                "vol_ann":      stats.get("vol_ann",    np.nan),
                "max_drawdown": stats.get("max_drawdown", np.nan),
                "n_months":     stats.get("n_months",   np.nan),
            }
            results.append(row)
            probs_store[name] = P
        except Exception as e:
            print(f"  FAILED variant {name}: {e}", flush=True)
            results.append({"variant": name, "sharpe_ann": np.nan,
                            "mean_ann": np.nan, "vol_ann": np.nan,
                            "max_drawdown": np.nan, "n_months": np.nan})

    res_df = pd.DataFrame(results).set_index("variant")
    res_df = res_df.sort_values("sharpe_ann", ascending=False)

    print("\n=================== Method C variant comparison ===================")
    print(res_df.round(4).to_string())

    # save table
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TABLE_DIR / "EXPLORE_method_c_variants.csv"
    res_df.to_csv(out_path)
    print(f"\nSaved comparison: {out_path}")

    # save best probs
    best_name = res_df.index[0]
    best_sharpe = res_df.iloc[0]["sharpe_ann"]
    best_probs = probs_store.get(best_name)
    if best_probs is not None:
        best_path = TABLE_DIR / "EXPLORE_method_c_best_probs.csv"
        best_probs.to_csv(best_path)
        print(f"Saved best-variant probs ({best_name}, Sharpe {best_sharpe:.3f}): {best_path}")

    print("\nDONE.")
    return res_df


if __name__ == "__main__":
    main()
