"""
gbt_baseline.py
===============
The "boring baseline" (architecture doc §5.3): gradient-boosted trees that
predict the TRUE delay from GP summary features. Defines the accuracy bar
the Stage-2 transformer must beat — if trees on ~10 summary numbers already
meet the bar, the transformer's case rests on microlensing, not delays.

Data     : scripts/gp_only_benchmark.ecsv (features + truth, 1000 systems)
Features : GP outputs + observables only — nothing from the truth record
Split    : GroupKFold by lens_index (5 folds) -> every row predicted
           out-of-fold by a model that never saw its lens system
Model    : HistGradientBoostingRegressor (native NaN handling), plus two
           quantile models (16/84%) for per-system error bars
Outputs  : metrics table on stdout (same columns as compare_fit_modes.py),
           out-of-fold predictions -> scripts/gbt_baseline_predictions.ecsv

Usage: python scripts/gbt_baseline.py
"""

import os
import sys
import numpy as np
from astropy.table import Table
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from roman_td.paths import benchmark_path, find_benchmark

QUALITIES = ["good", "broad", "multipeak", "fail"]

FEATURES = ["dt_gp", "dt_gp_err", "band_scatter", "n_bands",
            "flux_ratio", "flux_ratio_err", "snr_ref", "snr_img",
            "z_lens", "z_source", "n_images"]


def build_xy(tab):
    """Feature matrix (NaNs kept — HistGBR handles them), target, groups."""
    cols = [np.asarray(tab[c], dtype=float) for c in FEATURES]
    q = np.asarray(tab["quality"]).astype(str)
    cols += [(q == name).astype(float) for name in QUALITIES]
    X = np.column_stack(cols)
    # Target is the GP's RESIDUAL, not the absolute delay: trees are
    # piecewise-constant and waste all their capacity re-learning the
    # trivial "prediction ~ dt_gp" identity if asked for the absolute value
    # (measured: absolute-target GBT was ~2x WORSE than raw GP). Predicting
    # the correction means "predict 0" already equals raw-GP accuracy.
    y = (np.asarray(tab["true_delay"], dtype=float)
         - np.asarray(tab["dt_gp"], dtype=float))
    groups = np.asarray(tab["lens_index"], dtype=int)
    return X, y, groups


def metrics(residuals):
    r = residuals[np.isfinite(residuals)]
    return {"n": len(r), "median": np.median(r),
            "p68": np.percentile(np.abs(r), 68),
            "lt2": np.mean(np.abs(r) < 2) * 100,
            "lt5": np.mean(np.abs(r) < 5) * 100}


def main():
    tab = Table.read(find_benchmark("gp_only_benchmark.ecsv"))
    X, y, groups = build_xy(tab)
    print(f"{len(tab)} delays from {len(set(groups))} lens systems, "
          f"{X.shape[1]} features")

    # Out-of-fold predictions: median + 16/84% quantiles, grouped by lens
    pred = np.full(len(y), np.nan)
    q16 = np.full(len(y), np.nan)
    q84 = np.full(len(y), np.nan)
    gkf = GroupKFold(n_splits=5)
    # Early stopping guards against overfitting the ~850 rows per fold:
    # without it the trees fit noise and DEGRADE systems where the GP has
    # no systematic error to correct (measured on the "good" subset).
    params = dict(max_iter=500, learning_rate=0.05, max_leaf_nodes=31,
                  min_samples_leaf=20, random_state=42,
                  early_stopping=True, validation_fraction=0.2,
                  n_iter_no_change=20)
    for tr, te in gkf.split(X, y, groups):
        m = HistGradientBoostingRegressor(loss="absolute_error", **params)
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
        for arr, alpha in ((q16, 0.16), (q84, 0.84)):
            qm = HistGradientBoostingRegressor(loss="quantile",
                                               quantile=alpha, **params)
            qm.fit(X[tr], y[tr])
            arr[te] = qm.predict(X[te])

    res_gbt = pred - y
    # y is the residual (true - dt_gp), so raw-GP error = dt_gp - true = -y
    res_gp = -y
    quality = np.asarray(tab["quality"]).astype(str)

    # ── metrics table (same columns as compare_fit_modes.py) ───────────
    rows = {
        "raw GP": res_gp,
        "GBT": res_gbt,
        "raw GP (good)": res_gp[quality == "good"],
        "GBT (good)": res_gbt[quality == "good"],
        "GBT (not good)": res_gbt[quality != "good"],
    }
    print(f"\n{'method':<15} {'n':>5} {'median':>8} {'P68|res|':>9} "
          f"{'<2d':>6} {'<5d':>6}")
    for name, r in rows.items():
        s = metrics(r)
        print(f"{name:<15} {s['n']:>5} {s['median']:>+7.2f}d "
              f"{s['p68']:>8.2f}d {s['lt2']:>5.0f}% {s['lt5']:>5.0f}%")

    # ── error-bar calibration: is truth inside [q16, q84] ~68%? ────────
    inside = (y >= q16) & (y <= q84)
    print(f"\nquantile calibration: truth within [16%, 84%] band: "
          f"{np.mean(inside)*100:.0f}%  (target 68%)")
    half_width = (q84 - q16) / 2
    print(f"median half-width of predicted band: "
          f"{np.median(half_width):.2f} d")

    # ── what drives the predictions ─────────────────────────────────────
    from sklearn.inspection import permutation_importance
    m_full = HistGradientBoostingRegressor(loss="absolute_error", **params)
    m_full.fit(X, y)
    imp = permutation_importance(m_full, X, y, n_repeats=5, random_state=42,
                                 scoring="neg_mean_absolute_error")
    names = FEATURES + [f"quality={q}" for q in QUALITIES]
    order = np.argsort(imp.importances_mean)[::-1]
    print("\ntop features (permutation importance, days of MAE):")
    for j in order[:6]:
        print(f"  {names[j]:<18} {imp.importances_mean[j]:6.2f}")

    # Reconstruct absolute delays: model works in residual space (see
    # build_xy), stored values are dt_gp + predicted correction.
    dt_gp_col = np.asarray(tab["dt_gp"], dtype=float)
    out = Table({"lens_index": tab["lens_index"], "image": tab["image"],
                 "true_delay": np.asarray(tab["true_delay"], dtype=float),
                 "dt_gp": dt_gp_col, "gbt_pred": dt_gp_col + pred,
                 "gbt_q16": dt_gp_col + q16, "gbt_q84": dt_gp_col + q84,
                 "quality": quality, "residual": res_gbt})
    path = benchmark_path("gbt_baseline_predictions.ecsv")
    out.write(path, overwrite=True)
    print(f"\nSaved out-of-fold predictions -> {path}")


if __name__ == "__main__":
    main()
