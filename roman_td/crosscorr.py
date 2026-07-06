"""
crosscorr.py
============
Stage 1: fast GP-based cross-correlator for time-delay estimation.

No SN model, no sampler, no likelihood — each image's light curve is
interpolated with a Gaussian process (Matern-3/2), the GP means are
cross-correlated over trial lags, and the best-aligning lag is the delay
estimate. Runs in seconds per system, vs. minutes for the SNTD/BayeSN fits.

Role in the pipeline (documents/pipeline_architecture.md §2):
- coarse dt ± sigma  -> primes Stage 3's narrow t0 windows (upgrades "fast"
  mode from risky to trustworthy) and becomes a Stage 2 input feature
- flux ratio         -> total (macro x micro) magnification-ratio proxy
- quality flag       -> routing rule: "good" -> GP-primed fast fit,
  "broad"/"multipeak"/"fail" -> robust fit

Sign convention: dt_gp > 0 means the image arrives LATER than the reference
image (its light curve is shifted to later times), matching the sign of the
true delays (arrival_time_image - arrival_time_reference).

Input is the canonical photometry table (§6.1): columns
(mjd, filter, flux, fluxerr, zp, zpsys, image); the legacy column names
(time, band) are also accepted.
"""

import time as _time
import warnings

import numpy as np
from scipy.signal import find_peaks
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

__all__ = ["gp_cross_correlate"]

# GP length-scale bounds (days): the lower bound stops the GP from tracking
# noise point-to-point; the upper stops it over-smoothing the SN peak.
LS_BOUNDS = (3.0, 40.0)
GRID_DT = 0.5          # days; evaluation grid spacing (also the lag step)
MIN_OVERLAP_PTS = 20   # >= 10 days of valid overlap required per trial lag
MIN_CORR = 0.3         # below this peak correlation the estimate is "fail"
BRIGHT_FRAC = 0.3      # "bright" = GP mean above this fraction of its peak
MIN_BRIGHT_OVERLAP = 5 # bright regions of BOTH curves must overlap >= 2.5 d


def _colnames(tab):
    """Return (time_col, band_col) accepting canonical or legacy names."""
    cols = tab.colnames if hasattr(tab, "colnames") else list(tab.keys())
    tcol = "mjd" if "mjd" in cols else "time"
    bcol = "filter" if "filter" in cols else "band"
    return tcol, bcol


def _fit_gp_band(t, f, ferr, t_grid, n_draws, rng):
    """Fit a Matern-3/2 GP to one (image, band) light curve.

    Returns (mean, std, draws, valid) on t_grid, in the input flux units.
    `valid` marks grid points within one length scale of a real observation —
    outside that the GP is extrapolating toward the zero-flux prior mean and
    must not enter the cross-correlation.
    """
    scale = float(np.max(np.abs(f)))
    if scale <= 0 or not np.isfinite(scale):
        return None
    kernel = (ConstantKernel(1.0, (1e-3, 1e2))
              * Matern(length_scale=15.0, length_scale_bounds=LS_BOUNDS, nu=1.5))
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=(ferr / scale) ** 2,
        normalize_y=False,   # zero-mean prior: extrapolation decays to zero flux
        n_restarts_optimizer=0,
        random_state=int(rng.integers(2**31)),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp.fit(t[:, None], f / scale)
        mean, cov = gp.predict(t_grid[:, None], return_cov=True)
    std = np.sqrt(np.clip(np.diag(cov), 0.0, None))

    # Posterior draws by Cholesky — sklearn's sample_y goes through
    # numpy's multivariate_normal, whose internal SVD is ~100x slower.
    n = len(t_grid)
    eps = 1e-10 * max(float(np.trace(cov)) / n, 1e-30)
    for _ in range(6):
        try:
            chol = np.linalg.cholesky(cov + eps * np.eye(n))
            break
        except np.linalg.LinAlgError:
            eps *= 100.0
    else:
        return None
    draws = mean[:, None] + chol @ rng.standard_normal((n, n_draws))

    ls = float(gp.kernel_.k2.length_scale)
    valid = np.zeros(len(t_grid), dtype=bool)
    for ti in t:
        valid |= np.abs(t_grid - ti) <= ls
    return mean * scale, std * scale, draws * scale, valid


def _weighted_corr(a, b, w):
    """Weighted Pearson correlation. a: (n,) or (n, m); b, w: same leading n."""
    w = w / np.sum(w)
    if a.ndim == 1:
        a = a[:, None]
    if b.ndim == 1:
        b = b[:, None]
    am = a - np.sum(w[:, None] * a, axis=0)
    bm = b - np.sum(w[:, None] * b, axis=0)
    cov = np.sum(w[:, None] * am * bm, axis=0)
    va = np.sum(w[:, None] * am * am, axis=0)
    vb = np.sum(w[:, None] * bm * bm, axis=0)
    denom = np.sqrt(va * vb)
    with np.errstate(invalid="ignore", divide="ignore"):
        return cov / denom


def _xcorr_curve(curves_a, var_a, valid_a, curves_b, var_b, valid_b, lag_steps,
                 bright_a, bright_b, lag_mask=None):
    """Correlation vs lag. curves_*: (n,) mean or (n, m) draws on a common grid.

    C[j, col] compares A(t) with B(t + lag_j): a positive lag means B's
    features happen `lag` later than A's, i.e. B arrives later.

    A lag only scores if the BRIGHT parts of both curves overlap — normalized
    correlation of two flat zero-flux baselines is spuriously high, which
    otherwise produces fake peaks at extreme lags. `lag_mask` (bool per lag)
    restricts evaluation further (used to skip lags the mean curve rejected).
    """
    n = len(var_a)
    m = curves_a.shape[1] if curves_a.ndim > 1 else 1
    out = np.full((len(lag_steps), m), np.nan)
    for j, k in enumerate(lag_steps):
        if lag_mask is not None and not lag_mask[j]:
            continue
        # A[i] vs B[i + k]
        if n - abs(k) < MIN_OVERLAP_PTS:
            continue  # lag near/beyond grid length: slice(0, n-k) would wrap
        if k >= 0:
            sl_a, sl_b = slice(0, n - k), slice(k, n)
        else:
            sl_a, sl_b = slice(-k, n), slice(0, n + k)
        if np.count_nonzero(bright_a[sl_a] & bright_b[sl_b]) < MIN_BRIGHT_OVERLAP:
            continue
        ok = valid_a[sl_a] & valid_b[sl_b]
        if np.count_nonzero(ok) < MIN_OVERLAP_PTS:
            continue
        w = 1.0 / (var_a[sl_a][ok] + var_b[sl_b][ok])
        out[j] = _weighted_corr(curves_a[sl_a][ok], curves_b[sl_b][ok], w)
    return out.ravel() if m == 1 else out


def _refine_peak(lags_days, c):
    """Parabolic interpolation of the correlation maximum."""
    j = int(np.nanargmax(c))
    if 0 < j < len(c) - 1 and np.all(np.isfinite(c[j - 1:j + 2])):
        y0, y1, y2 = c[j - 1], c[j], c[j + 1]
        denom = y0 - 2 * y1 + y2
        if denom < 0:
            step = lags_days[1] - lags_days[0]
            return float(lags_days[j] + 0.5 * (y0 - y2) / denom * step)
    return float(lags_days[j])


def _classify(lags_days, c, dt_err):
    """Quality flag from the correlation curve + draw-scatter uncertainty.

    Deliberately NOT based on the correlation peak's width: smooth SN light
    curves always produce a wide correlation peak, so width reflects the
    curve shape, not the precision of the estimate. The draw scatter
    (dt_err) is the honest precision measure.
    """
    finite = np.isfinite(c)
    if not np.any(finite) or np.nanmax(c) < MIN_CORR:
        return "fail"
    cmax = float(np.nanmax(c))
    c_filled = np.where(finite, c, -1.0)
    peaks, _ = find_peaks(c_filled, height=0.8 * cmax)
    j_main = int(np.nanargmax(c))
    rivals = [p for p in peaks
              if abs(lags_days[p] - lags_days[j_main]) > 10.0]
    if rivals:
        return "multipeak"
    if not np.isfinite(dt_err) or dt_err > 5.0:
        return "broad"
    return "good"


def gp_cross_correlate(tab, images, bands=None, lag_range=(-200, 200),
                       n_draws=30, rng=None):
    """GP cross-correlation time-delay estimate for one lens system.

    Parameters
    ----------
    tab : astropy Table
        Canonical photometry table (§6.1): one row per (image, epoch, band)
        with columns (mjd|time, filter|band, flux, fluxerr, image).
    images : list of str
        Image labels present in the table (>= 2).
    bands : list of str or None
        Filters to use; None -> every filter in the table.
    lag_range : (float, float)
        Delay search range in days; must exceed the population's max delay.
    n_draws : int
        GP posterior draws for the uncertainty estimate.
    rng : np.random.Generator or None

    Returns
    -------
    dict — architecture §6.3:
        ref_image, t_peak_ref, wall_time_s, and per_image[img] with keys
        dt_gp, dt_gp_err, dt_per_band, flux_ratio, flux_ratio_err,
        flux_ratio_per_band, quality.
    """
    t_start = _time.time()
    if rng is None:
        rng = np.random.default_rng()
    tcol, bcol = _colnames(tab)
    if bands is None:
        bands = sorted(set(str(b) for b in tab[bcol]))

    t_all = np.asarray(tab[tcol], dtype=float)
    t_grid = np.arange(t_all.min() - 5.0, t_all.max() + 5.0, GRID_DT)
    lag_steps = np.arange(int(round(lag_range[0] / GRID_DT)),
                          int(round(lag_range[1] / GRID_DT)) + 1)
    lags_days = lag_steps * GRID_DT

    # --- GP per (image, band) -------------------------------------------
    gps = {}   # (img, band) -> (mean, std, draws, valid)
    for img in images:
        sel_img = np.asarray(tab["image"]) == img
        for band in bands:
            sel = sel_img & (np.asarray(tab[bcol]).astype(str) == band)
            if np.count_nonzero(sel) < 5:
                continue
            fit = _fit_gp_band(
                np.asarray(tab[tcol][sel], dtype=float),
                np.asarray(tab["flux"][sel], dtype=float),
                np.asarray(tab["fluxerr"][sel], dtype=float),
                t_grid, n_draws, rng,
            )
            if fit is not None:
                gps[(img, band)] = fit

    # --- reference image = brightest GP peak -----------------------------
    peak_flux = {
        img: max((float(np.max(gps[(img, b)][0])) for b in bands
                  if (img, b) in gps), default=-np.inf)
        for img in images
    }
    ref = max(peak_flux, key=peak_flux.get)

    ref_bands = [b for b in bands if (ref, b) in gps]
    best_band = max(ref_bands, key=lambda b: float(np.max(gps[(ref, b)][0])),
                    default=None)
    t_peak_ref = (float(t_grid[np.argmax(gps[(ref, best_band)][0])])
                  if best_band else np.nan)

    result = {
        "ref_image": ref,
        "t_peak_ref": t_peak_ref,
        "per_image": {},
    }

    # --- per image pair (ref, img) ---------------------------------------
    for img in images:
        if img == ref:
            continue
        shared = [b for b in bands if (ref, b) in gps and (img, b) in gps]
        entry = {
            "dt_gp": np.nan, "dt_gp_err": np.nan, "dt_per_band": {},
            "flux_ratio": np.nan, "flux_ratio_err": np.nan,
            "flux_ratio_per_band": {}, "quality": "fail",
        }
        result["per_image"][img] = entry
        if not shared:
            continue

        band_curves = []
        tau_samples = []
        for b in shared:
            mean_a, std_a, draws_a, valid_a = gps[(ref, b)]
            mean_b, std_b, draws_b, valid_b = gps[(img, b)]
            var_a, var_b = std_a ** 2, std_b ** 2
            bright_a = valid_a & (mean_a > BRIGHT_FRAC * np.max(mean_a[valid_a]))
            bright_b = valid_b & (mean_b > BRIGHT_FRAC * np.max(mean_b[valid_b]))

            c_mean = _xcorr_curve(mean_a, var_a, valid_a,
                                  mean_b, var_b, valid_b, lag_steps,
                                  bright_a, bright_b)
            lag_ok = np.isfinite(c_mean)
            if not np.any(lag_ok):
                continue
            entry["dt_per_band"][b] = _refine_peak(lags_days, c_mean)
            band_curves.append(c_mean)

            # Draws only evaluated at lags the mean curve accepted (speed).
            c_draws = _xcorr_curve(draws_a, var_a, valid_a,
                                   draws_b, var_b, valid_b, lag_steps,
                                   bright_a, bright_b, lag_mask=lag_ok)
            if c_draws.ndim > 1:
                for mcol in range(c_draws.shape[1]):
                    col = c_draws[:, mcol]
                    if np.any(np.isfinite(col)):
                        tau_samples.append(_refine_peak(lags_days, col))

            # flux ratio: GP peak of img / GP peak of ref, this band
            pa = float(np.max(mean_a[valid_a]))
            pb = float(np.max(mean_b[valid_b]))
            if pa > 0:
                r_draws = (np.max(draws_b[valid_b], axis=0)
                           / np.maximum(np.max(draws_a[valid_a], axis=0), 1e-30))
                entry["flux_ratio_per_band"][b] = {
                    "ratio": pb / pa,
                    "err": float(np.std(r_draws)),
                }

        if not band_curves:
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            c_comb = np.nanmean(np.vstack(band_curves), axis=0)
        if not np.any(np.isfinite(c_comb)):
            continue
        entry["dt_gp"] = _refine_peak(lags_days, c_comb)
        if len(tau_samples) >= 5:
            # Robust sigma (1.4826 * MAD): a few draw outliers must not
            # inflate the reported uncertainty.
            tau = np.asarray(tau_samples)
            entry["dt_gp_err"] = float(
                1.4826 * np.median(np.abs(tau - np.median(tau))))
        else:
            entry["dt_gp_err"] = np.nan
        entry["quality"] = _classify(lags_days, c_comb,
                                     entry["dt_gp_err"]
                                     if np.isfinite(entry["dt_gp_err"]) else np.inf)

        ratios = entry["flux_ratio_per_band"]
        if ratios:
            vals = np.array([v["ratio"] for v in ratios.values()])
            errs = np.array([max(v["err"], 1e-6) for v in ratios.values()])
            w = 1.0 / errs ** 2
            entry["flux_ratio"] = float(np.sum(w * vals) / np.sum(w))
            entry["flux_ratio_err"] = float(np.sqrt(1.0 / np.sum(w)))

    result["wall_time_s"] = _time.time() - t_start
    return result
