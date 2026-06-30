"""
bayesn_wrapper.py
=================
BayeSN + SNTD two-stage time-delay fitter for Roman lensed SN populations.

Previously: the reusable functions from sntd_bayesn_population.py
New import:  from roman_td.bayesn_wrapper import fit_system, ensure_registered, ...

Why BayeSN instead of SALT2?
-----------------------------
SALT2's color parameter c conflates intrinsic SN color with host galaxy dust.
BayeSN parameterises them separately (theta for intrinsic shape, AV+RV for dust),
which gives less-biased magnification estimates — important when you want to
disentangle macro-lensing magnification from the SN's true brightness.

Two-stage fitting strategy
--------------------------
Series fit (flux space):
  - Fits all images simultaneously using the full multi-band SED
  - Constrains amplitude (→ magnification), theta, hostebv, t0, and dt_ offsets
  - Amplitude-sensitive: good for magnification, but amplitude and time delay
    are partially degenerate (brighter image could be earlier or more magnified)

Color fit (flux ratio between adjacent bands):
  - Amplitude cancels in ratios → pure time-delay + dust constraint
  - Uses theta posterior from series as a Gaussian prior to break color-dust degeneracy
  - This is the final time-delay estimate

Performance
-----------
Both stages use dynesty nested sampling. The main cost is the number of
likelihood evaluations: O(npoints × n_live × n_params). With npoints=500
and 6 parameters, expect ~50,000 BayeSN evaluations per stage.
Reducing npoints to 100-200 is usually sufficient for well-sampled Roman curves.
"""

from __future__ import annotations

import os
import time
import warnings
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import astropy.table as at
import scipy.stats
from tqdm import tqdm

ZP = 27.0      # must match simulate.py
ZPSYS = "ab"
MWEBV = 0.0


def log(msg: str) -> None:
    tqdm.write(f"[BAYESN] {msg}")


def _to_builtin(x):
    """Convert numpy scalars/arrays to Python builtins for JSON serialization."""
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def _safe_get(block, name: str, default=None):
    """Duck-typed accessor for SNTD result blocks.

    SNTD blocks behave like objects in some versions and dicts in others;
    missing fields can raise either AttributeError or KeyError.
    """
    try:
        return getattr(block, name)
    except (AttributeError, KeyError):
        pass
    try:
        return block[name]
    except Exception:
        pass
    try:
        return block.get(name, default)
    except Exception:
        return default


def _wquantile(x: np.ndarray, q: float, w: np.ndarray) -> float:
    """Weighted quantile via linear interpolation of the weighted CDF."""
    idx = np.argsort(x)
    cdf = np.cumsum(w[idx]) / w.sum()
    return float(np.interp(q, cdf, x[idx]))


def ensure_registered(bayesn_yaml: str, filters_yaml: str) -> None:
    """Register BayeSN source and Roman bandpasses in the sncosmo registry.

    Called once per worker process. joblib spawns new processes for each
    parallel job, so each worker's sncosmo registry starts empty — we cannot
    register in the main process and expect workers to inherit it.
    The try/get guard prevents double-registration within a single process.
    """
    import sncosmo
    try:
        sncosmo.get_source("bayesn")
        return  # already registered in this process
    except Exception:
        pass

    from roman_td.bayesncosmo import BAYESNSource
    bsource = BAYESNSource(model_name=bayesn_yaml, use_epsilon=False, fit_epsilon=False)
    sncosmo.register(bsource, name="bayesn", data_class=sncosmo.Source, force=True)

    import yaml
    from sncosmo import Bandpass
    with open(filters_yaml) as f:
        cfg = yaml.safe_load(f)
    root = cfg["filters_root"]
    for name, meta in cfg["filters"].items():
        wave, thr = np.loadtxt(os.path.join(root, meta["path"]), unpack=True)
        if np.nanmedian(wave) < 100:   # microns → Angstrom
            wave = wave * 1e4
        sncosmo.registry.register(Bandpass(wave, thr, name=name), force=True)
        sncosmo.registry.register(Bandpass(wave, thr, name=name.lower()), force=True)
    sncosmo.get_magsystem("ab")


def make_model(z: float):
    """Create a fresh BayeSN sncosmo model with MW + host dust components.

    MW dust: FixedRV (RV=3.1) — Milky Way extinction is fixed by dust maps
    Host dust: FreeRV — host galaxy RV is a free parameter, typically 1-4
    theta: BayeSN's intrinsic shape/brightness parameter (like SALT2's x1 but
           physically calibrated; theta~0 is average, positive = brighter/slower)
    """
    import sncosmo
    from roman_td.bayesncosmo import FixedRVDust, FreeRVDust
    model = sncosmo.Model(
        "bayesn",
        effects=[FixedRVDust(r_v=3.1), FreeRVDust()],
        effect_names=["mw", "host"],
        effect_frames=["obs", "rest"],
    )
    model.set(z=z, mwebv=MWEBV)
    return model


def usable_bands(bands: List[str], z: float) -> List[str]:
    """Return bands whose rest-frame coverage falls inside the BayeSN spectral range.

    BayeSN is trained on rest-frame ~3800–8700 Å. At high redshift some Roman
    filters shift outside this range and the model extrapolates unreliably.
    """
    import sncosmo
    src = sncosmo.get_source("bayesn")
    lo, hi = src.minwave(), src.maxwave()
    keep = []
    for b in bands:
        bp = sncosmo.get_bandpass(b)
        if bp.minwave() / (1 + z) >= lo and bp.maxwave() / (1 + z) <= hi:
            keep.append(b)
    return keep


def make_misn_from_table(tab: at.Table, images: List[str], object_name: str):
    """Build a SNTD MISN object from a photometry table with an 'image' column."""
    import sntd
    image_tabs = []
    for img in images:
        sub = tab[tab["image"] == img]
        if len(sub) == 0:
            raise ValueError(f"Missing image '{img}' in simulated table.")
        image_tabs.append(sub)
    return sntd.table_factory(image_tabs, telescopename="Roman", object_name=object_name)


def peak_mjd(tab: at.Table, img: str) -> float:
    """Return the MJD of peak flux for a given image label (across all bands)."""
    sub = tab[tab["image"] == img]
    idx = np.argmax(np.array(sub["flux"], dtype=float))
    return float(sub["mjd"][idx])


def estimate_amplitude_bounds(
    tab: at.Table, model, ref_img: str
) -> Tuple[float, Tuple[float, float]]:
    """Data-driven amplitude guess from the reference image's peak flux.

    The 300x range on the bounds is wide enough to cover lensing magnification
    uncertainties while avoiding the amplitude=0 singularity that makes BayeSN
    return NaN bandflux (which poisons the log-likelihood).
    """
    sub = tab[tab["image"] == ref_img]
    i = int(np.argmax(np.array(sub["flux"], dtype=float)))
    band, mjd, flux = str(sub["filter"][i]), float(sub["mjd"][i]), float(sub["flux"][i])
    orig_amp, orig_t0 = model.get("amplitude"), model.get("t0")
    model.set(amplitude=1.0, t0=mjd)
    f1 = float(model.bandflux(band, mjd, zp=ZP, zpsys=ZPSYS))
    model.set(amplitude=orig_amp, t0=orig_t0)
    amp = max(flux, 1e-30) / max(f1, 1e-300)
    return amp, (amp / 300.0, amp * 300.0)


def fit_system(
    misn,
    tab: at.Table,
    z: float,
    images: List[str],
    npoints_series: int,
    npoints_color: int,
    ncpu_fit: int,
    delay_window: float = 40.0,
) -> Tuple[Any, Any, Dict[str, float], str]:
    """Run the BayeSN two-stage series → color fit on a MISN object.

    Parameters
    ----------
    npoints_series / npoints_color : int
        dynesty live points per stage. 500 is conservative; 100-200 is usually
        sufficient for well-sampled Roman light curves (5-day cadence).
        Halving npoints roughly halves wall-clock time.
    delay_window : float
        Half-width of the search window for each time delay offset, in days.
        Initialized from peak-flux offsets ± delay_window. 40 days is
        generous for galaxy-scale lenses (typical delays are 1-100 days).

    Returns
    -------
    (params_s, res_s), (params_c, res_c) or None, timing_dict, ref_image_label
    """
    import sntd.fitting

    timing: Dict[str, float] = {}

    peaks = {img: peak_mjd(tab, img) for img in images}
    fluxes = {img: float(np.max(tab[tab["image"] == img]["flux"])) for img in images}
    ref = max(fluxes, key=fluxes.get)   # brightest image = reference
    others = [i for i in images if i != ref]
    nimage = len(images)

    model = make_model(z)
    amp_guess, amp_bounds = estimate_amplitude_bounds(tab, model, ref)
    t0_ref = peaks[ref]

    series_vparams = ["t0", "theta", "hostebv", "amplitude"] + [f"dt_{img}" for img in others]
    series_bounds: Dict[str, Any] = {
        "t0": (t0_ref - 1.5 * delay_window, t0_ref + 1.5 * delay_window),
        "theta": (-3.0, 3.0),
        "hostebv": (0.0, 1.0),
        "amplitude": amp_bounds,
    }
    for img in others:
        off = peaks[img] - peaks[ref]
        series_bounds[f"dt_{img}"] = (off - delay_window, off + delay_window)

    # ----- SERIES FIT -----
    t_start = time.time()
    params_s, res_s, model = sntd.fitting.nest_series_lc(
        misn["table"], model, nimage, series_vparams, series_bounds,
        ref=ref, use_MLE=False, npoints=npoints_series,
        priors={"theta": lambda x: float(scipy.stats.norm.pdf(x, 0, 1))},
        modelcov=False,
    )
    timing["series_sec"] = time.time() - t_start

    vp_s = list(res_s.vparam_names)
    theta_idx = vp_s.index("theta")
    t0_idx = vp_s.index("t0")

    theta_lo, theta_med, theta_hi = params_s[theta_idx]
    theta_std = max((theta_hi - theta_lo) / 2.0, 1e-3)

    # Use 2-sigma t0 range from the series posterior to tighten the color fit bounds.
    # This is how the two-stage method gains speed: the color fit samples a much
    # smaller t0 volume than it would with flat priors.
    samples = np.asarray(res_s.samples)
    weights = np.asarray(res_s.weights)
    t0_2lo = _wquantile(samples[:, t0_idx], 0.0228, weights)
    t0_2hi = _wquantile(samples[:, t0_idx], 0.9772, weights)

    bands_in_data = sorted(set(str(b) for b in misn["table"]["band"]))
    color_pairs = [(bands_in_data[j], bands_in_data[j + 1])
                   for j in range(len(bands_in_data) - 1)]

    if not color_pairs:
        timing["color_sec"] = 0.0
        timing["total_sec"] = timing["series_sec"]
        return (params_s, res_s), None, timing, ref

    # Build color table without pre-shifting (static=True); SNTD handles shifts
    misn.color_table(
        [p[0] for p in color_pairs],
        [p[1] for p in color_pairs],
        time_delays={img: 0.0 for img in images},
        referenceImage=ref,
        static=True,
    )

    if len(misn.color.table) == 0:
        timing["color_sec"] = 0.0
        timing["total_sec"] = timing["series_sec"]
        return (params_s, res_s), None, timing, ref

    # ----- COLOR FIT -----
    # hostr_v is free here: the color curve shape constrains it independently
    # of amplitude, which is why it was not well-constrained in series.
    color_vparams = ["t0", "theta", "hostebv", "hostr_v"] + [f"dt_{img}" for img in others]
    color_bounds: Dict[str, Any] = {
        "t0": (t0_2lo, t0_2hi),
        "theta": (float(theta_med) - 5.0 * theta_std, float(theta_med) + 5.0 * theta_std),
        "hostebv": (0.0, 1.0),
        "hostr_v": (1.0, 4.0),
    }
    for img in others:
        color_bounds[f"dt_{img}"] = series_bounds[f"dt_{img}"]

    theta_prior = scipy.stats.norm(float(theta_med), float(theta_std)).pdf

    t_start = time.time()
    params_c, res_c, model = sntd.fitting.nest_color_lc(
        misn.color.table, model, nimage, color_pairs, color_vparams, color_bounds,
        ref=ref, use_MLE=False, npoints=npoints_color,
        priors={"theta": theta_prior}, modelcov=False,
    )
    timing["color_sec"] = time.time() - t_start
    timing["total_sec"] = timing["series_sec"] + timing["color_sec"]

    return (params_s, res_s), (params_c, res_c), timing, ref


def result_to_rows(res: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert a process_one_lens result dict to benchmark table rows.

    Re-references true delays to whichever image SNTD chose as reference
    (the brightest one, not necessarily image-a) so true and fitted delays
    are always measured from the same baseline.
    """
    rows = []
    images = res["images"]
    true0 = res["true_delays"]

    fit = res["fit_delays"]
    ref_fit = None
    for img, v in fit.items():
        if np.isclose(float(v), 0.0):
            ref_fit = img
            break
    if ref_fit is None or ref_fit not in true0:
        ref_fit = images[0]

    for img in images:
        if img == ref_fit:
            continue
        td_true = true0.get(img, np.nan) - true0.get(ref_fit, 0.0)
        td_fit = float(fit.get(img, np.nan))
        td_err = res["fit_delay_errors"].get(img, [np.nan, np.nan])
        if np.ndim(td_err) == 0:
            td_err = [td_err, td_err]
        rows.append({
            "lens_index": res["lens_index"], "image": img,
            "z_lens": res["z_lens"], "z_source": res["z_source"],
            "n_images": res["n_images"],
            "true_delay": td_true, "fit_delay": td_fit,
            "fit_err_lo": float(td_err[0]), "fit_err_hi": float(td_err[-1]),
            "residual": td_fit - td_true,
        })
    return rows


def save_benchmark(rows: List[Dict[str, Any]], path: str) -> None:
    if not rows:
        return
    at.Table(rows).write(path, overwrite=True)
