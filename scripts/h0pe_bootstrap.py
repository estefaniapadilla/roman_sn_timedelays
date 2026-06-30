#!/usr/bin/env python3
"""
h0pe_bootstrap.py
=================
BayeSN + SNTD time-delay measurement on bootstrapped SN H0pe photometry.

Previously: sntd_bayesn.py
New location: scripts/h0pe_bootstrap.py

This is NOT a general-purpose tool. It is hardcoded for SN H0pe:
  - JWST/NIRCam bands (F090W, F150W, F200W, F277W, F356W, F444W)
  - Exactly 3 images labelled a, b, c
  - Source redshift z=1.78
  - t0 bounds and delay windows tuned to the H0pe system

For Roman population analysis, use run_bayesn_population.py instead.

Performance note: ncpu=64 in fit_iteration() is aggressive and suited to
a cluster node with many cores. On a laptop, set ncpu=4 or ncpu=-1 (all cores).

Run example:
python scripts/h0pe_bootstrap.py \\
  --in_glob "/path/bootstraps/*.ecsv" \\
  --bayesn_yaml "/path/BAYESN.YAML" \\
  --filters_yaml "/path/filters.yaml" \\
  --outdir "/path/boot_out" \\
  --summary_csv "/path/boot_out/summary.csv"
"""

from __future__ import annotations

import os
import sys
# Add repo root to path so both roman_td and bayesncosmo are importable
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import glob
import json
import pickle
import time
import traceback
from typing import Dict, Any, Optional, List

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for cluster jobs
import matplotlib.pyplot as plt

import numpy as np
import astropy.table as at
import scipy.stats

from tqdm import tqdm

import sncosmo
import sntd
import sntd.fitting as _sntd_fitting
from roman_td.bayesncosmo import BAYESNSource, FixedRVDust, FreeRVDust

# corner is optional; degrades gracefully if not installed
try:
    import corner
    _HAS_CORNER = True
except Exception:
    _HAS_CORNER = False


def log(msg: str) -> None:
    tqdm.write(f"[H0PE BOOTSTRAP] {msg}")


def _to_builtin(x):
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def _safe_get(block, name: str, default=None):
    """Duck-typed accessor for SNTD result blocks (see bayesn_wrapper._safe_get)."""
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


def assign_visits(tab: at.Table, dt_tol: float = 1.0) -> at.Table:
    """Cluster MJDs per image into visits within dt_tol days, replace with mean MJD.

    H0pe observations sometimes have two visits within the same night.
    SNTD treats each MJD as a distinct epoch, so near-simultaneous visits
    cause degenerate color curves. Clustering prevents this.
    """
    tab = tab.copy()
    visit_col = np.zeros(len(tab), dtype=float)

    for img in np.unique(tab["image"]):
        idx = np.where(tab["image"] == img)[0]
        sub = tab[idx]
        order = np.argsort(sub["mjd"])
        idx = idx[order]
        mjd = np.array(tab["mjd"][idx], float)

        visit_id = np.zeros(len(mjd), dtype=int)
        v = 0
        start = 0
        for i in range(1, len(mjd)):
            if (mjd[i] - mjd[start]) > dt_tol:
                v += 1
                start = i
            visit_id[i] = v

        for vid in np.unique(visit_id):
            mask = visit_id == vid
            visit_col[idx[mask]] = float(np.mean(mjd[mask]))

    tab["mjd"] = visit_col
    tab.sort(["image", "mjd", "filter"])
    return tab


def preprocess_phot_table(
    tab: at.Table,
    visit_dt_tol: float = 1.0,
    force_ab: bool = True,
    img_map: Optional[Dict[str, str]] = None,
) -> at.Table:
    tab = tab.copy()

    if force_ab:
        tab["zpsys"] = ["ab"] * len(tab)

    tab["filter"] = [str(f).upper() for f in tab["filter"]]

    # BayeSN spectral range ~3800-8700 Å rest-frame. At z=1.78 only these
    # NIRCam bands have rest-frame coverage inside that range.
    BAYESN_BANDS = {"F090W", "F150W", "F200W", "F277W", "F356W", "F444W"}
    tab = tab[np.isin(tab["filter"], list(BAYESN_BANDS))]

    # NaN zp poisons sncosmo bandflux → NaN log-likelihood → dynesty fails
    zp_arr = np.array(tab["zp"], dtype=float)
    if not np.all(np.isfinite(zp_arr)):
        tab["zp"] = np.where(np.isfinite(zp_arr), zp_arr, float(np.nanmedian(zp_arr)))

    if img_map is None:
        img_map = {"2a": "a", "2b": "b", "2c": "c", "a": "a", "b": "b", "c": "c"}
    tab["image"] = [img_map.get(str(x), str(x)) for x in tab["image"]]

    tab = assign_visits(tab, dt_tol=visit_dt_tol)
    return tab


def peak_mjd_image_b(tab: at.Table) -> float:
    """Return MJD of peak flux for image b (the reference image in H0pe fits)."""
    sub = tab[tab["image"] == "b"]
    if len(sub) == 0:
        raise ValueError("No image 'b' rows found in photometry table.")
    idx = np.argmax(np.array(sub["flux"], dtype=float))
    return float(sub["mjd"][idx])


def make_misn_from_table(tab: at.Table, object_name: str = "SN_H0pe"):
    image_tabs = []
    for img in ["a", "b", "c"]:
        sub = tab[tab["image"] == img]
        if len(sub) == 0:
            raise ValueError(f"Missing image '{img}' in this file.")
        image_tabs.append(sub)
    return sntd.table_factory(image_tabs, telescopename="JWST", object_name=object_name)


def register_bayesn_source(yaml_path: str, name: str = "bayesn") -> None:
    log("Registering BayeSN source")
    bsource = BAYESNSource(model_name=yaml_path, use_epsilon=False, fit_epsilon=False)
    sncosmo.register(bsource, name=name, data_class=sncosmo.Source, force=True)


def register_jwst_bandpasses(filters_yaml: str) -> None:
    import yaml
    from sncosmo import Bandpass

    log(f"Registering JWST bandpasses from {filters_yaml}")
    with open(filters_yaml) as f:
        cfg = yaml.safe_load(f)

    root = cfg["filters_root"]
    for name, meta in cfg["filters"].items():
        wave, thr = np.loadtxt(os.path.join(root, meta["path"]), unpack=True)
        if np.nanmedian(wave) < 100:
            wave = wave * 1e4
        sncosmo.registry.register(Bandpass(wave, thr, name=name), force=True)
        sncosmo.registry.register(Bandpass(wave, thr, name=name.lower()), force=True)
    sncosmo.get_magsystem("ab")
    log("JWST bandpasses registered.")


def _extract_res_dict(res) -> Dict[str, Any]:
    out = {}
    for a in ["samples", "weights", "logl", "vparam_names"]:
        v = getattr(res, a, None)
        if v is not None:
            try:
                out[a] = _to_builtin(v)
            except Exception:
                out[a] = "EXTRACT_FAILED"
    errors = getattr(res, "errors", None)
    if errors is not None:
        try:
            out["errors"] = {k: _to_builtin(v) for k, v in errors.items()}
        except Exception:
            out["errors"] = "EXTRACT_FAILED"
    return out


def extract_fit_payload(fit_obj: Any, method: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"method": method}
    block = getattr(fit_obj, method)

    for key in ["time_delays", "time_delay_errors", "magnifications", "magnification_errors"]:
        val = _safe_get(block, key, None)
        if val is not None:
            out[key] = _to_builtin(val)

    post = {}
    if method in ("color", "series"):
        fits = _safe_get(block, "fits", None)
        res = _safe_get(fits, "res", None) if fits is not None else None
        if res is not None:
            post = _extract_res_dict(res)
        else:
            post["error"] = f"fits.res not found on {method} block"
    elif method == "parallel":
        images = getattr(fit_obj, "images", {})
        for img_name, img in images.items():
            fits = _safe_get(img, "fits", None)
            res = _safe_get(fits, "res", None) if fits is not None else None
            if res is not None:
                post[img_name] = _extract_res_dict(res)

    out["posterior"] = post
    return out


def save_corner_plot(block: Any, outfile: str, label: str) -> None:
    if not _HAS_CORNER:
        log("corner not installed; skipping corner plot")
        return
    try:
        samples = _safe_get(block, "samples", None)
        if samples is None:
            res = _safe_get(block, "results", None)
            if res is not None and hasattr(res, "samples"):
                samples = res.samples
        if samples is None:
            log(f"No posterior samples found for {label}; skipping corner plot")
            return
        fig = corner.corner(np.asarray(samples))
        fig.suptitle(label)
        fig.savefig(outfile, dpi=200)
        plt.close(fig)
        log(f"Saved corner plot → {outfile}")
    except Exception as e:
        log(f"Corner plot failed for {label}: {e}")


def save_lightcurve_plot(misn: Any, outfile: str) -> None:
    try:
        fig = misn.plot_object()
        fig.savefig(outfile, dpi=200)
        plt.close(fig)
        log(f"Saved light curve plot → {outfile}")
    except Exception as e:
        log(f"Light curve plot failed: {e}")


def _wquantile(x: np.ndarray, q: float, w: np.ndarray) -> float:
    idx = np.argsort(x)
    cdf = np.cumsum(w[idx]) / w.sum()
    return float(np.interp(q, cdf, x[idx]))


def fit_iteration(
    misn: Any,
    z: float,
    t0_approx: float,
    verbose: bool,
    npoints_series: int = 500,
    npoints_color: int = 500,
) -> tuple:
    """Run BayeSN series + color fit on a H0pe MISN.

    Bounds are tuned for H0pe: dt_a in [70,150] days, dt_c in [0,80] days.
    For a general system use bayesn_wrapper.fit_system() instead.
    """
    timing: Dict[str, float] = {}

    log("Running SERIES fit")
    t_start = time.time()

    dust_mw = FixedRVDust(r_v=3.1)
    dust_host = FreeRVDust()
    model = sncosmo.Model("bayesn",
        effects=[dust_mw, dust_host],
        effect_names=["mw", "host"],
        effect_frames=["obs", "rest"],
    )
    model.set(z=z, mwebv=0.019)  # H0pe MW extinction is small but non-zero

    series_bounds = {
        "t0": np.array([-110, 110]),
        "dt_a": np.array([70, 150]),
        "dt_c": np.array([0, 80]),
        "mu": (0.1, 2),
        "theta": (-3, 3),
        "hostebv": (0, 1),
        # Narrow amplitude bound avoids the amplitude≈0 singularity that causes
        # BayeSN to return NaN bandflux and poisons the log-likelihood.
        "amplitude": (5e-20, 1e-16),
    }

    fit_series = sntd.single_model_series_delays(
        misn, model,
        ["t0", "theta", "hostebv", "amplitude"],
        shared_parameters=["theta", "hostebv"],
        referenceImage="b",
        bounds=series_bounds,
        npoints=npoints_series,
        use_MLE=False,
        modelcov=False,
        priors={"theta": lambda x: float(scipy.stats.norm.pdf(x, 0, 1))},
        t0_guess={"a": t0_approx - 115, "b": t0_approx, "c": t0_approx - 50},
        ncpu=64,
    )

    timing["series_sec"] = time.time() - t_start
    log(f"Series fit time: {timing['series_sec']:.2f} sec")

    series_block = getattr(fit_series, "series", None)
    series_fits = _safe_get(series_block, "fits", None)
    series_res = _safe_get(series_fits, "res", None) if series_fits is not None else None

    if series_res is None:
        raise RuntimeError("Series fit produced no posterior (fits.res is None).")

    samples = np.asarray(series_res.samples)
    weights = np.asarray(series_res.weights)
    vp = list(series_res.vparam_names)

    theta_idx = vp.index("theta")
    theta_val = _wquantile(samples[:, theta_idx], 0.5, weights)
    theta_std = np.sqrt(
        np.sum(weights * (samples[:, theta_idx] - theta_val) ** 2) / np.sum(weights)
    )
    log(f"Series theta (weighted median): {theta_val:.4f}  std: {theta_std:.4f}")

    t0_idx = vp.index("t0_b")
    t0_med = _wquantile(samples[:, t0_idx], 0.5, weights)
    t0_lo = _wquantile(samples[:, t0_idx], 0.0228, weights)
    t0_hi = _wquantile(samples[:, t0_idx], 0.9772, weights)
    log(f"Series t0_b: {t0_med:.2f}  2σ bounds [{t0_lo:.2f}, {t0_hi:.2f}]")

    log(f"Running COLOR fit (theta prior: mean={theta_val:.4f} std={theta_std:.4f})")
    t_start = time.time()

    theta_prior = scipy.stats.norm(theta_val, theta_std).pdf
    theta_bound = (theta_val - 5 * theta_std, theta_val + 5 * theta_std)

    color_bounds = {
        "t0": np.array([t0_lo - t0_med, t0_hi - t0_med]),
        "dt_a": np.array([80, 150]),
        "dt_c": np.array([0, 80]),
        "hostebv": (0, 1),
        "hostr_v": (1.0, 4.0),
        "amplitude": (5e-20, 1e-16),
        "theta": theta_bound,
    }

    fit_color = sntd.single_model_color_delays(
        fit_series, model,
        ["t0", "hostebv", "hostr_v", "theta"],
        shared_parameters=["hostebv", "hostr_v"],
        referenceImage="b",
        bounds=color_bounds,
        t0_guess={"b": t0_med},
        npoints=npoints_color,
        ncpu=64,
        priors={"theta": theta_prior},
    )

    timing["color_sec"] = time.time() - t_start
    log(f"Color fit time: {timing['color_sec']:.2f} sec")
    timing["total_sec"] = timing["series_sec"] + timing["color_sec"]

    return fit_series, fit_color, timing


def save_summary(summary_csv: str, name: str, fit_color: Any) -> None:
    block = getattr(fit_color, "color")
    td = _safe_get(block, "time_delays", {})
    td_err = _safe_get(block, "time_delay_errors", {})
    mu = _safe_get(block, "magnifications", None) or _safe_get(block, "mu", {})
    mu_err = _safe_get(block, "magnification_errors", None) or _safe_get(block, "mu_err", {})
    if mu is None:
        mu = {}
    if mu_err is None:
        mu_err = {}

    log(f"Summary fields for {name}: td={'OK' if td else 'EMPTY'}, mu={'OK' if mu else 'MISSING/EMPTY'}")

    write_header = (not os.path.exists(summary_csv)) or (os.path.getsize(summary_csv) == 0)
    os.makedirs(os.path.dirname(summary_csv), exist_ok=True)

    with open(summary_csv, "a") as f:
        if write_header:
            f.write("iteration,time_delays,time_delay_errors,magnifications,magnification_errors\n")
        f.write(
            f"{name},"
            f"{json.dumps(td, default=_to_builtin)},"
            f"{json.dumps(td_err, default=_to_builtin)},"
            f"{json.dumps(mu, default=_to_builtin)},"
            f"{json.dumps(mu_err, default=_to_builtin)}\n"
        )
    log(f"Updated summary CSV → {summary_csv}")


def _pkl_res(outdir: str, name: str, block: Any, label: str) -> None:
    fits = _safe_get(block, "fits", None)
    if fits is None:
        return
    res = _safe_get(fits, "res", None)
    if res is not None:
        res_data = {}
        for field in ["samples", "weights", "logl", "vparam_names", "errors",
                      "niter", "ncall", "logz", "logzerr", "information"]:
            val = getattr(res, field, None)
            if val is not None:
                res_data[field] = val
        pkl_path = os.path.join(outdir, f"{name}__{label}_res.pkl")
        try:
            with open(pkl_path, "wb") as f:
                pickle.dump(res_data, f)
            log(f"Saved {label} res → {pkl_path}")
        except Exception as e:
            log(f"WARNING: could not pickle {label} res ({type(e).__name__}: {e})")

    tab = _safe_get(fits, "table", None)
    if tab is not None:
        pkl_tab = os.path.join(outdir, f"{name}__{label}_fittab.pkl")
        try:
            with open(pkl_tab, "wb") as f:
                pickle.dump(tab, f)
            log(f"Saved {label} fit table → {pkl_tab}")
        except Exception as e:
            log(f"WARNING: could not pickle {label} fit table ({type(e).__name__}: {e})")


def save_full_outputs(outdir, name, fit_series, fit_color, timing):
    os.makedirs(outdir, exist_ok=True)
    payload = {
        "iteration": name, "timing_sec": timing,
        "series": extract_fit_payload(fit_series, "series"),
        "color": extract_fit_payload(fit_color, "color"),
    }
    json_path = os.path.join(outdir, f"{name}__fit_payload.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, default=_to_builtin)
    log(f"Saved fit payload → {json_path}")

    series_block = getattr(fit_series, "series", None)
    if series_block is not None:
        _pkl_res(outdir, name, series_block, "series")
    color_block = getattr(fit_color, "color", None)
    if color_block is not None:
        _pkl_res(outdir, name, color_block, "color")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_glob", required=True)
    ap.add_argument("--bayesn_yaml", required=True)
    ap.add_argument("--filters_yaml", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--summary_csv", required=True)
    ap.add_argument("--visit_dt_tol", type=float, default=1.0)
    ap.add_argument("--z", type=float, default=1.78)  # H0pe redshift
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    register_bayesn_source(args.bayesn_yaml, name="bayesn")
    register_jwst_bandpasses(args.filters_yaml)

    files = sorted(glob.glob(args.in_glob))
    if len(files) == 0:
        raise SystemExit(f"No files matched: {args.in_glob}")

    log(f"Found {len(files)} bootstrap files")
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.dirname(args.summary_csv), exist_ok=True)

    total_runtime = 0.0
    pbar = tqdm(files, desc="Bootstraps", unit="file")
    for i, fp in enumerate(pbar):
        name = os.path.splitext(os.path.basename(fp))[0]
        pbar.set_postfix_str(name)
        log("==========================================")
        log(f"Processing {i+1}/{len(files)} → {name}")

        try:
            tab = at.Table.read(fp)
            tab = preprocess_phot_table(tab, visit_dt_tol=args.visit_dt_tol)
            misn = make_misn_from_table(tab, object_name="SN_H0pe")
            t0_approx = peak_mjd_image_b(tab)
            log(f"t0_approx from image-b peak flux: {t0_approx:.2f} MJD")

            iter_start = time.time()
            fit_series, fit_color, timing = fit_iteration(
                misn, z=args.z, t0_approx=t0_approx, verbose=args.verbose,
            )
            iter_time = time.time() - iter_start
            total_runtime += iter_time

            log(f"Iteration runtime: {iter_time/60:.2f} min")
            log(f"Average per bootstrap: {(total_runtime/(i+1))/60:.2f} min")

            save_summary(args.summary_csv, name, fit_color)
            save_full_outputs(args.outdir, name, fit_series, fit_color, timing)
            log(f"Finished {name}")

        except Exception:
            log(f"ERROR in {name}")
            errfile = os.path.join(args.outdir, f"{name}__ERROR.txt")
            os.makedirs(args.outdir, exist_ok=True)
            with open(errfile, "w") as f:
                f.write(traceback.format_exc())
            log(f"Error saved → {errfile}")

    log("All done.")


if __name__ == "__main__":
    main()
