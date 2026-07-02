"""
sntd_wrapper.py
===============
Bridge between slsim lens populations and SNTD for time-delay measurement
using the SALT2-extended SN Ia spectral template.

Previously: lensed_sn_td.py
New import:  from roman_td.sntd_wrapper import measure_one, register_roman_bands, DEFAULT_DEPTH_5SIG

Pipeline
--------
1. Extract noisy multi-band light curves from a slsim lens object
   (calls lenstronomy ray-tracing internally). Measured cost: sub-second
   per system in practice — NOT the bottleneck; see fit_time_s below.
2. Build a SNTD MISN (Multi-Image SuperNova) object
3. Run SNTD fit_data(method="parallel"): fits each image independently with
   SALT2-extended using nestle nested sampling (not emcee/MCMC — verified
   against the installed sntd package, which imports nestle and has no
   emcee usage outside its unused old_fitting.py), then reports t0 offsets
   as time delays. This nested-sampling step is the actual dominant cost,
   from tens of seconds to multiple hours per system depending on bounds
   width and sampling budget (see measure_one's t0_window/npoints/maxcall).

Speed vs accuracy trade-off vs BayeSN
--------------------------------------
SALT2's color parameter c conflates intrinsic SN color with host dust,
which biases magnification estimates (though less so for delay measurement).
See bayesn_wrapper.py for the more accurate but slower alternative.
"""

import numpy as _np
# sntd 2.x uses deprecated numpy type aliases (np.float, np.int, np.bool)
# removed in numpy 1.24. Monkey-patching here avoids modifying sntd source.
_np.float = float
_np.int = int
_np.bool = bool

import sncosmo
import sntd
import numpy as np

# nestle weights underflow to 0.0 in float64 when the posterior is much
# narrower than the prior (high-SNR data). exp(logwt - logz) rounds to zero,
# which breaks both nestle.mean_and_cov and SNTD's weighted_quantile call
# (both divide by the weight sum). Fix at the source: after nestle.sample
# returns, recompute weights in log-space to recover the true relative values.
def _patch_nestle_weights():
    try:
        import nestle
        _orig_sample = nestle.sample

        def _robust_sample(*args, **kwargs):
            res = _orig_sample(*args, **kwargs)
            # Always renormalize in log-space. The default weights are
            # exp(logwt - logz), which underflows to 0 when the posterior
            # is narrow (high-SNR data). SNTD multiplies two images' weight
            # arrays together for the delay estimate, so even subnormal
            # weights (~1e-318) produce a product (~1e-636) below float64
            # minimum (~5e-324), giving zero and then NaN in weighted_quantile.
            # Log-space renormalization keeps weights in a normal float range.
            logwt = res.logvol + res.logl
            finite = np.isfinite(logwt)
            if finite.any():
                lw = np.full_like(logwt, -np.inf)
                lw[finite] = logwt[finite] - logwt[finite].max()
                w = np.exp(lw)
                res.weights[:] = w / w.sum()
            return res

        nestle.sample = _robust_sample
    except ImportError:
        pass

_patch_nestle_weights()
from astropy.table import Table
from collections import OrderedDict
import time

# slsim uses uppercase band names; sncosmo ships Roman bands as lowercase
SLSIM_TO_SNCOSMO = {
    "F062": "f062",
    "F087": "f087",
    "F106": "f106",
    "F129": "f129",
    "F158": "f158",
    "F184": "f184",
}

# Per-visit 5-sigma point-source depths for Roman HLTDS deep tier (AB mag).
# Derived from a lenstronomy Roman noise model — deeper than the Hounsell+2018
# scaling used in simulate.py (which targets the wide tier).
DEFAULT_DEPTH_5SIG = {
    "F062": 26.7,
    "F087": 26.9,
    "F106": 27.3,
    "F129": 27.3,
    "F158": 27.6,
    "F184": 28.0,
}

# ZP=25.0 matches the SNANA/sncosmo AB flux convention.
# Must be consistent with whatever ZP is used to generate the photometry.
ZP = 25.0

# Convenience bundles for measure_one()'s independent t0_window/npoints/maxcall
# knobs. Prefer calling measure_one() with the raw knobs directly when trying
# to isolate which one actually matters — these presets are for normal runs.
FIT_PRESETS = {
    "robust": dict(t0_window=None, npoints=None, maxcall=None),
    "fast": dict(t0_window=30.0, npoints=100, maxcall=8000),
}


def register_roman_bands(speclite_filters=None):
    """Verify Roman bandpasses are present in the sncosmo registry.

    sncosmo ships Roman bands as lowercase names (f062, f087, …).
    Raises RuntimeError if any band is missing — upgrade sncosmo or
    register manually from the roman_td/filters.yaml file.
    """
    for slsim_name, sncosmo_name in SLSIM_TO_SNCOSMO.items():
        try:
            sncosmo.get_bandpass(sncosmo_name)
        except Exception:
            raise RuntimeError(
                f"Roman band '{sncosmo_name}' not found in sncosmo registry. "
                f"Upgrade sncosmo or register bands manually."
            )
    print("Roman bands verified in sncosmo")


def _mag_to_flux(mag, zp=ZP):
    return 10.0 ** (-0.4 * (mag - zp))


def _flux_err_from_depth(depth_5sig, zp=ZP):
    """Convert 5-sigma limiting depth (AB mag) to 1-sigma flux uncertainty."""
    return _mag_to_flux(depth_5sig, zp) / 5.0


def extract_light_curves(lens, bands, obs_times, depth_5sig, rng, zp=ZP):
    """Extract noisy multi-band light curves for every lensed image.

    Performance note: lens.point_source_magnitude() calls lenstronomy
    ray-tracing to solve the lens equation for each band × epoch pair.
    At 5-day cadence over 400 days that is ~80 calls per band — this
    dominates runtime for the SNTD path.

    Microlensing is NOT simulated: each image gets a flat magnification
    from the macro-model. Time-varying chromatic microlensing would require
    a magnification map convolved with the SN photosphere size vs. wavelength.

    Returns
    -------
    image_tables : OrderedDict[str, Table]
        Keys "image_1", "image_2", … each with columns
        (time, band, flux, fluxerr, zp, zpsys).
    arrival_times : array
    bands_used : list[str]  (sncosmo names)
    image_snr : dict[str, float]  (peak SNR per image)
    """
    all_arrival_times = lens.point_source_arrival_times()
    arrival_times = np.array(all_arrival_times[0])
    n_images = len(arrival_times)

    mag_cache = {}
    for band in bands:
        try:
            mag_cache[band] = lens.point_source_magnitude(
                band=band, lensed=True, time=obs_times
            )
        except ValueError:
            pass

    image_tables = OrderedDict()
    image_snr = OrderedDict()
    bands_used = set()

    for img_idx in range(n_images):
        rows = []
        for band in bands:
            if band not in mag_cache:
                continue
            sncosmo_band = SLSIM_TO_SNCOSMO.get(band, band)
            sigma = _flux_err_from_depth(depth_5sig.get(band, 26.0), zp)
            img_mags = mag_cache[band][0][img_idx]

            for t_idx, t in enumerate(obs_times):
                m = img_mags[t_idx]
                if not np.isfinite(m):
                    continue
                flux_true = _mag_to_flux(m, zp)
                flux_obs = flux_true + rng.normal(0, sigma)
                rows.append({
                    "time": float(t), "band": sncosmo_band,
                    "flux": float(flux_obs), "fluxerr": float(sigma),
                    "zp": float(zp), "zpsys": "ab",
                })
                bands_used.add(sncosmo_band)

        name = f"image_{img_idx + 1}"
        if rows:
            tbl = Table(rows)
            max_snr = float(np.max(np.abs(tbl["flux"] / tbl["fluxerr"])))
            image_tables[name] = tbl
            image_snr[name] = max_snr

    return image_tables, arrival_times, sorted(bands_used), image_snr


def measure_one(
    lens,
    bands,
    z_source,
    cadence_days=5.0,
    depth_5sig=None,
    rng=None,
    time_range=(-50, 300),
    model_name="salt2-extended",
    lens_index=0,
    max_delay_days=150.0,
    min_image_snr=10.0,
    t0_window=None,
    t0_pad_days=50.0,
    npoints=None,
    maxcall=None,
):
    """Measure time delays for one lensed SN system using SNTD + SALT2-extended.

    Why "salt2-extended" and not "salt2" or "salt3"?
    - plain "salt2" cuts off at ~9200 Å rest, missing high-z infrared Roman bands
    - "salt3" has improved calibration but a narrower trained wavelength range
    - "salt2-extended" covers ~3000–11000 Å rest-frame, spanning all Roman filters

    Fit knobs (independent — vary one at a time to see which one actually
    drives speed/accuracy, rather than only comparing bundled presets; see
    FIT_PRESETS below for the "robust"/"fast" convenience bundles):
    - t0_window: None (default) -> t0 bounds span every kept image's own
      significant-SNR detection window (not ground truth), padded by
      t0_pad_days on each side. t0_guess is pinned to 0 for every image so
      SNTD's internal guess_amplitude/trial_fit re-centering can't shift this
      already-wide window again. Most robust against a bad initial guess.
      A float -> bounds are a small window (+/- t0_window) around 0, and
      SNTD's trial_fit=True does a quick per-image sncosmo.fit_lc (t0 +
      amplitude only, top-SNR bands) to re-center that window on each image's
      own data. Can fail if the quick pre-fit locks onto the wrong peak
      (aliasing, low S/N) since the sampler can't then see outside the
      narrow window.
    - npoints: nestle live points. None -> SNTD's own internal default
      (100-1000 depending on call site). Fewer live points is faster but a
      sparser ellipsoid is less likely to land inside a very narrow,
      high-SNR posterior peak within budget.
    - maxcall: hard cap on likelihood evaluations. None -> unbounded.

    Returns
    -------
    dict with keys: status, z_lens, z_source, n_images,
                    true_delays, fit_delays, fit_delay_errors, diagnostics
                    (diagnostics includes fit_time_s on success)
    """
    if depth_5sig is None:
        depth_5sig = DEFAULT_DEPTH_5SIG
    if rng is None:
        rng = np.random.default_rng()

    z_lens = float(lens.deflector_redshift)

    result = {
        "status": "fail", "z_lens": z_lens, "z_source": z_source,
        "n_images": 0, "true_delays": {}, "fit_delays": {},
        "fit_delay_errors": {}, "diagnostics": {},
    }

    t_total_start = time.time()
    print(f"  [Lens {lens_index}] z_l={z_lens:.3f}  z_s={z_source:.3f}")

    t0 = time.time()
    try:
        all_arrival_times = lens.point_source_arrival_times()
        arrival_times = np.array(all_arrival_times[0])
    except Exception as e:
        result["status"] = f"no_arrival_times: {e}"
        return result
    print(f"    arrival times: {time.time()-t0:.1f}s  ({len(arrival_times)} images)")

    n_images = len(arrival_times)
    result["n_images"] = n_images
    if n_images < 2:
        result["status"] = "single_image"
        return result

    min_t = float(np.min(arrival_times)) + time_range[0]
    max_t = float(np.max(arrival_times)) + time_range[1]
    obs_times = np.arange(min_t, max_t, cadence_days)
    print(f"    obs grid: {len(obs_times)} epochs over {max_t-min_t:.0f} days")

    t0 = time.time()
    try:
        image_tables, arrival_times, bands_used, image_snr = extract_light_curves(
            lens, bands, obs_times, depth_5sig, rng
        )
    except Exception as e:
        result["status"] = f"lc_extraction_failed: {e}"
        return result
    snr_str = ", ".join(f"{k}={v:.1f}" for k, v in image_snr.items())
    print(f"    light curves: {time.time()-t0:.1f}s  bands={bands_used}  SNR=[{snr_str}]")

    result["diagnostics"]["image_snr"] = image_snr
    result["diagnostics"]["bands_used"] = bands_used
    result["diagnostics"]["n_bands"] = len(bands_used)

    if not bands_used:
        result["status"] = "no_valid_bands"
        return result

    kept_tables = OrderedDict()
    kept_arrival = []
    dropped = []
    for i, (name, tbl) in enumerate(image_tables.items()):
        snr = image_snr.get(name, 0)
        if snr >= min_image_snr:
            kept_tables[name] = tbl
            kept_arrival.append(arrival_times[i])
        else:
            dropped.append(f"{name}(SNR={snr:.1f})")

    if dropped:
        result["diagnostics"]["dropped_images"] = dropped
        print(f"    dropped low-SNR images: {dropped}")

    if len(kept_tables) < 2:
        snr_str = ", ".join(f"{k}={v:.1f}" for k, v in image_snr.items())
        result["status"] = f"low_snr: only {len(kept_tables)} images with SNR>={min_image_snr} ({snr_str})"
        print(f"    SKIP: {result['status']}  ({time.time()-t_total_start:.1f}s total)")
        return result

    image_tables = kept_tables
    arrival_times = np.array(kept_arrival)

    true_delays = {}
    for i, name in enumerate(image_tables.keys()):
        true_delays[name] = float(arrival_times[i] - arrival_times[0])
    result["true_delays"] = true_delays
    print(f"    true delays: {true_delays}")

    # Skip systems with very large delays: the SN fades before the second image
    # rises, so the two light curves don't overlap and the sampler has nothing
    # to cross-correlate. These are also rare extreme-mass lens configurations
    # that are not the primary Roman science target.
    max_delay = max(true_delays.values())
    if max_delay > max_delay_days:
        result["status"] = f"delay_too_large: {max_delay:.1f} d > {max_delay_days:.0f} d limit"
        print(f"    SKIP: {result['status']}  ({time.time()-t_total_start:.1f}s total)")
        return result

    try:
        misn = sntd.MISN(telescopename="Roman", object_name="slsim_lens")
        misn.zl = z_lens
        misn.zs = z_source
        for name, table in image_tables.items():
            img_lc = sntd.image_lc()
            img_lc.table = table
            img_lc.bands = set(table["band"])
            img_lc.zpsys = "ab"
            misn.add_image_lc(img_lc, key=name)

        fit_kwargs = dict(
            snType="Ia",
            models=model_name,
            bands=bands_used,
            params=["t0", "x0", "x1", "c"],
            # SNTD never reads misn.zs/misn.zl automatically (verified: zero
            # references to either in sntd/fitting.py) — without this, the
            # model fits at its default z=0 regardless of the true source
            # redshift, which for this population ranges ~0.9-3.8 and causes
            # catastrophic template mismatch (degenerate posteriors, fits
            # that never converge).
            constants={"z": z_source},
            method="parallel",
            verbose=False,
            guess_amplitude=True,
        )

        if t0_window is None:
            # Bounds must span ALL image peaks, not just a window around one
            # guess — for large delays (e.g. 339 days) a narrow window centered
            # on a single estimate misses the other image(s) entirely and the
            # sampler explores empty likelihood space. Built from each kept
            # image's own significant-SNR detection times (not ground-truth
            # arrival times, which real data won't have). t0_guess=0 for every
            # image stops SNTD's internal guess_amplitude/trial_fit machinery
            # from re-centering this already-wide window around a second guess.
            detected_times = np.concatenate([
                np.asarray(tbl["time"])[np.abs(tbl["flux"] / tbl["fluxerr"]) >= min_image_snr]
                for tbl in image_tables.values()
            ])
            t0_lo = float(detected_times.min()) - t0_pad_days
            t0_hi = float(detected_times.max()) + t0_pad_days
            fit_kwargs.update(
                bounds={"t0": (t0_lo, t0_hi), "x1": (-3, 3), "c": (-0.3, 0.3)},
                t0_guess={name: 0.0 for name in image_tables},
                trial_fit=False,
            )
        else:
            # Small window re-centered per image by a quick sncosmo.fit_lc
            # (t0 + amplitude only, top-SNR bands) instead of full nested
            # sampling exploration.
            fit_kwargs.update(
                bounds={"t0": (-t0_window, t0_window), "x1": (-3, 3), "c": (-0.3, 0.3)},
                trial_fit=True,
            )

        if npoints is not None:
            fit_kwargs["npoints"] = npoints
        if maxcall is not None:
            fit_kwargs["maxcall"] = maxcall

        print(f"    t0_window={t0_window}  npoints={npoints}  maxcall={maxcall}  "
              f"t0 bounds: {fit_kwargs['bounds']['t0']}")
        t_fit_start = time.time()
        fit_result = sntd.fit_data(misn, **fit_kwargs)
        fit_time_s = time.time() - t_fit_start
        print(f"    SNTD fit: {fit_time_s:.1f}s")
        result["diagnostics"]["fit_time_s"] = fit_time_s
        result["diagnostics"]["t0_window"] = t0_window
        result["diagnostics"]["npoints"] = npoints
        result["diagnostics"]["maxcall"] = maxcall

        fit_delays = {}
        fit_delay_errors = {}
        image_names = list(image_tables.keys())

        par = getattr(fit_result, "parallel", None)
        if par is not None:
            td = par.time_delays
            td_err = getattr(par, "time_delay_errors", None)

            # Reference image (index 0) has delay = 0 by definition
            fit_delays[image_names[0]] = 0.0
            fit_delay_errors[image_names[0]] = [0.0, 0.0]

            for name in image_names[1:]:
                # par.time_delays can be a dict keyed by image name OR a list/array
                # depending on the SNTD version; handle both
                if isinstance(td, dict):
                    fit_delays[name] = float(td.get(name, np.nan))
                else:
                    idx = image_names.index(name) - 1
                    fit_delays[name] = float(td[idx]) if idx < len(td) else np.nan

                if td_err is not None:
                    err = td_err.get(name, [np.nan, np.nan]) if isinstance(td_err, dict) \
                          else (td_err[image_names.index(name) - 1]
                                if image_names.index(name) - 1 < len(td_err) else [np.nan, np.nan])
                    fit_delay_errors[name] = [float(err[0]), float(err[1])] \
                                             if hasattr(err, "__len__") else [float(err), float(err)]
                else:
                    fit_delay_errors[name] = [np.nan, np.nan]

        result["fit_delays"] = fit_delays
        result["fit_delay_errors"] = fit_delay_errors
        result["status"] = "ok"
        print(f"    OK: fit_delays={fit_delays}  ({time.time()-t_total_start:.1f}s total)")

    except Exception as e:
        import traceback
        result["status"] = f"fit_failed: {e}"
        print(f"    FIT FAILED: {e}  ({time.time()-t_total_start:.1f}s total)")
        traceback.print_exc()

    return result
