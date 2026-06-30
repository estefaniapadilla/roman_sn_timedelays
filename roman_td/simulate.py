"""
simulate.py
===========
Simulate Roman multi-image photometry from a slsim lens population.

Used by scripts/run_bayesn_population.py to generate synthetic light curves
that are then passed to the BayeSN fitter in bayesn_wrapper.py.

Key difference from sntd_wrapper.py's extract_light_curves()
-------------------------------------------------------------
- This module uses the BayeSN model itself to simulate photons, so the
  simulated and fitted SEDs are self-consistent (no model mismatch bias).
- sntd_wrapper uses slsim's lenstronomy ray-tracing to get magnitudes,
  which is more realistic but slower and SALT2-model-agnostic.

ZP note: ZP=27.0 here vs ZP=25.0 in sntd_wrapper — both are valid AB
zeropoints, they just scale the flux numbers differently. BayeSN's amplitude
parameter was defined with ZP=27 convention so we stay consistent here.

Microlensing note: each image is given a constant magnification mu_k from
the macro lens model. Time-varying chromatic microlensing is NOT included.
This is the dominant modelling systematic for quad lenses. A future version
should draw magnification maps and convolve with the SN photosphere size.
"""

from __future__ import annotations

import numpy as np
import astropy.table as at
from typing import Dict, Any, Optional, List, Tuple

# ZP=27.0 matches the BayeSN amplitude convention
ZP = 27.0
ZPSYS = "ab"
MWEBV = 0.0          # no MW extinction in simulations; add it for realistic mocks
SIM_MB = -19.4       # fiducial SN Ia peak absolute B-band mag; real scatter ~0.15 mag

# SNTD uses single-letter image labels internally; slsim uses "image_1", "image_2" etc.
# We label by arrival order (a = first arriving image).
IMG_LABELS = list("abcdefgh")

SURVEY_BANDS: Dict[str, List[str]] = {
    "time_domain_wide": ["F062", "F087", "F106", "F129", "F158"],
    "time_domain_deep": ["F087", "F106", "F129", "F158", "F184"],
}

# Approximate per-visit 5-sigma AB depths for the Roman HLTDS tiers,
# from Hounsell+2018 / Rose+2021 scaling. The deep-tier values in
# sntd_wrapper.DEFAULT_DEPTH_5SIG are deeper (from lenstronomy noise model).
DEFAULT_DEPTH_5SIG: Dict[str, float] = {
    "F062": 26.4, "F087": 25.6, "F106": 25.5, "F129": 25.4,
    "F158": 25.4, "F184": 26.2, "F213": 25.0, "F146": 26.4,
}
try:
    # If the SNTD wrapper is available, use its deeper depth values so both
    # pipelines share the same noise model.
    from roman_td.sntd_wrapper import DEFAULT_DEPTH_5SIG as _SNTD_DEPTH
    DEFAULT_DEPTH_5SIG.update(_SNTD_DEPTH)
except Exception:
    pass


def lens_truth(lens) -> Optional[Dict[str, Any]]:
    """Extract ground-truth redshifts, time delays, and magnifications from slsim.

    Returns None if the lens object is missing required attributes.
    Delays are referenced to the first-arriving image (shortest arrival time).
    """
    try:
        zS = float(lens.source_redshift_list[0])
    except Exception:
        return None
    try:
        zL = float(lens.deflector_redshift)
    except Exception:
        zL = np.nan

    def _first(x):
        x = np.asarray(x, dtype=object)
        # slsim returns a list-of-arrays (one per source) in recent versions
        if x.dtype == object or x.ndim > 1:
            return np.asarray(x[0], dtype=float)
        return np.asarray(x, dtype=float)

    try:
        t_arr = _first(lens.point_source_arrival_times())
        mu = np.abs(_first(lens.point_source_magnification()))
    except Exception:
        return None

    if len(t_arr) < 2 or len(t_arr) != len(mu):
        return None

    order = np.argsort(t_arr)
    delays = t_arr[order] - t_arr[order][0]
    return {
        "z_source": zS, "z_lens": zL,
        "delays": delays, "mu": mu[order],
        "n_images": int(len(delays)),
    }


def set_amplitude_for_distance(model, zS: float, band: str) -> None:
    """Scale BayeSN gray amplitude so the SN sits at the correct distance.

    Tries model.set_source_peakabsmag first (requires BayeSN to support it);
    falls back to computing the distance modulus manually and matching peak flux.
    """
    import sncosmo
    try:
        model.set_source_peakabsmag(SIM_MB, "bessellb", "ab")
        return
    except Exception:
        pass
    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    m_target = SIM_MB + cosmo.distmod(zS).value
    model.set(amplitude=1.0)
    grid = np.linspace(model.mintime(), model.maxtime(), 60)
    f1 = np.nanmax(model.bandflux(band, grid, zp=ZP, zpsys=ZPSYS))
    f_target = 10 ** (-0.4 * (m_target - ZP))
    model.set(amplitude=max(f_target / max(f1, 1e-300), 1e-300))


def simulate_photometry(
    truth: Dict[str, Any],
    bands: List[str],
    cadence_days: float,
    depths: Dict[str, float],
    rng: np.random.Generator,
) -> Tuple[Optional[at.Table], Dict[str, Any]]:
    """Simulate Roman photometry for all lensed images of one system using BayeSN.

    Draws a random BayeSN realisation (theta, hostebv) and scales it to the
    correct distance, then adds Gaussian flux noise from the depth map.

    Returns
    -------
    tab : Table or None
        Combined photometry table with columns (mjd, filter, flux, fluxerr,
        zp, zpsys, image). None if fewer than 2 images are detectable.
    sim_info : dict
        Drawn SN parameters + truth dict for the kept images.
    """
    # Import here so this module does not require bayesn at import time
    from roman_td.bayesn_wrapper import make_model

    zS = truth["z_source"]
    delays, mu = truth["delays"], truth["mu"]

    model = make_model(zS)
    theta = float(rng.normal(0, 1))
    hostebv = float(min(rng.exponential(0.1), 1.0))
    model.set(theta=theta, hostebv=hostebv, hostr_v=3.1)
    set_amplitude_for_distance(model, zS, bands[len(bands) // 2])
    sim_params = {
        "theta": theta, "hostebv": hostebv,
        "amplitude": float(model.get("amplitude")),
    }

    t0 = 60000.0
    model.set(t0=t0)

    # Cover the full window: from before image-1 rises to after the most
    # delayed image fades. The +10 day buffers ensure we catch the tails.
    span_lo = t0 + model.mintime() - 10
    span_hi = t0 + model.maxtime() + delays[-1] + 10
    mjds = np.arange(span_lo, span_hi, cadence_days)

    rows = []
    kept_images, kept_delays, kept_mu = [], [], []
    for k, (dt, mu_k) in enumerate(zip(delays, mu)):
        label = IMG_LABELS[k]
        img_rows = []
        max_snr = 0.0
        for b in bands:
            ferr = 10 ** (-0.4 * (depths[b] - ZP)) / 5.0
            phase_ok = (
                ((mjds - t0 - dt) >= model.mintime()) &
                ((mjds - t0 - dt) <= model.maxtime())
            )
            t_obs = mjds[phase_ok]
            if len(t_obs) == 0:
                continue
            f_true = mu_k * model.bandflux(b, t_obs - dt, zp=ZP, zpsys=ZPSYS)
            f_true = np.where(np.isfinite(f_true), f_true, 0.0)
            f_obs = f_true + rng.normal(0, ferr, size=len(t_obs))
            max_snr = max(max_snr, float(np.max(f_true) / ferr) if len(f_true) else 0.0)
            for t, fo in zip(t_obs, f_obs):
                img_rows.append((t, b, fo, ferr, ZP, ZPSYS, label))

        # Require at least 5 observations above 5-sigma to call an image detectable
        if max_snr >= 5.0 and len(img_rows) >= 5:
            rows.extend(img_rows)
            kept_images.append(label)
            kept_delays.append(float(dt))
            kept_mu.append(float(mu_k))

    if len(kept_images) < 2:
        return None, sim_params

    tab = at.Table(
        rows=rows,
        names=["mjd", "filter", "flux", "fluxerr", "zp", "zpsys", "image"],
    )
    tab.sort(["image", "mjd", "filter"])

    # Re-reference delays to the first kept image (not necessarily image-1)
    kept_delays = np.array(kept_delays) - kept_delays[0]
    truth_kept = dict(truth)
    truth_kept.update({
        "images": kept_images,
        "delays": kept_delays,
        "mu": np.array(kept_mu),
        "n_images": len(kept_images),
        "t0": t0,
    })
    return tab, {**sim_params, "truth": truth_kept}
