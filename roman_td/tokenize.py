"""
tokenize.py
===========
Stage-2 input preparation: canonical photometry table -> transformer inputs.
Implements documents/transformer_explained.md §4 exactly. Used identically
at training and inference time — one code path, no train/serve skew.

Token layout (13 features per photometric measurement):

    [ phase_norm, flux_norm, fluxerr_norm,
      band_onehot(6), image_onehot(4), is_real ]

The two silent-failure traps this module exists to prevent:

1. ONE phase zero-point per SYSTEM (the reference image's peak). Phasing
   each image to its own peak would subtract the delay out of the input.
2. Flux normalized by the system's own brightest detection — NEVER by a
   cosmological distance modulus, which would divide out the magnification
   signal.

Pure numpy; no ML-framework dependency.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

# Fixed orderings — a trained network depends on these never changing.
BAND_ORDER = ["f062", "f087", "f106", "f129", "f158", "f184"]
N_IMAGE_SLOTS = 4
TOKEN_DIM = 3 + len(BAND_ORDER) + N_IMAGE_SLOTS + 1   # 3+6+4+1 = 14
GP_QUALITIES = ["good", "broad", "multipeak", "fail"]


def _colnames(tab):
    cols = tab.colnames if hasattr(tab, "colnames") else list(tab.keys())
    tcol = "mjd" if "mjd" in cols else "time"
    bcol = "filter" if "filter" in cols else "band"
    return tcol, bcol


def tokenize_system(
    tab,
    images: List[str],
    z_source: float,
    t_peak_ref: float,
    l_max: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Turn one system's photometry into a padded token array.

    Parameters
    ----------
    tab : astropy Table (canonical schema §6.1: mjd/filter/flux/fluxerr/image)
    images : image labels in slot order — REFERENCE IMAGE FIRST (slot 1).
        Must match the ordering used for the truth labels.
    z_source : source redshift (guaranteed known — survey design)
    t_peak_ref : GP peak time of the REFERENCE image (MJD). The single,
        system-wide phase zero-point (trap #1 above).
    l_max : pad/truncate to this many tokens; None = no padding.

    Returns
    -------
    tokens : (L, 14) float32  — L = l_max if given, else the token count
    mask   : (L,) bool        — True where the token is a real measurement
    """
    if len(images) > N_IMAGE_SLOTS:
        raise ValueError(f"more than {N_IMAGE_SLOTS} images: {images}")
    tcol, bcol = _colnames(tab)

    mjd = np.asarray(tab[tcol], dtype=float)
    flux = np.asarray(tab["flux"], dtype=float)
    ferr = np.asarray(tab["fluxerr"], dtype=float)
    band = np.char.lower(np.asarray(tab[bcol]).astype(str))
    img = np.asarray(tab["image"]).astype(str)

    # Trap #2: per-system internal flux scale (brightest detection anywhere
    # in the system). Keeps between-image ratios (magnification) intact.
    scale = float(np.nanmax(flux))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("no positive flux in system — cannot normalize")

    # Trap #1: one common zero-point; rest-frame via the known z_source.
    phase = (mjd - float(t_peak_ref)) / (1.0 + float(z_source))

    band_idx = {b: j for j, b in enumerate(BAND_ORDER)}
    img_idx = {name: j for j, name in enumerate(images)}

    keep = np.array([b in band_idx and im in img_idx
                     for b, im in zip(band, img)])
    n = int(np.count_nonzero(keep))
    L = l_max if l_max is not None else n
    tokens = np.zeros((L, TOKEN_DIM), dtype=np.float32)
    mask = np.zeros(L, dtype=bool)

    rows = np.flatnonzero(keep)[:L]   # truncate if over l_max (rare; log upstream)
    for out_i, i in enumerate(rows):
        tokens[out_i, 0] = phase[i]
        tokens[out_i, 1] = flux[i] / scale
        tokens[out_i, 2] = ferr[i] / scale
        tokens[out_i, 3 + band_idx[band[i]]] = 1.0
        tokens[out_i, 3 + len(BAND_ORDER) + img_idx[img[i]]] = 1.0
        tokens[out_i, -1] = 1.0
        mask[out_i] = True
    return tokens, mask


def scalar_features(
    z_source: float,
    z_lens: float,
    n_images: int,
    gp: Optional[Dict] = None,
    drop_gp: bool = False,
) -> np.ndarray:
    """Per-system scalar feature vector (fused after pooling; §4.2 of the
    transformer doc).

    Layout (fixed):
      [ z_source, z_lens, n_images,
        dt_gp slot2..4, dt_gp_err slot2..4, flux_ratio slot2..4,
        gp_band_scatter, quality_onehot(4) ]                    -> 16 floats

    gp : the §6.3 GP result dict for this system (per_image keyed by image
        label, slot order = token image order), or None.
    drop_gp : zero all GP-derived features and force quality to "fail" —
        the training-time GP-hint dropout, and the inference behavior when
        the GP failed. Missing per-image entries are zeros.
    """
    out = np.zeros(3 + 3 * (N_IMAGE_SLOTS - 1) + 1 + len(GP_QUALITIES),
                   dtype=np.float32)
    out[0], out[1], out[2] = float(z_source), float(z_lens), float(n_images)

    quality = "fail"
    if gp is not None and not drop_gp:
        per = gp.get("per_image", {})
        scatters = []
        # slot order: gp per_image entries follow the token image order
        # (reference first, then others) — slot 1 is the reference (dt=0).
        for slot, (name, e) in enumerate(per.items()):
            if slot >= N_IMAGE_SLOTS - 1:
                break
            if np.isfinite(e.get("dt_gp", np.nan)):
                out[3 + slot] = e["dt_gp"]
            if np.isfinite(e.get("dt_gp_err", np.nan)):
                out[3 + (N_IMAGE_SLOTS - 1) + slot] = e["dt_gp_err"]
            if np.isfinite(e.get("flux_ratio", np.nan)):
                out[3 + 2 * (N_IMAGE_SLOTS - 1) + slot] = e["flux_ratio"]
            dpb = list(e.get("dt_per_band", {}).values())
            if len(dpb) > 1:
                scatters.append(np.std(dpb))
        if scatters:
            out[3 + 3 * (N_IMAGE_SLOTS - 1)] = float(np.mean(scatters))
        # worst per-image flag = the system's flag
        flags = [e.get("quality", "fail") for e in per.values()] or ["fail"]
        quality = max(flags, key=GP_QUALITIES.index)

    out[3 + 3 * (N_IMAGE_SLOTS - 1) + 1 + GP_QUALITIES.index(quality)] = 1.0
    return out
