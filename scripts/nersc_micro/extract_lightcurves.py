#!/usr/bin/env python3
"""Extract 4-tier light curves from one precomputed lensed-SNe shard.

Runs on Perlmutter CPU nodes -- the microlensing magnification maps are
already cached inside each pickled ``Lens``, so no GPU is needed.

Per lens, per band (HLTDS wide+deep union):
    intrinsic (unlensed), macro-lensed, micro-lensed light curves on a
    common uniform time grid (observer-frame days).
Millilensing is stored as per-image constant factors (achromatic), not as
curves: tier 4 = micro - 2.5*log10(f) per image, applied downstream.

Output per shard:
    micro_lc_set_NN.pkl.gz   gzip pickle: list of per-lens dicts
    micro_lc_set_NN.json     manifest: config, counts, failures

Per-lens dict keys:
    shard_id, lens_index      provenance
    z_lens, z_source          floats
    theta_E                   Einstein radius [arcsec]
    n_images                  int
    dt_days                   (n_images,) observer-frame arrival times
    mu_macro                  (n_images,) signed macro magnification
    kappa, gamma1, gamma2     (n_images,) macro model at image positions
    kappa_star                (n_images,) stellar convergence at images
    milli_factors             (n_images,) millilensing |mu'/mu|
    times                     (n_epochs,) observer-frame days
    bands                     list of band names
    intrinsic                 (n_bands, n_epochs) unlensed magnitudes
    macro                     (n_bands, n_images, n_epochs) magnitudes
    micro                     (n_bands, n_images, n_epochs) magnitudes
"""

from __future__ import annotations

import argparse
import gzip
import json
import multiprocessing as mp
import os
import pickle
import socket
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from slsim.Pipelines import roman_speclite

# HLTDS wide (F062 F087 F106 F129) union deep (F106 F129 F158 F184).
BANDS = ["F062", "F087", "F106", "F129", "F158", "F184"]
GRID_SPACING_DAYS = 2.0
MAX_EPOCHS = 600  # cap for extreme (1+z)-dilated windows
SOURCE_GRID_PIXELS = 100  # matches map generation
SIGMA_MILLI = 0.0015
MILLI_SEED_BASE = 20260818


def read_lenses(shard_path, max_lenses=None):
    """Yield the Lens objects stored in one shard, one at a time."""
    with gzip.open(shard_path, "rb") as handle:
        count = 0
        while max_lenses is None or count < max_lenses:
            try:
                yield pickle.load(handle)
            except EOFError:
                return
            count += 1


def image_plane_kappa_gamma(lens, source_index=0):
    """Macro-model kappa, gamma1, gamma2 at each image position."""
    lens_model, kwargs_lens = lens.deflector_mass_model_lenstronomy(
        source_index=source_index
    )
    image_x, image_y = lens._point_source_image_positions(source_index=source_index)
    gamma1, gamma2 = lens_model.gamma(image_x, image_y, kwargs_lens)
    return lens_model.kappa(image_x, image_y, kwargs_lens), gamma1, gamma2


def millilensing_magnification_factors(
    kappa, gamma1, gamma2, sigma_kappa=SIGMA_MILLI, sigma_gamma=SIGMA_MILLI, rng=None
):
    """Millilensing-only magnification factor f = |mu_perturbed / mu_macro|."""
    rng = np.random.default_rng() if rng is None else rng
    kappa, gamma1, gamma2 = np.atleast_1d(kappa, gamma1, gamma2)
    d_kappa = rng.normal(0.0, sigma_kappa, size=kappa.shape)
    d_gamma1 = rng.normal(0.0, sigma_gamma, size=kappa.shape)
    d_gamma2 = rng.normal(0.0, sigma_gamma, size=kappa.shape)
    inverse_mu_macro = (1 - kappa) ** 2 - gamma1**2 - gamma2**2
    inverse_mu_perturbed = (
        (1 - kappa - d_kappa) ** 2
        - (gamma1 + d_gamma1) ** 2
        - (gamma2 + d_gamma2) ** 2
    )
    return np.abs(inverse_mu_macro / inverse_mu_perturbed)


def light_curve_window(lens, bands, source_index=0, pad=10.0):
    """Observer-frame window over which any band has a defined light curve.

    Union across bands: each band probes the (1+z)-stretched template support,
    widened on the late side by the largest relative time delay.
    """
    source = lens.source(source_index)
    template_times = np.asarray(source.point_source.source_dict["MJD"])  # rest frame
    probe = template_times * (1 + source.redshift)  # observer frame
    start, stop = np.inf, -np.inf
    for band in bands:
        mags = np.ravel(
            lens.point_source_magnitude(band=band, time=probe)[source_index]
        )
        defined = probe[np.isfinite(mags)]
        if defined.size:
            start = min(start, defined.min())
            stop = max(stop, defined.max())
    if not np.isfinite(start):
        raise RuntimeError("no band has a defined light curve")
    delays = lens.point_source_arrival_times()[source_index]
    return start - pad, stop + np.ptp(delays) + pad


def extract_one(task):
    """Compute the light-curve record for one lens. Returns (index, dict|err)."""
    shard_id, lens_index, lens = task
    try:
        source_index = 0
        start, stop = light_curve_window(lens, BANDS, source_index)
        n_epochs = min(int(np.ceil((stop - start) / GRID_SPACING_DAYS)) + 1, MAX_EPOCHS)
        times = np.linspace(start, stop, n_epochs)

        intrinsic, macro, micro = [], [], []
        for band in BANDS:
            intrinsic.append(
                np.ravel(
                    lens.point_source_magnitude(band=band, time=times)[source_index]
                )
            )
            macro.append(
                np.asarray(
                    lens.point_source_magnitude(band=band, time=times, lensed=True)[
                        source_index
                    ]
                )
            )
            micro.append(
                np.asarray(
                    lens.point_source_magnitude(
                        band=band,
                        time=times,
                        lensed=True,
                        microlensing=True,
                        kwargs_microlensing={
                            "kwargs_source_morphology": {
                                "grid_pixels": SOURCE_GRID_PIXELS
                            }
                        },
                    )[source_index]
                )
            )

        kappa, gamma1, gamma2 = image_plane_kappa_gamma(lens, source_index)
        rng = np.random.default_rng(MILLI_SEED_BASE + shard_id * 100_000 + lens_index)
        factors = millilensing_magnification_factors(kappa, gamma1, gamma2, rng=rng)

        magmaps = lens.microlensing_model_class(source_index).magmaps_images
        record = {
            "shard_id": shard_id,
            "lens_index": lens_index,
            "z_lens": float(lens.deflector_redshift),
            "z_source": float(lens.source_redshift_list[source_index]),
            "theta_E": float(lens.einstein_radius[source_index]),
            "n_images": int(lens.image_number[source_index]),
            "dt_days": np.asarray(
                lens.point_source_arrival_times()[source_index], dtype=np.float32
            ),
            "mu_macro": np.asarray([m.mu_ave for m in magmaps], dtype=np.float32),
            "kappa": np.asarray(kappa, dtype=np.float32),
            "gamma1": np.asarray(gamma1, dtype=np.float32),
            "gamma2": np.asarray(gamma2, dtype=np.float32),
            "kappa_star": np.asarray(
                [m._kappa_star for m in magmaps], dtype=np.float32
            ),
            "milli_factors": np.asarray(factors, dtype=np.float32),
            "times": times.astype(np.float32),
            "bands": list(BANDS),
            "intrinsic": np.asarray(intrinsic, dtype=np.float32),
            "macro": np.asarray(macro, dtype=np.float32),
            "micro": np.asarray(micro, dtype=np.float32),
        }
        return lens_index, record
    except Exception as error:  # one bad lens must not cost the shard
        return lens_index, {
            "_failed": True,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
        }


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", type=Path, required=True,
                    help="Directory holding lensed_sne_1000deg2_set_NN.pkl.gz")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--shard-id", type=int, default=None,
                    help="Default: SLURM_ARRAY_TASK_ID, else 0.")
    ap.add_argument("--max-lenses", type=int, default=None,
                    help="Smoke test: process only the first N lenses.")
    ap.add_argument("--workers", type=int,
                    default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    shard_id = args.shard_id
    if shard_id is None:
        shard_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))

    roman_speclite.configure_roman_filters()

    shard_path = args.run_dir / f"lensed_sne_1000deg2_set_{shard_id:02d}.pkl.gz"
    if not shard_path.is_file():
        raise FileNotFoundError(shard_path)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_file = args.output_dir / f"micro_lc_set_{shard_id:02d}.pkl.gz"
    manifest_file = args.output_dir / f"micro_lc_set_{shard_id:02d}.json"
    if out_file.exists() and not args.overwrite:
        raise FileExistsError(f"{out_file} exists; use --overwrite to regenerate.")

    started = time.time()
    print(f"shard {shard_id:02d}  host={socket.gethostname()}  "
          f"workers={args.workers}  bands={BANDS}", flush=True)

    tasks = (
        (shard_id, i, lens)
        for i, lens in enumerate(read_lenses(shard_path, args.max_lenses))
    )

    records, failures = [], []
    with mp.Pool(processes=args.workers) as pool:
        for lens_index, result in pool.imap_unordered(extract_one, tasks, chunksize=1):
            if result.get("_failed"):
                result["lens_index"] = lens_index
                failures.append(result)
                print(f"lens {lens_index} FAILED: {result['error_type']}: "
                      f"{result['error_message']}", flush=True)
            else:
                records.append(result)
            done = len(records) + len(failures)
            if done % 25 == 0:
                rate = done / (time.time() - started)
                print(f"  {done} done  ({rate * 3600:.0f} lenses/hr)", flush=True)

    records.sort(key=lambda r: r["lens_index"])
    part = out_file.with_suffix(".part")
    with gzip.open(part, "wb", compresslevel=4) as handle:
        pickle.dump(records, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(part, out_file)

    manifest = {
        "shard_id": shard_id,
        "source_shard": str(shard_path),
        "bands": BANDS,
        "grid_spacing_days": GRID_SPACING_DAYS,
        "max_epochs": MAX_EPOCHS,
        "source_grid_pixels": SOURCE_GRID_PIXELS,
        "sigma_milli": SIGMA_MILLI,
        "milli_seed_base": MILLI_SEED_BASE,
        "stored_lenses": len(records),
        "failed_lenses": failures,
        "num_failed": len(failures),
        "elapsed_seconds": time.time() - started,
        "output_size_bytes": out_file.stat().st_size,
        "python_executable": sys.executable,
    }
    manifest_file.write_text(json.dumps(manifest, indent=2, default=str) + "\n")

    print(f"done: {len(records)} lenses, {len(failures)} failed, "
          f"{out_file.stat().st_size / 1024**2:.1f} MiB, "
          f"{(time.time() - started) / 3600:.2f} hr", flush=True)


if __name__ == "__main__":
    main()
