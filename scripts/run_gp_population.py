"""
run_gp_population.py
====================
Stage 1 driver: GP cross-correlation time-delay estimates over a slsim lens
population, benchmarked against truth (documents/pipeline_architecture.md §2).

Deliberately mirrors run_sntd_population.py — same pickle, bands, cadence,
depths, and per-lens RNG seeding — so gp_benchmark.ecsv rows are directly
comparable, lens by lens, with delay_benchmark_fast/robust.ecsv.

Differences from the SNTD driver:
- no SNTD fit: extract_light_curves -> to_canonical -> gp_cross_correlate
- keeps images down to peak SNR >= 5 (the physical fit demands 10); the GP
  should be benchmarked on the fainter systems the fitter skips
- no max-delay cut: the GP's lag range covers +/-200 d, and large-delay
  systems are exactly where a cheap coarse estimate is most useful

Run: python scripts/run_gp_population.py   (no arguments)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pickle
import time
import warnings

import numpy as np
from astropy.table import Table
from joblib import Parallel, delayed

warnings.filterwarnings("ignore", message=r".*bandpass.*outside spectral range.*")
warnings.filterwarnings("ignore", message=r".*invalid value encountered.*")
warnings.filterwarnings("ignore", message=r".*divide by zero.*")

from roman_td.sntd_wrapper import (
    extract_light_curves, register_roman_bands, DEFAULT_DEPTH_5SIG,
)
from roman_td.simulate import to_canonical
from roman_td.crosscorr import gp_cross_correlate

# ── survey config: identical to run_sntd_population.py ─────────────────────
PICKLE = "/home/epadill/time_delays/data/roman_deep_lens_population_compat.pkl"
BANDS = ["F087", "F106", "F129", "F158", "F184"]
CADENCE = 5.0
TIME_RANGE = (-50, 300)     # obs window around first/last arrival, days
N_SYSTEMS = 1000            # stop after this many successful estimates (0 = all)
SEED = 42                   # same seed offset as the SNTD driver
N_JOBS = 8
MIN_IMAGE_SNR = 5.0         # GP works below the fitter's SNR >= 10 cut
LAG_RANGE = (-200, 200)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "gp_benchmark.ecsv")


def process_one_lens(i, lens):
    """Simulate photometry and run the GP cross-correlator for one system."""
    warnings.filterwarnings("ignore")
    out = {"lens_index": i, "status": "ok"}

    try:
        z_source = float(lens.source_redshift_list[0])
        z_lens = float(lens.deflector_redshift)
    except Exception:
        out["status"] = "bad_lens"
        return out

    try:
        arrival_times = np.array(lens.point_source_arrival_times()[0])
    except Exception as e:
        out["status"] = f"no_arrival_times: {e}"
        return out
    if len(arrival_times) < 2:
        out["status"] = "single_image"
        return out

    try:
        true_mu_all = np.abs(np.asarray(
            lens.point_source_magnification()[0], dtype=float))
    except Exception:
        true_mu_all = None

    rng = np.random.default_rng(SEED + i)
    obs_times = np.arange(arrival_times.min() + TIME_RANGE[0],
                          arrival_times.max() + TIME_RANGE[1], CADENCE)
    try:
        image_tables, arrival_times, bands_used, image_snr = extract_light_curves(
            lens, BANDS, obs_times, DEFAULT_DEPTH_5SIG, rng)
    except Exception as e:
        out["status"] = f"lc_extraction_failed: {e}"
        return out

    kept = {n: t for n, t in image_tables.items()
            if image_snr.get(n, 0.0) >= MIN_IMAGE_SNR}
    if len(kept) < 2:
        out["status"] = f"low_snr: {len(kept)} images with SNR>={MIN_IMAGE_SNR}"
        return out

    # Truth per kept image, indexed by original slsim position (the image
    # name encodes it, so this survives the SNR cut).
    arr = {n: float(arrival_times[int(n.rsplit('_', 1)[1]) - 1]) for n in kept}
    mu = ({n: float(true_mu_all[int(n.rsplit('_', 1)[1]) - 1]) for n in kept}
          if true_mu_all is not None
          and max(int(n.rsplit('_', 1)[1]) for n in kept) <= len(true_mu_all)
          else None)

    tab = to_canonical(kept)
    try:
        gp = gp_cross_correlate(tab, list(kept), lag_range=LAG_RANGE, rng=rng)
    except Exception as e:
        out["status"] = f"gp_failed: {e}"
        return out

    ref = gp["ref_image"]
    rows = []
    for img, e in gp["per_image"].items():
        dt_true = arr[img] - arr[ref]
        per_band = list(e["dt_per_band"].values())
        mu_true = (mu[img] / mu[ref]) if mu and mu[ref] > 0 else np.nan
        rows.append({
            "lens_index": i, "image": img, "ref_image": ref,
            "z_lens": z_lens, "z_source": z_source, "n_images": len(kept),
            "snr_ref": image_snr.get(ref, np.nan),
            "snr_img": image_snr.get(img, np.nan),
            "true_delay": dt_true,
            "dt_gp": e["dt_gp"], "dt_gp_err": e["dt_gp_err"],
            "residual": e["dt_gp"] - dt_true,
            "band_scatter": float(np.std(per_band)) if len(per_band) > 1 else np.nan,
            "n_bands": len(per_band),
            "quality": e["quality"],
            "true_mu_ratio": mu_true,
            "flux_ratio": e["flux_ratio"], "flux_ratio_err": e["flux_ratio_err"],
            "mu_residual": e["flux_ratio"] - mu_true,
            "wall_time_s": gp["wall_time_s"],
        })
    out["rows"] = rows
    return out


def main():
    register_roman_bands()
    t_start = time.time()

    with open(PICKLE, "rb") as f:
        pop = pickle.load(f)["lens_population"]
    print(f"Loaded {len(pop)} lenses")
    print(f"Running GP cross-correlation with {N_JOBS} workers...")

    results_gen = Parallel(n_jobs=N_JOBS, verbose=5, return_as="generator_unordered")(
        delayed(process_one_lens)(i, lens) for i, lens in enumerate(pop)
    )

    rows, fail_counts, n_done = [], {}, 0
    for res in results_gen:
        if res["status"] != "ok":
            reason = res["status"].split(":")[0]
            fail_counts[reason] = fail_counts.get(reason, 0) + 1
            continue
        rows.extend(res["rows"])
        n_done += 1
        if n_done % 25 == 0:
            Table(rows).write(OUTPUT_FILE, overwrite=True)
            print(f"  [checkpoint] {n_done} systems -> {OUTPUT_FILE}")
        if N_SYSTEMS and n_done >= N_SYSTEMS:
            print(f"  Reached N_SYSTEMS={N_SYSTEMS}, stopping.")
            break

    elapsed = time.time() - t_start
    tab = Table(rows)
    tab.write(OUTPUT_FILE, overwrite=True)

    res_arr = np.array([r["residual"] for r in rows if np.isfinite(r["residual"])])
    qual = [r["quality"] for r in rows]
    good = np.array([r["residual"] for r in rows
                     if r["quality"] == "good" and np.isfinite(r["residual"])])
    wall = np.array([r["wall_time_s"] for r in rows if np.isfinite(r["wall_time_s"])])

    print(f"\n{'='*60}")
    print(f"GP-measured {n_done} systems, {len(rows)} delays in {elapsed/60:.1f} min")
    if len(res_arr):
        print(f"All delays   : median={np.median(res_arr):+.2f} d  "
              f"std={np.std(res_arr):.2f} d  "
              f"|res|<2d: {np.mean(np.abs(res_arr) < 2)*100:.0f}%  "
              f"|res|<5d: {np.mean(np.abs(res_arr) < 5)*100:.0f}%")
    if len(good):
        print(f"quality=good : median={np.median(good):+.2f} d  "
              f"std={np.std(good):.2f} d  "
              f"|res|<2d: {np.mean(np.abs(good) < 2)*100:.0f}%  (n={len(good)})")
    print(f"quality mix  : {dict(zip(*np.unique(qual, return_counts=True)))}")
    if len(wall):
        print(f"wall/system  : median={np.median(wall):.1f}s")
    print(f"Saved -> {OUTPUT_FILE}")
    if fail_counts:
        print("\nFailure summary:")
        for reason, count in sorted(fail_counts.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
