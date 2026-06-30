"""
run_sntd_population.py
======================
Driver: load a slsim lens population pickle, measure time delays with
SNTD + SALT2-extended, and write a benchmark table (fitted vs. true delay).

Previously: fit_lc.py
New location: scripts/run_sntd_population.py

When to use this vs run_bayesn_population.py
---------------------------------------------
Use this script first: it is ~10x faster (MCMC vs. nested sampling) and
gives a good sanity check on the population before committing to the slower
BayeSN run.  SALT2's time delays are slightly less accurate in dusty systems
because its color parameter c conflates dust with intrinsic SN color, but for
delay measurement (not magnification) the difference is usually small.

Supports parallel execution via joblib.
"""

import sys
import os
# Add repo root to path so both roman_td and bayesncosmo are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pickle
import numpy as np
import warnings
import time
from astropy.table import Table
from joblib import Parallel, delayed

warnings.filterwarnings("ignore", message=r".*bandpass.*outside spectral range.*")
warnings.filterwarnings("ignore", message=r".*invalid value encountered.*")
warnings.filterwarnings("ignore", message=r".*divide by zero.*")

from roman_td.sntd_wrapper import measure_one, register_roman_bands, DEFAULT_DEPTH_5SIG

# ── survey config: match the population you simulated ────────────────────────
PICKLE = "/home/epadill/time_delays/data/roman_deep_lens_population_compat.pkl"
SURVEY = "time_domain_deep"
if SURVEY == "time_domain_deep":
    BANDS = ["F087", "F106", "F129", "F158", "F184"]
else:
    BANDS = ["F062", "F087", "F106", "F129", "F158"]

CADENCE = 5.0       # days between visits
N_SYSTEMS = 50      # stop after this many successful fits (0 = all)
SEED = 42
N_JOBS = 1          # parallel workers (-1 = all cores)
SAVE_EVERY = 10

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "delay_benchmark.ecsv")


def process_one_lens(i, lens, bands, cadence, depth, seed_offset):
    """Wrapper for parallel execution — each worker gets its own RNG seed."""
    import warnings
    warnings.filterwarnings("ignore")

    try:
        zS = float(lens.source_redshift_list[0])
    except Exception:
        return None

    rng = np.random.default_rng(seed_offset + i)
    return measure_one(
        lens, bands, zS,
        cadence_days=cadence,
        depth_5sig=depth,
        rng=rng,
        lens_index=i,
    )


def main():
    register_roman_bands()
    t_start = time.time()

    with open(PICKLE, "rb") as f:
        payload = pickle.load(f)
    pop = payload["lens_population"]
    print(f"Loaded {len(pop)} lenses")

    print(f"\nRunning with {N_JOBS} workers...")
    all_results = Parallel(n_jobs=N_JOBS, verbose=10)(
        delayed(process_one_lens)(i, lens, BANDS, CADENCE, DEFAULT_DEPTH_5SIG, SEED)
        for i, lens in enumerate(pop)
    )

    fail_counts = {}
    rows = []
    n_done = 0

    for i, res in enumerate(all_results):
        if res is None:
            continue
        if res["status"] != "ok":
            reason = res["status"].split(":")[0]
            fail_counts[reason] = fail_counts.get(reason, 0) + 1
            continue

        for img, td_true in res["true_delays"].items():
            if img == "image_1":
                continue
            td_fit = res["fit_delays"].get(img, np.nan)
            td_err = res["fit_delay_errors"].get(img, [np.nan, np.nan])
            rows.append({
                "lens_index": i, "image": img,
                "z_lens": res["z_lens"], "z_source": res["z_source"],
                "n_images": res["n_images"],
                "true_delay": td_true, "fit_delay": td_fit,
                "fit_err_lo": td_err[0] if np.ndim(td_err) else td_err,
                "fit_err_hi": td_err[1] if np.ndim(td_err) else td_err,
                "residual": td_fit - td_true,
            })
        n_done += 1

    elapsed = time.time() - t_start
    tab = Table(rows)
    tab.write(OUTPUT_FILE, overwrite=True)
    res_arr = np.array([r["residual"] for r in rows if np.isfinite(r["residual"])])

    print(f"\n{'='*60}")
    print(f"Measured {n_done} systems, {len(rows)} delays in {elapsed/60:.1f} min")
    if len(res_arr):
        print(f"Delay residual: median={np.median(res_arr):+.2f} d  "
              f"std={np.std(res_arr):.2f} d  "
              f"|res|<2d: {np.mean(np.abs(res_arr)<2)*100:.0f}%")
    print(f"Saved -> {OUTPUT_FILE}")
    if fail_counts:
        print(f"\nFailure summary:")
        for reason, count in sorted(fail_counts.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
