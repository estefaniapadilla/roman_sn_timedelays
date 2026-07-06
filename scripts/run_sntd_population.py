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

from roman_td.sntd_wrapper import measure_one, register_roman_bands, DEFAULT_DEPTH_5SIG, FIT_PRESETS

# ── survey config: match the population 1 simulated ────────────────────────
PICKLE = "/home/epadill/time_delays/data/roman_deep_lens_population_compat.pkl"
SURVEY = "time_domain_deep"
if SURVEY == "time_domain_deep":
    BANDS = ["F087", "F106", "F129", "F158", "F184"]
else:
    BANDS = ["F062", "F087", "F106", "F129", "F158"]

CADENCE = 5.0       # days between visits
N_SYSTEMS = 1000    # stop after this many successful fits (0 = no limit, run the full population)
SEED = 42
N_JOBS = 32         # use all cores; set to -1 to auto-detect
SAVE_EVERY = 10
FIT_MODE = "gp"  # "robust" (wide bounds, slow) | "fast" (trial_fit-recentered, capped) | "gp" (GP-primed windows, robust fallback)

from roman_td.paths import benchmark_path
OUTPUT_FILE = benchmark_path(f"delay_benchmark_{FIT_MODE}.ecsv")


def process_one_lens(i, lens, bands, cadence, depth, seed_offset, fit_mode):
    """Wrapper for parallel execution — each worker gets its own RNG seed."""
    import warnings
    warnings.filterwarnings("ignore")

    try:
        zS = float(lens.source_redshift_list[0])
    except Exception:
        return None

    rng = np.random.default_rng(seed_offset + i)
    result = measure_one(
        lens, bands, zS,
        cadence_days=cadence,
        depth_5sig=depth,
        rng=rng,
        lens_index=i,
        **FIT_PRESETS[fit_mode],
    )
    if result is not None:
        result["lens_index"] = i
    return result


def main():
    register_roman_bands()
    t_start = time.time()

    with open(PICKLE, "rb") as f:
        payload = pickle.load(f)
    pop = payload["lens_population"]
    print(f"Loaded {len(pop)} lenses")

    print(f"\nRunning with {N_JOBS} workers, fit_mode={FIT_MODE!r}...")
    results_gen = Parallel(n_jobs=N_JOBS, verbose=10, return_as="generator_unordered")(
        delayed(process_one_lens)(i, lens, BANDS, CADENCE, DEFAULT_DEPTH_5SIG, SEED, FIT_MODE)
        for i, lens in enumerate(pop)
    )

    fail_counts = {}
    rows = []
    n_done = 0

    for res in results_gen:
        if res is None:
            continue
        if res["status"] != "ok":
            reason = res["status"].split(":")[0]
            fail_counts[reason] = fail_counts.get(reason, 0) + 1
            continue

        fit_time_s = res["diagnostics"].get("fit_time_s", np.nan)
        for img, td_true in res["true_delays"].items():
            if img == "image_1":
                continue
            td_fit = res["fit_delays"].get(img, np.nan)
            td_err = res["fit_delay_errors"].get(img, [np.nan, np.nan])
            mu_true = res.get("true_mu_ratio", {}).get(img, np.nan)
            mu_fit = res.get("fit_mu_ratio", {}).get(img, np.nan)
            mu_err = res.get("fit_mu_ratio_errors", {}).get(img, [np.nan, np.nan])
            rows.append({
                "lens_index": res["lens_index"], "image": img,
                "z_lens": res["z_lens"], "z_source": res["z_source"],
                "n_images": res["n_images"],
                "true_delay": td_true, "fit_delay": td_fit,
                "fit_err_lo": td_err[0] if np.ndim(td_err) else td_err,
                "fit_err_hi": td_err[1] if np.ndim(td_err) else td_err,
                "residual": td_fit - td_true,
                "true_mu_ratio": mu_true, "fit_mu_ratio": mu_fit,
                "mu_err_lo": mu_err[0] if np.ndim(mu_err) else mu_err,
                "mu_err_hi": mu_err[1] if np.ndim(mu_err) else mu_err,
                "mu_residual": mu_fit - mu_true,
                "fit_mode": FIT_MODE, "fit_time_s": fit_time_s,
                # gp mode only: which route ran + GP overhead (else blank/nan)
                "mode_effective": res["diagnostics"].get("mode_effective", ""),
                "gp_time_s": res["diagnostics"].get("gp_time_s", np.nan),
            })
        n_done += 1

        if n_done % 5 == 0 and rows:
            Table(rows).write(OUTPUT_FILE, overwrite=True)
            print(f"  [checkpoint] {n_done} fits done -> {OUTPUT_FILE}")

        if N_SYSTEMS and n_done >= N_SYSTEMS:
            print(f"  Reached N_SYSTEMS={N_SYSTEMS} successful fits, stopping early "
                  f"(remaining in-flight workers will finish their current fit but "
                  f"won't be collected).")
            break

    elapsed = time.time() - t_start
    tab = Table(rows)
    tab.write(OUTPUT_FILE, overwrite=True)
    res_arr = np.array([r["residual"] for r in rows if np.isfinite(r["residual"])])
    time_arr = np.array([r["fit_time_s"] for r in rows if np.isfinite(r["fit_time_s"])])

    print(f"\n{'='*60}")
    print(f"Measured {n_done} systems, {len(rows)} delays in {elapsed/60:.1f} min  (fit_mode={FIT_MODE!r})")
    if len(res_arr):
        print(f"Delay residual: median={np.median(res_arr):+.2f} d  "
              f"std={np.std(res_arr):.2f} d  "
              f"|res|<2d: {np.mean(np.abs(res_arr)<2)*100:.0f}%")
    mu_res_arr = np.array([r["mu_residual"] for r in rows if np.isfinite(r["mu_residual"])])
    if len(mu_res_arr):
        print(f"Mu-ratio residual ({len(mu_res_arr)} images): "
              f"median={np.median(mu_res_arr):+.3f}  std={np.std(mu_res_arr):.3f}")
    if len(time_arr):
        print(f"Per-system fit time: median={np.median(time_arr):.1f}s  "
              f"mean={np.mean(time_arr):.1f}s  max={np.max(time_arr):.1f}s")
    print(f"Saved -> {OUTPUT_FILE}")
    if fail_counts:
        print(f"\nFailure summary:")
        for reason, count in sorted(fail_counts.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
