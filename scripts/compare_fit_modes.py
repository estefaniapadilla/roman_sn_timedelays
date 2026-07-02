"""
compare_fit_modes.py
=====================
Runs the same fixed subset of lens systems through several measure_one()
knob variants with identical noise realizations, so residual accuracy and
per-system wall-clock time can be compared head-to-head instead of across
uncontrolled runs.

VARIANTS defaults to the "robust"/"fast" convenience presets, but each entry
is just a kwargs dict for measure_one()'s independent t0_window/npoints/
maxcall knobs — add more entries here to isolate which knob actually drives
a speed or accuracy difference (e.g. narrow window with an uncapped sampler,
or the wide window with capped npoints) rather than only ever comparing the
two bundled presets.

Subset is drawn from an existing delay_benchmark_*.ecsv (spanning small to
large true delays, including the widest one available) so every system is
already known to pass the SNR/delay cuts and fit successfully under the
original approach.

Usage: python compare_fit_modes.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pickle
import numpy as np
from astropy.table import Table, vstack
from joblib import Parallel, delayed

from roman_td.sntd_wrapper import measure_one, register_roman_bands, DEFAULT_DEPTH_5SIG, FIT_PRESETS
from run_sntd_population import PICKLE, BANDS, CADENCE, SEED

# Reference table to draw a representative lens_index subset from (must exist —
# run_sntd_population.py or a prior compare run writes one of these).
REFERENCE_ECSV = os.path.join(os.path.dirname(__file__), "delay_benchmark_robust.ecsv")
if not os.path.exists(REFERENCE_ECSV):
    REFERENCE_ECSV = os.path.join(os.path.dirname(__file__), "delay_benchmark.ecsv")

N_COMPARE = 12       # number of lens systems to test, spanning the delay range
N_JOBS = 6           # modest — avoid contention with any full population run
VARIANTS = dict(FIT_PRESETS)  # {label: measure_one kwargs} — extend for isolated knob tests

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "fit_mode_comparison.ecsv")


def select_lens_indices(n):
    ref = Table.read(REFERENCE_ECSV)
    by_lens = {}
    for row in ref:
        by_lens.setdefault(int(row["lens_index"]), float(row["true_delay"]))
    lens_ids = list(by_lens.keys())
    lens_ids.sort(key=lambda i: by_lens[i])

    # Evenly spaced across the delay-sorted list, plus force-include the
    # single widest-delay system so the robust-vs-fast comparison actually
    # exercises the failure mode fast mode is riskiest for.
    picks = {lens_ids[int(round(x))] for x in np.linspace(0, len(lens_ids) - 1, n - 1)}
    picks.add(lens_ids[-1])
    return sorted(picks, key=lambda i: by_lens[i])


def process_one(i, lens, label, mode_kwargs, seed_offset):
    import warnings
    warnings.filterwarnings("ignore")

    try:
        zS = float(lens.source_redshift_list[0])
    except Exception:
        return None

    # Same seed per lens regardless of variant -> identical noise realization,
    # so any difference in outcome is attributable to the fitting knobs only.
    rng = np.random.default_rng(seed_offset + i)
    result = measure_one(
        lens, BANDS, zS,
        cadence_days=CADENCE,
        depth_5sig=DEFAULT_DEPTH_5SIG,
        rng=rng,
        lens_index=i,
        **mode_kwargs,
    )
    if result is not None:
        result["lens_index"] = i
        result["fit_mode"] = label
    return result


def main():
    register_roman_bands()

    lens_indices = select_lens_indices(N_COMPARE)
    print(f"Comparing variants={list(VARIANTS)} on {len(lens_indices)} systems: {lens_indices}")

    with open(PICKLE, "rb") as f:
        payload = pickle.load(f)
    pop = payload["lens_population"]

    jobs = [
        (i, pop[i], label, mode_kwargs)
        for i in lens_indices
        for label, mode_kwargs in VARIANTS.items()
    ]
    print(f"Running {len(jobs)} fits with {N_JOBS} workers...")
    results = Parallel(n_jobs=N_JOBS, verbose=10)(
        delayed(process_one)(i, lens, label, mode_kwargs, SEED) for i, lens, label, mode_kwargs in jobs
    )

    rows = []
    for res in results:
        if res is None:
            continue
        fit_time_s = res["diagnostics"].get("fit_time_s", np.nan)
        if res["status"] != "ok":
            rows.append({
                "lens_index": res["lens_index"], "fit_mode": res["fit_mode"],
                "image": "", "true_delay": np.nan, "fit_delay": np.nan,
                "residual": np.nan, "fit_time_s": fit_time_s,
                "status": res["status"],
            })
            continue
        for img, td_true in res["true_delays"].items():
            if img == "image_1":
                continue
            td_fit = res["fit_delays"].get(img, np.nan)
            rows.append({
                "lens_index": res["lens_index"], "fit_mode": res["fit_mode"],
                "image": img, "true_delay": td_true, "fit_delay": td_fit,
                "residual": td_fit - td_true, "fit_time_s": fit_time_s,
                "status": "ok",
            })

    tab = Table(rows)
    tab.write(OUTPUT_FILE, overwrite=True)
    print(f"\nSaved -> {OUTPUT_FILE}")

    print(f"\n{'='*60}")
    for mode in VARIANTS:
        sub = tab[tab["fit_mode"] == mode]
        ok = sub[sub["status"] == "ok"]
        res_arr = np.array(ok["residual"])
        res_arr = res_arr[np.isfinite(res_arr)]
        time_arr = np.array(sub["fit_time_s"])
        time_arr = time_arr[np.isfinite(time_arr)]
        if len(res_arr):
            print(f"[{mode}] {len(ok)}/{len(sub)} ok  "
                  f"residual median={np.median(res_arr):+.2f}d std={np.std(res_arr):.2f}d")
        else:
            print(f"[{mode}] {len(ok)}/{len(sub)} ok  no successful fits")
        if len(time_arr):
            print(f"          fit time median={np.median(time_arr):.1f}s "
                  f"mean={np.mean(time_arr):.1f}s max={np.max(time_arr):.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
