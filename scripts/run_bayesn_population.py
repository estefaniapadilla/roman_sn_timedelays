"""
run_bayesn_population.py
========================
BayeSN + SNTD two-stage time-delay measurement on a slsim lens population.

Previously: sntd_bayesn_population.py (main() + process_one_lens())
New location: scripts/run_bayesn_population.py
Reusable functions moved to: roman_td/bayesn_wrapper.py and roman_td/simulate.py

Per lens system:
  1. Pull true delays / magnifications / redshifts from the slsim lens object
  2. Simulate multi-image Roman photometry using the BayeSN model itself
     (self-consistent simulation: same model used for both sim and fit)
  3. Build an SNTD MISN and run the two-stage fit:
       series fit  → theta + amplitude + t0 posterior (magnification proxy)
       color fit   → time delays with theta prior from series
  4. Compare fitted vs. true delays; write a benchmark table (delay_benchmark.ecsv)

Performance tips
----------------
- Set --n_jobs=-1 to use all cores; keep --ncpu_fit=1 when n_jobs > 1 to avoid
  oversubscribing (dynesty is single-threaded per fit).
- Reduce --npoints_series and --npoints_color to 100-200 for a 3-5x speedup
  with modest accuracy loss on well-sampled Roman light curves.
- return_as="generator" (joblib ≥1.3) lets us save incrementally so a crash
  does not lose all results.

Run example:
python scripts/run_bayesn_population.py \\
  --pickle /path/roman_deep_lens_population_compat.pkl \\
  --survey time_domain_deep \\
  --bayesn_yaml /path/BAYESN.YAML \\
  --filters_yaml /path/filters.yaml \\
  --outdir /path/pop_out \\
  --n_jobs 8 --n_systems 50
"""

from __future__ import annotations

import sys
import os
# Add repo root to path so both roman_td and bayesncosmo are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import pickle
import time
import argparse
import traceback
import warnings
from typing import Dict, Any, Optional, List

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

warnings.filterwarnings("ignore", message=r".*bandpass.*outside spectral range.*")
warnings.filterwarnings("ignore", message=r".*invalid value encountered.*")
warnings.filterwarnings("ignore", message=r".*divide by zero.*")

from roman_td.bayesn_wrapper import (
    ensure_registered, usable_bands, make_misn_from_table,
    fit_system, result_to_rows, save_benchmark, _to_builtin, log,
)
from roman_td.simulate import (
    lens_truth, simulate_photometry,
    DEFAULT_DEPTH_5SIG, SURVEY_BANDS,
)


def process_one_lens(
    i: int,
    lens,
    bands: List[str],
    cadence: float,
    depths: Dict[str, float],
    seed: int,
    bayesn_yaml: str,
    filters_yaml: str,
    npoints_series: int,
    npoints_color: int,
    ncpu_fit: int,
    outdir: Optional[str],
) -> Dict[str, Any]:
    """End-to-end processing for one lens system (runs inside a joblib worker)."""
    warnings.filterwarnings("ignore")
    out: Dict[str, Any] = {"lens_index": i, "status": "ok"}
    _t0_lens = time.time()

    try:
        # Registration must happen inside each worker because joblib spawns new
        # processes whose sncosmo registries are empty.
        ensure_registered(bayesn_yaml, filters_yaml)

        truth = lens_truth(lens)
        if truth is None:
            out["status"] = "bad_lens: missing redshift/delays/magnifications"
            return out
        if truth["n_images"] < 2:
            out["status"] = "too_few_images"
            return out

        zS = truth["z_source"]
        bands_ok = usable_bands(bands, zS)
        if len(bands_ok) < 2:
            out["status"] = f"no_bands_in_coverage: z={zS:.2f}"
            return out

        rng = np.random.default_rng(seed + i)
        tab, sim_info = simulate_photometry(truth, bands_ok, cadence, depths, rng)
        if tab is None:
            out["status"] = "undetected: <2 images above 5sigma"
            return out

        truth_kept = sim_info["truth"]
        images = truth_kept["images"]

        misn = make_misn_from_table(tab, images, object_name=f"lens_{i:05d}")
        fit_series, fit_color, timing, ref = fit_system(
            misn, tab, zS, images,
            npoints_series=npoints_series,
            npoints_color=npoints_color,
            ncpu_fit=ncpu_fit,
        )

        # Use color fit delays if available; fall back to series if color was skipped
        # (color is skipped when fewer than 2 bands are in the data)
        params_fit, res_fit = fit_color if fit_color is not None else fit_series
        vp_fit = list(res_fit.vparam_names)
        td: Dict[str, Any] = {ref: 0.0}
        td_err: Dict[str, Any] = {ref: [0.0, 0.0]}
        for _img in [im for im in images if im != ref]:
            _key = f"dt_{_img}"
            if _key in vp_fit:
                _lo, _med, _hi = params_fit[vp_fit.index(_key)]
                td[_img] = float(_med)
                td_err[_img] = [float(_med - _lo), float(_hi - _med)]
            else:
                td[_img] = float("nan")
                td_err[_img] = [float("nan"), float("nan")]

        out.update({
            "z_lens": truth_kept["z_lens"], "z_source": zS,
            "n_images": truth_kept["n_images"],
            "images": images, "bands": bands_ok,
            "true_delays": {img: float(d) for img, d in zip(images, truth_kept["delays"])},
            "true_mu": {img: float(m) for img, m in zip(images, truth_kept["mu"])},
            "fit_delays": {k: _to_builtin(v) for k, v in td.items()},
            "fit_delay_errors": {k: _to_builtin(v) for k, v in td_err.items()},
            "sim_params": {k: v for k, v in sim_info.items() if k != "truth"},
            "timing": timing,
        })

        if outdir is not None:
            payload_path = os.path.join(outdir, f"lens_{i:05d}__payload.json")
            with open(payload_path, "w") as f:
                json.dump(out, f, indent=2, default=_to_builtin)

        out["elapsed_sec"] = time.time() - _t0_lens
        return out

    except Exception:
        out["status"] = f"fit_error: {traceback.format_exc(limit=3).splitlines()[-1]}"
        out["elapsed_sec"] = time.time() - _t0_lens
        if outdir is not None:
            try:
                with open(os.path.join(outdir, f"lens_{i:05d}__ERROR.txt"), "w") as f:
                    f.write(traceback.format_exc())
            except Exception:
                pass
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pickle", required=True)
    ap.add_argument("--survey", default="time_domain_wide",
                    choices=list(SURVEY_BANDS.keys()))
    ap.add_argument("--bayesn_yaml", required=True)
    ap.add_argument("--filters_yaml", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--cadence", type=float, default=5.0)
    ap.add_argument("--n_systems", type=int, default=50,
                    help="Stop after this many successful fits (0 = all)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_jobs", type=int, default=1)
    ap.add_argument("--ncpu_fit", type=int, default=1,
                    help="CPUs per dynesty fit. Keep 1 when n_jobs > 1.")
    ap.add_argument("--npoints_series", type=int, default=500)
    ap.add_argument("--npoints_color", type=int, default=500)
    ap.add_argument("--save_every", type=int, default=10)
    ap.add_argument("--save_payloads", action="store_true",
                    help="Write per-lens JSON payloads + error tracebacks")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    benchmark_path = os.path.join(args.outdir, "delay_benchmark.ecsv")
    payload_dir = args.outdir if args.save_payloads else None

    t_start = time.time()
    with open(args.pickle, "rb") as f:
        payload = pickle.load(f)
    pop = payload["lens_population"] if isinstance(payload, dict) else payload
    log(f"Loaded {len(pop)} lenses from {args.pickle}")

    log("Verifying BayeSN + filter registration in main process...")
    ensure_registered(args.bayesn_yaml, args.filters_yaml)
    log("Registration OK.")

    bands = SURVEY_BANDS[args.survey]
    log(f"Survey={args.survey}  bands={bands}  cadence={args.cadence}d  "
        f"n_jobs={args.n_jobs}  ncpu_fit={args.ncpu_fit}")

    target = args.n_systems if args.n_systems > 0 else len(pop)

    tasks = (
        delayed(process_one_lens)(
            i, lens, bands, args.cadence, DEFAULT_DEPTH_5SIG, args.seed,
            args.bayesn_yaml, args.filters_yaml,
            args.npoints_series, args.npoints_color, args.ncpu_fit,
            payload_dir,
        )
        for i, lens in enumerate(pop)
    )

    rows: List[Dict[str, Any]] = []
    fail_counts: Dict[str, int] = {}
    n_done = 0
    total_lens_sec = 0.0
    n_timed = 0

    parallel = Parallel(n_jobs=args.n_jobs, return_as="generator")
    pbar = tqdm(parallel(tasks), total=len(pop), desc="Lenses", unit="lens")
    for res in pbar:
        elapsed = res.get("elapsed_sec", 0.0)
        if elapsed > 0:
            total_lens_sec += elapsed
            n_timed += 1

        if res["status"] != "ok":
            reason = res["status"]
            fail_counts[reason] = fail_counts.get(reason, 0) + 1
            avg_s = total_lens_sec / n_timed if n_timed else 0.0
            pbar.set_postfix(ok=n_done, fails=sum(fail_counts.values()), avg_s=f"{avg_s:.1f}s")
            continue

        rows.extend(result_to_rows(res))
        n_done += 1
        avg_s = total_lens_sec / n_timed if n_timed else 0.0
        pbar.set_postfix(ok=n_done, delays=len(rows), avg_s=f"{avg_s:.1f}s")

        if n_done % args.save_every == 0:
            save_benchmark(rows, benchmark_path)
            log(f"Checkpoint: {n_done} systems, {len(rows)} delays -> {benchmark_path}")

        if n_done >= target:
            log(f"Reached target of {target} successful systems; stopping.")
            break
    pbar.close()

    elapsed = time.time() - t_start
    save_benchmark(rows, benchmark_path)
    res_arr = np.array([r["residual"] for r in rows if np.isfinite(r["residual"])])

    print(f"\n{'='*60}")
    print(f"Measured {n_done} systems, {len(rows)} delays in {elapsed/60:.1f} min")
    if len(res_arr):
        print(f"Delay residual: median={np.median(res_arr):+.2f} d  "
              f"std={np.std(res_arr):.2f} d  "
              f"|res|<2d: {np.mean(np.abs(res_arr) < 2)*100:.0f}%")
    print(f"Saved -> {benchmark_path}")
    if fail_counts:
        print("\nFailure summary:")
        for reason, count in sorted(fail_counts.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
