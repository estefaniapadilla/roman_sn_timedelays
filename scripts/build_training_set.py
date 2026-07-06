"""
build_training_set.py
=====================
Stage-0 training-set builder (architecture doc §1.4 / §6.2): mass-produce
(photometry, truth record) pairs for the Stage-2 transformer.

Per lens system:
  - SN parameters drawn ONCE (seeded by lens index) — realizations are
    noise instances of the same supernova, never different SNe
  - N noise realizations simulated with the BayeSN model (tier 0: mu forced
    to 1; tier 1: macro magnification applied — no microlensing yet)
  - the Stage-1 GP runs on each realization; its result is stored in the
    truth record (t_peak_ref is the tokenization phase zero-point, so it
    MUST come from the GP, exactly as at inference time — never from truth)
  - train/val/test assigned per LENS SYSTEM (deterministic hash, 80/10/10);
    all realizations of a lens share its split — no leakage

Outputs (under outputs/training/tier{T}/):
  lens_00042_r00__phot.ecsv   canonical photometry table (§6.1)
  lens_00042_r00__truth.json  truth record (§6.2) + "gp" block + "split"
  index.ecsv                  one row per example: files, split, metadata

Usage:
  python scripts/build_training_set.py --n_systems 200 --realizations 3
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import time
import pickle
import hashlib
import argparse
import warnings
from typing import Optional

import numpy as np
from astropy.table import Table
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

from roman_td.paths import DATA_DIR, TRAINING_DIR
from roman_td.bayesn_wrapper import ensure_registered, usable_bands, _to_builtin
from roman_td.simulate import (lens_truth, simulate_photometry,
                               DEFAULT_DEPTH_5SIG, SURVEY_BANDS)
from roman_td.crosscorr import gp_cross_correlate

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")


def split_of(lens_index: int, seed: int) -> str:
    """Deterministic 80/10/10 split BY LENS SYSTEM (not realization)."""
    h = int(hashlib.md5(f"{seed}:{lens_index}".encode()).hexdigest(), 16) % 100
    return "train" if h < 80 else ("val" if h < 90 else "test")


def process_one_lens(i, lens, args, bayesn_yaml, filters_yaml):
    warnings.filterwarnings("ignore")
    ensure_registered(bayesn_yaml, filters_yaml)

    truth = lens_truth(lens)
    if truth is None or truth["n_images"] < 2:
        return []
    zS = truth["z_source"]
    bands_ok = usable_bands(SURVEY_BANDS[args.survey], zS)
    if len(bands_ok) < 2:
        return []

    if args.tier == 0:
        truth = dict(truth)
        truth["mu"] = np.ones_like(np.asarray(truth["mu"], dtype=float))

    # SN drawn once per lens: realizations share the supernova
    prng = np.random.default_rng(args.seed + i)
    sn_params = {"theta": float(prng.normal(0, 1)),
                 "hostebv": float(min(prng.exponential(0.1), 1.0))}

    split = split_of(i, args.seed)
    tier_dir = os.path.join(args.outdir, f"tier{args.tier}")
    os.makedirs(tier_dir, exist_ok=True)

    index_rows = []
    for r in range(args.realizations):
        rng = np.random.default_rng(args.seed + i * 1000 + r + 1)
        tab, sim_info = simulate_photometry(
            truth, bands_ok, args.cadence, DEFAULT_DEPTH_5SIG, rng,
            sn_params=sn_params)
        if tab is None:
            continue
        tk = sim_info["truth"]

        gp_block: Optional[dict] = None
        try:
            gp = gp_cross_correlate(tab, tk["images"], rng=rng)
            gp_block = {
                "ref_image": gp["ref_image"],
                "t_peak_ref": gp["t_peak_ref"],
                "per_image": {img: {k: e[k] for k in
                              ("dt_gp", "dt_gp_err", "dt_per_band",
                               "flux_ratio", "flux_ratio_err", "quality")}
                              for img, e in gp["per_image"].items()},
            }
        except Exception:
            pass   # gp None -> loader uses drop_gp features for this example

        base = f"lens_{i:05d}_r{r:02d}"
        phot_path = os.path.join(tier_dir, base + "__phot.ecsv")
        truth_path = os.path.join(tier_dir, base + "__truth.json")
        tab.write(phot_path, overwrite=True)

        record = {
            "lens_index": i, "realization": r, "tier": args.tier,
            "split": split,
            "z_lens": tk["z_lens"], "z_source": zS,
            "images": tk["images"],
            "delays": list(tk["delays"]),
            "mu_macro": list(tk["mu"]),
            # tiers 0-1: zero microlensing BY LABEL (network must learn to
            # report "none"), not by masking — architecture §3.6
            "micro_amp": {img: 0.0 for img in tk["images"]},
            "sn_params": {**sn_params,
                          "hostr_v": 3.1,
                          "amplitude": sim_info["amplitude"],
                          "t0": tk["t0"]},
            "sim_config": {"survey": args.survey, "bands": bands_ok,
                           "cadence": args.cadence,
                           "depths": {b: DEFAULT_DEPTH_5SIG[b] for b in bands_ok},
                           "seed": args.seed, "sim_model": "bayesn"},
            "gp": gp_block,
        }
        with open(truth_path, "w") as f:
            json.dump(record, f, indent=1, default=_to_builtin)

        index_rows.append({
            "lens_index": i, "realization": r, "tier": args.tier,
            "split": split, "z_lens": float(tk["z_lens"]),
            "z_source": float(zS), "n_images": int(tk["n_images"]),
            "n_obs": len(tab),
            "gp_ok": gp_block is not None,
            "phot": os.path.relpath(phot_path, args.outdir),
            "truth": os.path.relpath(truth_path, args.outdir),
        })
    return index_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pickle", default=os.path.join(
        DATA_DIR, "roman_deep_lens_population_compat.pkl"))
    ap.add_argument("--survey", default="time_domain_deep",
                    choices=list(SURVEY_BANDS.keys()))
    ap.add_argument("--bayesn_yaml", default=os.path.join(REPO_ROOT, "BAYESN.YAML"))
    ap.add_argument("--filters_yaml", default=os.path.join(REPO_ROOT, "filters.yaml"))
    ap.add_argument("--outdir", default=TRAINING_DIR)
    ap.add_argument("--tier", type=int, default=1, choices=[0, 1],
                    help="0 = SN only (mu forced to 1), 1 = SN + macro")
    ap.add_argument("--n_systems", type=int, default=0,
                    help="lens systems to attempt (0 = whole population)")
    ap.add_argument("--realizations", type=int, default=3,
                    help="noise realizations per lens (same SN)")
    ap.add_argument("--cadence", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_jobs", type=int, default=8)
    args = ap.parse_args()

    t0 = time.time()
    with open(args.pickle, "rb") as f:
        pop = pickle.load(f)["lens_population"]
    n = args.n_systems if args.n_systems > 0 else len(pop)
    print(f"Building tier-{args.tier} training set: {n} lenses x "
          f"{args.realizations} realizations, {args.n_jobs} workers")

    results = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(process_one_lens)(i, lens, args, args.bayesn_yaml,
                                  args.filters_yaml)
        for i, lens in enumerate(pop[:n])
    )
    rows = [r for lens_rows in results for r in lens_rows]

    tier_dir = os.path.join(args.outdir, f"tier{args.tier}")
    index_path = os.path.join(tier_dir, "index.ecsv")
    Table(rows).write(index_path, overwrite=True)

    splits = [r["split"] for r in rows]
    names, counts = np.unique(splits, return_counts=True)
    print(f"\n{len(rows)} examples from {len(set(r['lens_index'] for r in rows))} "
          f"systems in {(time.time()-t0)/60:.1f} min")
    print(f"splits: {dict(zip(names, counts))}")
    print(f"gp_ok : {sum(r['gp_ok'] for r in rows)}/{len(rows)}")
    print(f"index -> {index_path}")


if __name__ == "__main__":
    main()
