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
from roman_td.micro import MicroCurves
from roman_td.simulate import (lens_truth, simulate_photometry,
                               IMG_LABELS, SURVEY_BANDS, SURVEY_CADENCE,
                               SURVEY_DEPTH_5SIG)
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
        cadence = (args.cadence if args.cadence is not None
                   else SURVEY_CADENCE[args.survey])
        depths = SURVEY_DEPTH_5SIG[args.survey]
        tab, sim_info = simulate_photometry(
            truth, bands_ok, cadence, depths, rng,
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
                           "cadence": cadence,
                           "depths": {b: depths[b] for b in bands_ok},
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


def _micro_labels(tab, images, mc: MicroCurves, zS: float, t0: float):
    """Per-image microlensing labels evaluated at the ACTUAL observed epochs.

    Returns (micro_amp, micro_block):
      micro_amp[img]   signed mean delta-mag over that image's observations
                       (averaged across bands) — the head's scalar target
      micro_block[img] per-band mean/std delta-mag + kappa_star/kappa_tot/shear
    """
    from roman_td.micro import REST_DAYS_EXPLOSION_TO_PEAK
    micro_amp, micro_block = {}, {}
    tk_delays = None  # delays already baked into tab via simulate; recompute t_expl per row
    for k, img in enumerate(images):
        sel = np.asarray(tab["image"]) == img
        per_band_mean, per_band_std, all_dm = {}, {}, []
        for b in mc.bands:
            bsel = sel & (np.asarray(tab["filter"]) == b)
            if not bsel.any():
                continue
            mjd = np.asarray(tab["mjd"])[bsel]
            # observed mjd -> days since explosion (mirror simulate.py)
            t_expl = (mjd - t0 - _row_delay(tab, images, img)) + \
                REST_DAYS_EXPLOSION_TO_PEAK * (1.0 + zS)
            dm = mc(b, k, t_expl)
            per_band_mean[b] = float(np.mean(dm))
            per_band_std[b] = float(np.std(dm))
            all_dm.append(dm)
        micro_amp[img] = float(np.mean(np.concatenate(all_dm))) if all_dm else 0.0
        micro_block[img] = {
            "mean_dmag_per_band": per_band_mean,
            "std_dmag_per_band": per_band_std,
            "kappa_star": float(mc.kappa_star[k]),
            "kappa_tot": float(mc.kappa_tot[k]),
            "shear": float(mc.shear[k]),
        }
    return micro_amp, micro_block


def _row_delay(tab, images, img):
    """Delay of image `img` relative to the first kept image (from tab meta)."""
    # delays are re-referenced in simulate; reconstruct from the truth carried
    # alongside — passed via tab.meta to avoid recomputation here
    return tab.meta["delays"][images.index(img)]


def _emit_example(tab, sim_info, sn_params, i, r, split, tier, tier_dir,
                  zS, bands_ok, cadence, depths, args, rng,
                  micro_amp=None, micro_block=None):
    """Write one (phot, truth) example; return its index row."""
    tk = sim_info["truth"]
    os.makedirs(tier_dir, exist_ok=True)

    gp_block = None
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
        pass

    if micro_amp is None:
        micro_amp = {img: 0.0 for img in tk["images"]}

    base = f"lens_{i:05d}_r{r:02d}"
    phot_path = os.path.join(tier_dir, base + "__phot.ecsv")
    truth_path = os.path.join(tier_dir, base + "__truth.json")
    tab.write(phot_path, overwrite=True)

    record = {
        "lens_index": i, "realization": r, "tier": tier,
        "split": split,
        "z_lens": tk["z_lens"], "z_source": zS,
        "images": tk["images"],
        "delays": list(tk["delays"]),
        "mu_macro": list(tk["mu"]),
        "micro_amp": micro_amp,
        "micro": micro_block,
        "sn_params": {**sn_params, "hostr_v": 3.1,
                      "amplitude": sim_info["amplitude"], "t0": tk["t0"]},
        "sim_config": {"survey": args.survey, "bands": bands_ok,
                       "cadence": cadence,
                       "depths": {b: depths[b] for b in bands_ok},
                       "seed": args.seed, "sim_model": "bayesn",
                       "micro_source": (os.path.basename(args.micro_dir)
                                        if micro_block is not None else None)},
        "gp": gp_block,
    }
    with open(truth_path, "w") as f:
        json.dump(record, f, indent=1, default=_to_builtin)

    return {
        "lens_index": i, "realization": r, "tier": tier,
        "split": split, "z_lens": float(tk["z_lens"]),
        "z_source": float(zS), "n_images": int(tk["n_images"]),
        "n_obs": len(tab), "gp_ok": gp_block is not None,
        "phot": os.path.relpath(phot_path, args.outdir),
        "truth": os.path.relpath(truth_path, args.outdir),
    }


def process_one_micro_pair(npz_path, args, bayesn_yaml, filters_yaml):
    """Matched micro-off/on twins for one external (npz) lens.

    Same lens, SN, and per-realization noise seed produce a tier-1 (macro
    only) and tier-2 (macro x microlensing) example; the only within-pair
    difference is microlensing. Skips a realization if micro changes which
    images are detectable (the delay comparison must be image-matched).
    """
    warnings.filterwarnings("ignore")
    ensure_registered(bayesn_yaml, filters_yaml)

    mc = MicroCurves(npz_path)
    i = int(os.path.basename(npz_path).split("_")[1].split("__")[0])
    truth = mc.truth()
    if truth["n_images"] < 2:
        return [], {"skip_nimg": 1}
    zS = truth["z_source"]
    bands_ok = usable_bands(SURVEY_BANDS[args.survey], zS)
    if len(bands_ok) < 2:
        return [], {"skip_bands": 1}

    prng = np.random.default_rng(args.seed + i)
    sn_params = {"theta": float(prng.normal(0, 1)),
                 "hostebv": float(min(prng.exponential(0.1), 1.0))}
    split = split_of(i, args.seed)
    cadence = (args.cadence if args.cadence is not None
               else SURVEY_CADENCE[args.survey])
    depths = SURVEY_DEPTH_5SIG[args.survey]

    rows, stats = [], {"pairs": 0, "skip_undetected": 0, "skip_mismatch": 0}
    for r in range(args.realizations):
        seed_r = args.seed + i * 1000 + r + 1
        twins = {}
        for tier, micro_model in ((1, None), (2, mc)):
            rng = np.random.default_rng(seed_r)
            tab, sim_info = simulate_photometry(
                truth, bands_ok, cadence, depths, rng,
                sn_params=sn_params, micro=micro_model)
            if tab is None:
                break
            tab.meta["delays"] = list(sim_info["truth"]["delays"])
            twins[tier] = (tab, sim_info, rng)
        if len(twins) < 2:
            stats["skip_undetected"] += 1
            continue
        if twins[1][1]["truth"]["images"] != twins[2][1]["truth"]["images"]:
            stats["skip_mismatch"] += 1
            continue

        # tier-1 (macro-only) twin
        tab1, info1, rng1 = twins[1]
        rows.append(_emit_example(
            tab1, info1, sn_params, i, r, split, 1,
            os.path.join(args.outdir, "tier1"),
            zS, bands_ok, cadence, depths, args, rng1))
        # tier-2 (microlensed) twin, with real per-image micro labels
        tab2, info2, rng2 = twins[2]
        images2 = info2["truth"]["images"]
        m_amp, m_blk = _micro_labels(tab2, images2, mc, zS, info2["truth"]["t0"])
        rows.append(_emit_example(
            tab2, info2, sn_params, i, r, split, 2,
            os.path.join(args.outdir, "tier2"),
            zS, bands_ok, cadence, depths, args, rng2,
            micro_amp=m_amp, micro_block=m_blk))
        stats["pairs"] += 1
    return rows, stats


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
    ap.add_argument("--cadence", type=float, default=None,
                    help="flat cadence override (days); default = per-filter "
                         "CCS spec (SURVEY_CADENCE[--survey])")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_jobs", type=int, default=8)
    ap.add_argument("--micro_dir", default=None,
                    help="directory of *__micro.npz curves; switches to "
                         "matched-pair mode (tier1 macro / tier2 microlensed "
                         "twins written under --outdir), external population")
    args = ap.parse_args()

    if args.micro_dir is not None:
        return _run_micro_pairs(args)

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


def _run_micro_pairs(args):
    """Matched micro-off/on twin build from an external npz curve directory."""
    import glob
    t0 = time.time()
    npzs = sorted(glob.glob(os.path.join(args.micro_dir, "*__micro.npz")))
    if args.n_systems > 0:
        npzs = npzs[:args.n_systems]
    print(f"Micro matched-pair build: {len(npzs)} lenses x "
          f"{args.realizations} realizations, {args.n_jobs} workers")
    print(f"  curves : {args.micro_dir}")
    print(f"  outdir : {args.outdir} (tier1 = macro, tier2 = microlensed)")

    results = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(process_one_micro_pair)(p, args, args.bayesn_yaml,
                                        args.filters_yaml)
        for p in npzs
    )
    rows = [r for lens_rows, _ in results for r in lens_rows]
    stats = {}
    for _, st in results:
        for k, v in st.items():
            stats[k] = stats.get(k, 0) + v

    for tier in (1, 2):
        trows = [r for r in rows if r["tier"] == tier]
        if not trows:
            continue
        tier_dir = os.path.join(args.outdir, f"tier{tier}")
        Table(trows).write(os.path.join(tier_dir, "index.ecsv"), overwrite=True)

    n_pairs = sum(1 for r in rows if r["tier"] == 1)
    splits = [r["split"] for r in rows if r["tier"] == 1]
    names, counts = np.unique(splits, return_counts=True)
    print(f"\n{n_pairs} matched pairs ({2*n_pairs} examples) from "
          f"{len(set(r['lens_index'] for r in rows))} lenses "
          f"in {(time.time()-t0)/60:.1f} min")
    print(f"splits (by pair): {dict(zip(names, counts))}")
    print(f"gp_ok : {sum(r['gp_ok'] for r in rows)}/{len(rows)}")
    print(f"skips : {stats}")
    print(f"index -> {os.path.join(args.outdir, 'tier{1,2}', 'index.ecsv')}")


if __name__ == "__main__":
    main()
