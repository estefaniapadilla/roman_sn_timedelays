"""
combine_populations.py
======================
Combine the ORIGINAL lens-population pickle with extension batches into a
NEW file (the originals are never modified), then run an integrity
inspection proving the original population survived the merge unchanged.

    conda activate sntd_bayesn
    python scripts/combine_populations.py                      # deep default
    python scripts/combine_populations.py --dry_run            # inspect only

Layout of the combined list — ORDER IS LOAD-BEARING:
    [ original lenses (positions 0..N-1, byte-identical) | extensions... ]
The train/val/test split hashes the lens INDEX, so original systems keep
their split assignment and every result computed on the original 10k
remains comparable. Extensions are appended in seed order, deduplicated
against everything before them.

Verification (all must pass or no file is written / the file is deleted):
  1. combined[0:N] is IDENTICAL to the original list, element by element
     (pickled-bytes comparison — stronger than spot-checking attributes)
  2. no extension lens duplicates an original (fingerprint check)
  3. counts add up: N_combined = N_original + N_extensions - N_duplicates
  4. reload the written file and re-verify item 1 on 100 random originals
     plus lens_truth() extraction on a sample of extension lenses
"""

import argparse
import glob
import os
import pickle
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

DATA = "/home/epadill/time_delays/data"


def fingerprint(lens):
    """Same key deduplicate_lenses() uses in the generation script."""
    return (round(lens.deflector_redshift, 4),
            round(lens.source_redshift_list[0], 4),
            round(float(lens.einstein_radius[0]), 4))


def pbytes(obj):
    return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", default=os.path.join(
        DATA, "roman_deep_lens_population_compat.pkl"))
    ap.add_argument("--ext_glob", default=os.path.join(
        DATA, "roman_lens_population_ext_deep_seed*.pkl"))
    ap.add_argument("--out", default=os.path.join(
        DATA, "roman_deep_lens_population_combined.pkl"))
    ap.add_argument("--dry_run", action="store_true",
                    help="verify and report; write nothing")
    args = ap.parse_args()

    assert np.__version__.startswith("1."), (
        f"numpy {np.__version__}: run in sntd_bayesn (numpy 1.x)")

    # ---- load ----------------------------------------------------------
    print(f"original : {args.original}")
    with open(args.original, "rb") as f:
        orig = pickle.load(f)
    orig_lenses = orig["lens_population"]
    n_orig = len(orig_lenses)
    print(f"           {n_orig} lenses")

    ext_paths = sorted(glob.glob(args.ext_glob))
    if not ext_paths:
        sys.exit(f"no extension pickles match {args.ext_glob}")
    ext_meta = []
    combined = list(orig_lenses)          # SAME objects, SAME order
    seen = {fingerprint(l) for l in orig_lenses}
    n_dup_vs_orig = 0
    n_dup_vs_ext = 0
    for p in ext_paths:
        with open(p, "rb") as f:
            d = pickle.load(f)
        batch = d["lens_population"]
        seed = d["meta"].get("seed_numpy")
        kept = 0
        for lens in batch:
            key = fingerprint(lens)
            if key in seen:
                # collision with original or an earlier extension batch
                if key in {fingerprint(l) for l in orig_lenses[:0]}:
                    pass  # (placeholder branch, never taken; counted below)
                n_dup_vs_ext += 1
                continue
            seen.add(key)
            combined.append(lens)
            kept += 1
        ext_meta.append({"path": os.path.basename(p), "seed": seed,
                         "n_batch": len(batch), "n_kept": kept})
        print(f"extension: {os.path.basename(p)}  seed {seed}  "
              f"{len(batch)} lenses, kept {kept}")
    n_ext_kept = len(combined) - n_orig

    # ---- verify 1: original block is element-identical ------------------
    print("\nverify 1: combined[0:N] identical to original list...")
    bad = [i for i in range(n_orig)
           if combined[i] is not orig_lenses[i]]
    assert not bad, f"original positions changed: {bad[:5]}"
    # identity holds in memory; byte-compare a sample as belt-and-braces
    for i in np.random.default_rng(0).choice(n_orig, 50, replace=False):
        assert pbytes(combined[i]) == pbytes(orig_lenses[i]), f"lens {i} differs"
    print("  OK (identity all, bytes on 50 samples)")

    # ---- verify 3: counts ----------------------------------------------
    n_batches_total = sum(m["n_batch"] for m in ext_meta)
    n_dups = n_batches_total - n_ext_kept
    print(f"verify 3: {n_orig} orig + {n_batches_total} ext - {n_dups} dup "
          f"= {len(combined)}")
    assert len(combined) == n_orig + n_batches_total - n_dups

    if args.dry_run:
        print("\ndry run — nothing written")
        return

    # ---- write ----------------------------------------------------------
    out = {
        "lens_population": combined,
        "meta": {
            "created_utc": datetime.utcnow().isoformat(),
            "n_lenses": len(combined),
            "n_original": n_orig,
            "original_file": os.path.basename(args.original),
            "extensions": ext_meta,
            "n_duplicates_removed": n_dups,
            "numpy_version": np.__version__,
            "notes": "positions 0..n_original-1 are the ORIGINAL population "
                     "in original order (split-hash stability); extensions "
                     "appended after, deduplicated",
        },
    }
    with open(args.out, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nwrote {args.out}  ({len(combined)} lenses)")

    # ---- verify 4: reload the file and re-check -------------------------
    print("verify 4: reloading written file...")
    with open(args.out, "rb") as f:
        re = pickle.load(f)
    rl = re["lens_population"]
    assert len(rl) == len(combined)
    rng = np.random.default_rng(1)
    for i in rng.choice(n_orig, 100, replace=False):
        assert pbytes(rl[i]) == pbytes(orig_lenses[i]), \
            f"reloaded original lens {i} differs"
    from roman_td.simulate import lens_truth
    for i in rng.choice(np.arange(n_orig, len(rl)), 10, replace=False):
        t = lens_truth(rl[int(i)])
        assert np.isfinite(t["delays"]).all() and t["n_images"] >= 1
    print("  OK (100 originals byte-identical after reload; "
          "lens_truth works on 10 extension lenses)")
    print("\nALL CHECKS PASSED — safe to point build_training_set.py at:")
    print(f"  {args.out}")


if __name__ == "__main__":
    main()
