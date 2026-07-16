"""
compare_micro_pairs.py
======================
First GP-bias-under-microlensing measurement (pipeline doc ledger row 6).

Reads the matched micro-off (tier1) / micro-on (tier2) twins written by
`build_training_set.py --micro_dir` and, per image, compares the Stage-1
GP delay error between the twins. Twins share lens, SN, and noise seed,
so the paired difference (err_on - err_off) is caused by microlensing
alone.

Usage:
  python scripts/compare_micro_pairs.py --data outputs/training/tier2_pilot
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import glob
import argparse

import numpy as np


def gp_delays_rel(rec, img0):
    """GP delay of every image relative to img0 (GP ref may differ)."""
    gp = rec.get("gp")
    if gp is None:
        return None
    d = {gp["ref_image"]: 0.0}
    for img, e in gp["per_image"].items():
        d[img] = float(e["dt_gp"])
    if img0 not in d:
        return None
    return {img: v - d[img0] for img, v in d.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="outputs/training/tier2_pilot")
    args = ap.parse_args()

    rows = []   # one row per (pair, non-ref image)
    for p1 in sorted(glob.glob(os.path.join(args.data, "tier1", "*__truth.json"))):
        p2 = p1.replace("tier1", "tier2")
        if not os.path.exists(p2):
            continue
        r1, r2 = json.load(open(p1)), json.load(open(p2))
        images = r1["images"]
        truth = dict(zip(images, r1["delays"]))
        img0 = images[0]
        g1 = gp_delays_rel(r1, img0)
        g2 = gp_delays_rel(r2, img0)
        if g1 is None or g2 is None:
            continue
        for img in images[1:]:
            if img not in g1 or img not in g2:
                continue
            blk = (r2.get("micro") or {}).get(img, {})
            stds = list(blk.get("std_dmag_per_band", {}).values())
            rows.append({
                "base": os.path.basename(p1).replace("__truth.json", ""),
                "img": img,
                "dt_true": truth[img],
                "err_off": g1[img] - truth[img],
                "err_on": g2[img] - truth[img],
                "micro_std": float(np.mean(stds)) if stds else 0.0,
                "micro_amp": float(r2["micro_amp"].get(img, 0.0)),
            })

    if not rows:
        print("no comparable pairs found");  return
    e_off = np.array([r["err_off"] for r in rows])
    e_on = np.array([r["err_on"] for r in rows])
    d = e_on - e_off                     # pure micro perturbation
    mstd = np.array([r["micro_std"] for r in rows])

    def p68(x):  return float(np.percentile(np.abs(x), 68))

    n = len(rows)
    print(f"{n} image-delays from matched pairs   ({args.data})\n")
    print(f"{'':>22}  {'micro OFF':>10}  {'micro ON':>10}")
    print(f"{'P68 |err| (d)':>22}  {p68(e_off):>10.2f}  {p68(e_on):>10.2f}")
    print(f"{'median err (d)':>22}  {np.median(e_off):>10.2f}  {np.median(e_on):>10.2f}")
    print(f"{'P95 |err| (d)':>22}  {np.percentile(np.abs(e_off),95):>10.2f}  "
          f"{np.percentile(np.abs(e_on),95):>10.2f}")

    print(f"\npaired micro perturbation (err_on - err_off), noise cancelled:")
    print(f"  median |d|: {np.median(np.abs(d)):.2f} d    P68 |d|: {p68(d):.2f} d    "
          f"P95: {np.percentile(np.abs(d),95):.2f} d")
    print(f"  median signed: {np.median(d):+.2f} d  (common-mode bias)")

    # split by micro variability of the image (quiet vs active)
    q = np.median(mstd[mstd > 0]) if (mstd > 0).any() else 0.0
    lo, hi = mstd <= q, mstd > q
    print(f"\nby image micro variability (split at std_dmag={q:.3f} mag):")
    print(f"  quiet  (n={lo.sum():3d}): P68 |perturb| {p68(d[lo]):.2f} d")
    print(f"  active (n={hi.sum():3d}): P68 |perturb| {p68(d[hi]):.2f} d")

    # worst cases
    print(f"\n5 largest perturbations:")
    for j in np.argsort(-np.abs(d))[:5]:
        r = rows[j]
        print(f"  {r['base']} img {r['img']}: dt_true={r['dt_true']:7.2f}  "
              f"err {r['err_off']:+7.2f} -> {r['err_on']:+7.2f}  "
              f"(micro std {r['micro_std']:.3f}, amp {r['micro_amp']:+.2f} mag)")


if __name__ == "__main__":
    main()
