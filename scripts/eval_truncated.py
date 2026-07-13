"""
eval_truncated.py
=================
The truncation evaluation: who can still measure delays when the survey
window clips the light curves? (architecture §3.6 evaluation battery)

Two stages, two envs:

  # 1. clip val curves + re-run the GP on them (env: sntd_bayesn, ~1 h)
  python scripts/eval_truncated.py --stage clip

  # 2. score GP / full-model / crop-model on the clipped systems (roman_ml)
  python scripts/eval_truncated.py --stage score \
      --models deep31k_full_v4 deep31k_crop_v4

Stage `clip` writes a builder-format mini dataset to
outputs/eval/truncated_val/tier1 (clipped __phot.ecsv, truth JSON with the
clipped-curve GP block, index.ecsv), so stage `score` reuses the standard
LensedSNDataset/tokenizer path — the realistic inference chain: hints and
the phase zero-point come from the GP run on the SAME clipped data the
model sees. Truth delays are unchanged (targets are the real delays).

Per-system truncation category (recorded in the index):
  mild    — every image keeps its peak epoch
  peak    — reference keeps its peak, >=1 other image loses it
  severe  — reference image loses its peak
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from astropy.table import Table

from roman_td.paths import TRAINING_DIR, REPO_ROOT

OUT_ROOT = os.path.join(REPO_ROOT, "outputs", "eval", "truncated_val")
WINDOW_DAYS = 365.0


# ── stage 1: clip + GP ───────────────────────────────────────────────────────
def peak_mjd_per_image(tab):
    """MJD of each image's brightest point (any band)."""
    out = {}
    img = np.asarray(tab["image"]).astype(str)
    mjd = np.asarray(tab["mjd"], dtype=float)
    flux = np.asarray(tab["flux"], dtype=float)
    for m in set(img):
        sel = img == m
        out[m] = float(mjd[sel][np.argmax(flux[sel])])
    return out


def clip_window(tab, lens_index, window=WINDOW_DAYS):
    """Deterministic observer window for this system (seeded by lens index).

    Same sliding range as the training crop augmentation; if the curve is
    too short to slide a window (< 0.4 W), return it unclipped.
    """
    rng = np.random.default_rng(9000 + int(lens_index))
    mjd = np.asarray(tab["mjd"], dtype=float)
    lo, hi = float(mjd.min()), float(mjd.max())
    if hi - 0.7 * window <= lo - 0.3 * window:
        return tab, 1.0
    start = rng.uniform(lo - 0.3 * window, hi - 0.7 * window)
    keep = (mjd >= start) & (mjd <= start + window)
    if keep.sum() < 10:          # pathological draw: keep the curve instead
        return tab, 1.0
    return tab[keep], float(keep.sum()) / len(keep)


def stage_clip(args):
    from roman_td.crosscorr import gp_cross_correlate

    src = os.path.join(TRAINING_DIR, "tier1_deep31k")
    idx = Table.read(os.path.join(src, "tier1", "index.ecsv"))
    sel = (np.asarray(idx["split"]).astype(str) == "val")
    if not args.all_realizations:
        sel &= np.char.endswith(np.asarray(idx["phot"]).astype(str),
                                "_r00__phot.ecsv")
    idx = idx[sel]
    if args.limit:
        idx = idx[:args.limit]
    outdir = os.path.join(OUT_ROOT, "tier1")
    os.makedirs(outdir, exist_ok=True)

    rows = []
    n_gp_fail = 0
    for k, row in enumerate(idx):
        tab = Table.read(os.path.join(src, str(row["phot"])))
        with open(os.path.join(src, str(row["truth"]))) as f:
            rec = json.load(f)

        peaks_before = peak_mjd_per_image(tab)
        ctab, kept = clip_window(tab, row["lens_index"])
        mjd = np.asarray(ctab["mjd"], dtype=float)
        peak_kept = {m: bool((mjd.min() <= p <= mjd.max())
                     and (m in set(np.asarray(ctab["image"]).astype(str))))
                     for m, p in peaks_before.items()}
        # ref image per ORIGINAL gp (brightest); category from peak survival
        ref0 = rec["gp"]["ref_image"]
        if all(peak_kept.values()):
            cat = "mild"
        elif peak_kept.get(ref0, False):
            cat = "peak"
        else:
            cat = "severe"

        gp_block, gp_ok = None, False
        try:
            gp = gp_cross_correlate(ctab, rec["images"],
                                    rng=np.random.default_rng(1234 + k))
            gp_block = {
                "ref_image": gp["ref_image"],
                "t_peak_ref": gp["t_peak_ref"],
                "per_image": {img: {kk: e[kk] for kk in
                              ("dt_gp", "dt_gp_err", "dt_per_band",
                               "flux_ratio", "flux_ratio_err", "quality")}
                              for img, e in gp["per_image"].items()},
            }
            gp_ok = True
        except Exception:
            n_gp_fail += 1

        base = f"trunc_{int(row['lens_index']):05d}"
        ctab.write(os.path.join(outdir, base + "__phot.ecsv"),
                   overwrite=True)
        rec_out = dict(rec)
        rec_out["gp"] = gp_block if gp_ok else rec["gp"]  # keep anchor format
        rec_out["gp_clipped_ok"] = gp_ok
        rec_out["trunc"] = {"kept_frac": kept, "category": cat,
                            "peak_kept": peak_kept}
        with open(os.path.join(outdir, base + "__truth.json"), "w") as f:
            json.dump(rec_out, f)
        rows.append({"lens_index": int(row["lens_index"]),
                     "phot": f"tier1/{base}__phot.ecsv",
                     "truth": f"tier1/{base}__truth.json",
                     "split": "val", "gp_ok": gp_ok,
                     "trunc_cat": cat, "kept_frac": kept})
        if (k + 1) % 50 == 0:
            print(f"{k+1}/{len(idx)}  gp_fail so far: {n_gp_fail}",
                  flush=True)

    Table(rows=rows).write(os.path.join(OUT_ROOT, "tier1", "index.ecsv"),
                           overwrite=True)
    # NOTE: index lives inside tier1/ so LensedSNDataset(data_dir=.../tier1)
    # resolves phot paths relative to OUT_ROOT as the builder layout does.
    cats = [r["trunc_cat"] for r in rows]
    print(f"\nclipped {len(rows)} systems -> {outdir}")
    print(f"GP on clipped curves: {len(rows)-n_gp_fail} ok, {n_gp_fail} FAIL "
          f"({n_gp_fail/len(rows)*100:.1f}%)")
    for c in ("mild", "peak", "severe"):
        print(f"  {c}: {cats.count(c)}")


# ── stage 2: score ───────────────────────────────────────────────────────────
def stage_score(args):
    import torch
    from torch.utils.data import DataLoader
    from roman_td.ml_data import LensedSNDataset, collate_dynamic
    from roman_td.transformer import LensedSNTransformer
    from roman_td.tokenize import DT_SCALE

    data_dir = os.path.join(OUT_ROOT, "tier1")
    idx = Table.read(os.path.join(data_dir, "index.ecsv"))
    n_all = len(idx)
    n_gp_fail = int((~np.asarray(idx["gp_ok"], dtype=bool)).sum())
    print(f"clipped systems: {n_all}  (GP failed on {n_gp_fail} = "
          f"{n_gp_fail/n_all*100:.1f}% — no anchor, excluded below; these "
          "route to Stage 3 at inference)\n")

    ds = LensedSNDataset(data_dir, "val", l_max=512)
    dl = DataLoader(ds, batch_size=64, num_workers=2,
                    collate_fn=collate_dynamic)
    cats = {int(r["lens_index"]): str(r["trunc_cat"]) for r in idx}

    def gather(model):
        P, T, HN, LI = [], [], [], []
        with torch.no_grad():
            for b in dl:
                out = model(b["tokens"], b["mask"], b["scalars"])
                m = b["dt_mask"].bool()
                P.append(out["dt"][m].numpy() * DT_SCALE)
                T.append(b["dt"][m].numpy() * DT_SCALE)
                HN.append(b["scalars"][:, 3:6][m].numpy() * DT_SCALE)
                li = b["lens_index"].unsqueeze(1).expand(-1, 3)
                LI.append(li[m].numpy())
        return [np.concatenate(x) for x in (P, T, HN, LI)]

    arms = {}
    for name in args.models:
        ck = torch.load(os.path.join(REPO_ROOT, "outputs", "models", name,
                                     "best.pt"), weights_only=False)
        model = LensedSNTransformer()
        model.load_state_dict(ck["model"])
        model.eval()
        arms[name] = gather(model)

    # GP arm from any gather (hints identical across models)
    p0, t0, hn0, li0 = arms[args.models[0]]
    hmask = hn0 != 0

    def report(label, pred, truth, sel):
        e = np.abs(pred[sel] - truth[sel])
        if not len(e):
            print(f"  {label:26s} n=0")
            return
        print(f"  {label:26s} n={len(e):4d}  P68={np.percentile(e,68):7.2f}d"
              f"  <2d={np.mean(e<2)*100:3.0f}%  >30d={np.mean(e>30)*100:4.1f}%")

    catarr = np.array([cats[int(i)] for i in li0])
    for cat in ("mild", "peak", "severe", None):
        sel = np.ones(len(t0), bool) if cat is None else catarr == cat
        print(f"category: {cat or 'ALL'}  "
              f"({len(set(li0[sel].astype(int)))} systems)")
        report("raw GP hint (clipped)", hn0, t0, sel & hmask)
        for name in args.models:
            p, t, _, _ = arms[name]
            report(name, p, t, sel)
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["clip", "score"])
    ap.add_argument("--all_realizations", action="store_true",
                    help="clip: use all val realizations (default r00 only)")
    ap.add_argument("--limit", type=int, default=None,
                    help="clip: cap systems (smoke test)")
    ap.add_argument("--models", nargs="+",
                    default=["deep31k_full_v4", "deep31k_crop_v4"])
    args = ap.parse_args()
    if args.stage == "clip":
        stage_clip(args)
    else:
        stage_score(args)


if __name__ == "__main__":
    main()
