"""
eval_micro_pairs.py
===================
Frozen transformer on the matched micro-off/on twins (tier-2 pilot).

Scores every head on both twins of each pair and writes one CSV row per
delay-bearing image: delay (vs truth), mag-ratio (vs TRUE MACRO ratio —
the gap under micro is the contamination), micro head (vs true labels;
a tiers-0/1-trained model should output ~0 — the documented null).

External population -> no leakage: all splits are scored.

Run in roman_ml:
  conda run -n roman_ml python scripts/eval_micro_pairs.py \
      --data outputs/training/tier2_pilot --models deep31k_full_v4
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import csv
import argparse

import numpy as np

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
LN10_04 = 0.4 * np.log(10.0)          # ln flux-ratio per delta-mag


def gather_tier(tier_dir, model):
    """Run the model over ALL splits of one tier dir.

    Returns {(lens_index, realization): example-dict}; dataset order maps
    1:1 onto ds.index rows (shuffle=False), asserted below.
    """
    import torch
    from torch.utils.data import DataLoader
    from roman_td.ml_data import LensedSNDataset, collate_dynamic
    from roman_td.tokenize import DT_SCALE, HINT_DT_SLICE

    out = {}
    for split in ("train", "val", "test"):
        ds = LensedSNDataset(tier_dir, split, l_max=512)
        if len(ds) == 0:
            continue
        dl = DataLoader(ds, batch_size=32, shuffle=False,
                        collate_fn=collate_dynamic)
        pos = 0
        with torch.no_grad():
            for b in dl:
                o = model(b["tokens"], b["mask"], b["scalars"])
                B = len(b["lens_index"])
                for j in range(B):
                    row = ds.index[pos + j]
                    li = int(b["lens_index"][j])
                    assert li == int(row["lens_index"]), \
                        f"order mismatch at {pos + j}"
                    out[(li, int(row["realization"]))] = {
                        "split": str(row["split"]),
                        "dt_mask": b["dt_mask"][j].numpy().astype(bool),
                        "dt_true": b["dt"][j].numpy() * DT_SCALE,
                        "gp": b["scalars"][j, HINT_DT_SLICE].numpy() * DT_SCALE,
                        "tf": o["dt"][j].numpy() * DT_SCALE,
                        "tf_sig": np.exp(o["dt_logsig"][j].numpy()) * DT_SCALE,
                        "logmu_true": b["logmu"][j].numpy(),
                        "logmu_tf": o["logmu"][j].numpy(),
                        "micro_true": b["micro_amp"][j].numpy(),   # (4,) mag
                        "micro_tf": o["micro_amp"][j].numpy(),
                    }
                pos += B
    return out


def p68(x):
    return float(np.percentile(np.abs(x), 68)) if len(x) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="outputs/training/tier2_pilot")
    ap.add_argument("--models", nargs="+", default=["deep31k_full_v4"])
    ap.add_argument("--outdir", default="outputs/eval/micro_pairs")
    args = ap.parse_args()

    import torch
    from roman_td.transformer import LensedSNTransformer

    os.makedirs(args.outdir, exist_ok=True)
    for name in args.models:
        ck = torch.load(os.path.join(REPO_ROOT, "outputs", "models", name,
                                     "best.pt"), weights_only=False)
        model = LensedSNTransformer()
        model.load_state_dict(ck["model"])
        model.eval()

        off = gather_tier(os.path.join(args.data, "tier1"), model)
        on = gather_tier(os.path.join(args.data, "tier2"), model)
        keys = sorted(set(off) & set(on))
        print(f"\n=== {name}: {len(keys)} matched pairs ===")

        rows = []
        for k in keys:
            a, b = off[k], on[k]
            # twins must agree on truth (same lens/SN/images by construction)
            if not (np.array_equal(a["dt_mask"], b["dt_mask"]) and
                    np.allclose(a["dt_true"], b["dt_true"])):
                continue
            for s in np.flatnonzero(a["dt_mask"]):
                rows.append({
                    "lens_index": k[0], "realization": k[1],
                    "split": a["split"], "img_slot": int(s),
                    "dt_true": a["dt_true"][s],
                    "gp_off": a["gp"][s], "gp_on": b["gp"][s],
                    "tf_off": a["tf"][s], "tf_on": b["tf"][s],
                    "tf_sig_off": a["tf_sig"][s], "tf_sig_on": b["tf_sig"][s],
                    "logmu_true": a["logmu_true"][s],
                    "logmu_off": a["logmu_tf"][s],
                    "logmu_on": b["logmu_tf"][s],
                    # micro slots include the ref image at 0 -> non-ref = s+1
                    "micro_true": b["micro_true"][s + 1],
                    "micro_tf_off": a["micro_tf"][s + 1],
                    "micro_tf_on": b["micro_tf"][s + 1],
                    "ref_micro_true": b["micro_true"][0],
                })

        csv_path = os.path.join(args.outdir, f"{name}.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"{len(rows)} image-delays -> {csv_path}\n")

        g = {c: np.array([r[c] for r in rows]) for c in rows[0]}
        t = g["dt_true"]

        # ---- delays ----
        print(f"{'delay arm':>14}  {'P68|err| off':>12}  {'P68|err| on':>11}"
              f"  {'P68|on-off|':>11}  {'med signed':>10}")
        for lab, po, pn in (("GP hint", g["gp_off"], g["gp_on"]),
                            ("transformer", g["tf_off"], g["tf_on"])):
            hm = (po != 0) & (pn != 0)   # hinted slots only for GP arm
            d = pn[hm] - po[hm]
            print(f"{lab:>14}  {p68(po[hm]-t[hm]):>11.2f}d  "
                  f"{p68(pn[hm]-t[hm]):>10.2f}d  {p68(d):>10.2f}d  "
                  f"{np.median(d):>+9.2f}d")

        # ---- mag-ratio contamination (mag units) ----
        e_off = (g["logmu_off"] - g["logmu_true"]) / LN10_04
        e_on = (g["logmu_on"] - g["logmu_true"]) / LN10_04
        dmicro = g["micro_true"] - g["ref_micro_true"]   # expected contam (mag)
        print(f"\nmag-ratio vs TRUE MACRO (mag): "
              f"P68 off={p68(e_off):.3f}  on={p68(e_on):.3f}  "
              f"paired P68={p68(e_on - e_off):.3f}")
        ok = np.abs(dmicro) > 1e-6
        if ok.sum() > 2:
            c = np.corrcoef(e_on[ok] - e_off[ok], -dmicro[ok])[0, 1]
            print(f"  contamination vs -(micro_img - micro_ref): "
                  f"corr={c:+.2f} over n={ok.sum()} "
                  f"(+1 = ratio head absorbs micro as expected)")

        # ---- micro head (expected null for tiers-0/1-trained models) ----
        print(f"\nmicro head (mag): pred on twins with TRUE micro "
              f"P68={p68(g['micro_true']):.3f}:")
        print(f"  off: mean={g['micro_tf_off'].mean():+.4f} "
              f"std={g['micro_tf_off'].std():.4f}   "
              f"on: mean={g['micro_tf_on'].mean():+.4f} "
              f"std={g['micro_tf_on'].std():.4f}")
        if ok.sum() > 2:
            c = np.corrcoef(g["micro_tf_on"], g["micro_true"])[0, 1]
            print(f"  corr(pred, true) on micro twins: {c:+.2f} "
                  f"(~0 expected before tier-2 training)")


if __name__ == "__main__":
    main()
