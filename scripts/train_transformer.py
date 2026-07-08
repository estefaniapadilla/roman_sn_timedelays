"""
train_transformer.py
====================
Train the Stage-2 transformer on a tier directory produced by
build_training_set.py. Runs in the `roman_ml` env (NOT sntd_bayesn).

Smoke test (2 min, CPU):
  python scripts/train_transformer.py --limit 64 --epochs 3 --name smoke

Real run:
  python scripts/train_transformer.py --epochs 200 --name tier1_v1

Outputs under outputs/models/<name>/:
  best.pt        checkpoint with the lowest validation NLL
  config.json    every hyperparameter + data fingerprint (reproducibility)
  history.csv    per-epoch train/val loss
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import time
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from roman_td.paths import TRAINING_DIR, REPO_ROOT
from roman_td.ml_data import LensedSNDataset
from roman_td.transformer import LensedSNTransformer, loss_fn


def run_epoch(model, loader, opt=None):
    training = opt is not None
    model.train(training)
    totals, n = {}, 0
    with torch.set_grad_enabled(training):
        for batch in loader:
            out = model(batch["tokens"], batch["mask"], batch["scalars"])
            parts = loss_fn(out, batch)
            if training:
                opt.zero_grad()
                parts["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            bs = len(batch["dt"])
            n += bs
            for k, v in parts.items():
                totals[k] = totals.get(k, 0.0) + float(v.detach()) * bs
    return {k: v / n for k, v in totals.items()}


def delay_metrics(model, loader):
    """Residual stats for the delay head (same columns as the benchmarks)."""
    model.eval()
    res, sig = [], []
    with torch.no_grad():
        for batch in loader:
            out = model(batch["tokens"], batch["mask"], batch["scalars"])
            m = batch["dt_mask"].bool()
            res.append((out["dt"] - batch["dt"])[m].numpy())
            sig.append(out["dt_logsig"][m].exp().numpy())
    res = np.concatenate(res) if res else np.array([])
    sig = np.concatenate(sig) if sig else np.array([])
    if not len(res):
        return {}
    return {"n": len(res), "median": float(np.median(res)),
            "p68": float(np.percentile(np.abs(res), 68)),
            "lt2": float(np.mean(np.abs(res) < 2) * 100),
            "cov1sig": float(np.mean(np.abs(res) < sig) * 100)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(TRAINING_DIR, "tier1"))
    ap.add_argument("--name", default="run")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gp_dropout", type=float, default=0.2)
    ap.add_argument("--crop_prob", type=float, default=0.0,
                    help="P(random observer-window crop) per train example; "
                         "0 = full curves (original behavior)")
    ap.add_argument("--crop_window", type=float, default=365.0,
                    help="observing-window length for crops (days)")
    ap.add_argument("--l_max", type=int, default=512)
    ap.add_argument("--patience", type=int, default=20,
                    help="early stop after this many epochs w/o val improvement")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap examples per split (smoke tests)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    outdir = os.path.join(REPO_ROOT, "outputs", "models", args.name)
    os.makedirs(outdir, exist_ok=True)

    tr = LensedSNDataset(args.data, "train", l_max=args.l_max,
                         gp_dropout=args.gp_dropout, seed=args.seed,
                         limit=args.limit, crop_prob=args.crop_prob,
                         crop_window_days=args.crop_window)
    va = LensedSNDataset(args.data, "val", l_max=args.l_max,
                         gp_dropout=0.0, limit=args.limit)
    print(f"train {len(tr)}  val {len(va)}  (data: {args.data})")
    tl = DataLoader(tr, batch_size=args.batch, shuffle=True,
                    num_workers=args.workers)
    vl = DataLoader(va, batch_size=args.batch, num_workers=args.workers)

    model = LensedSNTransformer()
    n_par = sum(p.numel() for p in model.parameters())
    print(f"model: {n_par/1e6:.2f} M parameters")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    with open(os.path.join(outdir, "config.json"), "w") as f:
        json.dump({**vars(args), "n_params": n_par,
                   "n_train": len(tr), "n_val": len(va)}, f, indent=1)

    best, best_epoch = float("inf"), -1
    hist_path = os.path.join(outdir, "history.csv")
    with open(hist_path, "w") as f:
        f.write("epoch,train_total,val_total,val_dt,lr\n")

    for epoch in range(args.epochs):
        t0 = time.time()
        trm = run_epoch(model, tl, opt)
        vam = run_epoch(model, vl)
        sched.step()
        with open(hist_path, "a") as f:
            f.write(f"{epoch},{trm['total']:.4f},{vam['total']:.4f},"
                    f"{vam['dt']:.4f},{sched.get_last_lr()[0]:.2e}\n")
        marker = ""
        if vam["total"] < best:
            best, best_epoch = vam["total"], epoch
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val": vam}, os.path.join(outdir, "best.pt"))
            marker = "  *best*"
        print(f"epoch {epoch:3d}  train {trm['total']:7.3f}  "
              f"val {vam['total']:7.3f}  ({time.time()-t0:.0f}s){marker}")
        if epoch - best_epoch >= args.patience:
            print(f"early stop: no val improvement for {args.patience} epochs")
            break

    # reload best and report delay accuracy on val
    ckpt = torch.load(os.path.join(outdir, "best.pt"), weights_only=False)
    model.load_state_dict(ckpt["model"])
    dm = delay_metrics(model, vl)
    print(f"\nbest epoch {ckpt['epoch']}  val NLL {best:.3f}")
    if dm:
        print(f"val delays: n={dm['n']}  median={dm['median']:+.2f}d  "
              f"P68={dm['p68']:.2f}d  <2d: {dm['lt2']:.0f}%  "
              f"|res|<1sig: {dm['cov1sig']:.0f}%  "
              f"(GBT bar: beat 1.59d P68 on good systems)")
    print(f"saved -> {outdir}")


if __name__ == "__main__":
    main()
