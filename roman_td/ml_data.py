"""
ml_data.py
==========
PyTorch Dataset for the Stage-2 transformer. Reads the training set written
by scripts/build_training_set.py (index.ecsv + per-example __phot.ecsv /
__truth.json) through roman_td.tokenize — the SAME code path used at
inference, so there is no train/serve skew.

Runs in the `roman_ml` env (torch + numpy + astropy only — no fitting stack).

Targets, all in the tokenizer's slot order (REFERENCE IMAGE = slot 0; the
"non-reference slots" below are slots 1..3):
    dt      (3,)  observer-frame delay vs reference, days/DT_SCALE + dt_mask
    logmu   (3,)  log(mu_i / mu_ref)                        + same mask
    micro_amp / micro_slope (4,) per image incl. reference  + micro_mask
    theta, dust (scalars; dust = host E(B-V))
Tiers 0-1: micro targets are ZERO, not masked — the network must learn to
report "no microlensing" (architecture §3.6).
"""

import json
import os
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from astropy.table import Table

from roman_td.tokenize import (tokenize_system, scalar_features,
                               N_IMAGE_SLOTS, TOKEN_DIM, DT_SCALE)

N_OTHER = N_IMAGE_SLOTS - 1   # non-reference slots


def collate_dynamic(batch):
    """Default-style collate, then trim token padding to the batch max.

    Real tokens are contiguous at the front (tokenize.py), so slicing off
    the all-padding tail changes nothing the model computes — padding is
    attention-masked and never enters the loss. Attention cost is O(L^2):
    trimming 512 -> ~200-250 typical is a 2-4x epoch speedup.
    """
    out = {k: torch.stack([b[k] for b in batch]) for k in batch[0]}
    l_real = int(out["mask"].sum(dim=1).max())
    l_keep = max(l_real, 1)
    out["tokens"] = out["tokens"][:, :l_keep]
    out["mask"] = out["mask"][:, :l_keep]
    return out

# --- window-crop augmentation (off unless crop_prob > 0) -----------------
MIN_PTS_IMAGE = 5    # an image "survives" a crop with at least this many pts
MIN_PTS_TARGET = 3   # below this, an image's dt/logmu targets are masked
GP_KEEP_FRAC = 0.8   # crop removing >20% of points -> GP hint dropped too


class LensedSNDataset(Dataset):
    def __init__(self, data_dir: str, split: str, l_max: int = 512,
                 gp_dropout: float = 0.0, seed: int = 0,
                 limit: Optional[int] = None,
                 crop_prob: float = 0.0, crop_window_days: float = 365.0):
        """
        data_dir : e.g. outputs/training/tier1
        split    : "train" | "val" | "test"  (assigned per lens system)
        gp_dropout : probability of zeroing the GP hint features
            (training-time robustness; use 0 for val/test)
        limit    : cap the number of examples (smoke tests)
        crop_prob : probability of clipping an example to a random
            observer-time window (survey-edge realism); 0 = full curves,
            the original behavior
        crop_window_days : length of the simulated observing window (days)
        """
        self.data_dir = data_dir
        self.l_max = l_max
        self.gp_dropout = gp_dropout
        self.crop_prob = crop_prob
        self.crop_window_days = crop_window_days
        self.rng = np.random.default_rng(seed)

        idx = Table.read(os.path.join(data_dir, "index.ecsv"))
        idx = idx[np.asarray(idx["split"]).astype(str) == split]
        # examples whose GP failed have no phase zero-point -> skip (rare;
        # at inference these route to the physical fit anyway)
        idx = idx[np.asarray(idx["gp_ok"], dtype=bool)]
        if limit is not None:
            idx = idx[:limit]
        self.index = idx

    def __len__(self):
        return len(self.index)

    def _crop_window(self, tab, ref: str):
        """Clip the system to one shared observer-time window.

        Roman points at the SYSTEM: every visit sees all images, so there is
        exactly one window in MJD — a trailing image (peak at +dt) is then
        naturally caught later in its evolution, or missed entirely.
        Returns (possibly-clipped table, fraction of points kept).
        """
        tcol = "mjd" if "mjd" in tab.colnames else "time"
        mjd = np.asarray(tab[tcol], dtype=float)
        img = np.asarray(tab["image"]).astype(str)
        lo, hi = float(mjd.min()), float(mjd.max())
        W = float(self.crop_window_days)
        if hi - 0.7 * W <= lo - 0.3 * W:
            # curve spans < 0.4 W: the uniform() below would be invalid
            # (crashed on short-span systems in the 31k build) — and such
            # a curve is already effectively truncated, so don't crop
            return tab, 1.0
        for _ in range(5):
            # window slides from "tail clipped" to "rise clipped / trailing
            # image only" — mimics a SN exploding at a random time relative
            # to the survey campaign
            start = self.rng.uniform(lo - 0.3 * W, hi - 0.7 * W)
            keep = (mjd >= start) & (mjd <= start + W)
            n_ref = int(np.count_nonzero(keep & (img == ref)))
            n_ok = sum(int(np.count_nonzero(keep & (img == m))) >= MIN_PTS_IMAGE
                       for m in set(img))
            if n_ref >= MIN_PTS_IMAGE and n_ok >= 2:
                return tab[keep], float(np.count_nonzero(keep)) / len(keep)
        return tab, 1.0

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        row = self.index[i]
        # paths in index.ecsv are relative to the training root (outdir)
        root = os.path.dirname(self.data_dir.rstrip("/"))
        tab = Table.read(os.path.join(root, str(row["phot"])))
        with open(os.path.join(root, str(row["truth"]))) as f:
            rec = json.load(f)

        gp = rec["gp"]
        ref = gp["ref_image"]
        images = [ref] + [im for im in rec["images"] if im != ref]

        kept_frac = 1.0
        if self.crop_prob > 0 and self.rng.random() < self.crop_prob:
            tab, kept_frac = self._crop_window(tab, ref)

        tokens, mask = tokenize_system(tab, images, rec["z_source"],
                                       gp["t_peak_ref"], l_max=self.l_max)
        # GP hint dropped at random (robustness) OR because a heavy crop
        # means the full-curve GP answer would leak unseen data
        drop = ((self.gp_dropout > 0
                 and self.rng.random() < self.gp_dropout)
                or kept_frac < GP_KEEP_FRAC)
        scalars = scalar_features(rec["z_source"], rec["z_lens"],
                                  len(rec["images"]), gp=gp, drop_gp=drop)

        # ---- targets in slot order ------------------------------------
        # points per image AFTER any crop: point-starved images have
        # unmeasurable delay/magnification -> their loss slots are masked
        imgcol = np.asarray(tab["image"]).astype(str)
        n_pts = {m: int(np.count_nonzero(imgcol == m)) for m in set(imgcol)}
        delays = dict(zip(rec["images"], rec["delays"]))
        mus = dict(zip(rec["images"], rec["mu_macro"]))
        dt = np.zeros(N_OTHER, dtype=np.float32)
        logmu = np.zeros(N_OTHER, dtype=np.float32)
        dt_mask = np.zeros(N_OTHER, dtype=np.float32)
        for s, img in enumerate(images[1:][:N_OTHER]):
            if n_pts.get(img, 0) < MIN_PTS_TARGET:
                continue
            dt[s] = (delays[img] - delays[ref]) / DT_SCALE
            logmu[s] = np.log(max(mus[img], 1e-12) / max(mus[ref], 1e-12))
            dt_mask[s] = 1.0

        micro_amp = np.zeros(N_IMAGE_SLOTS, dtype=np.float32)
        micro_slope = np.zeros(N_IMAGE_SLOTS, dtype=np.float32)
        micro_mask = np.zeros(N_IMAGE_SLOTS, dtype=np.float32)
        for s, img in enumerate(images[:N_IMAGE_SLOTS]):
            micro_amp[s] = float(rec["micro_amp"].get(img, 0.0))
            micro_mask[s] = 1.0   # zero label, present — not masked

        return {
            "tokens": torch.from_numpy(tokens),
            "mask": torch.from_numpy(mask),
            "scalars": torch.from_numpy(scalars),
            "dt": torch.from_numpy(dt),
            "logmu": torch.from_numpy(logmu),
            "dt_mask": torch.from_numpy(dt_mask),
            "micro_amp": torch.from_numpy(micro_amp),
            "micro_slope": torch.from_numpy(micro_slope),
            "micro_mask": torch.from_numpy(micro_mask),
            "theta": torch.tensor(float(rec["sn_params"]["theta"])),
            "dust": torch.tensor(float(rec["sn_params"]["hostebv"])),
            "lens_index": torch.tensor(int(row["lens_index"])),
        }
