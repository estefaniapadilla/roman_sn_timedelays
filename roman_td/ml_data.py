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
    dt      (3,)  observer-frame delay vs reference, days   + dt_mask
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
                               N_IMAGE_SLOTS, TOKEN_DIM)

N_OTHER = N_IMAGE_SLOTS - 1   # non-reference slots


class LensedSNDataset(Dataset):
    def __init__(self, data_dir: str, split: str, l_max: int = 512,
                 gp_dropout: float = 0.0, seed: int = 0,
                 limit: Optional[int] = None):
        """
        data_dir : e.g. outputs/training/tier1
        split    : "train" | "val" | "test"  (assigned per lens system)
        gp_dropout : probability of zeroing the GP hint features
            (training-time robustness; use 0 for val/test)
        limit    : cap the number of examples (smoke tests)
        """
        self.data_dir = data_dir
        self.l_max = l_max
        self.gp_dropout = gp_dropout
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

        tokens, mask = tokenize_system(tab, images, rec["z_source"],
                                       gp["t_peak_ref"], l_max=self.l_max)
        drop = (self.gp_dropout > 0
                and self.rng.random() < self.gp_dropout)
        scalars = scalar_features(rec["z_source"], rec["z_lens"],
                                  len(rec["images"]), gp=gp, drop_gp=drop)

        # ---- targets in slot order ------------------------------------
        delays = dict(zip(rec["images"], rec["delays"]))
        mus = dict(zip(rec["images"], rec["mu_macro"]))
        dt = np.zeros(N_OTHER, dtype=np.float32)
        logmu = np.zeros(N_OTHER, dtype=np.float32)
        dt_mask = np.zeros(N_OTHER, dtype=np.float32)
        for s, img in enumerate(images[1:][:N_OTHER]):
            dt[s] = delays[img] - delays[ref]
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
