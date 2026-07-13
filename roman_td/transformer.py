"""
transformer.py
==============
The Stage-2 network (documents/transformer_explained.md §5): a set
transformer over photometric tokens with four heteroscedastic heads.

~1.1 M parameters at the defaults — deliberately small; do not scale up
before the GBT baseline and this size are saturated.

Runs in the `roman_ml` env. No positional encoding: token order is
meaningless (a set); time enters through the phase feature.
"""

import math
from typing import Dict

import torch
import torch.nn as nn

from roman_td.tokenize import TOKEN_DIM, N_IMAGE_SLOTS, HINT_DT_SLICE

N_OTHER = N_IMAGE_SLOTS - 1
SCALAR_DIM = 17   # roman_td.tokenize.scalar_features layout


class LensedSNTransformer(nn.Module):
    def __init__(self, d_model: int = 128, n_heads: int = 8,
                 n_layers: int = 4, d_ff: int = 512, dropout: float = 0.1,
                 d_fusion: int = 256, residual_dt: bool = True):
        """
        residual_dt : hinted delay slots output (GP hint + correction,
            tight sigma init 10 d) from head_corr; hint-less slots
            (dropout / GP fail / missing) output (absolute value, vague
            sigma init 100 d) from head_delay. Value AND sigma gated per
            slot by hint != 0 — gradients only reach the branch in use.
            Mirrors the GBT residual lesson (fixes.md 07-06): "predict 0"
            equals raw-GP accuracy, so training starts there and can only
            improve. Checkpoints trained before 2026-07-10 (deep_*_v1/v2)
            used False.
        """
        super().__init__()
        self.residual_dt = residual_dt
        self.embed = nn.Linear(TOKEN_DIM, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.fusion = nn.Sequential(
            nn.Linear(d_model + SCALAR_DIM, d_fusion), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_fusion, d_fusion), nn.GELU(),
        )
        # each head predicts (value, log_sigma) per target
        self.head_delay = nn.Linear(d_fusion, N_OTHER * 2)
        if residual_dt:
            # separate correction head: hinted slots use hint + corr,
            # hint-less slots use head_delay's absolute value. A single
            # shared unit for both regimes learns a biased compromise
            # (+25 d bias by epoch 1, first v3 attempt): "output ~0"
            # (hint present) and "output the full delay" (hint dropped)
            # pull the same weights in opposite directions.
            # (correction, log_sigma) per hinted slot — sigma is gated
            # too: one shared sigma output stalls (the few no-hint slots'
            # huge misses outshout the hinted majority, sigma stays ~100 d,
            # and at sigma ~100 d the miss/sigma^2 gradient is too weak to
            # stop the correction head wandering — observed as a growing
            # +3 d bias, first gated-v3 attempt)
            self.head_corr = nn.Linear(d_fusion, N_OTHER * 2)
            for lin in (self.head_delay, self.head_corr):
                nn.init.zeros_(lin.weight)
                nn.init.zeros_(lin.bias)
            # informed sigma init: anchored predictions START ~2 d
            # accurate, so claim sigma = 10 d (0.1 scaled) from step 1 —
            # strong restoring force on the correction immediately. The
            # no-hint branch keeps the vague sigma = 1 (100 d) init.
            with torch.no_grad():
                self.head_corr.bias.view(N_OTHER, 2)[:, 1] = math.log(0.1)
        self.head_mag = nn.Linear(d_fusion, N_OTHER * 2)
        self.head_micro = nn.Linear(d_fusion, N_IMAGE_SLOTS * 4)
        self.head_sn = nn.Linear(d_fusion, 4)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor,
                scalars: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        tokens  : (B, L, TOKEN_DIM)   mask: (B, L) bool, True = real
        scalars : (B, SCALAR_DIM)
        """
        B = tokens.shape[0]
        x = self.embed(tokens)
        x = torch.cat([self.cls.expand(B, -1, -1), x], dim=1)
        # transformer wants True = PADDING; CLS is never padding
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool,
                                     device=mask.device), ~mask], dim=1)
        h = self.encoder(x, src_key_padding_mask=pad)[:, 0]   # CLS output
        h = self.fusion(torch.cat([h, scalars], dim=1))

        d = self.head_delay(h).view(B, N_OTHER, 2)
        m = self.head_mag(h).view(B, N_OTHER, 2)
        u = self.head_micro(h).view(B, N_IMAGE_SLOTS, 4)
        s = self.head_sn(h)
        dt = d[..., 0]
        dt_logsig = d[..., 1]
        if self.residual_dt:
            hint = scalars[:, HINT_DT_SLICE]
            c = self.head_corr(h).view(B, N_OTHER, 2)
            # hint == 0 exactly <=> dropped, GP-failed, or missing slot
            # (real hints are never exactly 0.0)
            present = hint != 0
            dt = torch.where(present, hint + c[..., 0], dt)
            dt_logsig = torch.where(present, c[..., 1], dt_logsig)
        return {
            "dt": dt, "dt_logsig": dt_logsig,
            "logmu": m[..., 0], "logmu_logsig": m[..., 1],
            "micro_amp": u[..., 0], "micro_amp_logsig": u[..., 1],
            "micro_slope": u[..., 2], "micro_slope_logsig": u[..., 3],
            "theta": s[:, 0], "theta_logsig": s[:, 1],
            "dust": s[:, 2], "dust_logsig": s[:, 3],
        }


def _nll(pred, logsig, target, mask=None):
    """Gaussian negative log-likelihood, mean over valid entries.
    logsig clamped: exploding/vanishing sigmas destabilize early training."""
    logsig = logsig.clamp(-5.0, 5.0)
    nll = 0.5 * ((target - pred) / logsig.exp()) ** 2 + logsig
    if mask is None:
        return nll.mean()
    denom = mask.sum().clamp(min=1.0)
    return (nll * mask).sum() / denom


def loss_fn(out: Dict[str, torch.Tensor],
            batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Sum of per-head masked Gaussian NLLs (architecture §3.6)."""
    parts = {
        "dt": _nll(out["dt"], out["dt_logsig"], batch["dt"], batch["dt_mask"]),
        "logmu": _nll(out["logmu"], out["logmu_logsig"],
                      batch["logmu"], batch["dt_mask"]),
        "micro_amp": _nll(out["micro_amp"], out["micro_amp_logsig"],
                          batch["micro_amp"], batch["micro_mask"]),
        "micro_slope": _nll(out["micro_slope"], out["micro_slope_logsig"],
                            batch["micro_slope"], batch["micro_mask"]),
        "theta": _nll(out["theta"], out["theta_logsig"], batch["theta"]),
        "dust": _nll(out["dust"], out["dust_logsig"], batch["dust"]),
    }
    parts["total"] = sum(parts.values())
    return parts
