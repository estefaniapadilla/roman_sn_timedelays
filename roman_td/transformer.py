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

from typing import Dict

import torch
import torch.nn as nn

from roman_td.tokenize import TOKEN_DIM, N_IMAGE_SLOTS

N_OTHER = N_IMAGE_SLOTS - 1
SCALAR_DIM = 17   # roman_td.tokenize.scalar_features layout


class LensedSNTransformer(nn.Module):
    def __init__(self, d_model: int = 128, n_heads: int = 8,
                 n_layers: int = 4, d_ff: int = 512, dropout: float = 0.1,
                 d_fusion: int = 256):
        super().__init__()
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
        return {
            "dt": d[..., 0], "dt_logsig": d[..., 1],
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
