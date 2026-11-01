"""Vibration token encoder (README §9): [peak_only, log_psd_20_40] (2-D, per
window, station-normalized) → 64-D token.  Linear(2→32) GELU LN Linear(32→64)
GELU LN.  Tokens may ONLY come from these two features."""
import torch.nn as nn


class VibrationTokenEncoder(nn.Module):
    def __init__(self, in_dim=2, hidden=32, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, out_dim), nn.GELU(), nn.LayerNorm(out_dim))

    def forward(self, X_s):                    # [B,S,2]
        return self.net(X_s)                   # [B,S,64]
