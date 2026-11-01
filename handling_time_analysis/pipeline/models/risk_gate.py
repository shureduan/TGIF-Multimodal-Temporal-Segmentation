"""Risk gate (README §16-§17).

Input per window: r_j = [D_j, p_v, p_s, |p_v−p_s|, H(p_v), H(p_s)] → MLP → g_j.
Gate target (from DETACHED aux predictions vs y_bin):
    video wrong & vib right → 1.00   (vibration should intervene)
    video right & vib wrong → 0.00
    both right              → 0.00   (video is already correct; abstain)
    both wrong              → 0.00
Weighted BCE (pos_weight) because targets ≥0.5 are rare.
"""
import torch
import torch.nn as nn


def binary_entropy(p, eps=1e-8):
    eps = max(eps, torch.finfo(p.dtype).eps)
    p = p.clamp(eps, 1 - eps)
    return -(p * torch.log(p) + (1 - p) * torch.log(1 - p))


class RiskGate(nn.Module):
    def __init__(self, hidden=16, dropout=0.10):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(6, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, 1), nn.Sigmoid())

    def forward(self, D, p_v, p_s):            # [B,S] each → g [B,S]
        r = torch.stack([D, p_v, p_s, (p_v - p_s).abs(),
                         binary_entropy(p_v), binary_entropy(p_s)], dim=-1)
        return self.mlp(r).squeeze(-1)

    @staticmethod
    def gate_target(p_v, p_s, y_bin):
        """README §17 table; detached inputs."""
        v_ok = ((p_v.detach() >= 0.5).float() == y_bin)
        s_ok = ((p_s.detach() >= 0.5).float() == y_bin)
        tgt = torch.zeros_like(y_bin)
        tgt[(~v_ok) & s_ok] = 1.00
        tgt[v_ok & s_ok] = 0.00
        return tgt                              # video-right&vib-wrong / both-wrong → 0

    @staticmethod
    def gate_loss(g, target, pos_weight=4.0, eps=1e-7):
        g = g.clamp(eps, 1 - eps)
        w = torch.where(target >= 0.5, torch.full_like(target, pos_weight),
                        torch.ones_like(target))
        bce = -(target * torch.log(g) + (1 - target) * torch.log(1 - g))
        return (w * bce).sum() / w.sum().clamp_min(1e-6)
