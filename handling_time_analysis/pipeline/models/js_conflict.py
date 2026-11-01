"""Normalized JS divergence between the two binary heads (README §15).
D_j ∈ [0,1]; 0 = modalities agree, 1 = maximal conflict."""
import math

import torch


def _kl_bernoulli(p, q, eps=1e-8):
    eps = max(eps, torch.finfo(p.dtype).eps)
    p = p.clamp(eps, 1 - eps); q = q.clamp(eps, 1 - eps)
    return p * torch.log(p / q) + (1 - p) * torch.log((1 - p) / (1 - q))


def js_conflict(p_v, p_s):                     # [B,S] each → [B,S] in [0,1]
    m = 0.5 * (p_v + p_s)
    js = 0.5 * _kl_bernoulli(p_v, m) + 0.5 * _kl_bernoulli(p_s, m)
    return (js / math.log(2)).clamp(0.0, 1.0)
