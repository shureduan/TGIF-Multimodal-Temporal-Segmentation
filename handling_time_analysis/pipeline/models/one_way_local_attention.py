"""One-way local attention (README §10): Video is the ONLY Query; vibration
tokens are Key/Value only.  Each video timestep may attend only to windows that
cover it plus ±local_neighbor_windows adjacent windows (time mask −inf).

forward returns (C_hat [B,T,Dv], A [B,T,S]) where A is the head-averaged
attention used for the timestep gate g_t = Σ_j A_{t,j} g_j (README §18).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_local_mask(T, windows, neighbors=1):
    """[T,S] bool — True = allowed. Window j covers t if a<=t<b; ±neighbors by index."""
    S = len(windows)
    cover = np.zeros((T, S), bool)
    for j, (a, b) in enumerate(windows):
        cover[a:b, j] = True
    allow = cover.copy()
    for d in range(1, neighbors + 1):
        allow[:, :-d] |= cover[:, d:]          # neighbor j+d covers t → allow j
        allow[:, d:] |= cover[:, :-d]
    # rows with no coverage (shouldn't happen) → allow nearest window
    empty = ~allow.any(1)
    if empty.any():
        centers = np.array([(a + b) / 2 for a, b in windows])
        for t in np.where(empty)[0]:
            allow[t, int(np.abs(centers - t).argmin())] = True
    return allow


class OneWayLocalAttention(nn.Module):
    """README §4: Q/K are L2-normalized (cosine attention) and scaled by a fixed
    temperature τ — decouples the score scale from the vibration/video magnitudes
    so fusion doesn't fail from mismatched scales."""
    def __init__(self, video_dim=256, token_dim=64, attention_dim=64,
                 num_heads=2, dropout=0.10, temperature=0.20):
        super().__init__()
        assert attention_dim % num_heads == 0
        self.h, self.d = num_heads, attention_dim // num_heads
        self.tau = temperature
        self.q = nn.Linear(video_dim, attention_dim)
        self.k = nn.Linear(token_dim, attention_dim)
        self.v = nn.Linear(token_dim, attention_dim)
        self.out = nn.Linear(attention_dim, video_dim)     # W_O: 64 → 256
        self.drop = nn.Dropout(dropout)

    def forward(self, H_v, E_s, allow_mask):   # [B,T,256],[B,S,64],[T,S] bool
        B, T, _ = H_v.shape
        S = E_s.shape[1]
        q = self.q(H_v).view(B, T, self.h, self.d).transpose(1, 2)     # [B,h,T,d]
        k = self.k(E_s).view(B, S, self.h, self.d).transpose(1, 2)     # [B,h,S,d]
        v = self.v(E_s).view(B, S, self.h, self.d).transpose(1, 2)
        q = F.normalize(q, p=2, dim=-1); k = F.normalize(k, p=2, dim=-1)  # cosine
        sc = (q @ k.transpose(-1, -2)) / self.tau                      # [B,h,T,S]
        m = torch.as_tensor(allow_mask, device=H_v.device).view(1, 1, T, S)
        sc = sc.masked_fill(~m, float("-inf"))
        A = torch.softmax(sc, dim=-1)
        A = torch.nan_to_num(A, nan=0.0)       # rows fully masked → 0
        ctx = (self.drop(A) @ v).transpose(1, 2).reshape(B, T, self.h * self.d)  # [B,T,64]
        qn = q.norm(dim=-1).mean().detach()    # (all ~1 after normalize)
        return self.out(ctx), A.mean(1)        # C_hat [B,T,256], A [B,T,S]


def attention_entropy_loss(A, allow_mask, min_entropy=0.25, eps=1e-8):
    """README §10: penalize normalized attention entropy below a floor so a
    multi-candidate row can't collapse to ~[1,0,0,...]. A:[B,T,S], mask:[T,S]."""
    dev = A.device
    m = torch.as_tensor(allow_mask, device=dev, dtype=A.dtype)         # [T,S]
    ent = -(A * torch.log(A.clamp_min(eps))).sum(-1)                   # [B,T]
    cc = m.sum(-1).clamp_min(2.0)                                      # [T] candidate count
    norm_ent = ent / torch.log(cc).unsqueeze(0)                       # [B,T] ∈[0,1]
    return F.relu(min_entropy - norm_ent).mean(), norm_ent.mean().detach()
