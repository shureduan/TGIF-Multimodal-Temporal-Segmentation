"""Trainable projection for the two video streams used by TGIF DWA."""
from __future__ import annotations

import torch
from torch import nn


class GatedProjection(nn.Module):
    """Gate global/operator ROI features and project 1536 dimensions to 512."""

    def __init__(self, in_dim: int = 1536, out_dim: int = 512):
        super().__init__()
        if in_dim % 2:
            raise ValueError("the concatenated global/operator dimension must be even")
        half = in_dim // 2
        self.global_gate = nn.Sequential(
            nn.Linear(half, half // 4), nn.ReLU(),
            nn.Linear(half // 4, 1), nn.Sigmoid(),
        )
        self.operator_gate = nn.Sequential(
            nn.Linear(in_dim, half // 4), nn.ReLU(),
            nn.Linear(half // 4, 1), nn.Sigmoid(),
        )
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != 1536:
            raise ValueError(f"video features must end in 1536, got {tuple(x.shape)}")
        global_view, operator_view = x.chunk(2, dim=-1)
        global_gated = self.global_gate(global_view) * global_view
        operator_gated = self.operator_gate(x) * operator_view
        return self.proj(torch.cat([global_gated, operator_gated], dim=-1))
