"""Video-only TGIF baseline paired with the released DWA training recipe."""
from __future__ import annotations

import torch
from torch import nn

from .mstcn import MSTCNStages
from .gated_projection import GatedProjection


class VideoMSTCN(nn.Module):
    """Gated two-ROI projection (1536->512) followed by a four-stage MS-TCN."""

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.proj = GatedProjection()
        self.mstcn = MSTCNStages(
            in_dim=512, n_stages=4, n_layers=8, ch=64, n_classes=3, dropout=dropout,
        )

    def forward(self, video_1536: torch.Tensor) -> torch.Tensor:
        if video_1536.ndim != 3 or video_1536.shape[1] != 1536:
            raise ValueError(f"video must be [B,1536,T], got {tuple(video_1536.shape)}")
        projected = self.proj(video_1536.permute(0, 2, 1)).permute(0, 2, 1)
        return self.mstcn(projected)
