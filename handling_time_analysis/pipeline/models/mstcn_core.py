"""Unchanged shared MS-TCN blocks from the original TGIF implementation."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, ch, dropout=0.1):
        super().__init__()
        self.conv = nn.Conv1d(ch, ch, 3, padding=dilation, dilation=dilation)
        self.proj = nn.Conv1d(ch, ch, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.drop(self.proj(F.relu(self.conv(x))))

class SingleStage(nn.Module):
    def __init__(self, n_layers, ch, in_dim, n_classes, dropout=0.1):
        super().__init__()
        self.proj = nn.Conv1d(in_dim, ch, 1)
        self.layers = nn.ModuleList([DilatedResidualLayer(2**i, ch, dropout)
                                     for i in range(n_layers)])
        self.out = nn.Conv1d(ch, n_classes, 1)

    def forward(self, x):
        x = self.proj(x)
        for l in self.layers:
            x = l(x)
        return self.out(x)

class MSTCNStages(nn.Module):
    """4-stage MS-TCN over a [B, in_dim, T] feature; returns [S, B, C, T]."""
    def __init__(self, in_dim=512, n_stages=4, n_layers=8, ch=64, n_classes=3,
                 dropout=0.1):
        super().__init__()
        self.stage1 = SingleStage(n_layers, ch, in_dim, n_classes, dropout)
        self.stages = nn.ModuleList([
            SingleStage(n_layers, ch, n_classes, n_classes, dropout)
            for _ in range(n_stages - 1)])

    def forward(self, feat):                             # feat: [B, in_dim, T]
        out = self.stage1(feat); outs = [out]
        for s in self.stages:
            out = s(F.softmax(out, dim=1)); outs.append(out)
        return torch.stack(outs, dim=0)                  # [S, B, C, T]
