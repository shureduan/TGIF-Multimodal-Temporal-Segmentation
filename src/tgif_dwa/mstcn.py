"""Checkpoint-compatible MS-TCN building blocks used by the final model."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, channels, dropout=0.1):
        super().__init__()
        self.conv = nn.Conv1d(
            channels, channels, 3, padding=dilation, dilation=dilation
        )
        self.proj = nn.Conv1d(channels, channels, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, inputs):
        return inputs + self.drop(self.proj(F.relu(self.conv(inputs))))


class SingleStage(nn.Module):
    def __init__(self, n_layers, channels, in_dim, n_classes, dropout=0.1):
        super().__init__()
        self.proj = nn.Conv1d(in_dim, channels, 1)
        self.layers = nn.ModuleList(
            [
                DilatedResidualLayer(2**index, channels, dropout)
                for index in range(n_layers)
            ]
        )
        self.out = nn.Conv1d(channels, n_classes, 1)

    def forward(self, inputs):
        output = self.proj(inputs)
        for layer in self.layers:
            output = layer(output)
        return self.out(output)


class MSTCNStages(nn.Module):
    """Multi-stage TCN over ``[B,D,T]``; returns ``[S,B,C,T]``."""

    def __init__(
        self,
        in_dim=512,
        n_stages=4,
        n_layers=8,
        ch=64,
        n_classes=3,
        dropout=0.1,
    ):
        super().__init__()
        self.stage1 = SingleStage(n_layers, ch, in_dim, n_classes, dropout)
        self.stages = nn.ModuleList(
            [
                SingleStage(n_layers, ch, n_classes, n_classes, dropout)
                for _ in range(n_stages - 1)
            ]
        )

    def forward(self, features):
        output = self.stage1(features)
        outputs = [output]
        for stage in self.stages:
            output = stage(F.softmax(output, dim=1))
            outputs.append(output)
        return torch.stack(outputs, dim=0)
