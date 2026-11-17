"""TGIF dimension adapter for the maintained sensor-driven DWA.

TGIF supplies two 768-D VideoMAE ROI streams and six precomputed vibration
features per aligned video frame. This adapter keeps the gated video projection,
sensor-driven window controller, one-way video-Q/sensor-KV attention, shared
two-round MS-TCN, and idle/activity probe while configuring those TGIF dimensions.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn

from .iterative_dwa import WindowConfig
from .gated_projection import GatedProjection
from .signal_dwa import SignalAdaptiveDWA, SignalWindowConfig
from .wear_model import TaskPresenceProbe


TGIF_VIDEO_DIM = 1536
TGIF_SENSOR_DIM = 6
TGIF_PROJECTED_VIDEO_DIM = 512
TGIF_NUM_CLASSES = 3
TGIF_IDLE_CLASS = 0
TGIF_DEFAULT_FPS = 7.4933


def build_tgif_signal_config(feature_stride_seconds: float = 1.0 / TGIF_DEFAULT_FPS):
    """Use TGIF's existing six per-frame FFT features as DWA sensor descriptors."""
    if not math.isfinite(feature_stride_seconds) or feature_stride_seconds <= 0:
        raise ValueError('feature stride must be positive and finite')
    return SignalWindowConfig(
        channels=TGIF_SENSOR_DIM,
        samples_per_token=1,
        input_mode='precomputed_features',
        stride_seconds=float(feature_stride_seconds),
    )


class TGIFSignalDWA(nn.Module):
    """Two-round TGIF DWA with only dataset interface dimensions adapted.

    Inputs are `[T,1536]` gated global/operator features and normalized `[T,6]`
    TGIF vibration features. Window selection and attention are inherited from
    the current WEAR signal-driven implementation. Class 0 is TGIF idle, so the
    WEAR task-presence probe is interpreted as an idle/activity probe here.
    """

    def __init__(
        self,
        *,
        feature_stride_seconds: float = 1.0 / TGIF_DEFAULT_FPS,
        beta: float = 0.5,
    ) -> None:
        super().__init__()
        if not math.isfinite(beta) or beta < 0:
            raise ValueError('beta must be finite and nonnegative')
        config = build_tgif_signal_config(feature_stride_seconds)
        self.video_projection = GatedProjection()
        self.parent = SignalAdaptiveDWA(
            signal_config=config,
            video_dim=TGIF_PROJECTED_VIDEO_DIM,
            imu_dim=TGIF_SENSOR_DIM,
            attn_dim=128,
            n_classes=TGIF_NUM_CLASSES,
            n_rounds=2,
            dropout=0.0,
            n_stages=4,
            n_layers=8,
            ch=64,
            window=WindowConfig(
                feature_stride_seconds=feature_stride_seconds,
                initial_radius_seconds=round(2.0 / feature_stride_seconds) * feature_stride_seconds,
                expansion_step_seconds=feature_stride_seconds,
                max_escape_seconds=0.0,
            ),
        )
        self.probe = TaskPresenceProbe(TGIF_PROJECTED_VIDEO_DIM + TGIF_SENSOR_DIM)
        self.beta = float(beta)

    def forward(
        self,
        video: torch.Tensor,
        sensor: torch.Tensor,
        aux_valid: Optional[torch.Tensor] = None,
        *,
        return_diagnostics: bool = True,
    ):
        if video.ndim != 2 or video.shape[1] != TGIF_VIDEO_DIM:
            raise ValueError(f'video must be [T,{TGIF_VIDEO_DIM}], got {tuple(video.shape)}')
        if sensor.shape != (video.shape[0], TGIF_SENSOR_DIM):
            raise ValueError(f'sensor must be [T,{TGIF_SENSOR_DIM}], got {tuple(sensor.shape)}')
        projected = self.video_projection(video)
        parent_out = self.parent(
            projected, sensor, aux_valid=aux_valid, return_diagnostics=return_diagnostics
        )
        logits = parent_out['round_logits'][-1][-1, 0].T.clone()
        probe_sensor = sensor
        if aux_valid is not None:
            aux_valid = aux_valid.to(device=video.device, dtype=torch.bool)
            probe_sensor = sensor.masked_fill(~aux_valid[:, None], 0.0)
        p_idle = torch.sigmoid(self.probe(torch.cat([projected, probe_sensor], dim=-1)))
        if aux_valid is not None:
            p_idle = torch.where(aux_valid, p_idle, 0.5)
        logits[:, TGIF_IDLE_CLASS] += self.beta * torch.log(p_idle + 1e-6)
        active_indices = [i for i in range(TGIF_NUM_CLASSES) if i != TGIF_IDLE_CLASS]
        logits[:, active_indices] += self.beta * torch.log(1.0 - p_idle[:, None] + 1e-6)
        probabilities = torch.softmax(logits, dim=-1)
        return {
            'logits': logits,
            'probabilities': probabilities,
            'predictions': probabilities.argmax(dim=-1),
            'p_idle': p_idle,
            'projected_video': projected,
            'parent': parent_out,
        }
