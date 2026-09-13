"""Checkpoint-compatible WEAR final model.

The parent checkpoint contains the two-round DWA/MS-TCN model and the
normalization statistics.  The small task-presence probe is a separate frozen
checkpoint and adjusts only the final Round-1 logits.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .boundary_dwa import BoundaryUncertaintyDWA
from .iterative_dwa import WindowConfig


class TaskPresenceProbe(nn.Module):
    """Binary background/action probe used by the frozen WEAR final model."""

    def __init__(self, in_dim: int = 2648) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class WearFinalModel(nn.Module):
    """Frozen final WEAR composition used in the 18-fold evaluation.

    Input tensors are feature-grid sequences: video ``[T, 2048]`` and
    training-fold-normalized inertial windows ``[T, 600]``.  Class 18 is the
    explicit background class.
    """

    def __init__(
        self,
        *,
        beta: float = 0.5,
        uncertainty_seconds: float = 1.0,
        margin_prior: float = 0.5,
    ) -> None:
        super().__init__()
        self.parent = BoundaryUncertaintyDWA(
            video_dim=2048,
            imu_dim=600,
            attn_dim=128,
            n_classes=19,
            n_rounds=2,
            dropout=0.0,
            n_stages=4,
            n_layers=8,
            ch=64,
            window=WindowConfig(
                feature_stride_seconds=0.5,
                initial_radius_seconds=2.0,
                expansion_step_seconds=0.5,
                max_escape_seconds=0.0,
                saturation_tau=0.01,
                saturation_patience=2,
            ),
            uncertainty_seconds=uncertainty_seconds,
            margin_prior=margin_prior,
        )
        self.probe = TaskPresenceProbe(2648)
        self.beta = float(beta)

    def forward(
        self,
        video: torch.Tensor,
        inertial: torch.Tensor,
        aux_valid: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        if video.ndim != 2 or video.shape[1] != 2048:
            raise ValueError(f"video must have shape [T,2048], got {tuple(video.shape)}")
        if inertial.shape != (video.shape[0], 600):
            raise ValueError(
                f"inertial must have shape [T,600], got {tuple(inertial.shape)}"
            )

        parent_out = self.parent(video, inertial, aux_valid=aux_valid)
        logits = parent_out["round_logits"][1][-1, 0].T.clone()  # [T,19]
        p_background = torch.sigmoid(self.probe(torch.cat([video, inertial], dim=-1)))
        logits[:, :18] += self.beta * torch.log(1.0 - p_background[:, None] + 1e-6)
        logits[:, 18] += self.beta * torch.log(p_background + 1e-6)
        probabilities = torch.softmax(logits, dim=-1)
        return {
            "logits": logits,
            "probabilities": probabilities,
            "predictions": probabilities.argmax(dim=-1),
            "p_background": p_background,
            "parent": parent_out,
        }


def normalize_wear_inertial(
    inertial: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    """Apply the checkpoint's train-fold statistics to RAW600 input."""

    x = np.nan_to_num(np.asarray(inertial, dtype=np.float32))
    if x.ndim != 2 or x.shape[1] != 600:
        raise ValueError(f"inertial must have shape [T,600], got {x.shape}")
    mean = np.asarray(mean, dtype=np.float32).reshape(-1)
    std = np.asarray(std, dtype=np.float32).reshape(-1)
    if mean.shape != (12,) or std.shape != (12,):
        raise ValueError("normalization mean/std must each contain 12 channels")
    return ((x.reshape(-1, 12, 50) - mean[None, :, None]) /
            std.clip(min=1e-8)[None, :, None]).reshape(-1, 600).astype(np.float32)


def load_wear_final(
    parent_checkpoint: str | Path,
    probe_checkpoint: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> Tuple[WearFinalModel, np.ndarray, np.ndarray]:
    """Strictly load a paired parent/probe checkpoint and fold statistics.

    PyTorch checkpoint files are pickle-based.  Only load files from a trusted
    release and verify their SHA-256 values first.
    """

    device = torch.device(device)
    parent_payload = torch.load(parent_checkpoint, map_location=device, weights_only=False)
    probe_payload = torch.load(probe_checkpoint, map_location=device, weights_only=False)
    if not isinstance(parent_payload, dict) or "model" not in parent_payload:
        raise ValueError("parent checkpoint must be a mapping containing 'model'")
    if not isinstance(probe_payload, dict) or "model" not in probe_payload:
        raise ValueError("probe checkpoint must be a mapping containing 'model'")
    if "normalization_mean" not in parent_payload or "normalization_std" not in parent_payload:
        raise ValueError("parent checkpoint is missing train-fold normalization statistics")

    model = WearFinalModel().to(device)
    model.parent.load_state_dict(parent_payload["model"], strict=True)
    model.probe.load_state_dict(probe_payload["model"], strict=True)
    model.eval()
    mean = np.asarray(parent_payload["normalization_mean"], dtype=np.float32).reshape(-1)
    std = np.asarray(parent_payload["normalization_std"], dtype=np.float32).reshape(-1)
    if mean.shape != (12,) or std.shape != (12,):
        raise ValueError("unexpected normalization statistics in parent checkpoint")
    return model, mean, std
