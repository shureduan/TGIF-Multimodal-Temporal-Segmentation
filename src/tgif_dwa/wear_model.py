"""Checkpoint-compatible WEAR final model.

The parent checkpoint contains the two-round DWA/MS-TCN model and the
normalization statistics.  The small task-presence probe is a separate frozen
checkpoint and adjusts only the final Round-1 logits.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .boundary_dwa import BoundaryUncertaintyDWA
from .iterative_dwa import WindowConfig
from .wear_data import loso_subjects


TRAINING_PROTOCOL = "fixed_epoch_loso_v2"
DEFAULT_MODEL_CONFIG = {"beta": 0.5, "uncertainty_seconds": 1.0, "margin_prior": 0.5}


def validate_run_metadata(metadata: Dict[str, Any]) -> None:
    """Validate the split and fixed-budget contract before using v2 weights."""
    if not isinstance(metadata, dict) or metadata.get("protocol") != TRAINING_PROTOCOL:
        raise ValueError("unsupported checkpoint training protocol")
    required = {"run_id", "method", "fold", "seed", "probe_seed", "test_subject",
                "train_subjects", "model_config", "checkpoint_selection",
                "parent_epochs", "probe_epochs"}
    if not required.issubset(metadata):
        raise ValueError("checkpoint is missing run metadata")
    train, test = loso_subjects(metadata["fold"])
    if metadata["train_subjects"] != train or metadata["test_subject"] != test:
        raise ValueError("checkpoint train/test split does not match its LOSO fold")
    if metadata["checkpoint_selection"] != "fixed_epoch_last":
        raise ValueError("v2 checkpoints must use fixed-epoch selection")
    for name in ("seed", "probe_seed"):
        if type(metadata[name]) is not int or not 0 <= metadata[name] < 2**32:
            raise ValueError(f"invalid {name} in checkpoint")
    for name in ("parent_epochs", "probe_epochs"):
        minimum = 0 if name == "probe_epochs" and metadata["method"] != "FINAL_MODEL" else 1
        if type(metadata[name]) is not int or metadata[name] < minimum:
            raise ValueError(f"invalid {name} in checkpoint")
    if not isinstance(metadata["run_id"], str) or not metadata["run_id"]:
        raise ValueError("checkpoint run_id must be a nonempty string")
    if metadata["model_config"] != DEFAULT_MODEL_CONFIG:
        raise ValueError("unsupported v2 model configuration; use a separately versioned protocol")


def checkpoint_normalization(payload):
    mean = np.asarray(payload["normalization_mean"], dtype=np.float32).reshape(-1)
    std = np.asarray(payload["normalization_std"], dtype=np.float32).reshape(-1)
    if mean.shape != (12,) or std.shape != (12,):
        raise ValueError("unexpected normalization statistics in parent checkpoint")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError("normalization statistics must be finite with positive std")
    return mean, std


def build_wear_parent(
    *, uncertainty_seconds: float = 1.0, margin_prior: float = 0.5
) -> BoundaryUncertaintyDWA:
    """The shared parent factory used by both training and inference."""
    return BoundaryUncertaintyDWA(
        video_dim=2048, imu_dim=600, attn_dim=128, n_classes=19,
        n_rounds=2, dropout=0.0, n_stages=4, n_layers=8, ch=64,
        window=WindowConfig(feature_stride_seconds=0.5,
                            initial_radius_seconds=2.0, max_escape_seconds=0.0),
        uncertainty_seconds=uncertainty_seconds, margin_prior=margin_prior,
    )


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
    """WEAR parent/probe composition, compatible with historical checkpoints.

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
        self.parent = build_wear_parent(
            uncertainty_seconds=uncertainty_seconds,
            margin_prior=margin_prior,
        )
        self.probe = TaskPresenceProbe(2648)
        self.beta = float(beta)
        if not math.isfinite(self.beta) or self.beta < 0:
            raise ValueError("beta must be finite and nonnegative")
        self.run_metadata: Dict[str, Any] = {}

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
        probe_inertial = inertial
        if aux_valid is not None:
            aux_valid = aux_valid.to(device=video.device, dtype=torch.bool)
            probe_inertial = inertial.masked_fill(~aux_valid[:, None], 0.0)
        p_background = torch.sigmoid(self.probe(torch.cat([video, probe_inertial], dim=-1)))
        if aux_valid is not None:
            # The probe was trained on complete video+IMU pairs. At a missing
            # token use a neutral prior, leaving the parent's probabilities.
            p_background = torch.where(aux_valid, p_background, 0.5)
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

    parent_run = parent_payload.get("run_metadata")
    probe_run = probe_payload.get("run_metadata")
    if parent_run != probe_run:
        raise ValueError("parent/probe run metadata differ; use a pair from the same run")
    if parent_run is not None:
        validate_run_metadata(parent_run)
        if parent_run["method"] != "FINAL_MODEL":
            raise ValueError("expected FINAL_MODEL checkpoint pair")
        for payload, component, seed_key in ((parent_payload, "parent", "seed"),
                                              (probe_payload, "probe", "probe_seed")):
            if (payload.get("epoch") != parent_run[f"{component}_epochs"] - 1 or
                    payload.get("seed") != parent_run[seed_key]):
                raise ValueError(f"{component} checkpoint does not match its final epoch/seed")
        model_config = parent_run["model_config"]
    else:
        model_config = DEFAULT_MODEL_CONFIG
    model = WearFinalModel(**model_config).to(device)
    model.parent.load_state_dict(parent_payload["model"], strict=True)
    model.probe.load_state_dict(probe_payload["model"], strict=True)
    model.eval()
    model.run_metadata = dict(parent_run or {"protocol": "legacy_unversioned"})
    mean, std = checkpoint_normalization(parent_payload)
    return model, mean, std
