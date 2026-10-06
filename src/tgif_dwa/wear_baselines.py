"""Matched feature-grid baselines for the corrected fixed-epoch protocol."""

from pathlib import Path

import torch
from torch import nn

from .iterative_dwa import IterativeRawDWA, WindowConfig
from .mstcn import MSTCNStages
from .wear_model import checkpoint_normalization, load_wear_final, validate_run_metadata

BASELINE_METHODS = ("VIDEO_ONLY", "EARLY_CONCAT", "FIXED_WINDOW_ATTENTION")
METHODS = (*BASELINE_METHODS, "FINAL_MODEL")


class WearBaseline(nn.Module):
    """Same 4-stage/8-layer/64-channel MS-TCN recipe as the historical baselines."""

    def __init__(self, method: str):
        super().__init__()
        if method not in BASELINE_METHODS:
            raise ValueError(f"unknown baseline: {method}")
        self.method = method
        self.run_metadata = {}
        if method == "FIXED_WINDOW_ATTENTION":
            self.core = IterativeRawDWA(
                n_rounds=1, window=WindowConfig(max_escape_seconds=0.0)
            )
        else:
            self.mstcn = MSTCNStages(
                in_dim=2048 if method == "VIDEO_ONLY" else 2648,
                n_stages=4, n_layers=8, ch=64, n_classes=19,
            )

    def forward(self, video, inertial):
        if self.method == "FIXED_WINDOW_ATTENTION":
            stages = self.core(video, inertial, return_diagnostics=False)["logits"]
        else:
            features = video if self.method == "VIDEO_ONLY" else torch.cat([video, inertial], -1)
            stages = self.mstcn(features.T.unsqueeze(0))
        logits = stages[-1, 0].T
        probabilities = logits.softmax(-1)
        return {"round_logits": [stages], "logits": logits,
                "probabilities": probabilities, "predictions": probabilities.argmax(-1)}


def load_wear_predictor(parent: str | Path, probe=None, *, device="cpu"):
    """Load v2 baselines or a v2/legacy final parent/probe pair."""
    payload = torch.load(parent, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("parent checkpoint must be a mapping")
    metadata = payload.get("run_metadata")
    if metadata is not None:
        validate_run_metadata(metadata)
    method = (metadata or {}).get("method", "FINAL_MODEL")
    if method == "FINAL_MODEL":
        if probe is None:
            raise ValueError("FINAL_MODEL requires its matching --probe checkpoint")
        return load_wear_final(parent, probe, device=device)
    if probe is not None:
        raise ValueError("baseline inference does not use a background probe")
    if payload.get("epoch") != metadata["parent_epochs"] - 1 or payload.get("seed") != metadata["seed"]:
        raise ValueError("baseline checkpoint does not match its final epoch/seed")
    model = WearBaseline(method).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.run_metadata = metadata
    model.eval()
    mean, std = checkpoint_normalization(payload)
    return model, mean, std
