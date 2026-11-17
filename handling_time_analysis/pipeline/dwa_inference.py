"""Inference for the six-feature TGIF DWA checkpoints bundled in this repo."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .models.tgif_dwa.tgif_adaptation import TGIFSignalDWA, TGIF_DEFAULT_FPS


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DWA_DIR = REPO_ROOT / "tgif_dwa_reproducibility" / "dwa_code_and_weights"


def choose_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def infer_arrays(
    global_features: np.ndarray,
    operator_features: np.ndarray,
    vibration_features: np.ndarray,
    *,
    checkpoint_path: str | Path,
    video_norm_path: str | Path,
    valid_mask: Optional[np.ndarray] = None,
    device: str | torch.device = "cpu",
):
    """Predict an aligned sequence; missing vibration rows are masked, not filled."""
    global_features = np.asarray(global_features, dtype=np.float32)
    operator_features = np.asarray(operator_features, dtype=np.float32)
    vibration_features = np.asarray(vibration_features, dtype=np.float32)
    if global_features.ndim != 2 or global_features.shape[1] != 768:
        raise ValueError(f"global features must be [T,768], got {global_features.shape}")
    if operator_features.shape != global_features.shape:
        raise ValueError("operator features must have the same [T,768] shape as global features")
    if vibration_features.shape != (len(global_features), 6):
        raise ValueError(f"DWA vibration features must be [T,6], got {vibration_features.shape}")
    if not np.isfinite(global_features).all() or not np.isfinite(operator_features).all():
        raise ValueError("video features contain NaN/Inf")

    sensor_valid = np.isfinite(vibration_features).all(axis=1)
    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if valid_mask.shape != (len(global_features),):
            raise ValueError(f"valid mask must be [T], got {valid_mask.shape}")
        sensor_valid &= valid_mask

    with np.load(video_norm_path, allow_pickle=False) as norm:
        mean = np.asarray(norm["mean"], dtype=np.float32).reshape(-1)
        std = np.asarray(norm["std"], dtype=np.float32).reshape(-1)
    if mean.shape != (1536,) or std.shape != (1536,):
        raise ValueError("video normalization must contain 1536-D mean and std")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError("video normalization must be finite with positive standard deviations")

    payload = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=True)
    if payload.get("method") != "dwa":
        raise ValueError("checkpoint is not a TGIF DWA checkpoint")
    sensor_mean = np.asarray(payload["sensor_mean"], dtype=np.float32).reshape(1, 6)
    sensor_std = np.asarray(payload["sensor_std"], dtype=np.float32).reshape(1, 6)
    if not np.isfinite(sensor_mean).all() or not np.isfinite(sensor_std).all() or (sensor_std <= 0).any():
        raise ValueError("checkpoint sensor normalization must be finite with positive standard deviations")

    video = np.concatenate([global_features, operator_features], axis=1)
    video = ((video - mean[None]) / std[None]).astype(np.float32)
    sensor = np.zeros((len(video), 6), dtype=np.float32)
    sensor[sensor_valid] = (vibration_features[sensor_valid] - sensor_mean) / sensor_std

    model = TGIFSignalDWA(beta=0.0, feature_stride_seconds=1.0 / TGIF_DEFAULT_FPS)
    model.load_state_dict(payload["model"], strict=True)
    device = torch.device(device)
    model.to(device).eval()
    with torch.inference_mode():
        result = model(
            torch.from_numpy(np.ascontiguousarray(video)).to(device),
            torch.from_numpy(np.ascontiguousarray(sensor)).to(device),
            aux_valid=torch.from_numpy(sensor_valid).to(device),
            return_diagnostics=False,
        )
    probabilities = result["probabilities"].detach().cpu().numpy()
    return {
        "predictions": probabilities.argmax(axis=1).astype(np.int64),
        "probabilities": probabilities,
        "sensor_valid": sensor_valid,
        "split_seed": int(payload["split_seed"]),
        "best_epoch": int(payload["best_epoch"]),
    }
