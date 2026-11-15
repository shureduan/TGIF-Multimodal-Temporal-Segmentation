#!/usr/bin/env python3
"""Run a packaged TGIF DWA checkpoint on pre-extracted, aligned features."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from tgif_dwa.tgif_adaptation import TGIFSignalDWA, TGIF_DEFAULT_FPS

CLASS_NAMES = ('idle', 'sewing', 'handling')


def choose_device(name: str) -> torch.device:
    if name != 'auto':
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def run_inference(
    global_features: np.ndarray,
    operator_features: np.ndarray,
    vibration_features: np.ndarray,
    *,
    checkpoint_path: str | Path,
    video_norm_path: str | Path,
    valid_mask: Optional[np.ndarray] = None,
    device: str | torch.device = 'cpu',
):
    """Predict one aligned sequence; NaN vibration rows are skipped, never filled."""
    global_features = np.asarray(global_features, dtype=np.float32)
    operator_features = np.asarray(operator_features, dtype=np.float32)
    vibration_features = np.asarray(vibration_features, dtype=np.float32)
    if global_features.ndim != 2 or global_features.shape[1] != 768:
        raise ValueError(f'global features must be [T,768], got {global_features.shape}')
    if operator_features.shape != global_features.shape:
        raise ValueError(f'operator features must match global shape, got {operator_features.shape}')
    if vibration_features.shape != (len(global_features), 6):
        raise ValueError(f'vibration features must be [T,6], got {vibration_features.shape}')
    if not np.isfinite(global_features).all() or not np.isfinite(operator_features).all():
        raise ValueError('video features contain NaN/Inf')

    sensor_valid = np.isfinite(vibration_features).all(axis=1)
    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if valid_mask.shape != (len(global_features),):
            raise ValueError(f'valid mask must be [T], got {valid_mask.shape}')
        sensor_valid &= valid_mask
    if not sensor_valid.any():
        print('Warning: no valid vibration rows; the sensor context will be zero for all frames.')

    device = torch.device(device)
    payload = torch.load(Path(checkpoint_path), map_location='cpu', weights_only=True)
    if payload.get('method') != 'dwa':
        raise ValueError('checkpoint is not a TGIF DWA checkpoint')
    split_seed = int(payload['split_seed'])

    with np.load(video_norm_path, allow_pickle=False) as norm:
        mean = np.asarray(norm['mean'], dtype=np.float32).reshape(-1)
        std = np.asarray(norm['std'], dtype=np.float32).reshape(-1)
    if mean.shape != (1536,) or std.shape != (1536,) or not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError('video normalization must contain finite 1536-D mean/std')
    if (std <= 0).any():
        raise ValueError('video normalization std must be positive')

    video = np.concatenate([global_features, operator_features], axis=1)
    video = ((video - mean[None, :]) / std[None, :]).astype(np.float32)
    sensor_mean = np.asarray(payload['sensor_mean'], dtype=np.float32).reshape(1, 6)
    sensor_std = np.asarray(payload['sensor_std'], dtype=np.float32).reshape(1, 6)
    if not np.isfinite(sensor_mean).all() or not np.isfinite(sensor_std).all() or (sensor_std <= 0).any():
        raise ValueError('checkpoint has invalid sensor normalization')
    sensor = np.zeros((len(video), 6), dtype=np.float32)
    sensor[sensor_valid] = (vibration_features[sensor_valid] - sensor_mean) / sensor_std

    model = TGIFSignalDWA(beta=0.0, feature_stride_seconds=1.0 / TGIF_DEFAULT_FPS)
    model.load_state_dict(payload['model'], strict=True)
    model.to(device).eval()
    with torch.inference_mode():
        result = model(
            torch.from_numpy(np.ascontiguousarray(video)).to(device),
            torch.from_numpy(np.ascontiguousarray(sensor)).to(device),
            aux_valid=torch.from_numpy(sensor_valid).to(device),
            return_diagnostics=False,
        )
    probabilities = result['probabilities'].detach().cpu().numpy()
    predictions = probabilities.argmax(axis=1).astype(np.int64)
    return {
        'predictions': predictions,
        'probabilities': probabilities,
        'sensor_valid': sensor_valid,
        'split_seed': split_seed,
        'best_epoch': int(payload['best_epoch']),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--global-features', required=True, type=Path, help='[T,768] VideoMAE full-frame features')
    parser.add_argument('--operator-features', required=True, type=Path, help='[T,768] VideoMAE operator-ROI features')
    parser.add_argument('--vibration-features', required=True, type=Path, help='[T,6] aligned vibration features; NaN rows mean missing')
    parser.add_argument('--valid-mask', type=Path, help='optional [T] bool .npy mask; false rows are skipped')
    parser.add_argument('--seed', type=int, default=41, choices=(41, 11, 43, 17, 53), help='validation-split checkpoint to use')
    parser.add_argument('--output', type=Path, required=True, help='output prediction CSV')
    parser.add_argument('--device', default='cpu', help='cpu, cuda, mps, or auto; default: cpu')
    args = parser.parse_args()

    code_dir = Path(__file__).resolve().parent
    checkpoint = code_dir / 'weights' / f'seed_{args.seed}_best.pt'
    norm = code_dir / 'weights' / 'video_norm.npz'
    mask = np.load(args.valid_mask, allow_pickle=False) if args.valid_mask else None
    out = run_inference(
        np.load(args.global_features, allow_pickle=False),
        np.load(args.operator_features, allow_pickle=False),
        np.load(args.vibration_features, allow_pickle=False),
        checkpoint_path=checkpoint,
        video_norm_path=norm,
        valid_mask=mask,
        device=choose_device(args.device),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame', 'time_seconds', 'sensor_valid', 'prediction_id', 'prediction',
                         'p_idle', 'p_sewing', 'p_handling'])
        for i, (label, prob, valid) in enumerate(zip(out['predictions'], out['probabilities'], out['sensor_valid'])):
            writer.writerow([i, f'{i / TGIF_DEFAULT_FPS:.6f}', int(valid), int(label), CLASS_NAMES[label],
                             *[f'{float(p):.8f}' for p in prob]])
    print(f"seed={out['split_seed']} best_epoch={out['best_epoch']} frames={len(out['predictions'])} "
          f"valid_vibration={int(out['sensor_valid'].sum())}/{len(out['sensor_valid'])}")
    print(f'Predictions: {args.output}')


if __name__ == '__main__':
    main()
