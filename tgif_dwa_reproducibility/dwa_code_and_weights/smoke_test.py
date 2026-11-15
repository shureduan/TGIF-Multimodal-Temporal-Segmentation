#!/usr/bin/env python3
"""Load a released checkpoint and exercise inference, including masked gaps."""
from pathlib import Path
import numpy as np

from infer import run_inference

root = Path(__file__).resolve().parent
rng = np.random.default_rng(47)
length = 48
global_features = rng.normal(size=(length, 768)).astype(np.float32)
operator_features = rng.normal(size=(length, 768)).astype(np.float32)
vibration_features = rng.normal(size=(length, 6)).astype(np.float32)
vibration_features[8:11] = np.nan
result = run_inference(
    global_features,
    operator_features,
    vibration_features,
    checkpoint_path=root / 'weights/seed_41_best.pt',
    video_norm_path=root / 'weights/video_norm.npz',
)
assert result['predictions'].shape == (length,)
assert result['probabilities'].shape == (length, 3)
assert not result['sensor_valid'][8:11].any()
assert np.isfinite(result['probabilities']).all()
print('Smoke test passed: checkpoint loaded; NaN vibration rows were masked without interpolation.')
