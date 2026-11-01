"""Validated array contracts shared by the three paper subsections."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import time
import numpy as np

CLASSES = ('idle', 'sewing', 'handling')
FEATURE_RATE = 7.4933
ENCODER_ID = 'MCG-NJU/videomae-base'
ENCODER_REVISION = 'dc740ceda42fce44faed2ea03c6d447db72f6af9'
OFFSETS = np.array([-30,-26,-22,-18,-14,-10,-6,-2,2,6,10,14,18,22,26,30])

def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def validate_timestamps(timestamps):
    t = np.asarray(timestamps, dtype=np.float64)
    if t.ndim != 1 or not len(t) or not np.isfinite(t).all():
        raise ValueError('timestamps_sec must be a nonempty finite 1-D array')
    if len(t) > 1 and not np.all(np.diff(t) > 0):
        raise ValueError('timestamps_sec must be strictly increasing')
    return t

def load_video_features(directory):
    directory = Path(directory)
    t = validate_timestamps(np.load(directory / 'timestamps_sec.npy', allow_pickle=False))
    streams = []
    for name in ('global_features.npy', 'operator_features.npy'):
        x = np.load(directory / name, allow_pickle=False)
        if x.shape != (len(t), 768) or not np.isfinite(x).all():
            raise ValueError(f'{name} must be finite [T,768], aligned to timestamps')
        streams.append(x.astype(np.float32))
    meta_path = directory / 'metadata.json'
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return np.concatenate(streams, axis=1), t, meta

def device_for(name):
    import torch
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    if name == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    if name == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS requested but unavailable')
    return torch.device(name)

def synchronize(device):
    import torch
    if device.type == 'cuda': torch.cuda.synchronize(device)
    elif device.type == 'mps': torch.mps.synchronize()
