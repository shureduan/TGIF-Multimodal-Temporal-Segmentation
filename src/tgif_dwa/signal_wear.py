"""Versioned experimental WEAR adapter for sensor-driven window selection."""
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .signal_dwa import SignalAdaptiveDWA, SignalWindowConfig, SignalAblationConfig
from .wear_data import loso_subjects
from .wear_model import TaskPresenceProbe, WearFinalModel, checkpoint_normalization

PROTOCOL = 'signal_adaptive_loso_v3_candidate'
METHOD = 'SIGNAL_ADAPTIVE_DWA'
VARIANTS = {
    METHOD: SignalAblationConfig(),
    'NO_SENSOR_RESIZE': SignalAblationConfig(resize=False, contract=False, expand=False),
    'NO_CONTRACTION': SignalAblationConfig(contract=False),
    'NO_EXPANSION': SignalAblationConfig(expand=False),
    'UNIFORM_POOL': SignalAblationConfig(learned_attention=False),
}


def build_signal_parent(config=None, method=METHOD):
    if method not in VARIANTS:
        raise ValueError(f'unknown sensor-driven variant: {method}')
    return SignalAdaptiveDWA(signal_config=config or SignalWindowConfig(), ablation_config=VARIANTS[method], video_dim=2048,
                             imu_dim=600, attn_dim=128, n_classes=19, n_rounds=2,
                             dropout=0.0, n_stages=4, n_layers=8, ch=64)


class SignalWearModel(WearFinalModel):
    def __init__(self, signal_config=None, beta=0.5, method=METHOD):
        nn.Module.__init__(self)
        self.parent = build_signal_parent(signal_config, method)
        self.probe = TaskPresenceProbe(2648)
        self.beta = beta
        self.run_metadata = {}


def validate_signal_metadata(metadata):
    if not isinstance(metadata, dict) or metadata.get('protocol') != PROTOCOL or metadata.get('method') not in VARIANTS:
        raise ValueError('expected a sensor-adaptive v3 candidate checkpoint, not v2 weights')
    expected_ablation = asdict(VARIANTS[metadata['method']])
    if metadata.get('ablation_config', asdict(VARIANTS[METHOD]) if metadata['method'] == METHOD else None) != expected_ablation:
        raise ValueError('checkpoint ablation configuration does not match its method')
    train, test = loso_subjects(metadata['fold'])
    if metadata['train_subjects'] != train or metadata['test_subject'] != test:
        raise ValueError('checkpoint split mismatch')
    if metadata['checkpoint_selection'] != 'fixed_epoch_last' or not metadata['run_id']:
        raise ValueError('expected a fixed-epoch run identity')
    for name in ('parent_epochs', 'probe_epochs'):
        if type(metadata[name]) is not int or metadata[name] < 1:
            raise ValueError('invalid component epoch budget')
    for name in ('seed', 'probe_seed'):
        if type(metadata[name]) is not int or not 0 <= metadata[name] < 2**32:
            raise ValueError('invalid component seed')
    if metadata['probe_seed'] != (metadata['seed'] + metadata['fold'] * 100 + 2) % 2**32:
        raise ValueError('probe seed mismatch')
    if metadata['beta'] != 0.5:
        raise ValueError('unsupported probe coefficient')
    if metadata['window_config'] != asdict(SignalWindowConfig()):
        raise ValueError('use a separately named experiment for changed window settings')


def load_signal_wear(parent, probe, device='cpu'):
    parent = torch.load(parent, map_location='cpu', weights_only=False)
    probe = torch.load(probe, map_location='cpu', weights_only=False)
    metadata = parent.get('run_metadata')
    validate_signal_metadata(metadata)
    if metadata != probe.get('run_metadata'):
        raise ValueError('parent/probe metadata differ; use the same v3 run')
    for payload, kind, key in ((parent, 'parent', 'seed'), (probe, 'probe', 'probe_seed')):
        if payload.get('epoch') != metadata[kind + '_epochs'] - 1 or payload.get('seed') != metadata[key]:
            raise ValueError('checkpoint epoch or seed mismatch')
    model = SignalWearModel(SignalWindowConfig(**metadata['window_config']), metadata['beta'], metadata['method'])
    model.parent.load_state_dict(parent['model'], strict=True)
    model.probe.load_state_dict(probe['model'], strict=True)
    if not bool(model.parent.signal_calibrated):
        raise ValueError('checkpoint lacks training-fold signal calibration')
    for buffer in (model.parent.signal_center, model.parent.signal_scale, model.parent.signal_thresholds):
        if not torch.isfinite(buffer).all():
            raise ValueError('non-finite signal calibration')
    if (model.parent.signal_scale <= 0).any() or (model.parent.signal_thresholds <= 0).any():
        raise ValueError('invalid signal scales or thresholds')
    model.run_metadata = metadata
    mean, std = checkpoint_normalization(parent)
    return model.to(device).eval(), mean, std
