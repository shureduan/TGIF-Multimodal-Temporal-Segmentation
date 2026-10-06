"""Fixed-epoch training for the separately versioned sensor-driven candidate."""
from dataclasses import asdict
import json
from pathlib import Path
import platform
import time
import uuid

import numpy as np
import torch

from .signal_dwa import SignalWindowConfig
from .signal_wear import PROTOCOL, METHOD, build_signal_parent
from .wear_data import load_sequence, loso_subjects, fit_inertial_normalizer
from .wear_model import normalize_wear_inertial
from .wear_training import set_seed, stage_loss, training_tensors, train_probe


def run_signal_training(*, data_root, fold, output, seed=47, parent_epochs=30, probe_epochs=15, device='cpu'):
    if parent_epochs < 1 or probe_epochs < 1 or not 0 <= seed < 2**32:
        raise ValueError('invalid seed or epoch budget')
    train_ids, test_id = loso_subjects(fold)
    folder = Path(output) / METHOD / f'seed_{seed}' / f'split_{fold:02d}'
    folder.mkdir(parents=True, exist_ok=False)
    metadata = {'protocol': PROTOCOL, 'method': METHOD, 'run_id': uuid.uuid4().hex,
                'fold': fold, 'seed': seed, 'probe_seed': (seed + fold * 100 + 2) % 2**32,
                'train_subjects': train_ids, 'test_subject': test_id,
                'parent_epochs': parent_epochs, 'probe_epochs': probe_epochs,
                'checkpoint_selection': 'fixed_epoch_last', 'beta': .5,
                'window_config': asdict(SignalWindowConfig())}
    (folder / 'protocol.json').write_text(json.dumps(metadata, indent=2) + '\n')
    # The only subject reads in training. Calibration receives no labels.
    train = [load_sequence(data_root, subject) for subject in train_ids]
    mean, std = fit_inertial_normalizer(train)
    set_seed(seed)
    parent = build_signal_parent()
    calibration = parent.fit_signal_statistics([normalize_wear_inertial(s['inertial'], mean, std) for s in train])
    (folder / 'signal_calibration.json').write_text(json.dumps(calibration, indent=2) + '\n')
    parent.to(device)
    optimizer = torch.optim.Adam(parent.parameters(), lr=5e-4, weight_decay=1e-4)
    history = []
    for epoch in range(parent_epochs):
        started = time.monotonic(); parent.train(); losses = []
        for index in np.random.permutation(len(train)):
            video, sensor, labels = training_tensors(train[index], mean, std, device)
            rounds = parent(video, sensor, return_diagnostics=False)['round_logits']
            loss = .5 * stage_loss(rounds[0], labels) + stage_loss(rounds[1], labels)
            if not torch.isfinite(loss):
                raise RuntimeError('non-finite adaptive training loss')
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(parent.parameters(), 5)
            optimizer.step(); losses.append(float(loss.detach().cpu()))
        row = {'epoch': epoch, 'train_loss': float(np.mean(losses)), 'seconds': time.monotonic() - started}
        history.append(row)
        with (folder / 'parent_log.jsonl').open('a') as handle:
            handle.write(json.dumps(row) + '\n')
        print(f"{METHOD} epoch={epoch:02d} train_loss={row['train_loss']:.6f} seconds={row['seconds']:.1f}", flush=True)
    torch.save({'model': {k: v.detach().cpu().clone() for k, v in parent.state_dict().items()},
                'normalization_mean': mean, 'normalization_std': std, 'epoch': parent_epochs - 1,
                'seed': seed, 'run_metadata': metadata}, folder / 'parent.pt')
    probe_state, probe_history = train_probe(train, mean, std, probe_epochs, device,
                                             metadata['probe_seed'], folder / 'probe_log.jsonl')
    torch.save({'model': probe_state, 'epoch': probe_epochs - 1, 'seed': metadata['probe_seed'],
                'run_metadata': metadata}, folder / 'background_probe.pt')
    (folder / 'training.json').write_text(json.dumps({**metadata, 'parent_history': history,
        'probe_history': probe_history, 'runtime': {'python': platform.python_version(),
        'torch': torch.__version__, 'numpy': np.__version__, 'device': str(device)}}, indent=2) + '\n')
    return folder
