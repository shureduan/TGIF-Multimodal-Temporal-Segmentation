#!/usr/bin/env python3
"""Freeze and run matched LOSO training, then evaluate all held-out subjects.

Completed runs are reusable; interrupted/failed runs are archived and restarted
from the same seed. Parallel jobs have separate processes, checkpoints and logs.
All methods/seeds for one fold use the same device. No test scoring occurs until
all scheduled training jobs finish successfully.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import uuid

METHODS = ["FINAL_MODEL", "VIDEO_ONLY", "EARLY_CONCAT", "FIXED_WINDOW_ATTENTION"]
REPO = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def source_hashes(root):
    paths = [*root.joinpath('src').rglob('*.py'), *root.joinpath('scripts').glob('*.py')]
    return {str(p.relative_to(root)): digest(p) for p in sorted(paths)}


def dataset_hashes(root):
    paths = [root / kind / f'sbj_{i}.npy' for kind in ('video', 'imu') for i in range(18)]
    labels = sorted((root / 'label').glob('wear*.json'))
    if not labels or any(not path.is_file() for path in paths):
        raise ValueError('complete WEAR video/imu/label inputs are required')
    return {str(path.relative_to(root)): digest(path) for path in [*paths, *labels]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--workers', type=int, default=1, help='concurrent primary-device jobs')
    parser.add_argument('--threads', type=int, default=1, help='CPU threads per training process')
    parser.add_argument('--mps-folds', type=int, nargs='*', default=[],
                        help='optional folds assigned to a separate single-worker MPS queue')
    parser.add_argument('--seeds', nargs='+', type=int, default=[41, 47, 53])
    parser.add_argument('--parent-epochs', type=int, default=30)
    parser.add_argument('--probe-epochs', type=int, default=15)
    parser.add_argument('--retries', type=int, default=1, help='same-protocol retries after a failure')
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or len(args.seeds) < 2:
        parser.error('provide at least two distinct seeds')
    if min(args.seeds) < 0 or max(args.seeds) >= 2**32:
        parser.error('seeds must be in 0..2**32-1')
    if min(args.parent_epochs, args.probe_epochs, args.workers, args.threads) < 1:
        parser.error('epoch budgets, workers and threads must be positive')
    if args.retries < 0:
        parser.error('retries must be nonnegative')
    if any(f not in range(1, 19) for f in args.mps_folds) or len(set(args.mps_folds)) != len(args.mps_folds):
        parser.error('MPS folds must be distinct values in 1..18')
    if args.mps_folds and args.device != 'cpu':
        parser.error('a separate MPS queue requires the primary device to be cpu')
    output, data = args.output.resolve(), args.data_root.resolve()
    fold_devices = {str(f): ('mps' if f in args.mps_folds else args.device) for f in range(1, 19)}
    plan = {
        'protocol': 'fixed_epoch_loso_v2', 'methods': METHODS, 'folds': list(range(1, 19)),
        'seeds': args.seeds, 'parent_epochs': args.parent_epochs, 'probe_epochs': args.probe_epochs,
        'device': args.device, 'fold_devices': fold_devices, 'workers': args.workers,
        'threads': args.threads, 'retries': args.retries, 'data_root': str(data),
        'checkpoint_selection': 'fixed_epoch_last',
        'evaluation_schedule': 'after all methods/folds/seeds finish training',
        'source_sha256': source_hashes(REPO), 'data_sha256': dataset_hashes(data),
    }
    plan_path, snapshot = output / 'benchmark_plan.json', output / 'source_snapshot'
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != plan:
            raise ValueError('settings, source or input data changed; use a new output directory')
        if source_hashes(snapshot) != plan['source_sha256']:
            raise ValueError('frozen source snapshot changed')
    else:
        output.mkdir(parents=True, exist_ok=False)
        for name in ('src', 'scripts', 'configs'):
            shutil.copytree(REPO / name, snapshot / name,
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        plan_path.write_text(json.dumps(plan, indent=2) + '\n')
    jobs = [(method, seed, fold) for seed in args.seeds for fold in range(1, 19) for method in METHODS]
    print(f'Frozen {len(jobs)} training jobs at {plan_path}', flush=True)
    if args.plan_only:
        return
    environment = dict(os.environ, PYTHONPATH=str(snapshot / 'src'),
                       PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS=str(args.threads),
                       OPENBLAS_NUM_THREADS=str(args.threads), VECLIB_MAXIMUM_THREADS=str(args.threads))
    lock = threading.Lock()
    job_status = {f'{m}/seed_{s}/split_{f:02d}': {'state': 'queued', 'device': fold_devices[str(f)]}
                  for m, s, f in jobs}

    def save_status(phase='training'):
        payload = {'phase': phase, 'updated_utc': datetime.now(timezone.utc).isoformat(),
                   'completed': sum(r['state'] == 'complete' for r in job_status.values()),
                   'total': len(jobs), 'jobs': job_status}
        temporary = output / 'status.json.tmp'
        temporary.write_text(json.dumps(payload, indent=2) + '\n')
        temporary.replace(output / 'status.json')

    def execute(script, *arguments, log=None):
        command = [sys.executable, '-u', str(snapshot / 'scripts' / script), *map(str, arguments)]
        if log is None:
            subprocess.run(command, env=environment, check=True)
        else:
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open('a') as handle:
                subprocess.run(command, env=environment, check=True, stdout=handle, stderr=subprocess.STDOUT)

    def validate_completion(folder, method, seed, fold):
        metadata = json.loads((folder / 'training.json').read_text())
        expected = {'protocol': plan['protocol'], 'method': method, 'fold': fold, 'seed': seed,
                    'parent_epochs': args.parent_epochs,
                    'probe_epochs': args.probe_epochs if method == 'FINAL_MODEL' else 0,
                    'checkpoint_selection': 'fixed_epoch_last'}
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError(f'completed job metadata mismatch: {folder}')
        if metadata.get('runtime', {}).get('device') != fold_devices[str(fold)]:
            raise ValueError(f'completed job device mismatch: {folder}')
        if not (folder / 'parent.pt').is_file() or (
                method == 'FINAL_MODEL' and not (folder / 'background_probe.pt').is_file()):
            raise ValueError(f'completed job has missing weights: {folder}')

    def train_job(method, seed, fold):
        key = f'{method}/seed_{seed}/split_{fold:02d}'
        folder = output / 'checkpoints' / key
        started = time.monotonic()
        with lock:
            job_status[key].update(state='running', started_utc=datetime.now(timezone.utc).isoformat())
            save_status()
        try:
            if (folder / 'training.json').exists():
                validate_completion(folder, method, seed, fold)
            else:
                for attempt in range(args.retries + 1):
                    if folder.exists():
                        archive = output / 'interrupted' / f'{method}_{seed}_{fold}_{uuid.uuid4().hex[:8]}'
                        archive.parent.mkdir(parents=True, exist_ok=True)
                        folder.rename(archive)
                    print(f'Train {key} device={fold_devices[str(fold)]} attempt={attempt + 1}', flush=True)
                    try:
                        execute('train_wear.py', '--data-root', data, '--fold', fold, '--seed', seed,
                                '--method', method, '--parent-epochs', args.parent_epochs,
                                '--probe-epochs', args.probe_epochs, '--device', fold_devices[str(fold)],
                                '--output', output / 'checkpoints',
                                log=output / 'logs' / f'{method}_seed_{seed}_split_{fold:02d}.log')
                        validate_completion(folder, method, seed, fold)
                        break
                    except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
                        if attempt == args.retries:
                            raise
                        time.sleep(2)
            with lock:
                job_status[key].update(state='complete', seconds=time.monotonic() - started)
                save_status()
            print(f'Complete {key} seconds={time.monotonic() - started:.1f}', flush=True)
        except Exception as error:
            with lock:
                job_status[key].update(state='failed', error=str(error))
                save_status()
            raise

    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as primary, ThreadPoolExecutor(max_workers=1) as mps:
        futures = [(mps if fold in args.mps_folds else primary).submit(train_job, method, seed, fold)
                   for method, seed, fold in jobs]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as error:
                failures.append(str(error))
    if failures:
        save_status('training_failed')
        raise RuntimeError('test evaluation withheld because training failed: ' + '; '.join(failures))
    save_status('evaluating')
    weights = sorted((output / 'checkpoints').rglob('*.pt'))
    (output / 'checkpoint_manifest.json').write_text(json.dumps(
        {str(p.relative_to(output)): {'sha256': digest(p), 'bytes': p.stat().st_size} for p in weights},
        indent=2) + '\n')
    summaries = []
    for method in METHODS:
        for seed in args.seeds:
            predictions = []
            for fold in range(1, 19):
                folder = output / 'checkpoints' / method / f'seed_{seed}' / f'split_{fold:02d}'
                prediction = output / 'predictions' / method / f'seed_{seed}' / f'split_{fold:02d}.npz'
                arguments = ['--data-root', data, '--subject', f'sbj_{fold - 1}',
                             '--parent', folder / 'parent.pt', '--output', prediction,
                             '--device', fold_devices[str(fold)]]
                if method == 'FINAL_MODEL':
                    arguments.extend(['--probe', folder / 'background_probe.pt'])
                execute('infer_wear.py', *arguments,
                        log=output / 'logs' / 'inference.log')
                predictions.append(prediction)
            summary = output / 'metrics' / f'{method}_seed_{seed}.json'
            execute('evaluate_wear.py', *predictions, '--output', summary,
                    log=output / 'logs' / 'evaluation.log')
            summaries.append(summary)
    execute('analyze_wear_statistics.py', *summaries, '--output', output / 'paired_statistics.json')
    save_status('complete')
    print(f'Benchmark complete: {output / "paired_statistics.json"}', flush=True)


if __name__ == '__main__':
    main()
