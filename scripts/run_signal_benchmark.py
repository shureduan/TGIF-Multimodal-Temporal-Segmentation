#!/usr/bin/env python3
"""Frozen 18-fold x 3-seed sensor-DWA benchmark and retrained component study."""
import argparse
import fcntl
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import uuid

from run_wear_benchmark import digest, source_hashes, dataset_hashes
from tgif_dwa.signal_wear import PROTOCOL, METHOD, VARIANTS

REPO = Path(__file__).resolve().parents[1]
BASELINES = ['VIDEO_ONLY', 'EARLY_CONCAT', 'FIXED_WINDOW_ATTENTION', 'FINAL_MODEL']
SEEDS = [41, 47, 53]


def dump(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def baseline_identity(root, data):
    plan = json.loads((root / 'benchmark_plan.json').read_text())
    status = json.loads((root / 'status.json').read_text())
    assert plan['protocol'] == 'fixed_epoch_loso_v2'
    assert plan['seeds'] == SEEDS and plan['folds'] == list(range(1, 19))
    assert plan['parent_epochs'] == 30 and plan['probe_epochs'] == 15
    assert plan['checkpoint_selection'] == 'fixed_epoch_last'
    assert status['phase'] == 'complete' and status['completed'] == 216
    assert plan['data_sha256'] == data, 'baseline input data differ'
    for name, expected in plan['source_sha256'].items():
        assert digest(root / 'source_snapshot' / name) == expected
    manifest = json.loads((root / 'checkpoint_manifest.json').read_text())
    for name, expected in manifest.items():
        assert digest(root / name) == expected['sha256'], name
    artifacts = [root / 'benchmark_plan.json', root / 'checkpoint_manifest.json']
    for method in BASELINES:
        for seed in SEEDS:
            metrics = root / 'metrics' / f'{method}_seed_{seed}.json'
            saved = json.loads(metrics.read_text())
            assert len(saved['per_subject']) == 18
            assert all(r['method'] == method and r['seed'] == seed and r['protocol'] == plan['protocol'] for r in saved['per_subject'])
            artifacts.append(metrics)
            for fold in range(1, 19):
                folder = root / 'checkpoints' / method / f'seed_{seed}/split_{fold:02d}'
                trained = json.loads((folder / 'training.json').read_text())
                assert trained['fold'] == fold and trained['seed'] == seed and trained['method'] == method
                assert trained['parent_epochs'] == 30 and trained['checkpoint_selection'] == 'fixed_epoch_last'
                assert trained['runtime']['device'] == plan['fold_devices'][str(fold)]
                artifacts.extend([folder / 'training.json', root / 'predictions' / method / f'seed_{seed}/split_{fold:02d}.npz'])
    return plan, {str(p.relative_to(root)): digest(p) for p in artifacts}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cpu-workers', type=int, default=3)
    p.add_argument('--mps-workers', type=int, default=2)
    p.add_argument('--plan-only', action='store_true')
    args = p.parse_args()
    if min(args.cpu_workers, args.mps_workers) < 1:
        p.error('worker counts must be positive')
    output = args.output.resolve(); baseline = args.baseline.resolve(); data_root = args.data_root.resolve()
    data = dataset_hashes(data_root)
    old_plan, old_artifacts = baseline_identity(baseline, data)
    plan = {
        'protocol': PROTOCOL, 'study': 'sensor_dwa_full_18fold_3seed_v1',
        'methods': list(VARIANTS), 'ablations': {m: asdict(c) for m, c in VARIANTS.items()},
        'baseline_methods': BASELINES, 'baseline_root': str(baseline),
        'baseline_artifact_sha256': old_artifacts, 'data_root': str(data_root), 'data_sha256': data,
        'folds': list(range(1, 19)), 'seeds': SEEDS, 'parent_epochs': 30, 'probe_epochs': 15,
        'checkpoint_selection': 'fixed_epoch_last', 'fold_devices': old_plan['fold_devices'],
        'cpu_workers': args.cpu_workers, 'mps_workers': args.mps_workers, 'threads': 1, 'retries': 1,
        'evaluation_schedule': 'Only after every training job completes. No test-score retries or tuning.',
        'pilot': 'sbj_0/seed47 was observed before this study; unchanged full configuration; all 54 full runs retrained; include a sensitivity analysis excluding sbj_0.',
        'inference_diagnostics': ['PARENT_NO_PROBE', 'ROUND0'],
        'statistical_unit': '18 subjects; average three matched seed differences within subject before testing',
        'primary_metrics': ['macro_f1_19', 'map_at_0.5'],
        'test_families': {'baseline': BASELINES, 'retrained_components': [m for m in VARIANTS if m != METHOD],
                          'frozen_components': ['PARENT_NO_PROBE', 'ROUND0']},
        'multiplicity': 'Two-sided paired t tests with Holm correction separately within each predeclared family (comparators x 2 metrics). Exploratory LOSO inference.',
        'figures': {'timeline': {'fold': 2, 'seed': 47}, 'gt_final': {'fold': 10, 'seed': 47},
                    'selection': 'Same subjects/seeds as existing figures, independent of new scores'},
        'robustness': 'Full-model weights shared between sensor-resizing and seed-only interventions; shift detached Round-0 internal boundaries +/-0.5 and +/-1.0 seconds; report context L2.',
        'publication': 'Local outputs only. Inspect all results before any GitHub publication; do not suppress negative comparisons.',
        'numerical_improvement_rule': 'Mean-subject Macro-F1 and mAP@0.5 must both exceed the published fixed-margin FINAL_MODEL; report significance separately, never auto-push.',
        'source_sha256': source_hashes(REPO),
    }
    snapshot = output / 'source_snapshot'; plan_path = output / 'benchmark_plan.json'
    if plan_path.exists():
        assert json.loads(plan_path.read_text()) == plan, 'source/settings/data changed; use a new study directory'
        assert source_hashes(snapshot) == plan['source_sha256'], 'frozen source changed'
    else:
        output.mkdir(parents=True, exist_ok=False)
        for name in ('src', 'scripts', 'configs'):
            shutil.copytree(REPO / name, snapshot / name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        dump(plan_path, plan)
    print(f'Frozen {len(VARIANTS) * 54} new training jobs; 216 verified baseline runs reused.', flush=True)
    if args.plan_only:
        return
    run_lock = (output / '.runner.lock').open('a+')
    try:
        fcntl.flock(run_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        run_lock.close()
        raise RuntimeError('this study already has a running coordinator; inspect status.json instead of launching a second copy') from error
    dump(output / 'coordinator.json', {'pid': os.getpid(), 'started_utc': datetime.now(timezone.utc).isoformat()})
    environment = dict(os.environ, PYTHONPATH=str(snapshot / 'src'), PYTHONDONTWRITEBYTECODE='1',
                       OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
    lock = threading.Lock()
    # Entire full model first, followed by separately retrained controls; the
    # evaluation barrier still waits for every method, fold and seed.
    jobs = [(m, s, f) for m in VARIANTS for s in SEEDS for f in plan['folds']]
    status = {f'{m}/seed_{s}/split_{f:02d}': {'state': 'queued', 'device': plan['fold_devices'][str(f)]} for m,s,f in jobs}

    def save_status(phase='training', error=None):
        completed = sum(r['state'] == 'complete' for r in status.values())
        payload = {'phase': phase, 'updated_utc': datetime.now(timezone.utc).isoformat(), 'completed': completed,
                   'total': len(jobs), 'jobs': status, 'error': error}
        dump(output / 'status.json', payload)
        lines = ['传感器驱动 DWA 全量实验', f'阶段: {phase}', f'训练完成: {completed}/{len(jobs)}',
                 '18 folds × seeds 41/47/53 × 5 个重新训练版本；复用已核验的 216 个 baseline 训练结果。',
                 '固定 parent 30 / probe 15 epochs；全部训练完成后统一评估并生成表格、比较图及统计结果。',
                 '图表和最终结论尚未完成时，不代表已有全量提升结论。所有内容保留本地，TGIF 与公开 README 未改动。']
        active = [(k, r) for k,r in status.items() if r['state'] == 'running']
        lines += ['运行中: ' + k + ' (' + r['device'] + ')' for k,r in active]
        if error:
            lines += ['错误: ' + error]
        (output / '进度.txt').write_text('\n'.join(lines) + '\n')

    def execute(script, arguments, log):
        command = [sys.executable, '-u', str(snapshot / 'scripts' / script), *map(str, arguments)]
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('a') as handle:
            subprocess.run(command, cwd=snapshot, env=environment, stdout=handle, stderr=subprocess.STDOUT, check=True)

    def validate(folder, method, seed, fold):
        d = json.loads((folder / 'training.json').read_text())
        expected = {'protocol': PROTOCOL, 'method': method, 'seed': seed, 'fold': fold,
                    'parent_epochs': 30, 'probe_epochs': 15, 'checkpoint_selection': 'fixed_epoch_last',
                    'ablation_config': asdict(VARIANTS[method])}
        assert all(d.get(k) == v for k,v in expected.items()), f'completed-run identity mismatch: {folder}'
        assert d['runtime']['device'] == plan['fold_devices'][str(fold)]
        assert [r['epoch'] for r in d['parent_history']] == list(range(30))
        assert [r['epoch'] for r in d['probe_history']] == list(range(15))
        for name in ('parent.pt', 'background_probe.pt'):
            assert (folder / name).is_file()
        receipt = folder / 'completion_sha256.json'
        actual = {name: digest(folder / name) for name in ('parent.pt', 'background_probe.pt', 'training.json')}
        if receipt.exists():
            assert json.loads(receipt.read_text()) == actual, f'completed checkpoint changed: {folder}'
        else:
            dump(receipt, actual)

    def train(job):
        method, seed, fold = job; key = f'{method}/seed_{seed}/split_{fold:02d}'
        folder = output / 'checkpoints' / key; started = time.monotonic()
        with lock:
            status[key].update(state='running', started_utc=datetime.now(timezone.utc).isoformat()); save_status()
        try:
            if (folder / 'training.json').exists():
                validate(folder, method, seed, fold)
            else:
                for attempt in range(plan['retries'] + 1):
                    if folder.exists():
                        archive = output / 'interrupted' / f'{method}_{seed}_{fold}_{uuid.uuid4().hex[:8]}'
                        archive.parent.mkdir(parents=True, exist_ok=True); folder.rename(archive)
                    try:
                        print(f'Train {key} device={plan["fold_devices"][str(fold)]} attempt={attempt+1}', flush=True)
                        execute('train_wear_signal.py', ['--data-root', data_root, '--output', output / 'checkpoints',
                            '--fold', fold, '--seed', seed, '--method', method, '--device', plan['fold_devices'][str(fold)],
                            '--parent-epochs', 30, '--probe-epochs', 15], output / 'logs' / f'{method}_{seed}_{fold:02d}.log')
                        validate(folder, method, seed, fold); break
                    except Exception:
                        if attempt == plan['retries']:
                            raise
            with lock:
                status[key].update(state='complete', seconds=time.monotonic()-started); save_status()
            print(f'Complete {key} ({time.monotonic()-started:.1f}s)', flush=True)
        except Exception as error:
            with lock:
                status[key].update(state='failed', error=str(error)); save_status()
            raise

    def infer(job):
        method, seed, fold = job
        folder = output / 'checkpoints' / method / f'seed_{seed}/split_{fold:02d}'
        destination = output / 'predictions' / method / f'seed_{seed}/split_{fold:02d}.npz'
        receipt = destination.with_suffix('.receipt.json')
        identity = {'method': method, 'seed': seed, 'fold': fold, 'device': plan['fold_devices'][str(fold)],
                    'checkpoint_sha256': {name: digest(folder / name) for name in ('parent.pt', 'background_probe.pt')}}
        if receipt.exists():
            saved = json.loads(receipt.read_text())
            assert all(saved.get(k) == v for k,v in identity.items()), 'cached prediction checkpoint identity mismatch'
            assert destination.is_file() and digest(destination) == saved['prediction_sha256'], 'cached prediction changed'
            return
        execute('infer_wear_signal.py', ['--data-root', data_root, '--subject', f'sbj_{fold-1}',
            '--parent', folder / 'parent.pt', '--probe', folder / 'background_probe.pt',
            '--device', plan['fold_devices'][str(fold)], '--output', destination],
            output / 'logs' / f'infer_{method}_{seed}_{fold:02d}.log')
        dump(receipt, {**identity, 'prediction_sha256': digest(destination)})

    def parallel(action):
        failures = []
        with ThreadPoolExecutor(max_workers=args.cpu_workers) as cpu, ThreadPoolExecutor(max_workers=args.mps_workers) as mps:
            futures = [(mps if plan['fold_devices'][str(j[2])] == 'mps' else cpu).submit(action,j) for j in jobs]
            for f in as_completed(futures):
                try:
                    f.result()
                except Exception as e:
                    failures.append(repr(e))
        if failures:
            raise RuntimeError('; '.join(failures))

    try:
        save_status(); parallel(train)
        dump(output / 'checkpoint_manifest.json', {str(p.relative_to(output)): {'sha256': digest(p), 'bytes': p.stat().st_size}
             for p in sorted((output / 'checkpoints').rglob('*.pt'))})
        save_status('evaluating'); parallel(infer)
        save_status('analyzing_and_plotting')
        execute('analyze_signal_benchmark.py', ['--benchmark', output], output / 'logs/analysis.log')
        execute('plot_signal_benchmark.py', ['--benchmark', output], output / 'logs/plotting.log')
        review = json.loads((output / 'review_conclusions.json').read_text())
        assert review['total_metric_records'] == 594 and review['all_270_training_runs_complete']
        assert review['figures_pending'] is False and review['pdf_pages'] == 9
        figures = json.loads((output / 'figure_outputs.json').read_text())
        assert figures['pages'] == 9 and len(figures['figure_names']) == 9
        for name, expected in figures['sha256'].items():
            assert digest(output / name) == expected, f'figure output missing or changed: {name}'
        save_status('complete')
        print(f'Complete: {output}', flush=True)
    except Exception as error:
        save_status('failed', repr(error)); raise
    finally:
        run_lock.close()


if __name__ == '__main__':
    main()
