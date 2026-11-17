#!/usr/bin/env python3
"""Train the TGIF video baseline and signal-driven DWA from cached features.

The raw TGIF data, pre-extracted features, and WEAR temporal initialization are
not distributed here. Supply their paths explicitly; see README.md for the
record layout and the checkpoint-selection protocol.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, random, shutil, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

RECORDINGS = (
    'station1', 'station2', 'station2_clip2', 'station3', 'station3_clip2',
    'station4', 'station1_0_30', 'station1_90_120', 'station2_40_70',
    'station3_40_70', 'station4_40_70',
)
SEEDS = (41, 11, 43, 17, 53)
SEG_LEN = 2247
VIB_DIM = 6
NUM_CLASSES = 3
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline.models.tgif_dwa.tgif_adaptation import TGIFSignalDWA
from pipeline.models.tgif_dwa.segmentation_metrics import core_metrics, frames_to_segments
from pipeline.models.tgif_dwa.video_baseline import VideoMSTCN

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def set_model_seed():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

def load_records(feature_root, vibration_root, video_norm_path):
    feature_root = Path(feature_root).expanduser().resolve()
    vibration_root = Path(vibration_root).expanduser().resolve()
    video_norm_path = Path(video_norm_path).expanduser().resolve()
    nz = np.load(video_norm_path)
    mean = np.asarray(nz['mean'], dtype=np.float32).reshape(-1)
    std = np.maximum(np.asarray(nz['std'], dtype=np.float32).reshape(-1), 1e-6)
    if mean.shape != (1536,) or std.shape != (1536,) or not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError('VideoMAE norm must contain finite 1536-D mean/std')
    records, sources = [], {}
    for name in RECORDINGS:
        folder = feature_root / name
        paths = {
            'global': folder / 'global_features.npy',
            'operator': folder / 'operator_features.npy',
            'labels': folder / 'labels.npy',
            'timestamps': folder / 'timestamps_sec.npy',
        }
        g = np.load(paths['global'], mmap_mode='r')
        o = np.load(paths['operator'], mmap_mode='r')
        y = np.load(paths['labels'], mmap_mode='r')
        ts = np.load(paths['timestamps'], mmap_mode='r')
        if g.ndim != 2 or g.shape[1] != 768 or o.shape != g.shape or y.shape != (len(g),) or ts.shape != (len(g),):
            raise ValueError(f'{name}: unexpected VideoMAE feature/label/timestamp shapes')
        if not np.isin(np.unique(y), [-100, 0, 1, 2]).all():
            raise ValueError(f'{name}: labels must be -100/0/1/2')
        vib_path = vibration_root / f'{name}_vib2040.npy'
        has_vib = vib_path.is_file()
        vib = np.load(vib_path, mmap_mode='r') if has_vib else None
        if has_vib and (vib.ndim != 2 or vib.shape[1] != VIB_DIM or len(vib) != len(g)):
            raise ValueError(f'{name}: vibration features are not aligned [T,6]')
        if has_vib and not np.isfinite(vib).all():
            raise ValueError(f'{name}: processed vibration contains non-finite values')
        n = min(len(g), len(o), len(y), len(ts), len(vib) if has_vib else len(g))
        nseg = n // SEG_LEN
        if nseg != 6:
            raise ValueError(f'{name}: expected six complete 5-minute segments; found {nseg}')
        for i in range(nseg):
            a, b = i * SEG_LEN, (i + 1) * SEG_LEN
            video = np.concatenate([np.asarray(g[a:b], dtype=np.float32),
                                    np.asarray(o[a:b], dtype=np.float32)], axis=1)
            video = ((video - mean[None]) / std[None]).astype(np.float32)
            sensor = (np.asarray(vib[a:b], dtype=np.float32).copy() if has_vib
                      else np.zeros((SEG_LEN, VIB_DIM), dtype=np.float32))
            labels = np.asarray(y[a:b], dtype=np.int64).copy()
            if not np.isfinite(video).all():
                raise ValueError(f'{name}_seg{i:02d}: non-finite video features')
            records.append({
                'id': f'{name}_seg{i:02d}', 'recording': name,
                'video': np.ascontiguousarray(video), 'sensor': np.ascontiguousarray(sensor),
                'aux_valid': np.full(SEG_LEN, has_vib, dtype=np.bool_),
                'has_vib': has_vib, 'label': labels,
            })
        sources[name] = {
            'feature_dir': str(folder), 'global_shape': list(g.shape),
            'operator_shape': list(o.shape), 'labels_shape': list(y.shape),
            'timestamps_shape': list(ts.shape),
            'vibration_path': str(vib_path) if has_vib else None,
            'vibration_shape': list(vib.shape) if has_vib else None,
            'has_vibration': has_vib, 'segments': nseg,
            'inputs': {k: {'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha256(p)}
                       for k, p in paths.items()},
        }
        if has_vib:
            sources[name]['vibration_input'] = {
                'bytes': vib_path.stat().st_size, 'sha256': sha256(vib_path)}
    return records, mean, std, sources

def idle_frac(record):
    y = record['label']
    valid = y >= 0
    return float(np.mean(y[valid] == 0)) if valid.any() else 0.0

def split_records(records, split_seed):
    # Exact six-bucket idle-fraction split used in tgif_transfer_experiment.py.
    ranked = sorted(range(len(records)), key=lambda i: idle_frac(records[i]))
    bucket_size = len(ranked) / 6
    rng = random.Random(split_seed)
    val_idx = []
    for bucket in range(6):
        lo, hi = int(bucket * bucket_size), int((bucket + 1) * bucket_size)
        pool = [i for i in ranked[lo:hi] if i not in val_idx]
        val_idx.append(rng.choice(pool))
    val_set = set(val_idx)
    train = [r for i, r in enumerate(records) if i not in val_set]
    val = [r for i, r in enumerate(records) if i in val_set]
    assert len(train) == 60 and len(val) == 6
    return train, val

def vib_stats(train):
    arrays = [r['sensor'][r['aux_valid']] for r in train if r['has_vib']]
    if not arrays:
        raise ValueError('training split has no aligned vibration segments')
    rows = np.concatenate(arrays, axis=0)
    return rows.mean(0).astype(np.float32), (rows.std(0) + 1e-6).astype(np.float32)

def normalize_sensor(values, mean, std):
    return ((np.asarray(values, dtype=np.float32) - mean[None]) / std[None]).astype(np.float32)

def stages_loss(stages, labels):
    # Historical loss: sum over MS-TCN stages of unweighted CE + 0.15 raw-logit TMSE.
    loss = stages.new_tensor(0.0)
    for stage in stages:
        logits = stage[0].T
        ce = F.cross_entropy(logits, labels, ignore_index=-100)
        if logits.shape[0] > 1:
            tmse = torch.clamp((logits[1:] - logits[:-1]).square(), max=9.0).mean()
        else:
            tmse = logits.new_tensor(0.0)
        loss = loss + ce + 0.15 * tmse
    return loss

def tensors(record, sensor_mean, sensor_std, device):
    video = torch.from_numpy(record['video']).to(device)
    sensor = torch.from_numpy(normalize_sensor(record['sensor'], sensor_mean, sensor_std)).to(device)
    labels = torch.from_numpy(record['label']).to(device=device, dtype=torch.long)
    aux_valid = torch.from_numpy(record['aux_valid']).to(device=device, dtype=torch.bool)
    return video, sensor, labels, aux_valid

def build_model(method, train_records, sensor_mean, sensor_std, wear_checkpoint):
    if method == 'video':
        model = VideoMSTCN(dropout=0.1, roi_weights=None)
        transfer = {'source': None, 'copied_tensors': 0, 'copied_parameters': 0}
        return model, transfer, None
    if method != 'dwa':
        raise ValueError(method)
    model = TGIFSignalDWA(beta=0.0)  # legacy TGIF transfer had no idle/background probe
    # Match legacy dropout=0.1; adapter defaults were 0.0.
    for module in model.parent.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.1
    calibration_rows = [
        normalize_sensor(r['sensor'][r['aux_valid']], sensor_mean, sensor_std)
        for r in train_records if r['has_vib']
    ]
    calibration = model.parent.fit_signal_statistics(calibration_rows)
    if wear_checkpoint is None:
        raise ValueError('DWA training requires --wear-checkpoint for the temporal-layer initialization')
    wear_checkpoint = Path(wear_checkpoint).expanduser().resolve()
    # The trusted WEAR checkpoint stores NumPy normalization arrays alongside
    # tensors. Allow only NumPy's basic reconstruction types; do not enable
    # unrestricted pickle loading for a caller-supplied checkpoint.
    numpy_safe_types = [
        np._core.multiarray._reconstruct, np.ndarray, np.dtype,
        np.float32, np.float64, np.dtypes.Float32DType, np.dtypes.Float64DType,
    ]
    with torch.serialization.safe_globals(numpy_safe_types):
        payload = torch.load(wear_checkpoint, map_location='cpu', weights_only=True)
    src = payload['model']
    dst = model.state_dict()
    copied = {
        'parent.mstcn.' + key[len('mstcn.'):]: value
        for key, value in src.items()
        if key.startswith('mstcn.') and '.layers.' in key
        and 'parent.mstcn.' + key[len('mstcn.'):] in dst
        and dst['parent.mstcn.' + key[len('mstcn.'):]].shape == value.shape
    }
    if len(copied) != 128 or sum(v.numel() for v in copied.values()) != 528384:
        raise RuntimeError(f'WEAR temporal transfer mismatch: {len(copied)} tensors')
    model.load_state_dict(copied, strict=False)
    return model, {
        'source': str(wear_checkpoint), 'copied_tensors': len(copied),
        'copied_parameters': sum(v.numel() for v in copied.values()),
        'source_sha256': sha256(wear_checkpoint),
    }, calibration

def forward(model, method, record, sensor_mean, sensor_std, device):
    video, sensor, labels, aux_valid = tensors(record, sensor_mean, sensor_std, device)
    if method == 'video':
        stages = model(video.T.unsqueeze(0))
        return stages, labels
    result = model(video, sensor, aux_valid=aux_valid, return_diagnostics=False)
    # Supervise the final DWA round, as in the historical runner which exposed only final logits.
    return result['parent']['round_logits'][-1], labels

@torch.no_grad()
def evaluate(model, method, records, sensor_mean, sensor_std, device):
    model.eval()
    rows = []
    for record in records:
        video, sensor, labels, aux_valid = tensors(record, sensor_mean, sensor_std, device)
        if method == 'video':
            stages = model(video.T.unsqueeze(0))
            logits = stages[-1, 0].T
        else:
            result = model(video, sensor, aux_valid=aux_valid, return_diagnostics=False)
            logits = result['logits']
        pred = logits.argmax(-1).cpu().numpy()
        y = labels.cpu().numpy()
        valid = y >= 0
        raw_metrics = core_metrics(pred[valid], y[valid])
        pred_segments = frames_to_segments(pred[valid])
        target_segments = frames_to_segments(y[valid])
        metrics = {
            'acc': raw_metrics['accuracy'],
            'macro_f1': raw_metrics['macro_f1'],
            'edit': raw_metrics['edit'],
            'f1_10': raw_metrics['f1_10'],
            'f1_25': raw_metrics['f1_25'],
            'f1_50': raw_metrics['f1_50'],
            'seg_count_ratio': len(pred_segments) / max(len(target_segments), 1),
        }
        rows.append({'segment_id': record['id'], 'has_vibration': bool(record['has_vib']), **metrics})
    keys = ('acc', 'macro_f1', 'edit', 'f1_10', 'f1_25', 'f1_50', 'seg_count_ratio')
    summary = {k: float(np.mean([row[k] for row in rows])) for k in keys}
    vib_rows = [row for row in rows if row['has_vibration']]
    summary['n_validation_segments'] = len(rows)
    summary['n_vibration_validation_segments'] = len(vib_rows)
    summary['vibration_subset'] = (
        {k: float(np.mean([row[k] for row in vib_rows])) for k in keys}
        if vib_rows else None
    )
    summary['per_segment'] = rows
    return summary

def train_one(method, split_seed, records, epochs, output, device, threads, wear_checkpoint):
    folder = output / f'seed_{split_seed}' / method
    result_path = folder / 'result.json'
    checkpoint_path = folder / 'best.pt'
    if result_path.exists() and checkpoint_path.exists():
        return json.loads(result_path.read_text())
    folder.mkdir(parents=True, exist_ok=True)
    train_records, val_records = split_records(records, split_seed)
    sensor_mean, sensor_std = vib_stats(train_records)
    set_model_seed()
    model, transfer, calibration = build_model(method, train_records, sensor_mean, sensor_std,wear_checkpoint)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    best_f1, best_epoch, best_state = -1.0, -1, None
    history_path = folder / 'training_log.csv'
    with history_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'epoch', 'train_loss', 'val_acc', 'val_macro_f1', 'val_f1_50',
            'val_vib_f1_50', 'learning_rate', 'seconds'])
        writer.writeheader(); f.flush()
        for epoch in range(1, epochs + 1):
            started = time.monotonic()
            model.train()
            losses = []
            for idx in np.random.permutation(len(train_records)):
                stages, labels = forward(model, method, train_records[int(idx)],
                                         sensor_mean, sensor_std, device)
                loss = stages_loss(stages, labels)
                if not torch.isfinite(loss):
                    raise RuntimeError(f'non-finite {method} loss seed={split_seed} epoch={epoch}')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            scheduler.step()
            metrics = evaluate(model, method, val_records, sensor_mean, sensor_std, device)
            score = metrics['f1_50']
            if score > best_f1:
                best_f1, best_epoch = score, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                torch.save({
                    'model': best_state, 'method': method, 'split_seed': split_seed,
                    'model_seed': 0, 'best_epoch': best_epoch,
                    'best_validation_f1_50': best_f1,
                    'train_segments': [r['id'] for r in train_records],
                    'validation_segments': [r['id'] for r in val_records],
                    'sensor_mean': sensor_mean.tolist(), 'sensor_std': sensor_std.tolist(),
                    'transfer': transfer, 'signal_calibration': calibration,
                }, checkpoint_path)
            row = {
                'epoch': epoch, 'train_loss': float(np.mean(losses)),
                'val_acc': metrics['acc'], 'val_macro_f1': metrics['macro_f1'],
                'val_f1_50': score,
                'val_vib_f1_50': (metrics['vibration_subset']['f1_50']
                                  if metrics['vibration_subset'] else ''),
                'learning_rate': optimizer.param_groups[0]['lr'],
                'seconds': time.monotonic() - started,
            }
            writer.writerow(row); f.flush()
            if epoch == 1 or epoch % 10 == 0:
                print(f'{method} split={split_seed} epoch={epoch:03d}/{epochs} '
                      f'loss={row["train_loss"]:.4f} valAcc={metrics["acc"]:.2f} '
                      f'valF1@50={score:.2f} best={best_f1:.2f}@{best_epoch} '
                      f'sec={row["seconds"]:.1f}', flush=True)
    if best_state is None:
        raise RuntimeError('training did not produce a checkpoint')
    model.load_state_dict(best_state)
    final_metrics = evaluate(model, method, val_records, sensor_mean, sensor_std, device)
    result = {
        'method': method, 'split_seed': split_seed, 'model_seed': 0,
        'epochs': epochs, 'best_epoch': best_epoch,
        'selection': 'highest mean per-segment validation F1@0.5',
        'train_segments': [r['id'] for r in train_records],
        'validation_segments': [r['id'] for r in val_records],
        'n_train': len(train_records), 'n_validation': len(val_records),
        'train_vibration_segments': sum(r['has_vib'] for r in train_records),
        'validation_vibration_segments': sum(r['has_vib'] for r in val_records),
        'transfer': transfer, 'signal_calibration': calibration,
        'metrics': final_metrics,
    }
    result_path.write_text(json.dumps(result, indent=2) + '\n')
    with (folder / 'validation_segments.csv').open('w', newline='') as f:
        rows = final_metrics.pop('per_segment')
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    # Restore nested per-segment rows in JSON result while keeping CSV as a convenience.
    final_metrics['per_segment'] = rows
    result['metrics'] = final_metrics
    result_path.write_text(json.dumps(result, indent=2) + '\n')
    return result

def write_aggregates(output, runs):
    rows = []
    for run in runs:
        m = run['metrics']
        rows.append({
            'seed': run['split_seed'], 'method': run['method'],
            'best_epoch': run['best_epoch'], 'n_validation': run['n_validation'],
            'n_vibration_validation': run['validation_vibration_segments'],
            **{k: m[k] for k in ('acc','macro_f1','edit','f1_10','f1_25','f1_50','seg_count_ratio')},
            **({'vib_acc':m['vibration_subset']['acc'],
                'vib_macro_f1':m['vibration_subset']['macro_f1'],
                'vib_edit':m['vibration_subset']['edit'],
                'vib_f1_50':m['vibration_subset']['f1_50']}
               if m['vibration_subset'] else {}),
        })
    if rows:
        with (output / 'results.csv').open('w', newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
            writer.writeheader(); writer.writerows(rows)
    summary = {}
    for method in ('video','dwa'):
        rr = [r for r in runs if r['method'] == method]
        if not rr: continue
        metrics = ('acc','macro_f1','edit','f1_10','f1_25','f1_50','seg_count_ratio')
        summary[method] = {k:{
            'mean':float(np.mean([r['metrics'][k] for r in rr])),
            'std_population':float(np.std([r['metrics'][k] for r in rr])),
            'min':float(np.min([r['metrics'][k] for r in rr])),
            'max':float(np.max([r['metrics'][k] for r in rr])),
        } for k in metrics}
        vib = [r['metrics']['vibration_subset'] for r in rr if r['metrics']['vibration_subset']]
        if vib:
            summary[method]['validation_vibration_subset'] = {
                k:{'mean':float(np.mean([v[k] for v in vib])),
                   'std_population':float(np.std([v[k] for v in vib])),
                   'n_splits':len(vib)}
                for k in ('acc','macro_f1','edit','f1_50')
            }
    if all(m in summary for m in ('video','dwa')):
        summary['dwa_minus_video_mean'] = {
            k:summary['dwa'][k]['mean']-summary['video'][k]['mean']
            for k in ('acc','macro_f1','edit','f1_10','f1_25','f1_50')
        }
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--feature-root',type=Path,required=True,
                    help='Directory containing the 11 recording feature folders')
    ap.add_argument('--vibration-root',type=Path,required=True,
                    help='Directory containing *_vib2040.npy arrays with shape [T,6]')
    ap.add_argument('--video-norm',type=Path,required=True,
                    help='1536-D mean/std NPZ used for the global and operator feature streams')
    ap.add_argument('--wear-checkpoint',type=Path,
                    help='WEAR parent checkpoint supplying compatible MS-TCN residual layers; required for DWA')
    ap.add_argument('--quality-json',type=Path,
                    help='Optional vibration alignment/availability metadata JSON')
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--epochs',type=int,default=150)
    ap.add_argument('--seeds',type=int,nargs='+',default=list(SEEDS))
    ap.add_argument('--method',choices=('video','dwa','both'),default='both')
    ap.add_argument('--threads',type=int,default=4)
    ap.add_argument('--device',choices=('auto','cpu','cuda','mps'),default='auto')
    ap.add_argument('--preflight',action='store_true')
    args=ap.parse_args()
    if args.method in ('dwa','both') and args.wear_checkpoint is None and not args.preflight:
        ap.error('--wear-checkpoint is required when training DWA')
    torch.set_num_threads(args.threads)
    records, video_mean, video_std, sources = load_records(args.feature_root,args.vibration_root,args.video_norm)
    assert len(records)==66
    quality=json.loads(args.quality_json.read_text()) if args.quality_json else {}
    split_plan={}
    for seed in args.seeds:
        train,val=split_records(records,seed)
        split_plan[str(seed)]={
            'train_segments':[r['id'] for r in train],
            'validation_segments':[r['id'] for r in val],
            'train_vibration_segments':[r['id'] for r in train if r['has_vib']],
            'validation_vibration_segments':[r['id'] for r in val if r['has_vib']],
        }
    if args.preflight:
        print(json.dumps({
            'recordings':list(RECORDINGS),'n_segments':len(records),
            'segments_per_recording':6,
            'with_vibration':sorted({r['recording'] for r in records if r['has_vib']}),
            'without_vibration':sorted({r['recording'] for r in records if not r['has_vib']}),
            'splits':split_plan,'quality_metadata':quality,
        },indent=2))
        return
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    script_copy=output/'train_tgif_dwa.py'
    shutil.copy2(Path(__file__), script_copy)
    config={
        'protocol':'legacy_tgif_transfer_recipe_adapted_to_full_11_recordings',
        'epochs':args.epochs,'seeds':args.seeds,'model_seed':0,
        'optimizer':'Adam lr=5e-4 weight_decay=0; StepLR(step_size=30,gamma=0.5)',
        'loss':'unweighted CE + 0.15 * mean(clamp(raw_logit_delta^2,max=9)); sum across stages',
        'checkpoint_selection':'highest mean per-segment validation F1@0.5 each epoch',
        'training_batching':'one full 5-minute segment at a time; shuffle each epoch',
        'gradient_clip_norm':5.0,'dropout':0.1,
        'video_normalization':str(args.video_norm.expanduser().resolve()),
        'sensor_normalization':'train-fold only; mean/std over valid processed 6-D feature rows',
        'missing_sensor_policy':'aux_valid=false and exact zero sensor attention context; no zero-imputed sensor treated as valid',
        'video_recordings':list(RECORDINGS),'n_segments':len(records),
        'segment_length_frames':SEG_LEN,'feature_stride_seconds':0.133467,
        'vibration_root':str(args.vibration_root.expanduser().resolve()),'vibration_quality':quality,
        'dwa_probe':'disabled (beta=0), matching the legacy TGIF transfer model',
        'dwa_initialization':'copy shape-compatible WEAR MS-TCN temporal residual layers only',
        'seed_splits':split_plan,
    }
    (output/'run_config.json').write_text(json.dumps(config,indent=2)+'\n')
    (output/'data_manifest.json').write_text(json.dumps({
        'video_norm':{'path':str(args.video_norm.expanduser().resolve()),'sha256':sha256(args.video_norm.expanduser().resolve())},
        'wear_init':({'path':str(args.wear_checkpoint.expanduser().resolve()),
                      'sha256':sha256(args.wear_checkpoint.expanduser().resolve())}
                     if args.wear_checkpoint else None),
        'recordings':sources,
    },indent=2)+'\n')
    (output/'split_manifest.json').write_text(json.dumps(split_plan,indent=2)+'\n')
    if args.device == 'auto':
        device=torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        device=torch.device(args.device)
        if device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested but is unavailable')
        if device.type == 'mps' and not torch.backends.mps.is_available():
            raise RuntimeError('MPS was requested but is unavailable')
    config['compute_device'] = str(device)
    (output/'run_config.json').write_text(json.dumps(config,indent=2)+'\n')
    print(f'Loaded {len(records)} segments from {len(RECORDINGS)} recordings; device={device}; '
          f'threads={args.threads}; output={output}',flush=True)
    methods=('video','dwa') if args.method=='both' else (args.method,)
    runs=[]
    for seed in args.seeds:
        for method in methods:
            result=train_one(method,seed,records,args.epochs,output,device,args.threads,args.wear_checkpoint)
            runs.append(result)
            write_aggregates(output,runs)
    # Include previously completed matching runs when resuming.
    all_runs=[]
    for seed in args.seeds:
        for method in methods:
            p=output/f'seed_{seed}'/method/'result.json'
            if p.exists(): all_runs.append(json.loads(p.read_text()))
    summary=write_aggregates(output,all_runs)
    print('COMPLETE',json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    main()
