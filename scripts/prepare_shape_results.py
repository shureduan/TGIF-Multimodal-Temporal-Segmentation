#!/usr/bin/env python3
"""Build portable, hash-verified v4 figure inputs from the completed seed-47 study."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import statistics

REPO = Path(__file__).resolve().parents[1]
METHOD = 'ENVELOPE_SHAPE_DWA'
MAIN = ['VIDEO_ONLY', 'EARLY_CONCAT', 'FIXED_WINDOW_ATTENTION', 'FINAL_MODEL', 'SIGNAL_ADAPTIVE_DWA', METHOD]
METRICS = ['accuracy', 'macro_precision_19', 'macro_recall_19', 'macro_f1_19', 'background_f1', 'action_macro_f1',
           *[f'map_at_{x:.1f}' for x in (.3, .4, .5, .6, .7)], 'avg_map']
ODD = list(range(1, 19, 2))
CONTROLS = ['NO_SENSOR_RESIZE', 'NO_CONTRACTION']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


def prepare(source, controls, output, metadata):
    source, output = Path(source), Path(output)
    manifest = json.loads((source/'manifest.json').read_text())
    for name, digest in manifest['sha256'].items():
        if Path(name).name != name or sha(source/name) != digest:
            raise ValueError(f'Input table hash mismatch: {name}')
    records = read_csv(source/'per_subject_metrics.csv')
    expected = {(m, f) for m in MAIN+['SHAPE_PARENT_NO_PROBE', 'SHAPE_ROUND0'] for f in range(1, 19)}
    identity = [(r['method'], int(r['fold'])) for r in records]
    if len(identity) != 144 or set(identity) != expected or any(int(r['seed']) != 47 or r['subject'] != f"sbj_{int(r['fold'])-1}" for r in records):
        raise ValueError('Expected exactly 144 paired seed-47 records')
    selected = [r for r in read_csv(controls) if r['method'] in CONTROLS and int(r['seed']) == 47 and int(r['fold']) in ODD]
    keys = [(r['method'], int(r['fold'])) for r in selected]
    if len(keys) != 18 or set(keys) != {(m, f) for m in CONTROLS for f in ODD}:
        raise ValueError('Expected both matched controls on all nine odd folds, seed 47')
    for row in selected:
        if row['subject'] != f"sbj_{int(row['fold'])-1}" or row['protocol'] != 'signal_adaptive_loso_v3_candidate':
            raise ValueError('Wrong control subject or protocol')
    output.mkdir(parents=True, exist_ok=True)
    # Local absolute paths and audit artifacts are deliberately not release inputs.
    names = ['aggregate_metrics.csv', 'per_subject_metrics.csv', 'per_class_metrics.csv', 'paired_comparisons.csv',
             'paired_statistics.json', 'window_behavior.csv', 'inference_timing.csv', 'timeline_sbj_1.csv', 'timeline_sbj_9.csv']
    for name in names:
        if (source/name).exists():
            shutil.copyfile(source/name, output/name)
    write_csv(output/'control_metrics.csv', selected)
    paired = selected + [r for r in records if r['method'] in ['SIGNAL_ADAPTIVE_DWA', METHOD] and int(r['fold']) in ODD]
    rows = []
    for method in [*CONTROLS, 'SIGNAL_ADAPTIVE_DWA', METHOD]:
        q = [r for r in paired if r['method'] == method]
        rows.append({'method': method, 'n_subjects': 9, 'n_seeds': 1, 'seed': 47,
                     **{k: statistics.mean(float(r[k]) for r in q) for k in METRICS},
                     **{k+'_subject_sd': statistics.stdev(float(r[k]) for r in q) for k in METRICS}})
    write_csv(output/'ablation_aggregate_metrics.csv', rows)
    # Portable prediction identities, never machine-specific absolute paths.
    predictions = []
    for name, digest in manifest.get('prediction_sha256', {}).items():
        path = Path(name); relative = '/'.join(path.parts[-3:])
        predictions.append({'path': relative, 'sha256': digest})
    (output/'prediction_sources.json').write_text(json.dumps(predictions, indent=2)+'\n')
    release = {'schema_version': 1, 'release': 'wear-shape-v4-20261020', 'protocol': 'envelope_shape_loso_v4_pilot',
               'seed': 47, 'seeds': [47], 'main_folds': list(range(1, 19)), 'ablation_folds': ODD,
               'main_methods': MAIN, 'controls': CONTROLS, 'metric_records': 144,
               'aggregation': 'Equal subject means; sample SD across subjects; concatenated F1 pools all 18 subjects.',
               'component_comparison': 'Nine matched odd folds, seed 47; controller alternatives are independently trained. Frozen output diagnostics share v4 weights.',
               'timing': {'device': 'cpu', 'threads': 1, 'subjects': ['sbj_0', 'sbj_9', 'sbj_17'], 'warmup': 1, 'repeats': 5,
                          'scope': 'Complete two-round forward plus probe, excluding data loading, normalization and video feature extraction; models timed in fixed v3-then-v4 order.'},
               'data_sha256': metadata['data_sha256'], 'reference_fold_devices': metadata['fold_devices'],
               'training_source_sha256': metadata.get('training_source_sha256', metadata.get('source_sha256', {})),
               'scope': 'Exploratory single-seed comparison on the existing WEAR subjects; all directions of change retained.'}
    release['sha256'] = {q.name: sha(q) for q in sorted(output.iterdir()) if q.suffix in ('.csv', '.json') and q.name != 'manifest.json'}
    (output/'manifest.json').write_text(json.dumps(release, indent=2)+'\n')
    return release


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--controls-table', type=Path, default=REPO/'results/wear_signal_v3/per_subject_seed_metrics.csv')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.study/'source_data', args.controls_table, args.output, json.loads((args.study/'benchmark_plan.json').read_text()))
    print('Prepared 18-subject comparison and matched nine-subject controller study:', args.output)


if __name__ == '__main__':
    main()
