#!/usr/bin/env python3
"""Infer the published WEAR cohort from verified released weights, then score it."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from download_wear_weights import ensure_bundle, sha256
from download_signal_weights import read_manifest, validate_archive_record
from run_wear_benchmark import dataset_hashes
from analyze_signal_release import analyze, FOLDS, ODD, SEEDS, CONTROLS, METHOD, BASELINES

REPO = Path(__file__).resolve().parents[1]


def inference_input_hashes(root):
    """Ignore annotation split files that the loader never reads."""
    return {name:value for name,value in dataset_hashes(root).items()
            if not name.startswith('label/') or name in
            {'label/wear_split_18.json', *{f'label/wear_test_split_{i}.json' for i in range(1,7)}}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--v2-models', type=Path, default=REPO/'models/wear_v2')
    parser.add_argument('--v3-models', type=Path, default=REPO/'models/wear_signal_v3')
    parser.add_argument('--device', default='cpu', choices=['cpu','mps','cuda','reference'],
                        help='reference uses CPU on odd folds and Apple MPS on even folds')
    parser.add_argument('--allow-different-inputs', action='store_true', help='explicitly evaluate alternate prepared features')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    reference = json.loads((REPO/'results/wear_signal_v3/manifest.json').read_text())
    inputs = inference_input_hashes(args.data_root)
    expected_names = {f'{kind}/sbj_{i}.npy' for kind in ('video','imu') for i in range(18)} | {'label/wear_split_18.json', *{f'label/wear_test_split_{i}.json' for i in range(1,7)}}
    expected_inputs = {name:value for name,value in reference['data_sha256'].items() if name in expected_names}
    feature_count = sum(name.startswith(('video/','imu/')) for name in inputs)
    if feature_count != 36:
        raise ValueError('Expected all 36 feature arrays')
    if inputs != expected_inputs and not args.allow_different_inputs:
        raise ValueError('Inputs differ from the released feature hashes; see docs/data.md. '
                         'Use --allow-different-inputs only for a separately identified evaluation.')
    v2 = json.loads((REPO/'models/wear_v2/manifest.json').read_text())['archives']
    v3 = read_manifest(REPO/'models/wear_signal_v3/manifest.json')
    for archive in v2:
        ensure_bundle(archive, args.v2_models.resolve(), verify_only=True)
    for archive in v3:
        ensure_bundle(archive, args.v3_models.resolve(), verify_only=True, validator=validate_archive_record)
    identity = {'data_sha256':inputs, 'device':args.device,
                'checkpoint_manifests':{name:sha256(REPO/f'models/{name}/manifest.json') for name in ('wear_v2','wear_signal_v3')},
                'inference_source_sha256':{str(p.relative_to(REPO)):sha256(p) for p in
                    [*sorted((REPO/'src/tgif_dwa').glob('*.py')), REPO/'scripts/infer_wear.py', REPO/'scripts/infer_wear_signal.py']}}
    plan = args.output/'evaluation_plan.json'
    if plan.exists() and json.loads(plan.read_text()) != identity:
        raise ValueError('Evaluation inputs/settings/source changed; use a new output directory')
    plan.write_text(json.dumps(identity, indent=2)+'\n')
    jobs = [(m,s,f) for m in [*BASELINES,METHOD] for s in SEEDS for f in FOLDS]
    jobs += [(m,s,f) for m in CONTROLS for s in SEEDS for f in ODD]
    for number,(method,seed,fold) in enumerate(jobs,1):
        baseline = method in BASELINES
        folder = (args.v2_models if baseline else args.v3_models)/method/f'seed_{seed}/split_{fold:02d}'
        prediction = args.output/'predictions'/method/f'seed_{seed}/split_{fold:02d}.npz'
        receipt = prediction.with_suffix('.sha256')
        if receipt.exists():
            if not prediction.exists() or sha256(prediction) != receipt.read_text().strip():
                raise ValueError(f'Cached prediction changed: {prediction}')
            continue
        device = reference['reference_fold_devices'][str(fold)] if args.device=='reference' else args.device
        command = [sys.executable,str(REPO/'scripts'/('infer_wear.py' if baseline else 'infer_wear_signal.py')),
                   '--data-root',str(args.data_root),'--subject',f'sbj_{fold-1}',
                   '--parent',str(folder/'parent.pt'),'--device',device,'--output',str(prediction)]
        if not baseline or method=='FINAL_MODEL':
            command += ['--probe',str(folder/'background_probe.pt')]
        print(f'[{number}/{len(jobs)}] {method} seed {seed} fold {fold}',flush=True)
        subprocess.run(command,check=True)
        receipt.write_text(sha256(prediction)+'\n')
    analyze(args.output/'predictions',args.output/'predictions',args.output/'source_data')


if __name__ == '__main__':
    main()
