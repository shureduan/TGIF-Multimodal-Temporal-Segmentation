#!/usr/bin/env python3
"""Download/verify v4 and matching reference weights, infer, score and redraw six figures."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from download_wear_weights import ensure_bundle, sha256
from download_signal_weights import read_manifest as signal_manifest, validate_archive_record as signal_validator
from download_shape_weights import read_manifest as shape_manifest, validate_archive_record as shape_validator
from reproduce_signal_release import inference_input_hashes
from tgif_dwa.signal_study import measure
from prepare_shape_results import prepare, write_csv, MAIN, METHOD, CONTROLS, ODD

REPO = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--v2-models', type=Path, default=REPO/'models/wear_v2')
    parser.add_argument('--v3-models', type=Path, default=REPO/'models/wear_signal_v3')
    parser.add_argument('--v4-models', type=Path, default=REPO/'models/wear_shape_v4')
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda', 'reference'], default='cpu',
                        help='reference matches CPU odd/MPS even folds used for the published scores')
    parser.add_argument('--allow-different-inputs', action='store_true', help='explicitly evaluate separately identified prepared features')
    args = parser.parse_args()
    reference = json.loads((REPO/'results/wear_shape_v4/manifest.json').read_text())
    inputs = inference_input_hashes(args.data_root)
    expected_names = {f'{kind}/sbj_{i}.npy' for kind in ('video','imu') for i in range(18)} | {'label/wear_split_18.json', *{f'label/wear_test_split_{i}.json' for i in range(1,7)}}
    expected_inputs = {k:v for k,v in reference['data_sha256'].items() if k in expected_names}
    if inputs != expected_inputs or sum(k.startswith(('video/', 'imu/')) for k in inputs) != 36:
        if not args.allow_different_inputs:
            raise ValueError('Prepared inputs differ from release hashes; see docs/data.md or explicitly use --allow-different-inputs')
    source_hashes = {str(p.relative_to(REPO)):sha256(p) for p in [*sorted((REPO/'src/tgif_dwa').glob('*.py')),
                     *[REPO/'scripts'/name for name in ('infer_wear.py','infer_wear_signal.py','infer_wear_shape.py','analyze_shape_seed.py','reproduce_shape_release.py')]]}
    identity = {'data_sha256':inputs, 'device':args.device, 'source_sha256':source_hashes,
                'checkpoint_manifests':{name:sha256(REPO/f'models/{name}/manifest.json') for name in ('wear_v2','wear_signal_v3','wear_shape_v4')}}
    args.output = args.output.resolve(); args.output.mkdir(parents=True, exist_ok=True)
    receipt = args.output/'evaluation_plan.json'
    if receipt.exists() and json.loads(receipt.read_text()) != identity:
        raise ValueError('Settings, input or source changed; use a new output directory')
    receipt.write_text(json.dumps(identity, indent=2)+'\n')
    v2 = [r for r in json.loads((REPO/'models/wear_v2/manifest.json').read_text())['archives'] if r['seed']==47]
    v3 = [r for r in signal_manifest(REPO/'models/wear_signal_v3/manifest.json') if r['seed']==47]
    v4 = shape_manifest(REPO/'models/wear_shape_v4/manifest.json')
    for row in v2: ensure_bundle(row, args.v2_models.resolve())
    for row in v3: ensure_bundle(row, args.v3_models.resolve(), validator=signal_validator)
    for row in v4: ensure_bundle(row, args.v4_models.resolve(), validator=shape_validator)
    environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    references = {}; artifacts = {}; control_rows = []
    jobs = [(m,f) for m in MAIN for f in range(1,19)] + [(m,f) for m in CONTROLS for f in ODD]
    for number,(method,fold) in enumerate(jobs,1):
        baseline = method in MAIN[:4]
        models = args.v2_models if baseline else args.v4_models if method==METHOD else args.v3_models
        folder = models.resolve()/method/'seed_47'/f'split_{fold:02d}'
        prediction = args.output/'predictions'/method/'seed_47'/f'split_{fold:02d}.npz'
        digest_file = prediction.with_suffix('.sha256')
        if digest_file.exists():
            if not prediction.exists() or sha256(prediction)!=digest_file.read_text().strip():
                raise ValueError('Cached prediction changed: '+str(prediction))
        else:
            script = 'infer_wear.py' if baseline else 'infer_wear_shape.py' if method==METHOD else 'infer_wear_signal.py'
            device = reference['reference_fold_devices'][str(fold)] if args.device=='reference' else args.device
            command = [sys.executable, str(REPO/'scripts'/script), '--data-root', str(args.data_root.resolve()),
                       '--subject', f'sbj_{fold-1}', '--parent', str(folder/'parent.pt'), '--device', device, '--output', str(prediction)]
            if not baseline or method=='FINAL_MODEL': command += ['--probe', str(folder/'background_probe.pt')]
            print(f'[{number}/{len(jobs)}] {method} fold {fold} on {device}', flush=True)
            subprocess.run(command, check=True, env=environment)
            digest_file.write_text(sha256(prediction)+'\n')
        if method in MAIN and method!=METHOD:
            references[f'{method}/{fold}'] = str(prediction); artifacts[str(prediction)] = sha256(prediction)
        if method=='SIGNAL_ADAPTIVE_DWA': artifacts[str(folder/'parent.pt')] = sha256(folder/'parent.pt')
        if method in CONTROLS:
            with np.load(prediction, allow_pickle=False) as z:
                if z['method'].item()!=method or int(z['seed'])!=47 or int(z['fold'])!=fold or z['id'].item()!=f'sbj_{fold-1}':
                    raise ValueError('Control prediction identity differs')
                values,_ = measure(z['true'],z['probabilities'],z['id'].item())
                control_rows.append({'method':method,'source_method':method,'seed':47,'fold':fold,'subject':f'sbj_{fold-1}',
                                     'protocol':z['protocol'].item(),'run_id':z['run_id'].item(),**values})
    write_csv(args.output/'control_metrics.csv', control_rows)
    # The analyzer's paths are local execution metadata, never committed release inputs.
    plan = {'study':'envelope_shape_release_seed47','seed':47,'method':METHOD,'folds':list(range(1,19)),
            'data_root':str(args.data_root.resolve()),'data_sha256':inputs,'fold_devices':reference['reference_fold_devices'],
            'source_sha256':reference['training_source_sha256'],'reference_predictions':references,'reference_artifact_sha256':artifacts}
    (args.output/'benchmark_plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    # CPU latency loader expects the candidate checkpoints under output/checkpoints.
    checkpoint_dir = args.output/'checkpoints'
    if checkpoint_dir.exists() and checkpoint_dir.resolve()!=args.v4_models.resolve():
        raise ValueError('Output/checkpoints must refer to the selected v4 weights')
    if not checkpoint_dir.exists(): checkpoint_dir.symlink_to(args.v4_models.resolve(), target_is_directory=True)
    subprocess.run([sys.executable,str(REPO/'scripts/analyze_shape_seed.py'),'--benchmark',str(args.output)], check=True, env=environment)
    prepare(args.output/'source_data',args.output/'control_metrics.csv',args.output/'release_data',plan)
    subprocess.run([sys.executable,str(REPO/'scripts/plot_shape_results.py'),'--data-root',str(args.output/'release_data'),
                    '--output',str(args.output/'gallery')], check=True, env=environment)
    print('Reproduced all six figures:',args.output/'gallery/WEAR_envelope_shape_DWA_results.pdf')


if __name__ == '__main__':
    main()
