#!/usr/bin/env python3
"""Download SHA-256 verified WEAR sensor-driven DWA weights (stdlib only)."""
import argparse
import json
from pathlib import Path
from download_wear_weights import ensure_bundle, validate_file_records

REPO = Path(__file__).resolve().parents[1]
METHODS = ('SIGNAL_ADAPTIVE_DWA', 'NO_SENSOR_RESIZE', 'NO_CONTRACTION')
SEEDS = (41, 47, 53)


def validate_archive_record(archive):
    method, seed = archive['method'], archive['seed']
    if method not in METHODS or seed not in SEEDS:
        raise ValueError('Unknown sensor-driven release method or seed')
    folds = list(range(1, 19)) if method == METHODS[0] else list(range(1, 19, 2))
    if archive['folds'] != folds:
        raise ValueError('Wrong release cohort: full DWA uses 18 folds; controls use nine odd folds')
    expected = {f'{method}/seed_{seed}/split_{fold:02d}/{name}' for fold in folds
                for name in ('parent.pt', 'background_probe.pt')}
    return validate_file_records(archive, expected)


def read_manifest(path):
    manifest = json.loads(Path(path).read_text())
    if manifest['schema_version'] != 1 or manifest['protocol'] != 'signal_adaptive_loso_v3_candidate':
        raise ValueError('Expected a sensor-driven v3 release manifest')
    archives = manifest['archives']
    if len(archives) != 9 or {(r['method'], r['seed']) for r in archives} != {(m,s) for m in METHODS for s in SEEDS}:
        raise ValueError('Expected all nine method/seed bundles')
    for archive in archives:
        validate_archive_record(archive)
    return archives


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=REPO/'models/wear_signal_v3/manifest.json')
    parser.add_argument('--model-root', type=Path, default=REPO/'models/wear_signal_v3')
    parser.add_argument('--method', choices=METHODS, default=METHODS[0])
    parser.add_argument('--seed', type=int, choices=SEEDS, default=47)
    parser.add_argument('--all', action='store_true', help='all three seeds, full DWA and both published controls')
    parser.add_argument('--archive-dir', type=Path)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    for archive in read_manifest(args.manifest):
        if args.all or (archive['method'], archive['seed']) == (args.method, args.seed):
            ensure_bundle(archive, args.model_root.resolve(), args.archive_dir, args.verify_only,
                          validator=validate_archive_record)


if __name__ == '__main__':
    main()
