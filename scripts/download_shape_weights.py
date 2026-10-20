#!/usr/bin/env python3
"""Download the verified 18-fold, seed-47 envelope-shape DWA weight bundle."""
import argparse
import json
from pathlib import Path
from download_wear_weights import ensure_bundle, validate_file_records

REPO = Path(__file__).resolve().parents[1]
METHOD = 'ENVELOPE_SHAPE_DWA'
PROTOCOL = 'envelope_shape_loso_v4_pilot'


def validate_archive_record(archive):
    if archive['method'] != METHOD or archive['seed'] != 47 or archive['folds'] != list(range(1, 19)):
        raise ValueError('Expected envelope-shape DWA seed 47, all 18 folds')
    expected = {f'{METHOD}/seed_47/split_{fold:02d}/{name}' for fold in range(1, 19)
                for name in ('parent.pt', 'background_probe.pt')}
    return validate_file_records(archive, expected)


def read_manifest(path):
    manifest = json.loads(Path(path).read_text())
    if manifest['schema_version'] != 1 or manifest['protocol'] != PROTOCOL or manifest['seeds'] != [47] or len(manifest['archives']) != 1:
        raise ValueError('Expected the single-seed v4 release manifest')
    for archive in manifest['archives']:
        validate_archive_record(archive)
    return manifest['archives']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=REPO/'models/wear_shape_v4/manifest.json')
    parser.add_argument('--model-root', type=Path, default=REPO/'models/wear_shape_v4')
    parser.add_argument('--archive-dir', type=Path)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    for archive in read_manifest(args.manifest):
        ensure_bundle(archive, args.model_root.resolve(), args.archive_dir, args.verify_only, validator=validate_archive_record)
    print('Envelope-shape weights ready:', args.model_root.resolve())


if __name__ == '__main__':
    main()
