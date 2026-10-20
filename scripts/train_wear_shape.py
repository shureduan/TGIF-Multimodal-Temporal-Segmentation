#!/usr/bin/env python3
"""Train the envelope-shape DWA pilot with fixed training budgets."""
import argparse
from pathlib import Path
from tgif_dwa.shape_training import run_shape_training
from tgif_dwa.shape_wear import VARIANTS, METHOD


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fold', type=int, choices=range(1, 19), required=True)
    parser.add_argument('--seed', type=int, default=47)
    parser.add_argument('--parent-epochs', type=int, default=30)
    parser.add_argument('--probe-epochs', type=int, default=15)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--method', choices=list(VARIANTS), default=METHOD)
    print(run_shape_training(**vars(parser.parse_args())))


if __name__ == '__main__':
    main()
