#!/usr/bin/env python3
"""Train the experimental sensor-driven DWA; v2 releases are unchanged."""
import argparse
from pathlib import Path
from tgif_dwa.signal_training import run_signal_training
from tgif_dwa.signal_wear import VARIANTS, METHOD


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
    print(run_signal_training(**vars(parser.parse_args())))


if __name__ == '__main__':
    main()
