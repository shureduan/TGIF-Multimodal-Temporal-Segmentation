#!/usr/bin/env python3
"""Train a fixed-epoch WEAR LOSO fold without reading the outer test subject."""

import argparse
from pathlib import Path

from tgif_dwa.wear_baselines import METHODS
from tgif_dwa.wear_training import run_training


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(1, 19), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, default="FINAL_MODEL")
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--parent-epochs", type=positive_int, default=30)
    parser.add_argument("--probe-epochs", type=positive_int, default=15)
    parser.add_argument("--device", default="cpu")
    run_training(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
