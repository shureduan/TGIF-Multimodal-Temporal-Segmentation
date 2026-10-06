#!/usr/bin/env python3
"""Check the 18-subject WEAR inputs before starting a training run."""

import argparse
from pathlib import Path

import numpy as np

from tgif_dwa.wear_data import load_sequence


def check_subject(data_root, subject):
    sequence = load_sequence(data_root, subject)
    length = len(sequence["labels"])
    if length == 0:
        raise ValueError("empty feature sequence")
    for name in ("video", "inertial"):
        if not np.isfinite(sequence[name]).all():
            raise ValueError(f"{name} contains non-finite values; complete finite inputs are required")
    if not np.isin(sequence["labels"], np.arange(19)).all():
        raise ValueError("annotation label IDs must be in 0..18")
    return length


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    failures = []
    for index in range(18):
        subject = f"sbj_{index}"
        try:
            length = check_subject(args.data_root, subject)
        except (OSError, ValueError, KeyError, TypeError) as error:
            failures.append(subject)
            print(f"FAIL {subject}: {error}", flush=True)
        else:
            print(f"OK   {subject}: {length} aligned rows", flush=True)
    if failures:
        raise SystemExit(f"Input check failed for {len(failures)} subject(s); fix these before training.")
    print("All 18 subjects passed shape, alignment, finite-value and label checks.")
    print("This checks input compatibility, not feature provenance or experimental performance.")


if __name__ == "__main__":
    main()
