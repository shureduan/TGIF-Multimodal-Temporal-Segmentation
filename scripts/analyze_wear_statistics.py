#!/usr/bin/env python3
"""Analyze complete, matched 18-subject WEAR CSV or evaluation-JSON records."""

import argparse
import json
from pathlib import Path

from tgif_dwa.wear_statistics import analyze_records, read_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records, sources = read_records(args.inputs)
    report = {**analyze_records(records), "sources": sources}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for row in report["comparisons"]:
        low, high = row["ci95_unadjusted"]
        print(f"{row['metric']} | {row['comparison']} | "
              f"delta={row['mean_difference']:+.6f} | CI95=[{low:+.6f}, {high:+.6f}] | "
              f"p={row['p_two_sided']:.6f} | Holm={row['p_holm_six_comparisons']:.6f}")


if __name__ == "__main__":
    main()
