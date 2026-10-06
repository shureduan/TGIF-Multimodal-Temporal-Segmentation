#!/usr/bin/env python3
"""Export WEAR v2 subject/seed records and aggregate tables from a benchmark."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

METHODS = ["VIDEO_ONLY", "EARLY_CONCAT", "FIXED_WINDOW_ATTENTION", "FINAL_MODEL"]
METRICS = ["macro_f1_19", "background_f1", "action_macro_f1", "accuracy",
           *[f"map_at_{x:.1f}" for x in (.3, .4, .5, .6, .7)], "avg_map"]


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    benchmark = args.benchmark.resolve()
    output = args.output.resolve()
    if output == benchmark or output in benchmark.parents:
        parser.error("--output must not be the benchmark directory or an ancestor")
    plan = json.loads((benchmark / "benchmark_plan.json").read_text())
    status = json.loads((benchmark / "status.json").read_text())
    if (plan["protocol"] != "fixed_epoch_loso_v2" or status["phase"] != "complete"
            or set(plan["methods"]) != set(METHODS) or plan["folds"] != list(range(1, 19))):
        raise ValueError("Expected a complete 18-fold, four-method v2 benchmark")
    seeds = plan["seeds"]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be nonempty and distinct")
    summary = {
        "schema_version": 2, "protocol": plan["protocol"], "feature_stride_seconds": .5,
        "seeds": seeds, "n_subjects": 18, "n_training_runs": 18 * len(seeds) * 4,
        "training": {"parent_epochs": plan["parent_epochs"], "probe_epochs": plan["probe_epochs"],
                     "checkpoint_selection": "fixed_epoch_last", "outer_test_loaded_during_training": False},
        "aggregation": {
            "mean": "average seeds within each subject, then average 18 subjects",
            "std": "sample SD (ddof=1) of the 18 subject means, not a confidence interval",
            "concatenated_macro_f1": "concatenate 18 held-out sequences within each seed, then average seed scores",
        }, "methods": {},
    }
    records, table = [], []
    for method in METHODS:
        local, concat = [], {}
        for seed in seeds:
            saved = json.loads((benchmark / "metrics" / f"{method}_seed_{seed}.json").read_text())
            per_subject = sorted(saved["per_subject"], key=lambda row: row["fold"])
            if saved["n_sequences"] != 18 or [r["fold"] for r in per_subject] != list(range(1, 19)):
                raise ValueError(f"Incomplete fold coverage: {method}, seed {seed}")
            combined = np.zeros((19, 19), dtype=np.int64)
            for source in per_subject:
                row = dict(source)
                fold = row["fold"]
                identity = (row["protocol"], row["method"], row["seed"], row["subject"])
                if identity != (plan["protocol"], method, seed, f"sbj_{fold - 1}"):
                    raise ValueError(f"Metric identity mismatch: {identity}")
                path = benchmark / "predictions" / method / f"seed_{seed}" / f"split_{fold:02d}.npz"
                with np.load(path, allow_pickle=False) as data:
                    for key in ("protocol", "method", "seed", "fold", "run_id"):
                        if data[key].item() != row[key]:
                            raise ValueError(f"Prediction/metric {key} mismatch: {path}")
                    true, pred = data["true"], data["pred"]
                    if (true.ndim != 1 or true.shape != pred.shape or len(true) == 0
                            or not np.isin(true, np.arange(19)).all()
                            or not np.isin(pred, np.arange(19)).all()):
                        raise ValueError(f"Invalid frame labels: {path}")
                    cm = np.bincount(true * 19 + pred, minlength=361).reshape(19, 19)
                combined += cm
                denominator = cm.sum(0) + cm.sum(1)
                f1 = np.divide(2 * cm.diagonal(), denominator, out=np.zeros(19), where=denominator != 0)
                if (not np.isclose(f1.mean(), row["macro_f1_19"], rtol=0, atol=1e-12)
                        or not np.isclose(cm.trace() / cm.sum(), row["accuracy"], rtol=0, atol=1e-12)):
                    raise ValueError(f"Frame metrics do not match predictions: {path}")
                row.update(background_f1=float(f1[18]), action_macro_f1=float(f1[:18].mean()))
                local.append(row)
            denominator = combined.sum(0) + combined.sum(1)
            pooled = np.divide(2 * combined.diagonal(), denominator,
                               out=np.ones(19), where=denominator != 0).mean()
            if not np.isclose(pooled, saved["concatenated"]["macro_f1"], rtol=0, atol=1e-12):
                raise ValueError(f"Concatenated F1 mismatch: {method}, seed {seed}")
            concat[str(seed)] = saved["concatenated"]["macro_f1"]
        values = {}
        for metric in METRICS:
            subject_means = [np.mean([r[metric] for r in local if r["fold"] == fold]) for fold in range(1, 19)]
            values[metric] = {
                "mean": float(np.mean(subject_means)), "std": float(np.std(subject_means, ddof=1)),
                "seed_means": {str(seed): float(np.mean([r[metric] for r in local if r["seed"] == seed])) for seed in seeds},
            }
        values["concatenated_macro_f1"] = {"mean": float(np.mean(list(concat.values()))), "per_seed": concat}
        summary["methods"][method] = values
        table.append({"method": method, "macro_f1_subject_mean": values["macro_f1_19"]["mean"],
                      "macro_f1_subject_sd": values["macro_f1_19"]["std"],
                      "macro_f1_concat_seed_mean": values["concatenated_macro_f1"]["mean"],
                      **{key: values[key]["mean"] for key in METRICS if key != "macro_f1_19"}})
        records.extend(local)
    if not all(np.isfinite(r[k]) for r in records for k in METRICS):
        raise ValueError("Non-finite metric value")
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "wear_18fold_metrics.csv", records)
    write_csv(output / "wear_summary.csv", table)
    (output / "wear_aggregate.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"Exported {len(records)} subject/seed records and four-method summaries to {output}")


if __name__ == "__main__":
    main()
