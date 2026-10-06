"""Exploratory paired, subject-level comparisons of matched WEAR runs."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import stats

COMPARATORS = ("VIDEO_ONLY", "EARLY_CONCAT", "FIXED_WINDOW_ATTENTION")
METRICS = ("macro_f1_19", "map_at_0.5")
HISTORICAL_PROTOCOL = "historical_validation_selected_loso"
HISTORICAL_CSV_SHA256 = "1a7d4a97d66507db5855b385ae0732f77e70b904201ef912d8e3239a4d90a82a"


def read_records(paths):
    records, sources = [], []
    for filename in paths:
        path = Path(filename)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if path.suffix == ".csv":
            with path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            # The committed original CSV predates protocol metadata.
            for row in rows:
                if not row.get("protocol"):
                    if digest != HISTORICAL_CSV_SHA256:
                        raise ValueError("unrecognized CSV requires explicit protocol metadata")
                    row["protocol"] = HISTORICAL_PROTOCOL
        else:
            payload = json.loads(path.read_text())
            rows = payload["per_subject"]
        records.extend(rows)
        sources.append({"file": path.name, "sha256": digest})
    return records, sources


def holm_adjust(pvalues):
    values = np.asarray(pvalues, dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("p-values must be finite and in [0,1]")
    order = np.argsort(values)
    ordered = np.maximum.accumulate(values[order] * np.arange(len(values), 0, -1))
    adjusted = np.empty_like(values)
    adjusted[order] = np.minimum(ordered, 1.0)
    return adjusted.tolist()


def analyze_records(records, *, expected_folds=18):
    if expected_folds < 2:
        raise ValueError("at least two subject folds are required")
    methods = ("FINAL_MODEL", *COMPARATORS)
    indexed = {method: {} for method in methods}
    protocols = set()
    for row in records:
        method = row["method"]
        if method not in indexed:
            raise ValueError(f"unexpected method {method}")
        fold, subject, seed = int(row["fold"]), str(row["subject"]), int(row["seed"])
        if subject != f"sbj_{fold - 1}":
            raise ValueError("fold/subject identity mismatch")
        key = (fold, subject, seed)
        if key in indexed[method]:
            raise ValueError(f"duplicate method/fold/subject/seed: {method}, {key}")
        values = {metric: float(row[metric]) for metric in METRICS}
        if not all(np.isfinite(value) and 0 <= value <= 1 for value in values.values()):
            raise ValueError("metric scores must be finite fractions in [0,1]")
        indexed[method][key] = values
        protocols.add(row["protocol"])
    if len(protocols) != 1:
        raise ValueError("all compared rows must come from the same training protocol")
    keys = set(indexed["FINAL_MODEL"])
    if any(set(indexed[method]) != keys for method in COMPARATORS):
        raise ValueError("methods must contain exactly matching subject/fold/seed records")
    seeds = sorted({key[2] for key in keys})
    subjects = sorted({(key[0], key[1]) for key in keys})
    if len(subjects) != expected_folds:
        raise ValueError(f"expected {expected_folds} subject folds, got {len(subjects)}")
    if {fold for fold, _ in subjects} != set(range(1, expected_folds + 1)):
        raise ValueError("fold IDs must cover the complete expected subject set")
    if keys != {(fold, subject, seed) for fold, subject in subjects for seed in seeds}:
        raise ValueError("every subject must have every seed for every method")
    comparisons = []
    for metric in METRICS:
        for comparator in COMPARATORS:
            differences = np.asarray([
                [indexed["FINAL_MODEL"][(fold, subject, seed)][metric]
                 - indexed[comparator][(fold, subject, seed)][metric] for seed in seeds]
                for fold, subject in subjects
            ])
            # Repeated seeds on one subject are not independent subjects.
            paired = differences.mean(axis=1)
            mean = float(paired.mean())
            sd = float(paired.std(ddof=1))
            sem = sd / np.sqrt(len(subjects))
            width = float(stats.t.ppf(0.975, len(subjects) - 1) * sem)
            if sd == 0:
                statistic, pvalue = (0.0, 1.0) if mean == 0 else (None, 0.0)
            else:
                test = stats.ttest_1samp(paired, popmean=0.0, alternative="two-sided")
                statistic, pvalue = float(test.statistic), float(test.pvalue)
            comparisons.append({
                "metric": metric, "comparison": f"FINAL_MODEL - {comparator}",
                "n_subjects": len(subjects), "n_seeds": len(seeds),
                "mean_difference": mean, "difference_std": sd,
                "ci95_unadjusted": [mean - width, mean + width],
                "t_statistic": statistic, "degrees_of_freedom": len(subjects) - 1,
                "p_two_sided": pvalue, "zero_variance": sd == 0,
                "wins": int((paired > 0).sum()), "ties": int((paired == 0).sum()),
                "losses": int((paired < 0).sum()),
                "per_seed_mean_difference": {
                    str(seed): float(differences[:, i].mean()) for i, seed in enumerate(seeds)
                },
                "per_subject_mean_difference": {
                    subject: float(paired[i]) for i, (_, subject) in enumerate(subjects)
                },
            })
    adjusted = holm_adjust([row["p_two_sided"] for row in comparisons])
    for row, pvalue in zip(comparisons, adjusted):
        row["p_holm_six_comparisons"] = pvalue
    return {
        "protocol": next(iter(protocols)), "seeds": seeds, "n_subjects": len(subjects),
        "unit": "subject; mean matched differences across seeds before paired testing",
        "family": "three comparators x two metrics; Holm adjustment over all six tests",
        "interpretation": (
            "Exploratory comparisons. LOSO training sets overlap, so folds are not fully "
            "independent and nominal t-test intervals/p-values have this limitation. "
            "Historical checkpoint selection used the reported subjects; statistics do "
            "not repair that bias. Non-significance is not proof of equivalence or no effect."
        ),
        "comparisons": comparisons,
    }
