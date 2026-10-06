#!/usr/bin/env python3
"""Compute figure inputs from frozen WEAR v2 runs, with no fitting or selection."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

STAGES = ["ROUND0", "SEGMENT_ONLY", "DWA_ONLY", "FINAL_MODEL"]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path, rows):
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def shifted_boundaries(prediction, steps):
    import numpy as np
    starts = np.flatnonzero(np.r_[True, prediction[1:] != prediction[:-1]])
    classes = prediction[starts]
    boundaries = starts[1:].astype(int)
    shifted, lower = [], 1
    for j, x in enumerate(boundaries):
        upper = len(prediction) - (len(boundaries) - j)
        cut = max(lower, min(upper, int(x + steps)))
        shifted.append(cut)
        lower = cut + 1
    cuts = np.r_[0, shifted, len(prediction)]
    result = np.empty_like(prediction)
    for j, label in enumerate(classes):
        result[cuts[j]:cuts[j + 1]] = label
    return result


def cm_f1(matrix):
    import numpy as np
    matrix = np.asarray(matrix, dtype=np.float64)
    denominator = matrix.sum(0) + matrix.sum(1)
    values = np.divide(2 * matrix.diagonal(), denominator,
                       out=np.zeros(19), where=denominator != 0)
    return float(values.mean())


def diagnostic_job(benchmark, output, seed, fold):
    sys.path.insert(0, str(Path(benchmark) / "source_snapshot/src"))
    import numpy as np
    import torch
    from tgif_dwa.wear_data import load_sequence
    from tgif_dwa.wear_model import load_wear_final, build_wear_parent, normalize_wear_inertial

    started = time.monotonic()
    torch.set_num_threads(1)
    benchmark, output = Path(benchmark), Path(output)
    plan = json.loads((benchmark / "benchmark_plan.json").read_text())
    destination = output / "diagnostics" / f"seed_{seed}_fold_{fold:02d}.json"
    device = plan["fold_devices"][str(fold)]
    subject = f"sbj_{fold - 1}"
    folder = benchmark / "checkpoints/FINAL_MODEL" / f"seed_{seed}" / f"split_{fold:02d}"
    manifest = json.loads((benchmark / "checkpoint_manifest.json").read_text())
    hashes = {}
    for name in ("parent.pt", "background_probe.pt"):
        path = folder / name
        hashes[name] = sha(path)
        assert hashes[name] == manifest[str(path.relative_to(benchmark))]["sha256"]
    if destination.exists():
        cached = json.loads(destination.read_text())
        if (cached["seed"], cached["fold"], cached["subject"], cached["device"]) != (seed, fold, subject, device):
            raise ValueError(f"Cached diagnostic identity mismatch: {destination}")
        if cached["checkpoint_sha256"] != hashes:
            raise ValueError(f"Cached checkpoint mismatch: {destination}")
        return cached
    final, mean, std = load_wear_final(folder / "parent.pt", folder / "background_probe.pt", device=device)
    assert final.parent.uncertainty_seconds == 1.0 and final.parent.margin_prior == 0.5
    assert final.parent.window.initial_radius_seconds == 2.0 and final.beta == 0.5
    hard = build_wear_parent(uncertainty_seconds=0.0, margin_prior=0.5).to(device).eval()
    hard.load_state_dict(final.parent.state_dict(), strict=True)
    sequence = load_sequence(plan["data_root"], subject)
    truth = sequence["labels"]
    v = torch.as_tensor(sequence["video"], dtype=torch.float32, device=device)
    a = torch.as_tensor(normalize_wear_inertial(sequence["inertial"], mean, std), device=device)
    valid = torch.ones(len(v), dtype=torch.bool, device=device)
    stages, robustness, replay = [], [], []
    with torch.inference_mode():
        parent = final.parent(v, a)
        seed_labels = parent["round_predictions"][0].cpu().numpy()
        starts, ends = hard.labels_to_bounds(parent["round_predictions"][0])
        hard_context = hard.segment_seed_expand_attention(v, a, starts, ends, valid, return_diagnostics=False)["context"]
        hard_logits, _ = hard.classify(v, hard_context)
        dynamic_logits = parent["round_logits"][1][-1, 0].T
        pbg = torch.sigmoid(final.probe(torch.cat([v, a], -1)))
        calibrated = dynamic_logits.clone()
        calibrated[:, :18] += final.beta * torch.log(1 - pbg[:, None] + 1e-6)
        calibrated[:, 18] += final.beta * torch.log(pbg + 1e-6)
        logits = [parent["round_logits"][0][-1, 0].T, hard_logits[-1, 0].T, dynamic_logits, calibrated]
        for stage, z in zip(STAGES, logits):
            probabilities = z.softmax(-1).cpu().numpy()
            predicted = probabilities.argmax(-1)
            cm = np.bincount(truth * 19 + predicted, minlength=19 * 19).reshape(19, 19)
            stages.append({"stage": stage, "macro_f1_19": cm_f1(cm), "confusion_matrix": cm.tolist()})
            if stage == "FINAL_MODEL":
                path = benchmark / "predictions" / stage / f"seed_{seed}" / f"split_{fold:02d}.npz"
                with np.load(path, allow_pickle=False) as saved:
                    assert np.array_equal(truth, saved["true"])
                    maximum = float(np.abs(probabilities - saved["probabilities"]).max())
                    equal = bool(np.array_equal(predicted, saved["pred"]))
                    assert maximum < 1e-5 and equal, (stage, seed, fold, maximum)
                replay.append({"stage": stage, "max_probability_difference": maximum, "classes_equal": equal})
        diagnostics = parent["round_diagnostics"][1]
        masses = {name: float(diagnostics[name].mean().cpu()) for name in (
            "left_margin_attention_mass", "core_attention_mass", "right_margin_attention_mass")}
        assert abs(sum(masses.values()) - 1.0) < 1e-5
        for label, model, base in (("Hard segment", hard, hard_context),
                                   ("Boundary uncertainty", final.parent, parent["round_contexts"][1])):
            for steps in (-1, 1, -2, 2):
                perturbed = torch.as_tensor(shifted_boundaries(seed_labels, steps), device=device)
                left, right = model.labels_to_bounds(perturbed)
                context = model.segment_seed_expand_attention(v, a, left, right, valid,
                                                               return_diagnostics=False)["context"]
                value = float(torch.linalg.vector_norm(base - context, dim=1).mean().cpu())
                assert np.isfinite(value)
                robustness.append({"method": label, "jitter_seconds": abs(steps) * 0.5,
                                   "signed_steps": steps, "context_l2": value})
    row = {"seed": seed, "fold": fold, "subject": subject, "device": device,
           "run_id": final.run_metadata["run_id"], "checkpoint_sha256": hashes,
           "frames": len(truth), "stages": stages, "support": masses,
           "robustness": robustness, "replay_checks": replay,
           "seconds": time.monotonic() - started}
    dump(destination, row)
    return row


def main():
    import numpy as np
    import pandas as pd
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-workers", type=int, default=2)
    args = parser.parse_args()
    if args.cpu_workers < 1:
        parser.error("--cpu-workers must be positive")
    output = args.output.resolve()
    benchmark = args.benchmark.resolve()
    if output == benchmark or output in benchmark.parents:
        parser.error("--output must not be the benchmark directory or an ancestor")
    if output.exists() and any(output.iterdir()) and not (output / "figure_computation_plan.json").is_file():
        parser.error("Use an empty output directory for a new computation")
    output.mkdir(parents=True, exist_ok=True)
    plan = json.loads((benchmark / "benchmark_plan.json").read_text())
    status = json.loads((benchmark / "status.json").read_text())
    assert status["phase"] == "complete" and status["completed"] == 216
    assert plan["protocol"] == "fixed_epoch_loso_v2" and plan["seeds"] == [41, 47, 53]
    for relative, digest in plan["source_sha256"].items():
        assert sha(benchmark / "source_snapshot" / relative) == digest, relative
    for relative, digest in plan["data_sha256"].items():
        assert sha(Path(plan["data_root"]) / relative) == digest, relative
    settings = {
        "protocol": plan["protocol"], "seeds": plan["seeds"], "folds": plan["folds"],
        "benchmark": str(benchmark), "computation_sha256": sha(__file__),
        "checkpoint_manifest_sha256": sha(benchmark / "checkpoint_manifest.json"),
        "frozen_source_sha256": plan["source_sha256"],
        "stages": STAGES, "stage_scope": "Inference interventions on the frozen margin-trained parent; no retraining.",
        "jitter_rule": "Shift all internal Round-0 boundaries by -1,+1,-2,+2 grid steps; clamp to preserve nonempty segments.",
        "jitter_aggregation": "Mean L2 across frames, then signed shifts, seeds and subjects with equal subject weight.",
        "main_f1": "Preserve original figure definition: concatenate 18 subjects within each seed, then average 3 seed scores.",
        "main_map": "Average the 3 seed scores per subject, then average 18 subjects.",
        "timeline": {"fold": 2, "subject": "sbj_1", "seed": 47, "rule": "Same subject and seed as old figure, retained for direct comparison."},
        "gt_final": {"fold": 10, "subject": "sbj_9", "seed": 47, "rule": "Same subject and seed as old fourth figure."},
    }
    settings_path = output / "figure_computation_plan.json"
    if settings_path.exists():
        assert json.loads(settings_path.read_text()) == settings, "Computation settings changed; use a new folder."
    else:
        dump(settings_path, settings)
    source = output / "source_data"
    source.mkdir(exist_ok=True)
    methods = ["VIDEO_ONLY", "EARLY_CONCAT", "FIXED_WINDOW_ATTENTION", "FINAL_MODEL"]
    raw, aggregates, tiou, metric_sources = [], [], [], []
    for method in methods:
        summaries = []
        for seed in plan["seeds"]:
            path = benchmark / "metrics" / f"{method}_seed_{seed}.json"
            saved = json.loads(path.read_text())
            assert saved["n_sequences"] == 18
            assert {r["fold"] for r in saved["per_subject"]} == set(range(1, 19))
            assert all(r["seed"] == seed and r["method"] == method and r["protocol"] == plan["protocol"]
                       for r in saved["per_subject"])
            summaries.append(saved)
            raw.extend(saved["per_subject"])
            metric_sources.append({"path": str(path), "sha256": sha(path)})
        aggregates.append({"method": method,
            "macro_f1_concat": float(np.mean([x["concatenated"]["macro_f1"] for x in summaries])),
            **{metric: float(np.mean([x["mean_fold"][metric]["mean"] for x in summaries]))
               for metric in ("macro_f1_19", "map_at_0.5", "avg_map", "accuracy")}})
        for threshold in (0.3, 0.4, 0.5, 0.6, 0.7):
            tiou.append({"method": method, "tiou": threshold,
                         "map": float(np.mean([x["mean_fold"][f"map_at_{threshold:.1f}"]["mean"] for x in summaries]))})
    df = pd.DataFrame(raw)
    df.to_csv(source / "primary_per_subject_seed.csv", index=False)
    df.groupby(["method", "fold", "subject"])[["macro_f1_19", "map_at_0.5", "avg_map"]].mean().reset_index().to_csv(
        source / "primary_subject_means.csv", index=False)
    write_csv(source / "main_aggregate.csv", aggregates)
    write_csv(source / "map_by_tiou.csv", tiou)
    dump(source / "primary_metric_sources.json", metric_sources)
    # Keep the original subjects; no selection based on new final-model scores.
    timeline, line_rows, prediction_sources = [], [], []
    for method in methods:
        path = benchmark / "predictions" / method / "seed_47/split_02.npz"
        with np.load(path, allow_pickle=False) as p:
            assert p["id"].item() == "sbj_1" and p["seed"].item() == 47
            if method == methods[0]:
                reference_truth = p["true"].copy()
                timeline.extend({"time_seconds": i * .5, "track": "GT", "label": int(y)} for i, y in enumerate(reference_truth))
            else:
                assert np.array_equal(reference_truth, p["true"])
            timeline.extend({"time_seconds": i * .5, "track": method, "label": int(y)} for i, y in enumerate(p["pred"]))
        prediction_sources.append({"path": str(path), "sha256": sha(path)})
    path = benchmark / "predictions/FINAL_MODEL/seed_47/split_10.npz"
    with np.load(path, allow_pickle=False) as p:
        assert p["id"].item() == "sbj_9" and p["seed"].item() == 47
        line_rows = [{"time_seconds": i * .5, "gt": int(y), "final": int(z)} for i, (y, z) in enumerate(zip(p["true"], p["pred"]))]
    prediction_sources.append({"path": str(path), "sha256": sha(path)})
    write_csv(source / "temporal_example_sbj_1_seed47.csv", timeline)
    write_csv(source / "gt_final_sbj_9_seed47.csv", line_rows)
    dump(source / "qualitative_prediction_sources.json", prediction_sources)

    completed = []
    with ProcessPoolExecutor(max_workers=args.cpu_workers) as cpu, ProcessPoolExecutor(max_workers=1) as mps:
        futures = []
        for seed in plan["seeds"]:
            for fold in plan["folds"]:
                executor = mps if plan["fold_devices"][str(fold)] == "mps" else cpu
                futures.append(executor.submit(diagnostic_job, benchmark, output, seed, fold))
        for future in as_completed(futures):
            row = future.result()
            completed.append(row)
            dump(output / "progress.json", {"completed": len(completed), "total": 54,
                                           "latest_seed": row["seed"], "latest_fold": row["fold"]})
            print(f'Diagnostic {len(completed):02d}/54 seed={row["seed"]} fold={row["fold"]:02d} device={row["device"]} seconds={row["seconds"]:.1f}', flush=True)
    completed.sort(key=lambda x: (x["seed"], x["fold"]))
    assert len({(r["seed"], r["fold"]) for r in completed}) == 54
    abl, stage_rows, robust_rows, support_rows = [], [], [], []
    for stage in STAGES:
        concat_scores = []
        for seed in plan["seeds"]:
            matrices = [next(s["confusion_matrix"] for s in r["stages"] if s["stage"] == stage)
                        for r in completed if r["seed"] == seed]
            concat_scores.append(cm_f1(np.asarray(matrices).sum(0)))
        local = []
        for r in completed:
            score = next(s["macro_f1_19"] for s in r["stages"] if s["stage"] == stage)
            local.append(score)
            stage_rows.append({"stage": stage, "seed": r["seed"], "fold": r["fold"], "subject": r["subject"], "macro_f1_19": score})
        abl.append({"stage": stage, "macro_f1_concat": float(np.mean(concat_scores)),
                    "macro_f1_19": float(np.mean(local)),
                    **{f"concat_seed_{seed}": score for seed, score in zip(plan["seeds"], concat_scores)}})
    final_aggregate = next(x for x in aggregates if x["method"] == "FINAL_MODEL")
    assert abs(abl[-1]["macro_f1_concat"] - final_aggregate["macro_f1_concat"]) < 1e-10
    assert abs(abl[-1]["macro_f1_19"] - final_aggregate["macro_f1_19"]) < 1e-10
    for row in completed:
        identity = {k: row[k] for k in ("seed", "fold", "subject")}
        robust_rows.extend({**identity, **r} for r in row["robustness"])
        support_rows.append({**identity, **row["support"]})
    robust = pd.DataFrame(robust_rows)
    robust.to_csv(source / "boundary_jitter_per_subject_seed_shift.csv", index=False)
    robust.groupby(["method", "jitter_seconds"])["context_l2"].mean().reset_index().to_csv(source / "boundary_robustness.csv", index=False)
    support = pd.DataFrame(support_rows)
    support.to_csv(source / "support_per_subject_seed.csv", index=False)
    write_csv(source / "support.csv", [
        {"region": region, "prior": prior, "empirical_mass": float(support[key].mean()) if key else 0.0}
        for region, prior, key in [("Left margin", .5, "left_margin_attention_mass"),
                                  ("Core", 1., "core_attention_mass"),
                                  ("Right margin", .5, "right_margin_attention_mass"),
                                  ("Outside", 0., None)]])
    write_csv(source / "ablation.csv", abl)
    write_csv(source / "ablation_per_subject_seed.csv", stage_rows)
    dump(output / "calculation_checks.json", {
        "complete": True, "diagnostic_checkpoint_pairs": 54,
        "replay_checks": sum(len(r["replay_checks"]) for r in completed),
        "maximum_replay_probability_difference": max(c["max_probability_difference"] for r in completed for c in r["replay_checks"]),
        "all_replay_classes_equal": all(c["classes_equal"] for r in completed for c in r["replay_checks"]),
        "main_summary": aggregates, "ablation_summary": abl,
        "figure_data_sha256": {str(p.relative_to(output)): sha(p) for p in sorted(source.glob('*')) if p.is_file()},
    })
    dump(source / "manifest.json", {
        "protocol": plan["protocol"], "seeds": plan["seeds"], "n_subjects": 18,
        "metric_definitions": {"concatenated_macro_f1": settings["main_f1"], "mean_map": settings["main_map"]},
        "attention": {"round0_radius_seconds": 2.0, "round1_margin_seconds": 1.0,
                      "margin_prior": 0.5, "probe_beta": 0.5},
        "diagnostic_scope": settings["stage_scope"],
        "boundary_jitter": settings["jitter_rule"],
        "qualitative_examples": {"timeline": settings["timeline"], "gt_final": settings["gt_final"]},
        "sha256": {p.name: sha(p) for p in sorted(source.glob("*.csv"))},
    })
    print("All figure data computed and checked.", flush=True)


if __name__ == "__main__":
    main()
