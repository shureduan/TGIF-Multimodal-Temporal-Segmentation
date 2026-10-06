#!/usr/bin/env python3
"""Redraw the four WEAR v2 figures from hash-verified CSV inputs."""
from pathlib import Path
import hashlib
import json
import argparse

REPO = Path(__file__).resolve().parents[1]
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

SOURCE = REPO / "results/wear_figure_data"
OUTPUT = REPO / "outputs/wear_figures"
PDF = None
METHODS = ["VIDEO_ONLY", "EARLY_CONCAT", "FIXED_WINDOW_ATTENTION", "FINAL_MODEL"]
LABELS = dict(zip(METHODS, ["Video-only", "Early concat.", "Fixed attention", "Final model"]))
COLORS = dict(zip(METHODS, ["#4C78A8", "#F58518", "#54A24B", "#D1495B"]))
FIGURE_NAMES = ["wear_main_results", "wear_ablation_robustness", "wear_temporal_segmentation", "wear_gt_vs_final"]


def style():
    plt.rcParams.update({
        "font.family": "serif", "font.size": 9, "axes.titlesize": 10,
        "axes.labelsize": 9, "legend.fontsize": 7.5, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight",
    })


def save(fig, name):
    (OUTPUT / "figures").mkdir(parents=True, exist_ok=True)
    (OUTPUT / "pdfs").mkdir(exist_ok=True)
    fig.savefig(OUTPUT / "figures" / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT / "pdfs" / f"{name}.pdf", bbox_inches="tight")
    PDF.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main_results():
    aggregate = pd.read_csv(SOURCE / "main_aggregate.csv").set_index("method")
    subjects = pd.read_csv(SOURCE / "primary_subject_means.csv")
    tiou = pd.read_csv(SOURCE / "map_by_tiou.csv")
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.4), gridspec_kw={"width_ratios": [1.05, 1.45, 1.05]})
    metrics = ["macro_f1_concat", "map_at_0.5", "avg_map"]
    x = np.arange(3)
    for i, method in enumerate(METHODS):
        axes[0].bar(x + (i - 1.5) * .19, aggregate.loc[method, metrics].to_numpy(float), .19,
                    label=LABELS[method], color=COLORS[method])
    lower = max(0, np.floor(aggregate[metrics].to_numpy().min() * 20) / 20 - .05)
    upper = min(1, np.ceil(aggregate[metrics].to_numpy().max() * 20) / 20 + .025)
    axes[0].set_ylim(lower, upper)
    axes[0].set_xticks(x, ["Macro-F1\n(concat.)", "mAP@0.5", "Avg. mAP"])
    axes[0].set_ylabel("Score (3-seed mean)")
    axes[0].set_title("(a) Aggregate performance", fontweight="bold")
    axes[0].grid(axis="y", alpha=.25)
    final = subjects[subjects.method == "FINAL_MODEL"].sort_values("fold")
    for baseline, offset in zip(METHODS[:-1], [-.22, 0., .22]):
        base = subjects[subjects.method == baseline].sort_values("fold")
        assert np.array_equal(final.fold, base.fold)
        axes[1].bar(np.arange(18) + offset,
                    final["map_at_0.5"].to_numpy() - base["map_at_0.5"].to_numpy(),
                    .21, color=COLORS[baseline])
    axes[1].axhline(0, color="black", linewidth=.8)
    axes[1].set_xticks(np.arange(18), [f"s{i}" for i in range(18)], rotation=90)
    axes[1].set_ylabel("Final - baseline: paired $\\Delta$ mAP@0.5")
    axes[1].set_title("(b) Cross-subject differences", fontweight="bold")
    axes[1].grid(axis="y", alpha=.2)
    for method in METHODS:
        q = tiou[tiou.method == method].sort_values("tiou")
        axes[2].plot(q.tiou, q["map"], marker="o", linewidth=1.8, color=COLORS[method])
    axes[2].set_xlabel("Temporal IoU threshold")
    axes[2].set_ylabel("mAP (subject/seed mean)")
    axes[2].set_xticks([.3, .4, .5, .6, .7])
    axes[2].set_title("(c) Localization strictness", fontweight="bold")
    axes[2].grid(alpha=.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(.5, .048))
    fig.text(.5, .012, "fixed_epoch_loso_v2 | 18 subjects, seeds 41/47/53 | F1 concatenates subjects within each seed; mAP averages subjects.",
             ha="center", fontsize=7.5)
    fig.tight_layout(rect=(0, .15, 1, 1))
    save(fig, "wear_main_results")


def ablation():
    abl = pd.read_csv(SOURCE / "ablation.csv")
    robustness = pd.read_csv(SOURCE / "boundary_robustness.csv")
    support = pd.read_csv(SOURCE / "support.csv")
    assert abl.stage.tolist() == ["ROUND0", "SEGMENT_ONLY", "DWA_ONLY", "FINAL_MODEL"]
    fig, axes = plt.subplots(1, 3, figsize=(13.7, 4.35))
    values = abl.macro_f1_concat.to_numpy()
    padding = max(.0025, float(np.ptp(values)) * .18)
    axes[0].plot(range(4), values, marker="o", linewidth=2, color=COLORS["FINAL_MODEL"])
    for i, value in enumerate(values):
        axes[0].annotate(f"{value:.4f}", (i, value), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
    axes[0].set_xticks(range(4), ["Round 0", "Segment only", "+ Margin/prior", "+ Probe"], rotation=18)
    axes[0].set_ylim(values.min() - padding, values.max() + 1.6 * padding)
    axes[0].set_ylabel("Concat. Macro-F1 (3-seed mean)")
    axes[0].set_title("(a) Frozen-parent component outputs", fontweight="bold")
    axes[0].grid(axis="y", alpha=.25)
    for i, (method, color) in enumerate([("Hard segment", "#888888"), ("Boundary uncertainty", COLORS["FINAL_MODEL"])]):
        q = robustness[robustness.method == method].set_index("jitter_seconds").loc[[.5, 1.]]
        axes[1].bar(np.arange(2) + (i - .5) * .34, q.context_l2, .34, label=method, color=color)
    axes[1].set_xticks(np.arange(2), ["±0.5 s", "±1.0 s"])
    axes[1].set_ylabel("Context L2 change (lower is better)")
    axes[1].set_title("(b) Boundary-jitter robustness", fontweight="bold")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=.25)
    q = support.set_index("region").loc[["Left margin", "Core", "Right margin", "Outside"]]
    assert np.allclose(q.prior, [.5, 1., .5, 0.])
    assert abs(q.empirical_mass.sum() - 1) < 1e-5
    axes[2].bar(np.arange(4) - .18, q.prior, .36, label="Support prior", color="#E6B655")
    axes[2].bar(np.arange(4) + .18, q.empirical_mass, .36, label="Attention mass", color=COLORS["FINAL_MODEL"])
    axes[2].set_xticks(np.arange(4), ["Left\nmargin", "Core", "Right\nmargin", "Outside"])
    axes[2].set_ylim(0, 1.08)
    axes[2].set_title("(c) Soft support allocation", fontweight="bold")
    axes[2].legend(frameon=False)
    axes[2].grid(axis="y", alpha=.25)
    fig.text(.5, .015, "54 frozen v2 parents; inference-only interventions, no ablation retraining. Jitter and attention mass average subjects and seeds equally.",
             ha="center", fontsize=7.5)
    fig.tight_layout(rect=(0, .075, 1, 1))
    save(fig, "wear_ablation_robustness")


def timeline():
    data = pd.read_csv(SOURCE / "temporal_example_sbj_1_seed47.csv")
    tracks = ["GT", *METHODS]
    arrays = [data[data.track == track].sort_values("time_seconds").label.to_numpy(int) for track in tracks]
    assert len({len(x) for x in arrays}) == 1
    array = np.stack(arrays)
    palette = plt.get_cmap("tab20")(np.linspace(0, 1, 19))
    fig, ax = plt.subplots(figsize=(12, 2.95))
    ax.imshow(array, aspect="auto", interpolation="nearest", cmap=ListedColormap(palette), vmin=0, vmax=18)
    ax.set_yticks(range(5), ["Ground truth", *[LABELS[m] for m in METHODS]])
    ax.set_xticks(np.linspace(0, array.shape[1] - 1, 7), [f"{x:.0f}" for x in np.linspace(0, data.time_seconds.max(), 7)])
    ax.set_xlabel("Time (s)")
    ax.set_title("WEAR temporal segmentation | sbj_1, seed 47", fontweight="bold")
    for y in np.arange(.5, 5, 1):
        ax.axhline(y, color="white", linewidth=1.2)
    fig.text(.5, .015, "fixed_epoch_loso_v2 | Same subject and seed as the original example | Colors retain the original 19-class mapping.",
             ha="center", fontsize=7.5)
    fig.tight_layout(rect=(0, .085, 1, 1))
    save(fig, "wear_temporal_segmentation")


def gt_final():
    data = pd.read_csv(SOURCE / "gt_final_sbj_9_seed47.csv")
    with plt.rc_context(rc=plt.rcParamsDefault):
        plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.plot(data["time_seconds"], data["gt"], label="GT")
        ax.plot(data["time_seconds"], data["final"], label="Final")
        ax.set_title("WEAR sbj_9 (fold 10), seed 47 | GT vs Final")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Class ID")
        ax.set_yticks([0, 3, 6, 9, 12, 15, 18])
        ax.legend()
        fig.text(.5, .015, "fixed_epoch_loso_v2 | Same subject as the original GT/Final plot; class 18 is background.",
                 ha="center", fontsize=8)
        fig.tight_layout(rect=(0, .045, 1, 1))
        save(fig, "wear_gt_vs_final")


def main():
    global SOURCE, OUTPUT, PDF
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    SOURCE, OUTPUT = args.data_root.resolve(), args.output.resolve()
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    if manifest["protocol"] != "fixed_epoch_loso_v2":
        raise ValueError("Expected fixed_epoch_loso_v2 figure inputs")
    expected = {
        "main_aggregate.csv", "primary_subject_means.csv", "map_by_tiou.csv",
        "ablation.csv", "ablation_per_subject_seed.csv", "boundary_robustness.csv",
        "boundary_jitter_per_subject_seed_shift.csv", "support.csv",
        "support_per_subject_seed.csv", "temporal_example_sbj_1_seed47.csv",
        "gt_final_sbj_9_seed47.csv",
    }
    if not expected.issubset(manifest["sha256"]):
        raise ValueError("Figure input manifest is incomplete")
    for relative in sorted(expected):
        digest = hashlib.sha256((SOURCE / relative).read_bytes()).hexdigest()
        if digest != manifest["sha256"][relative]:
            raise ValueError(f"Figure input hash mismatch: {relative}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    style()
    output = OUTPUT / "WEAR_new_results_figures.pdf"
    with PdfPages(output) as PDF:
        main_results()
        ablation()
        timeline()
        gt_final()
    hashes = {str(p.relative_to(OUTPUT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in [*[OUTPUT / "figures" / f"{name}.png" for name in FIGURE_NAMES], output]}
    (OUTPUT / "figure_outputs.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print(f"Generated four PNGs and PDFs, plus {output}")


if __name__ == "__main__":
    main()
