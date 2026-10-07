# Results and evaluation scope

## TGIF sewing dataset

The TGIF homepage values are transcribed from the annotations in
`assets/tgif_video_vs_multimodal.png`, as confirmed by the project owner. They
are not estimated from bar heights and were not recomputed in this release pass.

| Metric | Video-only (%) | Multimodal (%) | Change (percentage points) |
|---|---:|---:|---:|
| Accuracy | 91.9 | 92.5 | +0.6 |
| Edit score | 89.7 | 91.4 | +1.7 |
| Macro-F1 | 92.3 | 92.7 | +0.4 |
| Idle recall | 93.6 | 94.2 | +0.6 |
| Sewing recall | 91.8 | 92.3 | +0.5 |
| Handling recall | 87.7 | 88.2 | +0.5 |

The source figure contains error bars, but their statistical definition is not
available in the release materials. This document therefore does not assign a
seed count, standard deviation, confidence interval, or significance test to
them. It also does not add F1@50 because that value is not present in the
specified figure.

The timeline `assets/tgif_timeline_seed43.png` is explicitly labeled seed 43 and
“val good-vib, concatenated.” It contains two time panels, each approximately
0–900 seconds, with GT, pure-video, and multimodal tracks for idle, sewing, and
handling. It is used as a qualitative prediction trace only; no specific
boundary correction is inferred from the image.

These figures are the release's authoritative TGIF display evidence. They retain
the method names Video-only and Multimodal. Runtime validation of the maintained
model code is reported separately and does not alter these confirmed results.

## WEAR

The current release adds sensor-driven DWA to four fixed-epoch reference models.
The main comparison is complete for `sbj_0`–`sbj_17`, seeds 41/47/53: 54 DWA
parent/probe runs and 216 reference runs. Prepared inputs are 2048-D I3D and
600-D raw IMU at 2 Hz. Parent/probe budgets are 30/15 epochs; each saves the final
epoch. Normalization and controller calibration use the 17 training subjects.
All released checkpoints completed training before their outer-test evaluation.

| Method | Subject Macro-F1 (mean ± SD) | Concatenated Macro-F1 | Accuracy | mAP@0.5 | Avg mAP |
|---|---:|---:|---:|---:|---:|
| Video-only MS-TCN | 0.6419 ± 0.1246 | 0.6903 | 0.7461 | 0.6057 | 0.6061 |
| Early concatenation | 0.7048 ± 0.2110 | 0.7443 | 0.7734 | 0.6811 | 0.6776 |
| Fixed-window attention | 0.7246 ± 0.2071 | 0.7635 | 0.7843 | 0.6880 | 0.6880 |
| Segment-guided Window Attention (SWA) | 0.7500 ± 0.1892 | 0.7748 | 0.8008 | 0.7154 | **0.7148** |
| **Sensor-driven DWA (final)** | **0.7520 ± 0.1782** | **0.7811** | **0.8040** | **0.7188** | 0.7147 |

DWA has the best overall performance across the main table, leading both F1
aggregations, accuracy and mAP@0.5. It increases subject-mean Macro-F1 and mAP@0.5 over fixed attention by 2.74 and
3.08 percentage points. Its accuracy is 0.8040, versus 0.7843 for fixed attention
and 0.8008 for the v2 SWA reference. These are descriptive means
of the entire model recipes. The highest mean in each column is bolded.

### Metrics and aggregation

- Subject scores: average three seeds within each subject; then compute the
  mean and sample SD (`ddof=1`) over subjects. Error bars show subject SD.
- Concatenated Macro-F1: concatenate subject predictions within each seed,
  compute a 19-class F1, then average the three scores. This weights sequence
  lengths differently from subject-mean F1.
- Frame precision/recall/F1 use all 19 labels and `zero_division=0`; background
  is class 18. Action Macro-F1 averages classes 0–17.
- Accuracy and TAL mAP weight subjects and seeds equally. TAL converts contiguous
  predictions to temporal segments. Avg mAP averages tIoU 0.3/0.4/0.5/0.6/0.7.
  These are 2 Hz feature-grid scores, not official 50 Hz record or detector scores.

All 432 method/subject/seed metric records, 12-metric summaries, class scores,
seed means and window measurements are in
[`results/wear_signal_v3/`](../results/wear_signal_v3/). The 432 include frozen
Round-0 and parent outputs; they are not 432 independently trained models.
The four baseline values are recalculated from the original v2 predictions.
Older v2-only sources remain under `results/wear_figure_data/` and
`results/wear_18fold_metrics.csv`, with their original protocol unchanged.

### Retrained component study

| Variant | Subjects × seeds | Macro-F1 | Accuracy | mAP@0.5 | Avg mAP |
|---|---:|---:|---:|---:|---:|
| No sensor resize | 9 × 3 | 0.7598 | 0.8006 | 0.7294 | 0.7307 |
| No contraction | 9 × 3 | 0.7477 | 0.7872 | 0.7254 | 0.7205 |
| Full DWA | 9 × 3 | 0.7672 | 0.8075 | 0.7443 | 0.7372 |

Each row uses the same nine subjects (`sbj_0,2,...,16`) and all three seeds.
Disabling resizing preserves both rounds, learned attention, MS-TCN, loss,
optimizer, normalization and separately trained probe. Full DWA has higher
mean F1 and mAP@0.5 on this cohort. The one-round fixed-attention baseline is
not substituted for this matched component control.

This is the completed component subset of the extended five-variant study.
The cohort was fixed by checkpoint availability before new test evaluation;
`NO_EXPANSION` and `UNIFORM_POOL` are not included as completed results.
The release manifest records that evaluation amendment. Their training options
remain available for extending the study. Full-model results use all 18 subjects;
the nine-subject component means must be compared within their own table.

### Frozen-model outputs and window behavior

Round 0, Round 1 without the probe, and final output share each full model's
weights. Their 18-subject Macro-F1 values are 0.7505, 0.7496 and 0.7520;
these are output diagnostics, not separately retrained architectures.

Window measurements confirm sensor-dependent resizing. Round 0 produces shorter,
equal-length and longer supports for 45.0%, 9.4% and 45.6% of queries; Round 1
produces 61.7%, 17.6% and 20.7%, averaged equally across subjects and seeds.
Equal length does not imply identical endpoints. The controller's statistical
stability flag is a sensor proxy defined in [method.md](method.md).

### Paired numerical comparisons

[`paired_statistics.json`](../results/wear_signal_v3/paired_statistics.json)
and its CSV version retain every comparison and both directions of difference.
Average seed differences within each subject before two-sided paired t tests;
95% t intervals are unadjusted. Holm families contain eight baseline tests,
eight planned retrained-control tests (unavailable tests assigned p=1), and four
frozen-output tests. The table includes the declared subject-exclusion check
for `sbj_0`, which was observed in the development pilot. These are exploratory
subject-level comparisons with overlapping LOSO training sets.

### WEAR figures

The original figure layout and qualitative subjects are retained, with the new
DWA added. No subject or seed was reselected using the new results.

| Figure | Contents |
|---|---|
| [Main comparison](../assets/wear_main_results.png) | Five models, paired subject differences and localization thresholds |
| [Components and windows](../assets/wear_ablation_robustness.png) | Nine-subject retrained controls; 18-subject frozen outputs and resize fractions |
| [Temporal segmentation](../assets/wear_temporal_segmentation.png) | `sbj_1`, seed 47, GT plus five prediction tracks |
| [GT and final models](../assets/wear_gt_vs_final.png) | `sbj_9`, seed 47, GT plus SWA and sensor-driven DWA |
| [All metrics](../assets/wear_all_metrics.png) | All 12 metrics, means and subject SD |
| [Classes and seeds](../assets/wear_class_and_seed.png) | Per-class F1 and seed-level subject means |

Run `python scripts/plot_signal_results.py` to create all six PNGs, individual
PDFs and a combined PDF without data access. The renderer checks input hashes.
Full checkpoint-to-result commands are in [reproduction.md](reproduction.md).
