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

The current results use `fixed_epoch_loso_v2` on `sbj_0`–`sbj_17`, with 18
leave-one-subject-out folds and seeds 41, 47 and 53. There are 216 training runs:
18 subjects × 3 seeds × 4 methods. Inputs are 2048-D I3D video features and
600-D raw inertial windows on the 2 Hz feature grid. IMU normalization uses
only the 17 training subjects in each fold.

The parent is trained for 30 epochs and the background probe for 15; both use
the final epoch. Training and inference share the ±2-second Round-0 support
and predicted-segment-plus-±1-second Round-1 support with priors 1.0/0.5.
Held-out subjects are not loaded during training or used for checkpoint
selection. All training finishes before outer-test evaluation.

| Method | Subject Macro-F1 (mean ± SD) | Concatenated Macro-F1 | Background F1 | Action Macro-F1 | Accuracy | mAP@0.5 | Avg mAP |
|---|---:|---:|---:|---:|---:|---:|---:|
| Video-only MS-TCN | 0.6419 ± 0.1246 | 0.6903 | 0.8171 | 0.6322 | 0.7461 | 0.6057 | 0.6061 |
| Early concatenation | 0.7048 ± 0.2110 | 0.7443 | 0.8153 | 0.6987 | 0.7734 | 0.6811 | 0.6776 |
| Fixed-window attention | 0.7246 ± 0.2071 | 0.7635 | 0.8133 | 0.7197 | 0.7843 | 0.6880 | 0.6880 |
| Final multimodal model | 0.7500 ± 0.1892 | 0.7748 | 0.8327 | 0.7454 | 0.8008 | 0.7154 | 0.7148 |

### Metric aggregation

- **Subject mean ± SD:** average the three seed scores within each subject,
  then compute the mean and sample SD (`ddof=1`) across the 18 subject means.
  The SD describes between-subject variation, not a confidence interval.
- **Concatenated Macro-F1:** concatenate all 18 held-out sequences within each
  seed, compute Macro-F1 over the fixed 19-class space, then average the three
  seed scores. The main and component figures use this definition. It differs
  from the subject-mean F1 because F1 is nonlinear and sequence lengths vary.
- Background F1, action Macro-F1, accuracy, mAP@0.5 and Avg mAP average subjects
  and seeds equally. Macro-F1 includes background (class 18); action Macro-F1
  covers classes 0–17. Per-subject scoring uses F1 = 0 for an absent class.
  The concatenated evaluator retains the official `zero_division=1` setting;
  all 19 classes occur in each pooled seed, so this does not affect these
  concatenated scores.
- TAL scoring converts contiguous frame predictions into segments. Avg mAP is
  the arithmetic mean over tIoU 0.3, 0.4, 0.5, 0.6 and 0.7.

The complete 216-row source is
[`wear_18fold_metrics.csv`](../results/wear_18fold_metrics.csv). The compact
[`wear_summary.csv`](../results/wear_summary.csv) and schema-v2
[`wear_aggregate.json`](../results/wear_aggregate.json) provide method means,
subject SDs, seed means and concatenated F1. Regenerate the tables with
`scripts/summarize_wear_results.py` from a completed benchmark. These files and
the figures describe the current protocol; older WEAR scores are retained in
Git history.

### Interpretation and scope

The final model has higher descriptive means than all three baselines. Relative
to fixed attention, subject-mean Macro-F1 is higher by 2.54 percentage points
and mAP@0.5 by 2.74 points. These differences do not establish statistical
superiority over fixed attention in the matched subject-level comparisons.
The provided statistics script averages seed differences within each subject
and applies Holm correction across three baselines × two metrics. These are
exploratory comparisons: LOSO training sets overlap and there are only 18
subjects and three seeds. Reproduction commands are in
[reproduction.md](reproduction.md).

Baselines use one round and no background probe, while the final recipe uses
two rounds and a probe. The main comparison therefore measures the whole
recipe rather than an isolated DWA window effect. These are 2 Hz feature-grid
results, not official 50 Hz record scores or official ActionFormer/TriDet
output evaluation. Fixed epoch selection removes held-out checkpoint selection
from this run; it does not create a new dataset independent of prior method
development. Differences from historical results also reflect changed training
and attention settings and should not be attributed to a single change.

### Component and boundary diagnostics

The component figure evaluates the same 54 frozen, margin-trained parents.
Only inference changes; no component is retrained or selected by its test score.

| Output | Concatenated Macro-F1 | Subject-mean Macro-F1 |
|---|---:|---:|
| Round 0 | 0.7680 | 0.7395 |
| Round 1, segment only (zero margin) | 0.7724 | 0.7463 |
| Round 1, margin and prior (DWA parent) | 0.7729 | 0.7471 |
| Final, with background probe | 0.7748 | 0.7500 |

The small increases are descriptive outputs of shared weights. In particular,
zero-margin inference changes the support of a model trained with a margin;
this is not a separately trained zero-margin baseline or a causal ablation.

Boundary perturbations shift every internal Round-0 boundary by −1, +1, −2 or
+2 feature steps (0.5 seconds per step), clamping the boundaries to preserve
nonempty segments. Context L2 change is averaged across frames, signed shifts,
seeds and subjects. Lower values mean less context change, not necessarily
better segmentation accuracy under jitter.

| Support | ±0.5-second jitter | ±1.0-second jitter |
|---|---:|---:|
| Hard segment | 0.4782 | 0.8437 |
| Boundary uncertainty | 0.4165 | 0.7426 |

| Region | Support prior | Mean attention mass |
|---|---:|---:|
| Left margin | 0.5 | 0.010753 |
| Core | 1.0 | 0.978635 |
| Right margin | 0.5 | 0.010611 |
| Outside | 0.0 | 0.000000 |

Attention mass averages frames within each sequence, then subjects and seeds
equally. Most mass remains in the predicted core; the support prior is a
multiplicative attention weight, not a prescribed mass fraction.

### Figures and numerical inputs

All four PNGs have corresponding committed CSV inputs and a hash manifest in
[`results/wear_figure_data/`](../results/wear_figure_data/). Redraw them with
`scripts/plot_wear_results.py`; no checkpoints or dataset access are needed.

- [`wear_main_results.png`](../assets/wear_main_results.png): concatenated F1,
  mean mAP, paired subject-level mAP changes and the tIoU sweep.
- [`wear_ablation_robustness.png`](../assets/wear_ablation_robustness.png):
  frozen-parent component outputs, boundary jitter and support mass.
- [`wear_temporal_segmentation.png`](../assets/wear_temporal_segmentation.png):
  `sbj_1`, fold 2, seed 47; ground truth and all four models.
- [`wear_gt_vs_final.png`](../assets/wear_gt_vs_final.png): `sbj_9`, fold 10,
  seed 47; ground truth and final-model class IDs (18 is background).

Both qualitative subjects and seed 47 are retained from the previous figures.
They are not selected or described as median cases under the new results.
