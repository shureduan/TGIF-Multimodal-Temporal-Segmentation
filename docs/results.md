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

The WEAR results are from the first 18 subjects (`sbj_0`–`sbj_17`), leave-one-
subject-out folds, and seed 47. Inputs are the 2 Hz official-style feature grid:
2048-D I3D video features and 600-D raw inertial windows. Fold-specific IMU
normalization uses the 17 training subjects only.

| Method | Macro-F1 mean ± std | Concatenated 19-class Macro-F1 | Background F1 | Action Macro-F1 | Accuracy | mAP@0.5 | Avg mAP |
|---|---:|---:|---:|---:|---:|---:|---:|
| Video only | 0.7292 ± 0.0861 | 0.7573 | 0.8328 | 0.7234 | 0.7929 | 0.6755 | 0.6814 |
| Early concatenation | 0.7820 ± 0.1888 | 0.7976 | 0.8272 | 0.7795 | 0.8092 | 0.7546 | 0.7506 |
| Fixed-window attention | 0.7911 ± 0.1818 | 0.8092 | 0.8435 | 0.7882 | 0.8238 | 0.7635 | 0.7568 |
| Final model | 0.7998 ± 0.1621 | 0.8109 | 0.8509 | 0.7970 | 0.8284 | 0.7756 | 0.7640 |

The mean and population standard deviation are computed across the 18 folds.
“Concatenated” scores the concatenated fold predictions over the fixed 19-class
space, including background. The per-fold source table is
`results/wear_18fold_metrics.csv`; the frozen aggregate is
`results/wear_aggregate.json`.

Important scope limits:

- The held-out subject in each fold is also used for best-epoch selection, so
  these are validation-selected LOSO results rather than untouched test scores.
- The record results are aligned with the official 19-class mapping but operate
  on the 2 Hz feature grid. They are not a reproduction of the official 50 Hz
  record evaluator.
- The TAL mAP converts contiguous MS-TCN labels into segments and averages over
  tIoU 0.3–0.7. It is not an evaluation of official ActionFormer or TriDet
  detector outputs.
- One training seed is available; no multi-seed WEAR claim is made.

The README uses three approved result figures:

- `assets/wear_main_results.png` summarizes aggregate performance,
  subject-level paired changes in mAP@0.5, and localization performance across
  tIoU thresholds.
- `assets/wear_ablation_robustness.png` shows incremental components,
  boundary-jitter robustness, and soft support allocation.
- `assets/wear_temporal_segmentation.png` is a qualitative 2,776-second trace
  with Ground truth, Video-only, Early concatenation, Fixed attention, and the
  Final model under a shared class-color mapping.

The frozen tables above remain the source for the exact values quoted in text.
