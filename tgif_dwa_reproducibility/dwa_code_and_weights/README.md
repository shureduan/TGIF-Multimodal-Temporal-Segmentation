# TGIF Dynamic-Window Attention (DWA)

This folder contains the TGIF DWA inference implementation, five validation-split checkpoints, and a small runner for aligned, pre-extracted features. It does not include TGIF video/sensor data, the VideoMAE backbone weights, or the upstream feature-extraction pipeline.

## Contents

- `tgif_dwa/`: DWA model implementation and its required modules.
- `weights/seed_*_best.pt`: one checkpoint for each split seed (41, 11, 43, 17, 53).
- `weights/video_norm.npz`: the 1536-D video-feature normalization used by this experiment.
- `infer.py`: per-sequence inference CLI.
- `smoke_test.py`: checkpoint-load and masked-gap smoke test.
- `requirements.txt`: minimal runtime dependencies.

## Model flow

For each aligned timestamp, the input contains a 768-D full-frame VideoMAE embedding, a 768-D operator-ROI embedding, and six vibration descriptors.

1. Concatenate the two video embeddings and apply the packaged 1536-D z-score normalization.
2. A learned gated projection applies a global self-gate and an operator cross-gate, then maps 1536-D to 512-D.
3. Round 0 starts with a local window of approximately ±2 seconds. A deterministic vibration-statistics controller clips the candidate support at detected changes, contracts redundant regions when their statistics remain equivalent and stable, and can expand unstable/undersampled supports within the same signal region.
4. One-way attention uses video queries and vibration keys/values. The resulting 6-D weighted vibration context is concatenated with the 512-D projected video feature and passed through the shared four-stage MS-TCN.
5. The Round-0 prediction defines a segment around each frame. Round 1 uses that predicted segment as its seed, applies the same signal-driven resizing and attention, then reuses the same MS-TCN. Round-1 logits are the output. The TGIF idle probe has `beta=0` in these checkpoints and does not alter the logits.

The support controller reads vibration statistics before learned Q/K attention. It does not use labels, attention weights, or predicted confidence to resize a window. Invalid sensor frames are excluded from attention and form window boundaries.

## Dataset and feature preparation

The validation experiment uses 11 TGIF recording/crop IDs and divides each into six non-overlapping 5-minute segments: 66 segments total, 2,247 feature rows per segment at approximately 7.4933 rows/s. Labels are `idle=0`, `sewing=1`, `handling=2`; `-100` marks unannotated frames and is excluded from the reported metrics. The five split seeds are 41, 11, 43, 17, and 53. Each split has 60 training and 6 validation segments, selected by stratifying segment idle fractions. This is segment-level validation: segments from the same recording, and in some cases overlapping crops, can occur on opposite sides of a split. It is not a held-out-worker or held-out-recording test.

**Video features.** The supplied experiment used two pre-extracted VideoMAE streams: the whole-frame crop and a fixed per-video operator crop. Each stream is 768-D. The upstream extractor uses centered 16-frame clips sampled every 4 source frames (about 7.5 anchors/s); features, labels, and timestamps must have identical row order and length. The operator crop comes from the project’s saved automatic ROI configuration. This DWA package consumes those arrays and does not run YOLO or VideoMAE itself.

**Vibration features.** Six per-frame descriptors are used: `rms_dyn`, `log_e_20_40`, `ratio_20_40`, `ratio_20_30`, `ratio_30_40`, and `dom_20_40`. The upstream builder reconstructs a timestamped 500-Hz grid from tri-axial acceleration, computes magnitude, subtracts a centered 2-second rolling baseline, and aligns sensor time to video using the sewing-label correlation. It retains a recording when vibration availability exceeds 30% and the alignment correlation is at least 0.20. At each video anchor it uses a centered 2-second window and Hann-windowed FFT power. The six features are dynamic RMS; log10 20–40 Hz power; 20–40 Hz power divided by 0.5–150 Hz power; 20–30 Hz and 30–40 Hz power divided by the same total; and the strongest 20–40 Hz frequency normalized to [0,1]. The historical feature-builder replaces missing raw samples by zero inside a window when at least 30% of samples are present, then linearly interpolates remaining missing per-frame feature rows. Therefore, the included checkpoints and published comparison figures correspond to that historical preprocessing. The inference runner below does **not** interpolate: NaN feature rows and/or rows marked false in `--valid-mask` are excluded from attention. This no-interpolation inference behavior is supported by the architecture, but it is not the preprocessing used to train these checkpoints; changing the missing-data policy can change predictions and does not reproduce the included validation scores.

**Normalization.** Sensor mean/std are stored inside each checkpoint and were fit on valid training-split vibration rows. The packaged `video_norm.npz` is the fixed 1536-D normalization file used by the saved experiment and was reused for all five splits, rather than re-fit independently within each split. Its exact source-feature snapshot is not recoverable from this delivery. Treat these metrics as a reproduction of the saved experiment, not as a fold-pure estimate. Refit video normalization within each training split and retrain both methods before claiming a leakage-controlled result.

## Install and run

Use Python 3.10+ with PyTorch and NumPy. From this folder:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python smoke_test.py
```

Prepare three NumPy arrays with the same temporal alignment:

- `global_features.npy`: `[T,768]` whole-frame VideoMAE features.
- `operator_features.npy`: `[T,768]` operator-ROI VideoMAE features.
- `vibration_features.npy`: `[T,6]` vibration descriptors; use NaN for rows with missing vibration data, or provide a separate boolean mask.

Run one saved split checkpoint (the default is seed 41):

```bash
python infer.py \
  --global-features inputs/global_features.npy \
  --operator-features inputs/operator_features.npy \
  --vibration-features inputs/vibration_features.npy \
  --seed 41 \
  --device cpu \
  --output predictions.csv
```

For explicit gap handling, supply `--valid-mask inputs/vibration_valid.npy`, a boolean `[T]` array. A row is considered valid only when the mask is true and all six vibration values are finite. Invalid values are never interpolated. Use `--device auto` to select CUDA, Apple MPS, or CPU when available.

The output CSV has one row per feature timestamp with frame index, time in seconds, sensor-valid flag, predicted class, and three class probabilities. To reproduce the saved validation predictions, run each 5-minute segment independently; running a whole recording as one sequence can change context near the segment boundaries. The five checkpoints are alternative split-specific models, not an ensemble. They were selected by validation F1@0.5; choose the checkpoint corresponding to the split you intend to inspect. They are research validation checkpoints, not a single model trained on all 66 segments.

## Reproducing the included comparison

The image-only comparison folder next to this code folder contains three figures: `primary_metrics_bar_comparison.png`, `confusion_matrices_pooled_validation.png`, and `long_temporal_comparison_seed43_station3_30min.png`. The bar chart reports the mean and population SD across five split-level metrics; within each split, metrics are first averaged over its six validation segments. Its vertical axis starts at 60%. The confusion matrices pool valid validation frames across the same five paired split seeds, retaining repeated segment appearances. The 30-minute temporal display shows station3 under seed 43, with class colors Sewing=red, Handling=blue, and Idle=green. Seed 43’s validation accuracy was 90.82% for Video Method and 92.24% for Multimodal Fusion. The displayed held-out `station3_seg02` predictions exactly match the saved paired validation predictions; its ground truth includes an approximately 43-second continuous Idle segment. Other 5-minute segments in the full timeline were used for training, so this figure is a qualitative full-sequence view rather than a fully held-out 30-minute evaluation. The numerical workbook contains the same six metrics.

This is a descriptive comparison, not an isolated causal ablation: DWA uses two video ROIs plus vibration and starts from transferred WEAR MS-TCN temporal layers; the full-frame video baseline uses only global video features and starts from scratch. The validation subsets are small and are not independent external test data.

## Model license and data

The checkpoints and implementation are provided for research use with this project. TGIF source video, vibration recordings, labels, VideoMAE backbone weights, and operator-ROI configuration are not bundled. Follow the dataset and upstream model terms before redistributing or using those assets.
