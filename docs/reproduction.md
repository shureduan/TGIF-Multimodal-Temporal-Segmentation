# Reproducing the WEAR v2 code

The current training protocol is `fixed_epoch_loso_v2`. It trains from prepared
WEAR features and saves the final epoch of each component. The WEAR figures in
the README and tables in `results/` use the completed 18-fold, three-seed run
under this protocol. Metric definitions and scope are in [results.md](results.md).
The checkpoint manifest at `models/wear_final/manifest.json` belongs to an older
bundle and does not reproduce these current scores. To run the published
pretrained weights without retraining, follow [models.md](models.md).

## 1. Install and check inputs

Run all commands from the repository root. The pinned environment uses Python
3.10 and the versions in `environment.yml` / `pyproject.toml`:

```bash
conda env create -f environment.yml
conda activate tgif-temporal
python -m unittest discover -s tests -v
```

Alternatively, in a Python 3.10 environment, use `python -m pip install -e .`.
The tests run on CPU with synthetic inputs; the optional checkpoint test is
skipped unless `WEAR_PARENT_CKPT` and `WEAR_PROBE_CKPT` are set.

Obtain the precomputed WEAR I3D and RAW600 features and annotations, and arrange
them as described in [data.md](data.md#wear). A complete run requires `sbj_0`
through `sbj_17`. Dataset files are not bundled. The pretrained checkpoints are
available separately through [the model release](models.md).

```bash
python scripts/check_wear_data.py --data-root /path/to/WEAR_prepared
```

This checks all subjects for readable arrays, dimensions, aligned lengths,
finite values and label range. It does not select epochs or tune the model.
The command-line training/inference path expects complete video/IMU pairs.
An explicit `aux_valid` mask is supported by the model API, but is not generated
automatically by these scripts.

## 2. Train, predict and evaluate one fold

This example uses the full epoch budget. Fold 1 holds out `sbj_0`; in general,
fold `f` holds out `sbj_{f-1}`. Use `--device cuda` for a suitable CUDA setup or
`--device mps` on Apple Silicon. CPU is the portable default.

```bash
OMP_NUM_THREADS=1 python scripts/train_wear.py \
  --data-root /path/to/WEAR_prepared \
  --fold 1 --seed 47 --method FINAL_MODEL \
  --parent-epochs 30 --probe-epochs 15 \
  --device cpu --output outputs/wear_training

python scripts/infer_wear.py \
  --data-root /path/to/WEAR_prepared --subject sbj_0 \
  --parent outputs/wear_training/FINAL_MODEL/seed_47/split_01/parent.pt \
  --probe outputs/wear_training/FINAL_MODEL/seed_47/split_01/background_probe.pt \
  --device cpu --output outputs/wear_split_01.npz

python scripts/evaluate_wear.py \
  outputs/wear_split_01.npz --output outputs/wear_split_01_metrics.json
```

For a short pipeline check, use `--parent-epochs 1 --probe-epochs 1` and a
separate output directory. Those weights are only a smoke test, not the full
experiment. The trainer refuses to overwrite an existing fold directory.

Each completed training directory contains `parent.pt`, `background_probe.pt`,
`protocol.json`, `training.json` and loss logs. Keep the parent/probe from the
same run together: the loader verifies their run, fold, seed, model
configuration and final-epoch metadata. Baselines do not create or accept a
probe. Their names are `VIDEO_ONLY`, `EARLY_CONCAT` and
`FIXED_WINDOW_ATTENTION`.

Inference writes `pred [T]`, `probabilities [T,19]`, run identity metadata,
`p_background [T]` for FINAL_MODEL, and `true [T]` when using `--data-root`.
With `--video features.npy --imu imu.npy`, inference needs no annotation file
and writes no ground truth; such outputs cannot be scored by the evaluator.

## 3. Run the complete matched benchmark

```bash
python scripts/run_wear_benchmark.py \
  --data-root /path/to/WEAR_prepared \
  --output outputs/wear_v2_benchmark \
  --device cpu --workers 1 --threads 1 \
  --seeds 41 47 53 --parent-epochs 30 --probe-epochs 15
```

This schedules 18 folds × 3 seeds × 4 methods = 216 training jobs. It freezes
source files, input hashes, seeds, device assignments and epoch budgets before
training. Test inference starts only after all scheduled training completes.
Allow several hours or longer depending on hardware; one full subject sequence
is processed at a time. Start with one worker to limit memory use.

Use `--plan-only` to freeze/check the plan without training. Rerun the identical
command to reuse completed jobs and restart interrupted jobs from their seeds;
it does not resume an intermediate optimizer state. A changed source tree,
dataset, device assignment or setting requires a new output directory.

The reference run used one CPU thread per job, three concurrent CPU workers,
and a separate one-worker MPS queue for even folds. To use that assignment on
an Apple Silicon machine, replace `--workers 1` above with:

```text
--workers 3 --mps-folds 2 4 6 8 10 12 14 16 18
```

All methods and seeds within a fold use the same device. A CPU-only or CUDA run
uses the same mathematical protocol, but hardware and numerical differences
can affect predicted boundaries and training trajectories; bitwise equality
across devices is not promised.

Outputs are written under the selected directory:

```text
benchmark_plan.json        settings and source/input SHA-256 values
source_snapshot/           frozen code and configuration
status.json                job progress and completion state
checkpoints/METHOD/seed_N/split_FF/
checkpoint_manifest.json   hashes of this run's saved weights
predictions/METHOD/seed_N/split_FF.npz
metrics/METHOD_seed_N.json per-subject and aggregate metrics
paired_statistics.json     matched subject-level comparisons
logs/                      training, inference and evaluation logs
```

To export the complete subject/seed CSV and the aggregate JSON/CSV tables:

```bash
python scripts/summarize_wear_results.py \
  --benchmark outputs/wear_v2_benchmark --output outputs/wear_tables
```

This reads the evaluation summaries and checks frame-level F1, accuracy and
concatenated F1 against all saved prediction sequences. It also derives
background/action F1 and averages seed scores within subjects before computing
subject means and sample SDs. The three exported files have the same schemas
as the published `results/wear_*.csv` and `results/wear_aggregate.json`.

To recompute the statistics from the 12 evaluation summaries:

```bash
python scripts/analyze_wear_statistics.py \
  outputs/wear_v2_benchmark/metrics/*.json \
  --output outputs/wear_v2_benchmark/paired_statistics.json
```

The evaluator accepts one method/seed/protocol at a time. `mean_fold` weights
subjects equally; `concatenated` pools frame predictions. Macro-F1 includes all
19 classes. TAL scores come from contiguous predictions on the 2 Hz grid, not
the official 50 Hz evaluation. Per-seed `mean_fold.std` uses population SD
(`ddof=0`). For a multi-seed subject summary, average each subject's seed scores
first, then compute the mean and sample SD (`ddof=1`) over the 18 subjects.

Statistics average matched seed differences within each subject, then use
18 paired subject differences and Holm correction across three comparators ×
two metrics. They are exploratory because LOSO training sets overlap.

### Redraw the published figures

The four figures can be reproduced directly from their committed numerical
inputs, independently of dataset access or training:

```bash
python -m pip install -e '.[plot]'
python scripts/plot_wear_results.py --output outputs/wear_figures
```

The renderer validates `results/wear_figure_data/manifest.json` and writes four
PNGs, individual PDFs and `WEAR_new_results_figures.pdf`. Numerical inputs are
fixed; font/rendering differences across platforms can change pixels. The
plotting extra pins pandas 2.3.3 and Matplotlib 3.10.9.

To regenerate those inputs from a completed benchmark, including the frozen-
parent component and jitter diagnostics:

```bash
python scripts/prepare_wear_figures.py \
  --benchmark outputs/wear_v2_benchmark \
  --output outputs/wear_figure_inputs --cpu-workers 2
python scripts/plot_wear_results.py \
  --data-root outputs/wear_figure_inputs/source_data \
  --output outputs/wear_figures_from_run
```

This verifies source, input and checkpoint hashes and performs inference with
the saved device assignment and weights. It does not train or select models.
The reference figure recipe requires all 18 folds and seeds 41/47/53; it retains
`sbj_1` and `sbj_9` with seed 47 for the qualitative examples. Reuse the same
command to resume completed diagnostics; changed inputs require a new output
folder. Runtime depends on sequence length and hardware. The local output also
contains computation metadata and per-pair diagnostic intermediates.

## 4. Protocol and checkpoint compatibility

Training and inference share `build_wear_parent()`: Round 0 uses ±2-second
support; Round 1 uses the detached predicted segment plus a ±1-second margin,
with core/margin priors of 1.0/0.5. Parent training uses
`0.5 × L_round0 + L_round1`; each round sums four-stage cross entropy and
`0.15 ×` truncated temporal smoothing. Adam uses learning rate `5e-4`, weight
decay `1e-4` and gradient clipping at 5. The separate background probe uses
Adam at `1e-3`, weight decay `1e-4` and binary cross entropy.

Parent/probe budgets are fixed at 30/15 epochs; both save their last epoch.
Training and IMU normalization use only the 17 training subjects. The probe's
seed is `(seed + fold * 100 + 2) % 2**32`. Baselines use the same parent budget,
optimizer and MS-TCN dimensions, but one round and no probe. Thus comparisons
with FINAL_MODEL measure the full two-round/probe recipe, not an isolated
window-only change. [method.md](method.md) describes the implementation.

`configs/wear_final.yaml` documents this fixed recipe; scripts do not take a
`--config` option. Use `--help` for supported arguments. Changing the YAML alone
does not change the model or training parameters.

Legacy unversioned parent/probe pairs remain loadable, but their identity must
be verified with the original manifest using `scripts/verify_model_files.py`.
That script and `models/wear_final/manifest.json` apply only to the old bundle,
not newly trained v2 checkpoints. New and legacy components cannot be mixed.
No public weight archive is required for training from scratch.

## 5. TGIF result and runtime scope

The two project-owner-confirmed original TGIF figures and their numeric
annotations are preserved as release assets; their hashes are tested. TGIF data
and weights are not part of the public runtime package at present, so current
code validation is reported through the maintained final-model implementation
without reinterpreting the confirmed TGIF results.
