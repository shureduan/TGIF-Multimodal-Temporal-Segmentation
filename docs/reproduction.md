# Reproducing sensor-driven DWA on WEAR

Choose the level you need: redraw figures from committed tables, evaluate
released weights on prepared features, or train from scratch. All commands run
from the repository root. The release covers 18 subjects × 3 seeds for the main
comparison and nine matched subjects × 3 seeds for two retrained controls.

## 1. Install and prepare inputs

```bash
conda env create -f environment.yml
conda activate tgif-temporal
python -m pip install -e '.[plot]'
python -m unittest discover -s tests -v
```

The pinned environment uses Python 3.10. Alternatively install into an existing
Python 3.10 environment with `python -m pip install -e '.[plot]'`.
Synthetic tests run on CPU; an optional legacy checkpoint test needs
`WEAR_PARENT_CKPT` and `WEAR_PROBE_CKPT`.

Follow [data.md](data.md#wear) to obtain and arrange I3D/RAW600 arrays and labels
for `sbj_0` through `sbj_17`, then validate them:

```bash
python scripts/check_wear_data.py --data-root /path/to/WEAR_prepared
```

Dataset files are not redistributed. This checks dimensions, alignment, finite
values and class range. CLI training and inference expect complete video/IMU
pairs. The model API additionally supports an explicit invalid-sensor mask.

## 2. Redraw all current figures (no dataset required)

```bash
python scripts/plot_signal_results.py --output outputs/wear_signal_figures
```

The renderer verifies `results/wear_signal_v3/manifest.json` and writes six PNGs,
six individual PDFs and `WEAR_sensor_DWA_results.pdf`. Numerical inputs are fixed;
font/rendering differences may change pixels. Original v2-only figures remain
reproducible with `scripts/plot_wear_results.py` and their original source folder.

## 3. Evaluate the released weights

For a single-fold startup example, see [models.md](models.md). To regenerate
all current numerical tables:

```bash
python scripts/download_signal_weights.py --all
python scripts/download_wear_weights.py --all

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python scripts/reproduce_signal_release.py \
  --data-root /path/to/WEAR_prepared \
  --output outputs/reproduced_signal_release --device cpu

python scripts/plot_signal_results.py \
  --data-root outputs/reproduced_signal_release/source_data \
  --output outputs/reproduced_signal_figures
```

This verifies all checkpoint files and prepared-input hashes, runs 324
method/seed/fold inferences (216 reference + 54 full DWA + 54 controls), and
exports 432 metric records including 108 frozen-model output diagnostics. It
then recomputes the paired comparisons, timelines, window statistics and all
figure inputs directly from predictions. There is no fitting or epoch selection.
Allow time for full-sequence inference; the default is one process to bound
memory usage. Repeating the same command resumes hash-verified predictions.
Changed data, code, devices or weights require a new output directory.

Reference inference uses **CPU on odd folds, Apple MPS on even folds**, matching
training for every model. On Apple Silicon, use `--device reference` for that
assignment. CPU and CUDA are supported alternatives; floating-point changes can
affect discrete boundaries, so cross-device bitwise equality is not promised.
The reference input hashes are in the result manifest. To intentionally evaluate
a different prepared feature set, use `--allow-different-inputs` with a new
output directory and identify that result separately.

If predictions already exist, recompute tables without inference:

```bash
python scripts/analyze_signal_release.py \
  --v2-predictions /path/to/v2/predictions \
  --v3-predictions /path/to/v3/predictions \
  --output outputs/rescored_signal/source_data
```

Each input root must contain `METHOD/seed_N/split_FF.npz`. The analyzer requires
all released identities, consistent ground truth and valid 19-class probability
arrays. It explicitly takes all 18 full-model folds and the same nine odd folds
for controls; additional completed controls do not change the published cohort.

## 4. Train one DWA fold from scratch

```bash
OMP_NUM_THREADS=1 python scripts/train_wear_signal.py \
  --data-root /path/to/WEAR_prepared --fold 1 --seed 47 \
  --parent-epochs 30 --probe-epochs 15 --device cpu \
  --output outputs/signal_training

python scripts/infer_wear_signal.py \
  --data-root /path/to/WEAR_prepared --subject sbj_0 \
  --parent outputs/signal_training/SIGNAL_ADAPTIVE_DWA/seed_47/split_01/parent.pt \
  --probe outputs/signal_training/SIGNAL_ADAPTIVE_DWA/seed_47/split_01/background_probe.pt \
  --device cpu --output outputs/signal_split_01.npz
python scripts/evaluate_wear.py outputs/signal_split_01.npz \
  --output outputs/signal_split_01_metrics.json
```

For a smoke test use one parent/probe epoch in a separate output directory;
those scores do not represent the published budget. Training refuses to
replace an existing fold directory. Inference saves final probabilities,
Round-0 and parent probabilities, run metadata and per-round initial/selected
bounds, contraction/expansion counts and stability flags.

The exact same controller and attention code runs in training and inference.
Normalization and descriptor calibration use only the 17 training subjects.
The parent and probe save their last epochs; no held-out checkpoint selection
is used. Controller hyperparameters and optimizer settings are specified in
[method.md](method.md) and embedded in the checkpoint metadata.

For matching controls, pass `--method NO_SENSOR_RESIZE` or
`--method NO_CONTRACTION`. The release uses folds 1,3,5,...,17 for each control
and seeds 41,47,53. Full DWA uses all folds and the same seeds. Retrain every
required pair before evaluation. Do not replace a retrained control with a
frozen inference intervention or a one-round baseline.

## 5. Extended benchmark and resuming training

The extended runner trains all five sensor-driven variants on all 18 folds:
270 new parent/probe pairs, plus 216 reusable v2 reference runs. Its scope is
larger than the current released component subset. For a fresh run:

```bash
python scripts/run_wear_benchmark.py \
  --data-root /path/to/WEAR_prepared --output outputs/wear_v2_benchmark \
  --device cpu --workers 1 --threads 1 \
  --seeds 41 47 53 --parent-epochs 30 --probe-epochs 15

python scripts/run_signal_benchmark.py \
  --data-root /path/to/WEAR_prepared --baseline outputs/wear_v2_benchmark \
  --output outputs/signal_full --cpu-workers 1 --mps-workers 1
```

For the reference mixed-device assignment, add
`--mps-folds 2 4 6 8 10 12 14 16 18` to the v2 command. New runs inherit its
fold-device assignment. The runner freezes code, input hashes and settings;
repeating the same command skips verified complete runs. Interrupted partial
jobs restart from their declared seed and budget. Settings/source changes
require a fresh output folder. Its built-in full analysis waits for all 270
runs and reports the extended study, not the nine-subject release subset.
To reproduce the published subset from its predictions, use the analyzer in
section 3. [Full-study specification](sensor_driven_full_study.md).

Old v2 commands and checkpoint compatibility are retained in
[wear_v2_reproduction.md](wear_v2_reproduction.md).

## 6. TGIF result and runtime scope

The two project-owner-confirmed original TGIF figures and their numeric
annotations are preserved as release assets; their hashes are tested. TGIF data
and weights are not part of the public runtime package at present, so current
code validation is reported through the maintained final-model implementation
without reinterpreting the confirmed TGIF results.
