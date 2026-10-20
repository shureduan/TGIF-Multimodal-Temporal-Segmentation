# Envelope-shape DWA (v4)

V4 replaces the sensor-window controller with a coarse envelope-and-shape
representation, then performs the same one-way Video Q -> Sensor K/V attention.
The new descriptor selects support; the fused attention values remain RAW600.
Video features, learned backbone, two rounds, background probe, loss, optimizer,
and fixed 30/15 parent/probe budgets are unchanged. Initial learned parameters
match v3 for the same seed. Serialized IDs remain `ENVELOPE_SHAPE_DWA` and
`envelope_shape_loso_v4_pilot` so the released checkpoints remain loadable.

## Signal representation and support selection

1. Reshape each normalized RAW600 token into 12 channels x 50 samples. Remove
   each channel's local DC mean, compute AC RMS, then apply `log1p`.
2. Scale each envelope channel by training-only median/IQR, with IQR floor 0.05.
   Preserve amplitude differences instead of normalizing each window to unit variance.
3. Split a local five-token neighborhood into three ordered temporal cells.
   Each cell contains channel envelope means and falling/flat/rising slope
   histograms; the slope dead zone is calibrated on training subjects only.
   The resulting descriptor has 144 dimensions (36 levels + 108 histogram bins).
4. Compare descriptors using 0.75 envelope / 0.25 histogram weighted squared
   distance. Calibrate the matching threshold from the 90th percentile of
   training-neighbor distances at one/two-token offsets, with minimum 0.05.
5. Round 0 starts at +/-2 seconds; Round 1 starts at the detached predicted
   segment. Search the seed plus at most 12 extra tokens (6 seconds) per side.
   Keep the contiguous matching region containing the query, stopping at the
   first mismatch. Every candidate is compared with the fixed query descriptor.
6. Missing-sensor rows stop both descriptor neighborhoods and attention support;
   invalid queries receive zero sensor context. Select support before Q/K projection.

This allows an active query to discard flat surroundings, while a long region
with similar coarse appearance can expand. The descriptor retains coarse
order and amplitude; it is an envelope/PAA and slope-histogram adaptation,
not a full HOG-1D classifier or a Shannon-entropy controller. RAW600 remains
available to attention, including signal information omitted by the controller.

## Reproduce figures without data or weights

```bash
python -m pip install -e '.[plot]'
python scripts/plot_shape_results.py --output outputs/wear_shape_figures
```

This validates the committed table hashes and writes six PNGs, six individual
PDFs and `WEAR_envelope_shape_DWA_results.pdf`. Numerical inputs live in
`results/wear_shape_v4/`; the earlier three-seed v3 gallery remains separate.

## Infer with released weights

Obtain WEAR features as described in [data.md](data.md). The download validates
both archive SHA-256 and all 36 checkpoint files. Dataset files are not included.

```bash
python scripts/download_shape_weights.py
python scripts/infer_wear_shape.py \
  --data-root /path/to/WEAR_prepared --subject sbj_0 \
  --parent models/wear_shape_v4/ENVELOPE_SHAPE_DWA/seed_47/split_01/parent.pt \
  --probe models/wear_shape_v4/ENVELOPE_SHAPE_DWA/seed_47/split_01/background_probe.pt \
  --device cpu --output outputs/shape_split_01.npz
python scripts/evaluate_wear.py outputs/shape_split_01.npz \
  --output outputs/shape_split_01_metrics.json
```

Fold `f` holds out `sbj_{f-1}`. Parent and probe must belong to the same run,
method, fold and seed; the loader rejects mismatched pairs and other protocols.

## Recompute the complete comparison from weights

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python scripts/reproduce_shape_release.py \
  --data-root /path/to/WEAR_prepared \
  --device cpu --output outputs/shape_reproduction
```

This automatically downloads/verifies seed-47 weights for all six methods and
the two v3 controls, then runs 126 complete-model inferences: 6 x 18 main cases
and 2 x 9 controls. It scores the outputs, recomputes 12 metrics, paired
comparisons, per-class results and CPU latency, then draws the six figures.
Completed predictions are hash-checked and reused on a rerun. Changed inputs,
source or settings require a new output directory. About 0.5 GiB of weights
are downloaded; prepared features are obtained separately.

Published evaluation used CPU on odd folds and Apple MPS on even folds for
every method. Use `--device reference` on an Apple MPS host for that assignment.
CPU/CUDA are portable inference paths; numerical results can vary with device
and library versions. Input hashes are checked by default; alternate prepared
features require `--allow-different-inputs` and constitute a separate evaluation.

## Retrain without held-out checkpoint selection

Train all 18 folds from the prepared inputs; no old benchmark directories are
required for this path. Run the following Python snippet from the repository:

```python
import subprocess
import sys
for fold in range(1, 19):
    subprocess.run([
        sys.executable, 'scripts/train_wear_shape.py',
        '--data-root', '/path/to/WEAR_prepared', '--output', 'outputs/shape_training',
        '--fold', str(fold), '--seed', '47', '--device', 'cpu',
        '--parent-epochs', '30', '--probe-epochs', '15',
    ], check=True)
```

The published training assignment was CPU for odd folds and MPS for even
folds; change `--device` accordingly to match it. Both components save the
fixed last epoch. Normalization and descriptor calibration read only the 17
training subjects. Existing fold folders are never overwritten. After training,
pass `--v4-models outputs/shape_training` to `reproduce_shape_release.py`.

The advanced `run_shape_seed.py` runner preserves the original experiment's
frozen source/data plan and resumable concurrent training. It additionally
requires complete v2/v3 benchmark folders; the commands above are the public
standalone training and reproduction entry points.

## Reading the comparison and component figures

The new main table uses all 18 subjects and only seed 47 for every row.
Subject Macro-F1 includes all 19 classes; mean and sample SD weight subjects
equally. Concatenated Macro-F1 pools the 18 sequences. Do not combine v4's
one-seed row with the preceding three-seed v3 table.

The controller component panel uses the same nine odd folds (`sbj_0,2,...,16`)
and seed 47 for all four independently trained recipes: no resizing, no
contraction, v3 controller and v4 shape controller. This compares complete
controller alternatives; it does not isolate the envelope and histogram
contributions individually. Round-0/parent/probe rows share trained v4 weights
and are labeled frozen-output diagnostics.

CPU timing covers whole two-round forward plus probe, after feature loading
and normalization: one CPU thread, one warm-up and five timed repeats on each
of `sbj_0`, `sbj_9`, `sbj_17`. V3 is timed before v4; the measured speedup is
specific to these prepared sequences and this CPU setup, not training or GPU
throughput. Full repetitions and all paired statistics are retained in the
result tables. The current result is an exploratory single-seed comparison.

References: [HOG-1D](https://ilab.usc.edu/publications/Zhao_Itti16tkde.html),
[PAA/SAX](https://www.cs.ucr.edu/~stelo/papers/DMKD03.pdf),
[vibration envelopes](https://journals.sagepub.com/doi/10.1177/1077546313501929).
