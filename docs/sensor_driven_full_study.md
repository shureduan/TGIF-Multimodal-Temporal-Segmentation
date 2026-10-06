# Sensor-driven DWA: full matched study

This study freezes the existing sensor-driven candidate and evaluates **18 LOSO
folds x seeds 41, 47, 53**, with 30 parent epochs and 15 probe epochs. Checkpoints
are always the final epoch. All new training must complete before test metrics
are computed. Normalization and sensor thresholds use only the 17 training
subjects in each fold. TGIF is outside this study.

## Separately trained models

| Method | Change relative to the full sensor-driven model |
|---|---|
| `SIGNAL_ADAPTIVE_DWA` | Both rounds resize sensor support before learned one-way attention. |
| `NO_SENSOR_RESIZE` | Preserve the initial seed support, with no signal-based contraction or expansion. |
| `NO_CONTRACTION` | Disable both signal-boundary clipping and redundancy trimming. Expansion cannot move an endpoint inward. |
| `NO_EXPANSION` | Preserve contraction; do not extend either endpoint beyond its contracted support. |
| `UNIFORM_POOL` | Preserve sensor resizing; replace learned attention with uniform averaging of the selected raw sensor values. |

Missing-sensor barriers remain active in every variant. All variants retain
the same two-round classifier, optimizer, initialization seed, train subjects,
loss, training budget and independently trained probe recipe. Uniform pooling
leaves Q/K and temperature unused. Variant settings are saved and validated in
both checkpoints; different variants cannot silently exchange checkpoint pairs.

There are **270 newly trained parent/probe pairs**. The full model is retrained
for all 54 runs, including the pilot's fold/seed. The addition of ablation
switches preserves all 27 stored full-model pilot output arrays exactly.

The previous complete v2 benchmark supplies 216 matched runs for video-only,
early concatenation, fixed-window attention and the published fixed-margin
model. Before reuse, the runner verifies input hashes, source snapshots,
checkpoint hashes, seeds, folds, epoch budgets and device assignments. The new
runs follow the same per-fold device mapping. Baseline metrics are recomputed
from their saved predictions and must agree with the original metrics.

## Output diagnostics are distinct from retrained ablations

`PARENT_NO_PROBE` and `ROUND0` are outputs of the trained full model. They are
reported separately from the four retrained controls above. Boundary-jitter
diagnostics also share the trained full model's weights and original Round-0
seeds between sensor resizing and seed-only support; they are inference
interventions, not another trained architecture comparison.

The former fixed margin/prior support plot is replaced by actual window
shrinkage, expansion, sizes and statistical-stability measurements. Statistical
stability is a signal proxy, not a guarantee of semantic information sufficiency.

## Metrics and statistical comparisons

All methods have frame accuracy, fixed-19-class macro precision/recall/F1,
background F1, action-only macro F1, mAP at tIoU 0.3/0.4/0.5/0.6/0.7, and average
mAP. Frame P/R/F1 use `zero_division=0`. The previous pooled F1 convention is
also exported separately: concatenate subjects within each seed and then
average the three seed scores. TAL uses contiguous predictions on the 2 Hz
feature grid; it is not a reproduction of official ActionFormer/TriDet outputs.

The primary inferential unit is subject, **not 54 independent fold-seed rows**.
First average three matched differences within each subject, then conduct
paired two-sided t tests on 18 subject differences. Report unadjusted 95% t
intervals and Holm-adjusted p values within each predeclared family:

- Four baselines x two primary metrics (Macro-F1 and mAP@0.5): eight tests.
- Four retrained component controls x two metrics: eight tests.
- Two frozen-model output diagnostics x two metrics: four tests.

The pilot already observed `sbj_0/seed47`. The full configuration is unchanged;
a sensitivity analysis excluding **all seeds of sbj_0** is declared before the
full run. LOSO training sets overlap, so these nominal statistics remain
exploratory. Figures retain sbj_1 and sbj_9 at seed 47 from the old examples,
rather than selecting new examples by performance.

The numerical comparison with the published model is declared in advance:
both mean-subject Macro-F1 and mAP@0.5 must be higher. Significance and the
sensor-resize component contrast are assessed separately. All negative and
non-significant comparisons remain in the report. Nothing is pushed or
published automatically.

## Running or resuming

Install the candidate checkout with plotting dependencies:

```bash
python -m pip install -e '.[plot]'
```

With an existing complete v2 benchmark:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python scripts/run_signal_benchmark.py \
  --data-root /path/to/WEAR_prepared \
  --baseline /path/to/complete_wear_v2_benchmark \
  --output outputs/sensor_dwa_full \
  --cpu-workers 2 --mps-workers 1
```

The same command resumes completed jobs after verifying identities and file
hashes. Partial jobs are archived and restarted from the same seed and budget.
No score-based retry is permitted. Source/settings/data changes require a new
study directory. Each worker trains in a separate process; MPS and primary
non-MPS devices have separate worker queues. Keep the machine awake and avoid
running another copy of the same study at the same time.

For a fresh reproduction, first run `scripts/run_wear_benchmark.py` to generate
the complete v2 baselines. Use `--workers 3 --mps-folds 2 4 6 8 10 12 14 16 18`
for this Mac's original mixed CPU/MPS assignment; omit `--mps-folds` for a
CPU-only study. New methods inherit the baseline study's fold-device mapping.

The runner writes `status.json`, per-job logs and `进度.txt`. Once training is
complete it performs inference, validates all 594 method/subject/seed records,
computes component diagnostics and paired tests, and creates nine PNG figures
plus `WEAR_sensor_DWA_full_comparison.pdf`. Numerical sources and checksums live
in `source_data/`. `先看这里.txt` and `review_conclusions.json` summarize the
outcome, including limitations. PDF pages are also rendered for final visual
review before any publication decision.
