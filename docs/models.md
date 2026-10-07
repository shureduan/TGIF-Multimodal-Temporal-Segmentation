# Pretrained WEAR models

## Sensor-driven DWA (current)

The [sensor-driven v3 release](https://github.com/shureduan/TGIF-Multimodal-Temporal-Segmentation/releases/tag/wear-signal-v3-20261007)
contains the exact downstream weights for the current WEAR tables:

| Method | Fold coverage | Seeds | Parent/probe pairs |
|---|---|---|---:|
| `SIGNAL_ADAPTIVE_DWA` | 1–18 | 41, 47, 53 | 54 |
| `NO_SENSOR_RESIZE` | 1,3,5,...,17 | 41, 47, 53 | 27 |
| `NO_CONTRACTION` | 1,3,5,...,17 | 41, 47, 53 | 27 |

Nine archives total 530.2 MiB compressed. Downloading only the default full DWA,
seed 47, requires 88.4 MiB. Archives contain weights, training-fold IMU
normalization, sensor calibration and matching run metadata. They contain no
training data or feature-extraction weights. Downloaded weights stay outside Git.

```bash
# Quick start: full DWA, seed 47, all 18 folds.
python scripts/download_signal_weights.py

# All current-model and published-control bundles.
python scripts/download_signal_weights.py --all
python scripts/download_signal_weights.py --all --verify-only

# A single control/seed (nine matched folds).
python scripts/download_signal_weights.py --method NO_SENSOR_RESIZE --seed 47
```

The downloader checks the committed [manifest](../models/wear_signal_v3/manifest.json),
each archive and each checkpoint's size and SHA-256 before installation. Repeated
commands reuse verified files. Conflicting local files are preserved and rejected.
For offline installation, download release assets and pass `--archive-dir`.
Use `--model-root` to choose an alternative location.

```text
models/wear_signal_v3/SIGNAL_ADAPTIVE_DWA/seed_47/split_01/
├── parent.pt
└── background_probe.pt
```

Fold `f` holds out `sbj_{f-1}`. Parent and probe must share fold, seed, method,
run ID and configuration. The loader checks these identities, final-epoch
metadata and calibration buffers. These are fold-specific LOSO models, not one
model trained on all subjects. The persisted protocol string remains
`signal_adaptive_loso_v3_candidate`; it identifies the exact trained algorithm.

```bash
python scripts/infer_wear_signal.py \
  --data-root /path/to/WEAR_prepared --subject sbj_0 \
  --parent models/wear_signal_v3/SIGNAL_ADAPTIVE_DWA/seed_47/split_01/parent.pt \
  --probe models/wear_signal_v3/SIGNAL_ADAPTIVE_DWA/seed_47/split_01/background_probe.pt \
  --device cpu --output outputs/wear_signal_split_01.npz
python scripts/evaluate_wear.py outputs/wear_signal_split_01.npz \
  --output outputs/wear_signal_split_01_metrics.json
```

Prepared I3D/RAW600 inputs are still required; see [data.md](data.md).
For all comparisons and retraining, see [reproduction.md](reproduction.md).

## Reference baselines and historical models

The unchanged [v2 release](https://github.com/shureduan/TGIF-Multimodal-Temporal-Segmentation/releases/tag/wear-v2-pretrained-20261006)
provides Video-only, Early concatenation, Fixed-window attention and the Segment-guided Window Attention (SWA, `FINAL_MODEL`). All use 18 subjects and three seeds.
Its twelve archives total 729.4 MiB. Obtain all reference weights with:

```bash
python scripts/download_wear_weights.py --all
python scripts/download_wear_weights.py --all --verify-only
```

Use `infer_wear.py` for these weights, with a probe only for `FINAL_MODEL`.
Use `infer_wear_signal.py` for new DWA and its controls; v2 and v3 pairs are not
interchangeable. [wear_v2_models.md](wear_v2_models.md) preserves the full v2
instructions and legacy manifest details. The old unversioned manifest under
`models/wear_final/` is not the current release.

## TGIF model availability

The confirmed TGIF figures and metrics are retained as the research result
record. TGIF data and a public model bundle are not distributed in the current
release candidate. This does not change the accepted result values; it only
means that the present downloadable inference example is the maintained WEAR
final-model path.

If a TGIF runnable bundle is published later, package its checkpoint together
with the exact ROI/feature configuration, class mapping, vibration
normalization, and input metadata. The bundle should use the same public DWA
interfaces or a clearly versioned dataset adapter without relabeling the
confirmed result experiment.
