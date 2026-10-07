# Model files (v2 reference)

> Historical v2 reference. Current sensor-driven DWA: [models.md](models.md).

## WEAR v2 checkpoints

The [WEAR v2 pretrained release](https://github.com/shureduan/TGIF-Multimodal-Temporal-Segmentation/releases/tag/wear-v2-pretrained-20261006) contains the exact downstream checkpoints used for the current
`fixed_epoch_loso_v2` result tables: 54 Final parent/probe pairs and 162 baseline
parents across 18 folds and seeds 41, 47 and 53. The 12 archives are split by
method and seed (729.4 MiB compressed in total). There is no new fitting or
checkpoint selection for this release.

Run from the repository root after installation:

```bash
# Default: Final model, seed 47, all 18 folds (~88 MiB).
python scripts/download_wear_weights.py

# Another seed or a baseline; each command downloads one 18-fold bundle.
python scripts/download_wear_weights.py --method FINAL_MODEL --seed 41
python scripts/download_wear_weights.py --method FIXED_WINDOW_ATTENTION --seed 47

# All 12 bundles, then verify installed files without accessing the network.
python scripts/download_wear_weights.py --all
python scripts/download_wear_weights.py --all --verify-only
```

The downloader uses the committed [v2 manifest](../models/wear_v2/manifest.json)
to verify each archive and every checkpoint before installation. Repeating a
command reuses already verified files. It rejects mismatching existing files;
move them aside or use a fresh `--model-root` instead of mixing experiments.
For offline installation, download the `.tar.gz` assets from the release page
and pass `--archive-dir /path/to/archives`. You can choose a different install
location with `--model-root /path/to/weights`.

Default layout:

```text
models/wear_v2/FINAL_MODEL/seed_47/split_01/
├── parent.pt
└── background_probe.pt
models/wear_v2/VIDEO_ONLY/seed_47/split_01/parent.pt
```

The parent contains DWA/MS-TCN parameters and the 12-channel normalization
statistics fitted on that fold's training subjects. The separate probe contains
the background/action classifier. Both carry identical `fixed_epoch_loso_v2`
run metadata. Inference checks their pairing, train/test identities, seed,
model configuration and final epoch. Keep both files from the same run.
Fold `f` must evaluate held-out `sbj_{f-1}`. These are LOSO checkpoints, not a
single model fitted to all subjects.

Training and inference share segment-plus-margin attention as specified in
[method.md](wear_v2_method.md#training-and-inference). Baselines contain only a parent;
run `scripts/infer_wear.py` without `--probe` for them. The archives contain
checkpoint files with the model parameters and normalization statistics.
Inputs still require prepared I3D video and RAW600 IMU features.

### Evaluate the pretrained Final model across 18 subjects

After downloading seed 47, this shell loop reproduces its evaluation summary:

```bash
for fold in $(seq 1 18); do
  split=$(printf '%02d' "$fold")
  subject="sbj_$((fold - 1))"
  python scripts/infer_wear.py \
    --data-root /path/to/WEAR_prepared --subject "$subject" \
    --parent "models/wear_v2/FINAL_MODEL/seed_47/split_${split}/parent.pt" \
    --probe "models/wear_v2/FINAL_MODEL/seed_47/split_${split}/background_probe.pt" \
    --device cpu --output "outputs/pretrained/FINAL_MODEL/seed_47/split_${split}.npz"
done
python scripts/evaluate_wear.py \
  outputs/pretrained/FINAL_MODEL/seed_47/split_*.npz \
  --output outputs/pretrained/FINAL_MODEL_seed_47.json
```

Repeat for seeds 41 and 53 to obtain the three-seed Final results. Baseline
comparisons additionally require their corresponding method/seed bundles and
inference without a probe. The reference used CPU for odd folds and MPS for
even folds; CPU-only inference is supported, but cross-device floating-point
changes may affect predicted boundaries. Compare metric definitions in
[results.md](results.md); subject-mean and concatenated F1 are different.

To retrain, use [reproduction.md](wear_v2_reproduction.md). New training runs write their
own `protocol.json`, `training.json` and checkpoint manifest outside this
pretrained release. PyTorch checkpoints use pickle serialization; load only
trusted files with matching release hashes.

## Legacy WEAR bundle

`models/wear_final/manifest.json` describes the older 18-fold set, not v2
training outputs. Its total size is 100,559,226 bytes (95.90 MiB): 76,079,484
bytes for 18 parents and 24,479,742 bytes for 18 probes. If that bundle is
available locally, verify it with `scripts/verify_model_files.py` before use.
The loader retains legacy-pair compatibility and rejects mixing a legacy
component with a versioned component.

The legacy bundle is not distributed by the v2 release. The legacy verifier
`scripts/verify_model_files.py` applies only to that historical manifest; use
`scripts/download_wear_weights.py --verify-only` for the current weights.

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
