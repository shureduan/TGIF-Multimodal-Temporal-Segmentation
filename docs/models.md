# Model files

## WEAR v2 checkpoints

Train from scratch using [the reproduction commands](reproduction.md). A final
model run writes:

```text
outputs/wear_training/FINAL_MODEL/seed_47/split_01/
├── parent.pt
├── background_probe.pt
├── protocol.json
└── training.json
```

The parent contains DWA/MS-TCN parameters and the 12-channel normalization
statistics fitted on that fold's training subjects. The separate probe contains
the background/action classifier. Both carry identical `fixed_epoch_loso_v2`
run metadata. Inference checks their pairing, train/test identities, seed,
model configuration and final epoch. Keep both files from the same run.

Training and inference share segment-plus-margin attention as specified in
[method.md](method.md#training-and-inference). The full benchmark writes its
own `checkpoint_manifest.json` with SHA-256 hashes. Baseline checkpoints contain
only a parent and are loaded without `--probe`.

## Legacy WEAR bundle

`models/wear_final/manifest.json` describes the older 18-fold set, not v2
training outputs. Its total size is 100,559,226 bytes (95.90 MiB): 76,079,484
bytes for 18 parents and 24,479,742 bytes for 18 probes. If that bundle is
available locally, verify it with `scripts/verify_model_files.py` before use.
The loader retains legacy-pair compatibility and rejects mixing a legacy
component with a versioned component.

Weights are not committed to Git and no public download URL is currently
provided. New users can train with the prepared WEAR data instead. PyTorch
checkpoints use pickle serialization; load only checkpoints from a trusted
source.

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
