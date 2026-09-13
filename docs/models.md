# Model files

## WEAR final model

Every LOSO fold needs two trusted checkpoint files:

```text
models/wear_final/split_01/
├── parent.pt
└── background_probe.pt
```

The parent checkpoint contains the DWA/MS-TCN parameters and the 12-channel
training-fold IMU normalization statistics. The probe checkpoint contains the
separate background/action classifier. Publishing only one of the two does not
reproduce final predictions.

The complete 18-fold set is 100,559,226 bytes (95.90 MiB):

- 18 parents: 76,079,484 bytes
- 18 probes: 24,479,742 bytes

Exact sizes and SHA-256 values are in `models/wear_final/manifest.json`. The
recommended release is one immutable archive attached to a tagged GitHub
Release, Zenodo record, or model repository, with the manifest alongside it.
The files should not be committed to ordinary Git history. No download link is
listed until a real upload is available.

PyTorch `.pt` checkpoints use pickle serialization. Verify the manifest and
load only files from the trusted project release.

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
