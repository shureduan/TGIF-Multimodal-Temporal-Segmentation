# Data preparation

## WEAR

The maintained final model consumes pre-extracted, time-aligned NumPy arrays:

```text
WEAR_prepared/
├── video/
│   ├── sbj_0.npy       # [T,2048] or [2048,T], I3D feature grid
│   └── ... sbj_17.npy
├── imu/
│   ├── sbj_0.npy       # [T,600] or [600,T]
│   └── ... sbj_17.npy
└── label/
    ├── wear_split_18.json
    └── wear_test_split_1.json ... wear_test_split_6.json
```

Each 600-D inertial row is interpreted as `[12,50]`: four sensors × three axes
× 50 samples from a one-second, 50 Hz window. Consecutive feature rows are 0.5
seconds apart. The released alignment maps feature index `t` to center time
`(t + 1.0) × 0.5` seconds. Unmatched frames use class 18 (background).

The loader accepts either array orientation, crops video and IMU to their common
length when they differ by no more than two rows, and fails on larger length
differences. The parent checkpoint stores 12 means and standard deviations fit
on its 17 training subjects. Video features are not normalized by this code.

Obtain WEAR data and preprocessing code from the
[WEAR project](https://mariusbock.github.io/wear/) and its
[official repository](https://github.com/mariusbock/wear). Dataset files are not
covered by this repository's eventual code license.

## TGIF

The internal sewing pipeline used two VideoMAE ROI streams and aligned
vibration:

```text
recording_id/
├── global_features.npy      # [T,768]
├── operator_features.npy    # [T,768]
├── labels.npy               # [T], idle/sewing/handling; may include ignored rows
└── timestamps_sec.npy       # [T]

vibration_features/
└── recording_id_vib.npy     # per-frame vibration features before window pooling
```

The historical pipeline used an effective video-feature rate of approximately
7.4933 Hz and built vibration summaries from the aligned 500 Hz signal. The
release must eventually freeze the exact VideoMAE revision, clip sampling,
global/operator ROI coordinates, image preprocessing, vibration cleaning,
dead-region masks, and train-only normalization statistics.

TGIF raw videos, annotations, ROI metadata, vibration recordings, and derived
features are laboratory assets and are not currently approved or packaged for
public download. Their absence blocks both pretrained inference and result
reproduction; a random-array model test would establish only tensor-interface
correctness.
