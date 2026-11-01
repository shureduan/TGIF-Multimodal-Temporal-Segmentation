# Data preparation

## WEAR

The maintained final model consumes pre-extracted, time-aligned NumPy arrays.
Use the official download's `processed` features and `annotations`, linked in
[the WEAR download instructions](https://github.com/mariusbock/wear#download).
Select the 2048-D video and 600-D inertial features on the 0.5-second grid, not
combined video+IMU features. Copy or symlink their subject arrays into `video/`
and `imu/` below, and the original annotation JSON files into `label/`. Preserve
subject filenames, feature values, sensor channel order and annotation IDs;
the loader handles either array orientation. This repository starts from those
features and does not extract I3D features from raw video.

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

Only `sbj_0` through `sbj_17` enter this protocol. Annotation lookup first checks
`wear_split_18.json`, then `wear_test_split_1.json` through
`wear_test_split_6.json`; together they must cover these subjects. The JSON's
`database[subject].annotations` entries need `segment: [start_seconds,
end_seconds]` and integer `label_id`. Preserve the mapping in
[`configs/wear_label_mapping.json`](../configs/wear_label_mapping.json).

Before training, run:

```bash
python scripts/check_wear_data.py --data-root /path/to/WEAR_prepared
```

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

The pipeline uses approximately 7.4933 video features per second and aligned
500-Hz vibration. The [TGIF handoff](../handling_time_analysis/) now pins the
VideoMAE revision, clip sampling, image processing and model statistics. Its
three scripts extract global/operator features, produce four vibration features
and classify activity intervals. Supply the recording-specific ROI and fixed
clock offset. See that folder for CSV formats and missing-sensor behavior.

TGIF raw videos, annotations, ROI calibrations, vibration recordings and derived
feature arrays remain laboratory assets. The repository distributes code and
trained segmentation-model bundles; the research group supplies recording data
and recording-specific calibration separately.
