# TGIF handling-time analysis: code for the three method subsections

Start here for the TGIF sewing/handling study. The three scripts below are
independent command-line entry points; all their implementation dependencies
are inside this folder. They work without the original author's project tree.

| Paper subsection | Run this script | Implementation |
|---|---|---|
| **1. Visual feature extraction** | [01_visual_feature_extraction.py](01_visual_feature_extraction.py) | [VideoMAE, two ROIs and clip sampling](pipeline/visual.py) |
| **2. Vibration feature extraction** | [02_vibration_feature_extraction.py](02_vibration_feature_extraction.py) | [Clock alignment, dynamic magnitude and FFT features](pipeline/vibration.py) |
| **3. Multimodal fusion and classification** | [03_multimodal_fusion_classification.py](03_multimodal_fusion_classification.py) | [Window construction and inference](pipeline/inference.py), [attention model](pipeline/models/dynamic_window_multimodal_mstcn.py), [MS-TCN](pipeline/models/mstcn_core.py) |

Classification produces **idle / sewing / handling** labels. The
[duration calculation](pipeline/handling.py) then converts contiguous labels into
half-open intervals and sums the predicted **handling time**. This is work
activity duration, separate from the computation times logged by each script.

## Install and get the trained model

Use Python 3.10 and run from this folder:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python download_models.py --seed 43 --archive /path/to/tgif_handling_seed43.zip
```

The default model is the historical TGIF adaptive-window attention checkpoint
for split seed 43. The laboratory-supplied ZIP contains its model, frozen video normalization,
matching training-split vibration statistics, metadata, and checksums. Other
available split seeds are `41 17 5 31 59`; install the matching archive with `--seed`.
[models.json](models.json) pins the archive names and SHA-256 values. The public
repository contains the code; trained TGIF bundles are currently supplied
within the research group. The private handoff ZIP already contains the six
installed bundles under `checkpoints/`, so its recipients can skip this step.

The first video extraction also downloads the frozen VideoMAE encoder.
`--local-files-only` reuses an already downloaded encoder on an offline machine.
The trained TGIF head is supplied by the research group; the VideoMAE encoder
is a separate public third-party download.

## 1. Visual feature extraction

Supply the laboratory video and its static operator ROI:

```bash
python 01_visual_feature_extraction.py \
  --video /path/to/recording.mp4 \
  --operator-roi X1 Y1 X2 Y2 \
  --output outputs/recording/visual --device cpu
```

Alternatively, use `--roi-json /path/to/roi.json` with the schema in
[examples/roi.example.json](examples/roi.example.json). Coordinates are pixels,
with exclusive right and bottom edges. Choose the ROI for that recording;
the synthetic example coordinates are not calibration for a laboratory video.

The two streams are the full frame and the operator crop, each resized to
224 x 224 and encoded into 768 features. There is one anchor every four source
frames, about 7.4933 Hz for a 29.97-fps recording. Each centered clip samples
16 frames at offsets `[-30,-26,...,-2,2,...,26,30]`. Edge frames are repeated.
Output files are:

```text
visual/
  global_features.npy      [T,768]
  operator_features.npy    [T,768]
  timestamps_sec.npy       [T], recording-relative seconds
  metadata.json           frame rate, coverage, encoder revision and timings
```

The exact encoder revision is pinned in [common.py](pipeline/common.py).
`qkv_bias=False` records the zero-attention-bias behavior of the existing TGIF
feature cache explicitly. All encoder tensors must load; unused pretraining
decoder tensors can be reported by Transformers. Use this feature protocol with
these trained heads. The pipeline is offline: centered clips use future frames.

## 2. Vibration feature extraction

Use the same video timestamp grid and a synchronization offset fixed from
independent clock matching or a calibration recording:

```bash
python 02_vibration_feature_extraction.py \
  --imu-dir /path/to/imu3 \
  --video-features outputs/recording/visual \
  --video-start 2026-05-21T13:55:00 \
  --offset-seconds FIXED_OFFSET \
  --output outputs/recording/vibration.npz
```

Each input file is named `imu3_YYYY-MM-DD_HH-MM.csv` and has columns
`timestamp,ax,ay,az` (additional columns are accepted). `timestamp` is in
milliseconds. Each minute is anchored independently on a 500-Hz clock grid;
gaps retain their true locations. Acceleration defaults to raw counts divided
by 16384; pass `--accel-scale 1` if it is already in g.

The offset convention is **sensor center time = video time + offset**. The
video start is the original recording's clock origin, including when visual
extraction begins later with `--start-seconds`. No action labels are read by
this stage. Offset selection and calibration must be completed before evaluating
new recordings.

A centered two-second rolling mean is subtracted from acceleration magnitude.
Each video anchor receives four features from a two-second Hann window:

1. Dynamic-signal RMS.
2. Peak-to-peak amplitude.
3. `log10` energy in 20-40 Hz.
4. The ratio of 20-40 Hz energy to 0.5-150 Hz energy.

The output contains `[T,4]` features, timestamps and validity flags. Interpolated
edge/gap values remain marked invalid. If no aligned signal is usable, the
script fails with a clock/availability message. `--allow-missing` explicitly
creates an all-invalid sensor input, which selects the model's video branch.

## 3. Multimodal fusion, classification and handling time

```bash
python 03_multimodal_fusion_classification.py \
  --video-features outputs/recording/visual \
  --vibration outputs/recording/vibration.npz \
  --bundle checkpoints/seed_43 --station-id station3 \
  --output outputs/recording/predictions --device cpu
```

The checkpoint recipe enables vibration for `station2`, `station3` and
`station4`. Other station IDs select its video branch. These identifiers also
select the stored training statistics. A new recording needs an appropriate
station/normalization calibration before being treated as the same setup.
Omit `--vibration` to run the video branch explicitly.

The two video streams are concatenated and projected from 1536 to 256 dimensions.
A preliminary video prediction defines segments (one-second smoothing,
two-second minimum, twenty-second maximum). Their lengths define the sensor
windows. Each window yields peak energy per second and mean linear 20-40 Hz
energy, normalized with the stored median/IQR and encoded into 64 dimensions.
Video is the **query**; vibration supplies **keys/values**. In this TGIF
configuration each frame reads the one token for its own predicted segment.
The risk gate controls the residual injection. A local mask,
auxiliary sewing probabilities and a conflict gate control sensor injection.
The normalized fused features enter the four-stage MS-TCN classifier.

The default chunk size is the historical 2247-feature, approximately five-minute
segment. The final incomplete chunk is also classified. Ground-truth labels
are not an inference input. Optional `--labels /path/to/labels.npy` computes
post-hoc accuracy and Macro-F1 after predictions have been produced; `-100`
means unannotated. Repeated timestamps, large gaps, incompatible feature rates,
wrong normalization dimensions and mismatched model/statistics are rejected.

Outputs:

| File | Use |
|---|---|
| `predictions.csv` | Time intervals, class, confidence, three probabilities and sensor-use flag |
| `activity_segments.csv` | Contiguous idle/sewing/handling intervals and duration in seconds |
| `handling_time_summary.json` | Total handling seconds/minutes, time fractions, segment counts and mean durations |
| `predictions.npz` | Arrays for plotting or downstream simulation |

Durations cover the timestamp interval recorded in `metadata.json`; they do not
silently include unobserved gaps. `handling_seconds` is the sum of intervals
predicted as class 2. Frame-grid resolution and segmentation errors remain part
of the estimate. Per-stage computation time and timing boundaries are logged
separately in the stage metadata.

## Examples and existing laboratory features

Run all three stages on generated, non-laboratory video and sensor input:

```bash
python demo.py --bundle checkpoints/seed_43 --output outputs/synthetic_demo
```

This exercises real frozen VideoMAE extraction and the trained TGIF model. It
verifies execution and file contracts; synthetic predictions are not an
accuracy benchmark.

If the laboratory already has the original cached features:

```bash
python export_cached_inputs.py --legacy-root /path/to/TGIF \
  --recording station3 --output outputs/cached_station3
# Then run stage 3 using outputs/cached_station3/visual and vibration.npz.
```

[export_lab_bundles.py](export_lab_bundles.py) is the one-time exporter for the
original lab checkpoints. Recipients normally install the laboratory ZIP with `download_models.py --archive` instead.
[source_manifest.json](source_manifest.json) records the source files and
hashes of the transferred implementation. All three stage scripts accept
`--help` from any working directory.

## Scope for the paper

This folder packages the **TGIF** attention pipeline and its trained 256-D
model definition. The separate **WEAR envelope-shape DWA v4** uses AC-RMS and
ordered slope histograms to control sensor windows; its entry points are
[shape_dwa.py](https://github.com/shureduan/TGIF-Multimodal-Temporal-Segmentation/blob/main/src/tgif_dwa/shape_dwa.py),
[shape_wear.py](https://github.com/shureduan/TGIF-Multimodal-Temporal-Segmentation/blob/main/src/tgif_dwa/shape_wear.py) and
[infer_wear_shape.py](https://github.com/shureduan/TGIF-Multimodal-Temporal-Segmentation/blob/main/scripts/infer_wear_shape.py). Specify the dataset and
model recipe when describing the method.

This handoff produces predictions, durations and metrics for the supplied
recording. Existing aggregate TGIF figures in `assets/` are a separate result
record; this command does not regenerate their numerical annotations.
Laboratory recordings, annotations, ROI calibrations and derived feature data
are supplied within the research group, not downloaded from this repository.
