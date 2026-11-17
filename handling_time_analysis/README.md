# TGIF handling-time analysis: code for the three method subsections

Start here for the TGIF sewing/handling study. The three scripts below form the
visual extraction → vibration extraction → fusion/classification workflow. The
original four-feature handling model remains available, and the third step now
also runs the six-feature signal-driven DWA model. DWA source and training code
are included in this folder; its released checkpoints are in the repository's
sibling `tgif_dwa_reproducibility/` folder.

| Paper subsection | Run this script | Implementation |
|---|---|---|
| **1. Visual feature extraction** | [01_visual_feature_extraction.py](01_visual_feature_extraction.py) | [Frozen VideoMAE full-frame/operator features](pipeline/visual.py); the DWA video branch's trainable [GatedProjection](pipeline/models/tgif_dwa/gated_projection.py) maps the concatenated 1536-D features to 512-D. |
| **2. Vibration feature extraction** | [02_vibration_feature_extraction.py](02_vibration_feature_extraction.py) | [Clock alignment and FFT features](pipeline/vibration.py); select `handling4` (legacy default) or `dwa6` (six-feature DWA input). |
| **3. Multimodal fusion and classification** | [03_multimodal_fusion_classification.py](03_multimodal_fusion_classification.py) | `--model handling` runs the original four-feature model; `--model tgif-dwa` runs the six-feature DWA model and one-way attention. |

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

The commands here install the original `handling4` model, which uses its
four-feature checkpoint bundles. The six-feature DWA checkpoints are already
provided in the sibling `tgif_dwa_reproducibility/` folder and are selected with
`--model tgif-dwa` in stage 3.

The default `handling4` model is the historical TGIF adaptive-window attention checkpoint
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

The released training cache's operator stream was generated with the project's
YOLO-based operator crop. This public stage currently accepts a fixed ROI from
the caller and does not run YOLO; use the original cached features or reproduce
the same crops when aiming to match the released training inputs. The learned
GatedProjection is applied to the two extracted streams during model training
and inference; it is not part of the frozen VideoMAE encoder.

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
By default, each video anchor receives four features from a two-second Hann window:

1. Dynamic-signal RMS.
2. Peak-to-peak amplitude.
3. `log10` energy in 20-40 Hz.
4. The ratio of 20-40 Hz energy to 0.5-150 Hz energy.

The output contains `[T,4]` features, timestamps and validity flags. Interpolated
edge/gap values remain marked invalid. If no aligned signal is usable, the
script fails with a clock/availability message. `--allow-missing` explicitly
creates an all-invalid sensor input. The legacy `handling4` model then uses its
video-only branch; DWA masks the sensor context to zero but retains its DWA
classifier, which is not the separately trained Video Method baseline.

For the DWA model, request its six-feature profile:

```bash
python 02_vibration_feature_extraction.py \
  --imu-dir /path/to/imu3 \
  --video-features outputs/recording/visual \
  --video-start 2026-05-21T13:55:00 \
  --offset-seconds FIXED_OFFSET \
  --feature-set dwa6 \
  --output outputs/recording/vibration_dwa6.npz
```

The output is `[T,6]`: dynamic RMS, log 20–40 Hz energy, 20–40 Hz/total
energy, 20–30 Hz/total energy, 30–40 Hz/total energy, and normalized dominant
frequency within 20–40 Hz. All six use the same centered two-second Hann-FFT
window at each video anchor. The supplied clock offset is independent of action
labels. Missing feature rows are interpolated for storage but retain
`valid=false`; DWA excludes those rows from attention.

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

### Current six-feature DWA path

Run the same third script with `--model tgif-dwa` and a `dwa6` vibration file:

```bash
python 03_multimodal_fusion_classification.py \
  --model tgif-dwa \
  --video-features outputs/recording/visual \
  --vibration outputs/recording/vibration_dwa6.npz \
  --seed 43 \
  --output outputs/recording/dwa_predictions \
  --device auto
```

The default checkpoint and 1536-D video normalization are loaded from
`tgif_dwa_reproducibility/dwa_code_and_weights/weights/`. The script writes the
same prediction, activity-segment, duration-summary, and NPZ files as the
original handling path. Use `--checkpoint` or `--video-normalization` to supply
alternative files.

For each aligned video frame, the model gates the full-frame and operator
VideoMAE embeddings and projects 1536 dimensions to 512. Round 0 starts from a
local ±2-second support. A sensor-only controller uses statistics of the six
vibration features to clip at signal changes, contract statistically redundant
regions, and expand unstable or undersampled support. Video queries attend
one-way to vibration keys; values remain the original six-dimensional feature
rows. The resulting six-dimensional context is concatenated with the 512-D
video representation before MS-TCN.

Round 1 obtains a seed interval from the detached Round-0 prediction of the
shared MS-TCN. Its start and end indices define the candidate support, which is
again resized from vibration statistics before attention. Each selected window
is therefore represented by two boundary indices, while its duration varies.
The six FFT features are **not recomputed** for each predicted interval; the
model attends over the already computed six-feature sequence. There is no
separate video pre-classifier in this DWA checkpoint. The two-second FFT window
and the DWA attention window are distinct operations.

`dwa6` is a label-free option for processing a new recording when its clock
offset is known. The released DWA checkpoints were trained on cached feature
arrays built by the historical timestamp builder, which selected offsets by
correlation with sewing labels and interpolated remaining feature gaps.
Therefore, `dwa6` is the appropriate deployment flow, but its inputs do not
recreate the exact historical validation arrays or guarantee the saved
validation scores. The 66-segment training caches and annotations are not
distributed.

### Training the released TGIF DWA recipe

Training is included in [train_tgif_dwa.py](train_tgif_dwa.py). It consumes the
same cached 66 five-minute feature segments used for the released checkpoints:
11 recording/crop IDs × 6 segments, with 60 train and 6 validation segments per
split. The five split seeds are `41, 11, 43, 17, 53`; each selects checkpoints
by its mean validation F1@0.5. The run uses 150 epochs by default, model seed 0,
and initializes the DWA model with compatible MS-TCN residual layers from a
WEAR parent checkpoint. TGIF data, the normalization file, labels, and this
WEAR initialization checkpoint are not included in the repository.

```bash
python train_tgif_dwa.py \
  --feature-root /path/to/videomae_pipeline/outputs/extracted_features \
  --vibration-root /path/to/multimodal/fusion/data_vib_ts \
  --video-norm /path/to/videomae_2roi/models/norm.npz \
  --wear-checkpoint /path/to/wear_parent_best.pt \
  --output outputs/tgif_dwa_training \
  --method both --epochs 150 --device auto
```

Each feature folder needs `global_features.npy`, `operator_features.npy`,
`labels.npy`, and `timestamps_sec.npy`; available vibration files are named
`<recording>_vib2040.npy` and must have shape `[T,6]`. Missing vibration
recordings are explicitly masked rather than treated as observed zeros. The
video normalization is read from the supplied 1536-D file, while vibration
mean/std and DWA signal thresholds are fitted on each split's training segments
only. VideoMAE is a frozen offline feature extractor; training updates the
gated projection, attention projections/temperature, and shared MS-TCN. The
inference checkpoints in the sibling reproducibility folder were generated by
this training recipe.

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

This folder packages both the original TGIF 256-D handling-time pipeline and
the newer six-feature, signal-driven TGIF DWA model, including training and
inference entry points. The separate **WEAR envelope-shape DWA v4** uses AC-RMS and
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
