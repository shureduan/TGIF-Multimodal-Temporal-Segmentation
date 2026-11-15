# TGIF DWA reproducibility package

This folder contains the TGIF DWA inference code, its five split-specific checkpoints, comparison figures, and the metric workbook. It is organized as a standalone addition to the main repository.

## Start here

- Read [`dwa_code_and_weights/README.md`](dwa_code_and_weights/README.md) for the model flow, input feature formats, preprocessing, checkpoint selection, and inference instructions.
- Run the packaged checkpoint smoke test from `dwa_code_and_weights/`:

  ```bash
  python -m pip install -r requirements.txt
  python smoke_test.py
  ```

- See [`comparison_figures/`](comparison_figures/) for the primary metric chart, pooled validation confusion matrices, and seed-43 temporal example. The workbook contains the numeric comparison.

## Data and evaluation

TGIF source video, vibration recordings, labels, VideoMAE backbone weights, and upstream feature-extraction assets are not included. The checkpoints consume aligned, pre-extracted features as described in the code README. The reported comparison is based on five segment-level validation splits; it is a reproducibility record for the saved experiment, not an independent held-out-worker evaluation. The code README documents preprocessing caveats and the validation protocol in detail.
