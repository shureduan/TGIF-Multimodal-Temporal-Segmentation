# Release-candidate verification

Verification date: 2026-09-12. Environment: Python 3.10.20, PyTorch 2.12.0,
NumPy 2.2.6, SciPy 1.15.3, and scikit-learn 1.7.2.

## Completed checks

- Built the Python wheel and installed it into an isolated temporary directory
  with `pip install . --target <temporary-dir> --no-deps --no-build-isolation`.
- Ran eight `unittest` checks: TGIF asset hashes, segmentation metrics, WEAR
  metrics, feature orientation/timestamps/normalization, two-round prediction
  seeding, boundary-margin support, missing-IMU zero context, and one private
  parent/probe checkpoint load. All passed.
- Compared the 36-entry WEAR model manifest against the private source files.
  Every byte size and SHA-256 value matched; total size was 100,559,226 bytes.
- Strictly loaded all 18 WEAR parent checkpoints and all 18 corresponding probes
  through `tgif_dwa.wear_model.load_wear_final`. All 18 pairs passed.
- Ran `scripts/infer_wear.py` on the complete `sbj_0` sequence (5,587 feature
  rows) with split-1 weights. `true`, `pred`, and the full `[5587,19]`
  probability array were bit-exact with the stored final-v1 prediction file
  (maximum absolute probability difference 0.0).
- Ran `scripts/evaluate_wear.py` over all 18 stored final predictions. It
  reproduced mean fold Macro-F1 0.7998145359 ± 0.1620696671, mean accuracy
  0.8283687466, mean mAP@0.5 0.7755878895, mean average mAP 0.7639940966, and
  concatenated 19-class Macro-F1 0.8108608949.
- Verified that both copied TGIF PNG files are byte-identical to the two source
  files selected by the project owner.
- Searched the release candidate for absolute `/Users/...` paths; none remain.

## Not run or not externally reproducible yet

- No large-scale training was started.
- A new conda environment was not created because the existing `videomae`
  environment already contained the audited runtime dependencies. The wheel
  build and isolated-target installation passed.
- TGIF model inference and metric recomputation were not run because the TGIF
  data and a public model bundle are outside the current release package. The
  confirmed result figures are treated as a separate research result record.
- Official WEAR 50 Hz record scoring and official detector-format TAL evaluation
  are not implemented.
- A WEAR three-track GT/video-only/multimodal timeline is still missing. The
  existing multi-track files have a class-rendering defect; the available valid
  class-ID trace contains GT and final only.
