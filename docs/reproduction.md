# Reproduction levels

## 1. Source and interface checks

From a clean environment:

```bash
conda env create -f environment.yml
conda activate tgif-temporal
python -m pip install -e .
python -m unittest discover -s tests -v
```

The tests cover tensor shapes, fixed and prediction-derived support, missing-IMU
zero context in the parent, metric fixtures, and the exact TGIF asset hashes.
They use synthetic arrays and do not establish experimental reproduction.

## 2. Existing-weight WEAR inference

Obtain the verified 18-fold model archive, then run
`scripts/verify_model_files.py`. A fold number maps to held-out subject
`sbj_{fold-1}`. For example, split 1 must run on `sbj_0` with the split-1 parent
and probe.

```bash
python scripts/infer_wear.py \
  --data-root /path/to/WEAR_prepared \
  --subject sbj_0 \
  --parent models/wear_final/split_01/parent.pt \
  --probe models/wear_final/split_01/background_probe.pt \
  --output outputs/final_model_split_01.npz
```

Expected output arrays are `pred [T]`, `probabilities [T,19]`,
`p_background [T]`, and (when `--data-root` is used) `true [T]`.

## 3. Recompute the saved WEAR feature-grid metrics

Run inference for all 18 fold/subject pairs and pass every output to:

```bash
python scripts/evaluate_wear.py \
  outputs/final_model_split_*.npz \
  --output outputs/final_model_18fold_metrics.json
```

Compare the evaluator's `mean_fold` values and concatenated Macro-F1 against
`results/wear_aggregate.json`. Exact agreement additionally requires the same
official feature arrays and annotation JSON files. The public
repository currently supplies neither data nor weights, so an external clean
clone cannot yet complete this level.

## 4. Train WEAR from scratch

```bash
python scripts/train_wear.py \
  --data-root /path/to/WEAR_prepared \
  --fold 1 \
  --output outputs/wear_training \
  --device cuda
```

The command trains the two-round parent for 30 epochs and a video+RAW600
background probe for 15 epochs. Parent loss is
`0.5 × L_round0 + L_round1`; each round loss sums four-stage cross entropy and
`0.15 ×` truncated temporal smoothing. Adam uses learning rate `5e-4`, weight
decay `1e-4`, gradient clipping at 5, and seed 47. The probe uses Adam at
`1e-3`, weight decay `1e-4`, and binary cross entropy.

The held-out LOSO subject is used for parent best-epoch selection, matching the
existing result rather than creating a new train/validation/test protocol.
Full training was not rerun during release preparation.

## 5. TGIF result and runtime scope

The two project-owner-confirmed original TGIF figures and their numeric
annotations are preserved as release assets; their hashes are tested. TGIF data
and weights are not part of the public runtime package at present, so current
code validation is reported through the maintained final-model implementation
without reinterpreting the confirmed TGIF results.
