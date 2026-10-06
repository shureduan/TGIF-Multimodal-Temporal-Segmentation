# Sensor-driven DWA candidate (local experiment)

This candidate implements the intended order:

```text
initial window
  -> inspect sensor statistics
  -> contract / expand the window
  -> video-query, sensor-key/value attention on the selected support
  -> concatenate video and raw sensor context
  -> shared MS-TCN
```

The released `fixed_epoch_loso_v2` weights and figures describe the earlier
prediction-segment-plus-fixed-margin method. They are not results for this
candidate. Its separate protocol is `signal_adaptive_loso_v3_candidate`.

## Window control before attention

The initial Round-0 window is +/-2 seconds. Round 1 starts from the previous
round's detached predicted segment. **Both rounds resize their initial window
before computing Q/K scores.** Predictions initialize geometry only; the resize
controller accepts sensor values, sensor validity and seed endpoints. It has
no video, class-label, attention-context or test-score input.

The candidate operational definition of sensor information uses six statistics
per raw channel: mean, log mean-square energy, and log squared-DFT band
magnitudes in four frequency bands. At 50 Hz these bands are (0,2], (2,5],
(5,12] and (12,25] Hz. These descriptors control the window only. The attention
value remains the full normalized RAW600 row, not the descriptor vector.
Descriptors are centered by their training median and scaled by training IQR
with a 0.05 floor. Up to 512 equally spaced rows per training subject enter
calibration. No held-out subject is used to fit these quantities.

For a window W, R(W) concatenates the temporal mean and standard deviation of
the standardized descriptors. The distance between two R vectors is root mean
squared coordinate difference. This is a statistical proxy, not a measurement
of semantic information or a guarantee that an action has been identified.

1. **Inspect state changes.** Compare descriptor moments over two tokens on
   either side of each candidate cut. Local maxima above the training 98th
   percentile create sensor-derived barriers. Invalid sensor rows also create
   barriers. A seed crossing a barrier contracts to the region containing its
   query. This offline boundary calculation uses neighboring sensor tokens;
   it is not a causal streaming algorithm.
2. **Remove redundancy.** Try halving each side of the current window toward
   the query. Accept a smaller candidate only if it has at least three tokens,
   its moments remain close to the original barrier-clipped reference window,
   and its two temporal halves have consistent moments. Repeat until a further
   halving fails. The reference is fixed during this search to prevent drift.
3. **Expand insufficient or unstable support.** If fewer than three tokens
   remain, or the two halves still differ too much, extend each available side
   by one 0.5-second feature step and reevaluate. Stop on stability, a sensor
   barrier, a sequence endpoint, or the twelve-step expansion budget. A window
   that reaches a limit without stability stays explicitly flagged as unstable.

The stability threshold is the training median half-to-half distance in local
nine-token windows. The redundancy threshold is the training median distance
between local nine-token and five-token windows. These are frozen candidate
rules, not hyperparameters selected by held-out accuracy. Windows near missing
data or endpoints may contain fewer than three valid tokens.

The search can contract or expand the same seed depending on the sensor
content. It is deterministic and discrete; gradients do not train its boundary
decisions. Q/K projections, temperature and MS-TCN train through the selected
context. Changing the video or Q/K weights while keeping the seed and sensor
fixed leaves the selected support unchanged.

## One-way attention and training

After support selection, attention computes `softmax(temperature * cosine(q,k))`
over valid sensor keys in that support. Video produces Q, sensor produces K,
and the original normalized RAW600 supplies V. There is no reverse attention
and no fixed core/margin prior in this candidate. "One-way" describes the
cross-modal direction, not temporal causality.

The two rounds share Q/K and MS-TCN. The parent loss remains
`0.5 * L_round0 + L_round1`, with the same stage losses and optimizer as v2.
The separate background probe also retains the original training recipe.
Parent/probe budgets are fixed at 30/15 epochs; normalization and signal
calibration use only the 17 training subjects. Both components save their last
epoch and carry matching versioned run metadata. The new loader rejects v2
checkpoints and mismatched pairs.

## Local commands

Install this candidate checkout, then train and infer one fold:

```bash
python -m pip install -e .
OMP_NUM_THREADS=1 python scripts/train_wear_signal.py \
  --data-root /path/to/WEAR_prepared --fold 1 --seed 47 \
  --parent-epochs 30 --probe-epochs 15 --device cpu \
  --output outputs/signal_training

python scripts/infer_wear_signal.py \
  --data-root /path/to/WEAR_prepared --subject sbj_0 \
  --parent outputs/signal_training/SIGNAL_ADAPTIVE_DWA/seed_47/split_01/parent.pt \
  --probe outputs/signal_training/SIGNAL_ADAPTIVE_DWA/seed_47/split_01/background_probe.pt \
  --device cpu --output outputs/signal_split_01.npz

python scripts/evaluate_wear.py outputs/signal_split_01.npz \
  --output outputs/signal_split_01_metrics.json
```

Inference saves each round's initial and selected bounds, sensor-boundary
clipping, redundancy trimming, expansion counts and final stability flags.
These diagnose whether sensor-driven resizing is actually executed. Statistics
are computed deterministically on CPU even when attention/MS-TCN run on MPS.

## What validation can establish

The tests establish expansion, contraction, dependence on sensor values,
independence from video/QK for fixed seeds, controller-before-attention order,
complete long-window support, finite gradients, missing-sensor behavior,
train/inference agreement, checkpoint pairing and train-only calibration.
They do not establish improved recognition or semantic boundary correctness.

The first pilot is predeclared as fold 1, seed 47, 30 parent epochs and 15 probe
epochs on CPU. Evaluation follows completed training. It can identify execution
problems and describe one subject's behavior; it cannot replace the full
18-fold, three-seed comparison or demonstrate statistical superiority. Any
future revision of the decision rules must be identified as a new experiment.
