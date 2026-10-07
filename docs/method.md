# Sensor-driven Dynamic Window Attention (WEAR)

The current WEAR model implements the following order:

```text
initial window
  -> inspect sensor statistics
  -> contract / expand the window
  -> video-query, sensor-key/value attention on the selected support
  -> concatenate video and raw sensor context
  -> shared MS-TCN
```

The released `fixed_epoch_loso_v2` weights and figures describe the earlier
prediction-segment-plus-fixed-margin method. The current sensor-driven results are in [results.md](results.md). The serialized
protocol remains `signal_adaptive_loso_v3_candidate` to load the released weights.

## Window control before attention

The initial Round-0 window is +/-2 seconds. Round 1 starts from the previous
round's detached predicted segment. **Both rounds resize their initial window
before computing Q/K scores.** Predictions initialize geometry only; the resize
controller accepts sensor values, sensor validity and seed endpoints. It has
no video, class-label, attention-context or test-score input.

The inherited entry-point name `fixed_window_attention` refers to the initial
fixed seed; this class overrides it and resizes that seed before attention.
Likewise, the inherited `max_escape_seconds=0` disables the old attention-context
saturation search only. Actual sensor-driven growth is controlled separately
by `SignalWindowConfig.max_expansion_steps=12` and is exercised in both rounds.

The operational definition of sensor information uses six statistics
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
between local nine-token and five-token windows. These are frozen controller
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
and no fixed core/margin prior in this model. "One-way" describes the
cross-modal direction, not temporal causality.

The two rounds share Q/K and MS-TCN. The parent loss remains
`0.5 * L_round0 + L_round1`, with the same stage losses and optimizer as v2.
The separate background probe also retains the original training recipe.
Parent/probe budgets are fixed at 30/15 epochs; normalization and signal
calibration use only the 17 training subjects. Both components save their last
epoch and carry matching versioned run metadata. The new loader rejects v2
checkpoints and mismatched pairs.

## Attention, fusion and background calibration

For a video query at t and sensor token i in its selected window W(t):

```text
q_t = normalize(W_q video_t)          # 2048 -> 128
k_i = normalize(W_k sensor_i)         # 600 -> 128
a_ti = softmax_i(temperature * q_t^T k_i), i in W(t)
c_t = sum_i a_ti * sensor_i           # original normalized RAW600 values
x_t = concat(video_t, c_t)            # 2648 dimensions
```

A four-stage MS-TCN with eight layers per stage and 64 channels produces each
round's logits. Invalid sensor tokens receive no attention. An empty valid
support returns zero context; the probe contributes a neutral prior at invalid
queries. The CLI expects complete input arrays; an explicit validity mask is
available through the model API.

The separate probe predicts background probability `p_bg` from video and RAW600.
The final Round-1 action logits receive `0.5 * log(1-p_bg+1e-6)` and the
background logit receives `0.5 * log(p_bg+1e-6)`, followed by softmax.

The parent uses Adam (learning rate 5e-4, weight decay 1e-4, gradient clipping 5).
Each round's loss sums stage cross entropy and 0.15 times truncated temporal
smoothing. The probe uses Adam (1e-3, weight decay 1e-4) and binary cross entropy.
Neither component selects an epoch using the held-out subject.

## Model variants and evaluation

SWA and DWA are two strategies for the sensor-window length problem: prediction-
guided support with a fixed margin, or sensor-driven adaptive support. Their
whole-model comparison preserves the two-round MS-TCN and background probe.
DWA additionally calibrates its signal controller on the training subjects and
replaces the SWA margin/prior rule in both rounds. Checkpoint IDs stay unchanged.

| Model | Support and fusion |
|---|---|
| Video-only | I3D to MS-TCN; one round, no probe |
| Early concatenation | I3D plus aligned RAW600; one round, no probe |
| Fixed-window attention | ±2-second support; one round, no probe |
| Segment-guided Window Attention (SWA) | Fixed Round 0; predicted segment plus ±1-second margin and 1.0/0.5 prior in Round 1; probe |
| Sensor-driven DWA | Sensor-controlled support in both rounds; learned one-way attention; probe |
| No sensor resize | DWA recipe with seed support preserved; independently retrained parent and probe |
| No contraction | DWA recipe with sensor clipping/trimming disabled; independently retrained parent and probe |

The whole-model baselines and matched two-round controls answer different
questions. See [results.md](results.md) for their respective evaluation cohorts.
The original fixed-margin implementation remains reproducible through
[wear_v2_method.md](wear_v2_method.md) and the unchanged v2 weights.

## Dataset adapters and result presentation

- TGIF uses global/operator ROI VideoMAE features and aligned vibration. Its
  confirmed Video-only and Multimodal results are presented using the two
  project-owner-approved original figures.
- This sensor-driven release, its new weights and new experiments apply to WEAR.

## Code map

- `src/tgif_dwa/signal_dwa.py`: descriptor calibration, window controller and attention.
- `src/tgif_dwa/signal_wear.py`: shared parent factory, probe composition and checkpoint pairing.
- `src/tgif_dwa/signal_training.py`: train-only normalization/calibration and fixed-epoch fitting.
- `scripts/train_wear_signal.py` / `scripts/infer_wear_signal.py`: train and infer one fold.
- `scripts/reproduce_signal_release.py`: verify weights/inputs, infer all released comparisons and score.
- `scripts/analyze_signal_release.py`: aggregate predictions using explicit matched cohorts.
- `scripts/plot_signal_results.py`: redraw all current WEAR figures from verified tables.

Commands are in [reproduction.md](reproduction.md).
