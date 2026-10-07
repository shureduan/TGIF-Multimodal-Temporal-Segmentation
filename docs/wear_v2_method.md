# Dynamic Window Attention (v2 reference)

> Historical v2 reference. Current sensor-driven DWA: [method.md](method.md).

## Motivation

Vibration and IMU features describe activity over a temporal interval. A fixed
window can include unrelated states near a boundary, while a short window can
miss the stable pattern needed by the sensor branch. DWA uses its own
preliminary multimodal segmentation from fixed-window attention and MS-TCN to
choose the sensor support for a second pass.
The prediction supplies boundaries only; the predicted class is not inserted
into the sensor feature.

## Current WEAR algorithm

The diagram in [the README](../README.md#dwa-pipeline) follows this inference
path. Round 0 and Round 1 are successive computations within one forward pass.

Let the aligned video and sensor sequences be
`V ∈ R^(T×Dv)` and `S ∈ R^(T×Ds)`. The WEAR release uses `Dv=2048`, `Ds=600`,
and a 0.5-second feature stride.

For video frame `t` and candidate sensor position `i`:

```text
q_t = normalize(W_q V_t)
k_i = normalize(W_k S_i)
score(t,i) = temperature × q_tᵀ k_i + log(prior(t,i) + 1e-6)
a(t,i) = softmax over valid i in the current support
c_t = Σ_i a(t,i) S_i
```

The projections produce 128-D query/key vectors and are used only to calculate
cosine scores. The value remains the normalized RAW600 sensor row; there is no
value projection. The attended context is concatenated with the unchanged
video feature, giving a 2648-D MS-TCN input: `[V_t, c_t]`. Feature normalization
prepares the sensor input and query/key scores; fusion itself is concatenation.

### Round 0

Each query attends to a fixed ±2-second neighborhood. At a 0.5-second stride,
this is at most nine sensor tokens. The shared four-stage, eight-layer-per-stage
MS-TCN produces the preliminary per-frame logits and labels.
The temporal prior is uniform in this round, so its constant log term is
omitted in the implementation.

### Round 1

Preliminary labels are detached and converted into contiguous predicted
segments. If `t` lies in `[l_t,u_t]`, Round 1 attends to that full segment plus a
one-second margin on both sides. Candidate logits receive a fixed temporal
prior: 1.0 inside the predicted segment and 0.5 in the margin. Positions outside
this support are excluded. Query/key projections and the MS-TCN are shared with
Round 0. Support is clipped to the sequence endpoints, and invalid sensor
positions are masked. The margin size is fixed; the support is dynamic because
the predicted segment boundaries and durations vary with the input.

This means one query normally has multiple key/value tokens. The softmax
distributes mass among those candidates. It should not be described as an
automatic sensor-trust or reliability estimator: no learned reliability gate is
present in the released final model.

### Final WEAR background prior

A separately trained probe consumes `[video, normalized RAW600]` and estimates
`p_bg(t)`. It adjusts only the final Round-1 logits:

```text
z_action(t) += 0.5 × log(1 - p_bg(t) + 1e-6)
z_background(t) += 0.5 × log(p_bg(t) + 1e-6)
```

The final probabilities are the softmax of these adjusted 19-class logits.

## Training and inference

The current `fixed_epoch_loso_v2` protocol uses the same
`build_wear_parent()` factory for training and final inference. Both use
`BoundaryUncertaintyDWA`: fixed ±2-second support in Round 0, then the predicted
segment plus a ±1-second margin and 1.0/0.5 core/margin prior in Round 1.
The parent is trained jointly in two rounds:

```text
total loss = 0.5 × L_round0 + 1.0 × L_round1
L_round = sum over MS-TCN stages [cross entropy + 0.15 × truncated TMSE]
```

Both paths derive windows from current, detached predictions. Ground-truth
labels provide training supervision; they do not define attention windows.
The background probe is trained separately on the same 17 training subjects
and frozen for final prediction. Parent and probe save the last epoch of fixed
30/15-epoch budgets; the held-out subject is not read during training or used
for checkpoint selection. See [the runnable protocol](wear_v2_reproduction.md).

The base `IterativeRawDWA` retains optional context-saturation expansion for
legacy experiments. The current fixed-margin model disables that branch and
rejects a positive `max_escape_seconds`; setting its margin to zero means
segment-only support. With an explicit invalid-sensor mask, empty support
produces zero context and the probe contributes a neutral prior at invalid
query positions. This is an API behavior; the CLI expects complete inputs.

## Dataset adapters and result presentation

- TGIF uses global/operator ROI VideoMAE features and aligned vibration. Its
  confirmed Video-only and Multimodal results are presented using the two
  project-owner-approved original figures.
- WEAR uses the official-style 2 Hz I3D/RAW600 feature grid. The released model,
  checkpoints, normalization, and inference/evaluation interfaces correspond to
  this implementation.

Historical result evidence and current release-code validation are recorded
separately. The method definition above follows the maintained final-model code.

## Code map

- `src/tgif_dwa/iterative_dwa.py`: query/key scoring, fixed/segment support,
  optional context-saturation expansion, prediction detachment, context
  computation, and shared MS-TCN calls.
- `src/tgif_dwa/boundary_dwa.py`: Round-1 segment support and 1.0/0.5 prior.
- `src/tgif_dwa/mstcn.py`: checkpoint-compatible MS-TCN blocks.
- `src/tgif_dwa/wear_model.py`: parent/probe composition and final logit update.
- `src/tgif_dwa/wear_training.py`: fixed-epoch training, loss, fold normalization,
  probe training, and checkpoint output.
- `scripts/train_wear.py`: single-fold training entry point.
- `scripts/run_wear_benchmark.py`: frozen multi-seed LOSO schedule and evaluation.
- `scripts/infer_wear.py` and `scripts/evaluate_wear.py`: public run interfaces.
