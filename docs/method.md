# Dynamic Window Attention

## Motivation

Vibration and IMU features describe activity over a temporal interval. A fixed
window can include unrelated states near a boundary, while a short window can
miss the stable pattern needed by the sensor branch. DWA uses its own
preliminary multimodal segmentation from fixed-window attention and MS-TCN to
choose the sensor support for a second pass.
The prediction supplies boundaries only; the predicted class is not inserted
into the sensor feature.

## Final WEAR inference algorithm

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

The released `scripts/train_wear.py` trains the parent jointly in two rounds
using `IterativeRawDWA` with `WindowConfig(max_escape_seconds=0.0)`: Round 0 has
the fixed ±2-second window, and Round 1 uses only the predicted segment, with
uniform temporal prior. The training loss is:

```text
total loss = 0.5 × L_round0 + 1.0 × L_round1
L_round = sum over MS-TCN stages [cross entropy + 0.15 × truncated TMSE]
```

The final inference wrapper loads those parent parameters into the
checkpoint-compatible `BoundaryUncertaintyDWA` and adds the fixed ±1-second
margin with the 1.0/0.5 temporal prior. The background probe is trained
separately on the training subjects and frozen for final prediction.

| Setting | Parent training | Final WEAR inference |
|---|---|---|
| Round 0 | Fixed ±2-second support | Fixed ±2-second support |
| Round-1 seed | Detached Round-0 prediction | Detached Round-0 prediction |
| Round-1 support | Predicted segment only | Predicted segment + ±1-second margin |
| Round-1 temporal prior | Uniform within segment | 1.0 in segment; 0.5 in margin |
| Background probe | Trained separately | Adjusts final Round-1 logits |

Both paths derive windows from current predictions. Ground-truth labels provide
training supervision; they do not define attention windows. Inference uses no
ground-truth windows or label caches.

The base `IterativeRawDWA` also supports optional context-saturation expansion
when `max_escape_seconds > 0`. That option is disabled in the released training
script; final inference uses the fixed-margin override described above.

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
- `scripts/train_wear.py`: segment-only parent training, loss, fold
  normalization, probe training, and checkpoint output.
- `scripts/infer_wear.py` and `scripts/evaluate_wear.py`: public run interfaces.
