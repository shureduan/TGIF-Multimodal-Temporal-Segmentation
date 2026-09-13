# Dynamic Window Attention

## Motivation

Vibration and IMU features describe activity over a temporal interval. A fixed
window can include unrelated states near a boundary, while a short window can
miss the stable pattern needed by the sensor branch. DWA uses the video model's
own preliminary segmentation to choose the sensor support for a second pass.
The prediction supplies boundaries only; the predicted class is not inserted
into the sensor feature.

## Released algorithm

Let the aligned video and sensor sequences be
`V ∈ R^(T×Dv)` and `S ∈ R^(T×Ds)`. The WEAR release uses `Dv=2048`, `Ds=600`,
and a 0.5-second feature stride.

For video frame `t` and candidate sensor position `i`:

```text
q_t = normalize(W_q V_t)
k_i = normalize(W_k S_i)
score(t,i) = temperature × q_tᵀ k_i + log prior_i
a(t,i) = softmax over valid i in the current support
c_t = Σ_i a(t,i) S_i
```

The projections produce 128-D query/key vectors and are used only to calculate
cosine scores. The value remains the normalized RAW600 sensor row; there is no
value projection. The attended context is concatenated with the unchanged
video feature, giving a 2648-D MS-TCN input.

### Round 0

Each query attends to a fixed ±2-second neighborhood. At a 0.5-second stride,
this is at most nine sensor tokens. The shared four-stage, eight-layer-per-stage
MS-TCN produces the preliminary per-frame logits and labels.

### Round 1

Preliminary labels are detached and converted into contiguous predicted
segments. If `t` lies in `[l_t,u_t]`, Round 1 attends to that full segment plus a
one-second margin on both sides. Candidate logits receive a fixed temporal
prior: 1.0 inside the predicted segment and 0.5 in the margin. Positions outside
this support are excluded. Query/key projections and the MS-TCN are shared with
Round 0.

This means one query normally has multiple key/value tokens. The softmax
distributes mass among those candidates. It should not be described as an
automatic sensor-trust or reliability estimator: no learned reliability gate is
present in the released final model.

### Final WEAR background prior

A separately trained probe consumes `[video, normalized RAW600]` and estimates
`p_bg(t)`. It adjusts only the final Round-1 logits:

```text
z_action(t) += 0.5 × log(1 - p_bg(t))
z_background(t) += 0.5 × log(p_bg(t))
```

The final probabilities are the softmax of these adjusted 19-class logits.

## Training and inference

The parent is trained jointly in two rounds:

```text
total loss = 0.5 × L_round0 + 1.0 × L_round1
L_round = sum over MS-TCN stages [cross entropy + 0.15 × truncated TMSE]
```

Round-1 windows come from detached Round-0 predictions during both training and
inference. Inference does not use ground-truth windows or label caches. The
background probe is trained separately on the training subjects and frozen for
final prediction.

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

- `src/tgif_dwa/iterative_dwa.py`: query/key scoring, fixed support, prediction
  detachment, context computation, and shared MS-TCN calls.
- `src/tgif_dwa/boundary_dwa.py`: Round-1 segment support and 1.0/0.5 prior.
- `src/tgif_dwa/mstcn.py`: checkpoint-compatible MS-TCN blocks.
- `src/tgif_dwa/wear_model.py`: parent/probe composition and final logit update.
- `scripts/train_wear.py`: loss, fold normalization, training, and checkpoint
  output.
- `scripts/infer_wear.py` and `scripts/evaluate_wear.py`: public run interfaces.
