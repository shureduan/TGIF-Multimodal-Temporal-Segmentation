"""Iterative Raw-Feature Dynamic Window Attention for WEAR.

Design goals
------------
1. Preserve the official pre-extracted features as much as possible:
   video stays [T, 2048], RAW IMU stays [T, 600].
2. Q/K projections exist ONLY to compute attention weights.
   The attention Value is the original RAW600 vector (no V projection).
3. No preliminary Conv1D window predictor.
4. Round 0 uses one fixed local window and produces an MS-TCN prediction.
5. Round >=1 uses the PREVIOUS ROUND'S detached predicted segment as the
   seed window, then expands beyond that segment in 0.5 s steps until the
   attention context saturates. Predicted segment boundaries are a seed,
   never a hard boundary.
6. The same attention projections and the same MS-TCN are reused across rounds.

Expected feature convention
---------------------------
video:  [T, 2048]
raw600: [T, 600]
feature stride: 0.5 s

The module intentionally performs no RAW600 sensor splitting, handcrafted
feature extraction, sensor embedding, video bottleneck projection, gate,
correction MLP, or separate preliminary classifier.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mstcn import MSTCNStages


@dataclass(frozen=True)
class WindowConfig:
    feature_stride_seconds: float = 0.5
    initial_radius_seconds: float = 2.0
    expansion_step_seconds: float = 0.5
    max_escape_seconds: float = 6.0
    saturation_tau: float = 0.01
    saturation_patience: int = 2


class IterativeRawDWA(nn.Module):
    """Two-round-by-default iterative Video-Q -> RAW-IMU-K/V DWA.

    Round 0
      fixed local window around each t
      -> attention-weighted RAW600 context
      -> concat(video_2048, context_600)
      -> shared MS-TCN
      -> prediction P0

    Round 1+
      previous detached prediction
      -> contiguous predicted segment containing t = seed window
      -> seed-and-expand Context-Saturation DW (may cross old boundaries)
      -> attention-weighted RAW600 context
      -> concat(video_2048, context_600)
      -> the SAME shared MS-TCN
      -> next prediction
    """

    def __init__(
        self,
        video_dim: int = 2048,
        imu_dim: int = 600,
        attn_dim: int = 128,
        n_classes: int = 19,
        n_rounds: int = 2,
        dropout: float = 0.0,
        n_stages: int = 4,
        n_layers: int = 8,
        ch: int = 64,
        window: WindowConfig = WindowConfig(),
        min_temperature: float = 1.0,
        max_temperature: float = 20.0,
        init_temperature: float = 5.0,
    ) -> None:
        super().__init__()
        if n_rounds < 1:
            raise ValueError("n_rounds must be >= 1")
        if not (min_temperature < init_temperature < max_temperature):
            raise ValueError("temperature must satisfy min < init < max")
        self.video_dim = int(video_dim)
        self.imu_dim = int(imu_dim)
        self.attn_dim = int(attn_dim)
        self.n_classes = int(n_classes)
        self.n_rounds = int(n_rounds)
        self.window = window

        # Minimal score-only projections. They do NOT replace the original features.
        self.q_proj = nn.Linear(video_dim, attn_dim, bias=False)
        self.k_proj = nn.Linear(imu_dim, attn_dim, bias=False)

        frac = (init_temperature - min_temperature) / (max_temperature - min_temperature)
        self.min_temperature = float(min_temperature)
        self.max_temperature = float(max_temperature)
        self.raw_temperature = nn.Parameter(torch.tensor(math.log(frac / (1.0 - frac))))

        # The classifier receives the untouched video feature and the RAW600
        # attention-weighted context directly.
        self.input_dropout = nn.Dropout(dropout)
        self.mstcn = MSTCNStages(
            in_dim=video_dim + imu_dim,
            n_stages=n_stages,
            n_layers=n_layers,
            ch=ch,
            n_classes=n_classes,
        )

        self._validate_window_config()

    # ------------------------------------------------------------------
    # Configuration / small helpers
    # ------------------------------------------------------------------
    def _validate_window_config(self) -> None:
        w = self.window
        if w.feature_stride_seconds <= 0:
            raise ValueError("feature_stride_seconds must be positive")
        if w.initial_radius_seconds < 0:
            raise ValueError("initial_radius_seconds must be >= 0")
        if w.expansion_step_seconds <= 0 or w.max_escape_seconds < 0:
            raise ValueError("invalid expansion configuration")
        if w.saturation_patience < 1:
            raise ValueError("saturation_patience must be >= 1")
        for name, seconds in (
            ("initial_radius_seconds", w.initial_radius_seconds),
            ("expansion_step_seconds", w.expansion_step_seconds),
            ("max_escape_seconds", w.max_escape_seconds),
        ):
            frames = seconds / w.feature_stride_seconds
            if abs(frames - round(frames)) > 1e-6:
                raise ValueError(f"{name}={seconds} must align to feature stride {w.feature_stride_seconds}")

    def effective_temperature(self) -> torch.Tensor:
        return self.min_temperature + (
            self.max_temperature - self.min_temperature
        ) * torch.sigmoid(self.raw_temperature)

    @property
    def initial_radius_steps(self) -> int:
        return int(round(self.window.initial_radius_seconds / self.window.feature_stride_seconds))

    @property
    def expansion_step_frames(self) -> int:
        return int(round(self.window.expansion_step_seconds / self.window.feature_stride_seconds))

    @property
    def max_escape_frames(self) -> int:
        return int(round(self.window.max_escape_seconds / self.window.feature_stride_seconds))

    def _validate_inputs(self, video: torch.Tensor, raw600: Optional[torch.Tensor]) -> None:
        if video.ndim != 2 or video.shape[-1] != self.video_dim:
            raise ValueError(f"video must be [T,{self.video_dim}], got {tuple(video.shape)}")
        if raw600 is not None:
            if raw600.ndim != 2 or raw600.shape != (video.shape[0], self.imu_dim):
                raise ValueError(
                    f"raw600 must be [T,{self.imu_dim}] aligned with video; got {tuple(raw600.shape)}"
                )

    @staticmethod
    def _safe_attention(scores: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Masked softmax with exact zero weights when no key is valid.

        scores: [..., K]
        valid:  broadcastable boolean [..., K]
        """
        valid = valid.bool()
        masked = scores.masked_fill(~valid, float("-inf"))
        weights = torch.softmax(masked, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        any_valid = valid.any(dim=-1, keepdim=True)
        return torch.where(any_valid, weights, torch.zeros_like(weights))

    def _project_qk(self, video: torch.Tensor, raw600: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Q/K normalized only for cosine scoring. Original V and I never get replaced.
        q = F.normalize(self.q_proj(video), dim=-1)
        k = F.normalize(self.k_proj(raw600), dim=-1)
        return q, k

    # ------------------------------------------------------------------
    # Segment extraction. This is always called on detached predictions.
    # ------------------------------------------------------------------
    @staticmethod
    @torch.no_grad()
    def labels_to_bounds(labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert contiguous labels [T] to per-frame seed segment [start,end]."""
        if labels.ndim != 1:
            raise ValueError("labels must be [T]")
        T = labels.numel()
        if T == 0:
            return labels.new_empty(0), labels.new_empty(0)
        change = torch.ones(T, dtype=torch.bool, device=labels.device)
        change[1:] = labels[1:] != labels[:-1]
        seg_starts = torch.nonzero(change, as_tuple=False).flatten()
        seg_ends = torch.cat([seg_starts[1:] - 1, labels.new_tensor([T - 1])])
        starts = torch.empty(T, dtype=torch.long, device=labels.device)
        ends = torch.empty(T, dtype=torch.long, device=labels.device)
        for s, e in zip(seg_starts.tolist(), seg_ends.tolist()):
            starts[s : e + 1] = s
            ends[s : e + 1] = e
        return starts, ends

    # ------------------------------------------------------------------
    # Round 0: fixed local window.
    # ------------------------------------------------------------------
    def fixed_window_attention(
        self,
        video: torch.Tensor,
        raw600: torch.Tensor,
        aux_valid: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        T = video.shape[0]
        q, k = self._project_qk(video, raw600)
        radius = self.initial_radius_steps
        W = 2 * radius + 1

        # Pad sequence and create [T,W,D] sliding windows. The Value remains RAW600.
        k_pad = F.pad(k, (0, 0, radius, radius))
        v_pad = F.pad(raw600, (0, 0, radius, radius))
        valid_pad = F.pad(aux_valid.float(), (radius, radius), value=0.0).bool()

        # unfold(0, W, 1) gives [T,D,W] for a 2-D input; transpose to [T,W,D].
        k_win = k_pad.unfold(0, W, 1).transpose(1, 2)
        v_win = v_pad.unfold(0, W, 1).transpose(1, 2)
        valid_win = valid_pad.unfold(0, W, 1)

        scores = self.effective_temperature() * torch.einsum("td,twd->tw", q, k_win)
        attn = self._safe_attention(scores, valid_win)
        context = torch.einsum("tw,twi->ti", attn, v_win)
        entropy = -(attn.clamp_min(1e-8).log() * attn).sum(-1)
        max_weight = attn.max(-1).values

        return {
            "context": context,
            "attention": attn,
            "attention_entropy": entropy,
            "max_attention": max_weight,
            "selected_escape_seconds": torch.zeros(T, device=video.device, dtype=video.dtype),
            "selected_window_start": (
                torch.arange(T, device=video.device) - radius
            ).clamp_min(0),
            "selected_window_end": (
                torch.arange(T, device=video.device) + radius
            ).clamp_max(T - 1),
            "seed_segment_start": torch.full((T,), -1, device=video.device, dtype=torch.long),
            "seed_segment_end": torch.full((T,), -1, device=video.device, dtype=torch.long),
            "seed_segment_duration_seconds": torch.full(
                (T,), float("nan"), device=video.device, dtype=video.dtype
            ),
        }

    # ------------------------------------------------------------------
    # Round >=1: previous predicted segment is the seed. The window can
    # expand outside the old segment until context saturation.
    # ------------------------------------------------------------------
    def segment_seed_expand_attention(
        self,
        video: torch.Tensor,
        raw600: torch.Tensor,
        seed_start: torch.Tensor,
        seed_end: torch.Tensor,
        aux_valid: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        T = video.shape[0]
        if seed_start.shape != (T,) or seed_end.shape != (T,):
            raise ValueError("seed_start/seed_end must be [T]")
        q_all, k_all = self._project_qk(video, raw600)

        context_out = video.new_zeros((T, self.imu_dim))
        entropy_out = video.new_zeros(T)
        maxw_out = video.new_zeros(T)
        selected_idx_out = torch.zeros(T, dtype=torch.long, device=video.device)
        selected_start_out = torch.zeros(T, dtype=torch.long, device=video.device)
        selected_end_out = torch.zeros(T, dtype=torch.long, device=video.device)

        # Predicted segments are contiguous; iterate each seed segment once.
        cursor = 0
        n_candidates = self.max_escape_frames // self.expansion_step_frames + 1
        candidate_escape_frames = [j * self.expansion_step_frames for j in range(n_candidates)]

        while cursor < T:
            s = int(seed_start[cursor].detach().item())
            e = int(seed_end[cursor].detach().item())
            if s != cursor:
                # Defensive check: per-frame bounds must describe a contiguous segmentation.
                raise ValueError(
                    f"invalid seed bounds at cursor {cursor}: start={s}; expected {cursor}"
                )
            if e < s or e >= T:
                raise ValueError(f"invalid seed segment [{s},{e}] for T={T}")

            query_idx = torch.arange(s, e + 1, device=video.device)
            q_seg = q_all[s : e + 1]

            # ---- non-differentiable selection: previous segment is expansion 0 ----
            with torch.no_grad():
                candidate_contexts: List[torch.Tensor] = []
                for escape_frames in candidate_escape_frames:
                    left = max(0, s - escape_frames)
                    right = min(T - 1, e + escape_frames)
                    k = k_all[left : right + 1].detach()
                    v = raw600[left : right + 1].detach()
                    valid = aux_valid[left : right + 1].bool()
                    scores = self.effective_temperature().detach() * (q_seg.detach() @ k.T)
                    attn = self._safe_attention(scores, valid[None, :].expand(len(q_seg), -1))
                    ctx = attn @ v
                    candidate_contexts.append(ctx)

                stack = torch.stack(candidate_contexts, dim=1)  # [Q,R,600]
                if stack.shape[1] == 1:
                    selected = torch.zeros(len(q_seg), dtype=torch.long, device=video.device)
                else:
                    delta = 1.0 - F.cosine_similarity(stack[:, 1:], stack[:, :-1], dim=-1)
                    selected = torch.full(
                        (len(q_seg),), stack.shape[1] - 1, dtype=torch.long, device=video.device
                    )
                    unresolved = torch.ones(len(q_seg), dtype=torch.bool, device=video.device)
                    streak = torch.zeros(len(q_seg), dtype=torch.long, device=video.device)
                    for j in range(delta.shape[1]):
                        small = delta[:, j] < self.window.saturation_tau
                        streak = torch.where(small, streak + 1, torch.zeros_like(streak))
                        stop = unresolved & (streak >= self.window.saturation_patience)
                        # delta[:,j] compares candidate j -> j+1, so stop at j+1.
                        selected[stop] = j + 1
                        unresolved[stop] = False

            # ---- differentiable final context for the selected expansion only ----
            seg_context = video.new_zeros((len(q_seg), self.imu_dim))
            seg_entropy = video.new_zeros(len(q_seg))
            seg_maxw = video.new_zeros(len(q_seg))
            seg_window_start = torch.empty(len(q_seg), dtype=torch.long, device=video.device)
            seg_window_end = torch.empty(len(q_seg), dtype=torch.long, device=video.device)

            for selected_j in selected.unique(sorted=True).tolist():
                row_mask = selected == selected_j
                rows = torch.nonzero(row_mask, as_tuple=False).flatten()
                escape_frames = candidate_escape_frames[int(selected_j)]
                left = max(0, s - escape_frames)
                right = min(T - 1, e + escape_frames)
                k = k_all[left : right + 1]
                v = raw600[left : right + 1]  # RAW600 is the Value. No V projection.
                valid = aux_valid[left : right + 1].bool()
                scores = self.effective_temperature() * (q_seg[rows] @ k.T)
                attn = self._safe_attention(scores, valid[None, :].expand(len(rows), -1))
                ctx = attn @ v
                ent = -(attn.clamp_min(1e-8).log() * attn).sum(-1)
                maxw = attn.max(-1).values

                seg_context = seg_context.index_copy(0, rows, ctx)
                seg_entropy = seg_entropy.index_copy(0, rows, ent)
                seg_maxw = seg_maxw.index_copy(0, rows, maxw)
                seg_window_start[rows] = left
                seg_window_end[rows] = right

            context_out = context_out.index_copy(0, query_idx, seg_context)
            entropy_out = entropy_out.index_copy(0, query_idx, seg_entropy)
            maxw_out = maxw_out.index_copy(0, query_idx, seg_maxw)
            selected_idx_out[query_idx] = selected
            selected_start_out[query_idx] = seg_window_start
            selected_end_out[query_idx] = seg_window_end
            cursor = e + 1

        selected_escape_seconds = (
            selected_idx_out.to(video.dtype)
            * self.expansion_step_frames
            * self.window.feature_stride_seconds
        )
        seed_duration_seconds = (
            (seed_end - seed_start + 1).to(video.dtype) * self.window.feature_stride_seconds
        )

        return {
            "context": context_out,
            "attention_entropy": entropy_out,
            "max_attention": maxw_out,
            "selected_escape_seconds": selected_escape_seconds,
            "selected_window_start": selected_start_out,
            "selected_window_end": selected_end_out,
            "seed_segment_start": seed_start,
            "seed_segment_end": seed_end,
            "seed_segment_duration_seconds": seed_duration_seconds,
        }

    # ------------------------------------------------------------------
    # Shared fusion/classification. No projection, gate, or correction.
    # ------------------------------------------------------------------
    def classify(self, video: torch.Tensor, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        fused = torch.cat([video, context], dim=-1)
        fused = self.input_dropout(fused)
        logits = self.mstcn(fused.T.unsqueeze(0))  # expected [S,1,C,T]
        return logits, fused

    @staticmethod
    @torch.no_grad()
    def final_prediction(logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 4:
            raise ValueError(f"expected MS-TCN logits [S,1,C,T], got {tuple(logits.shape)}")
        return logits[-1, 0].argmax(dim=0)

    # ------------------------------------------------------------------
    # Full iterative forward.
    # ------------------------------------------------------------------
    def forward(
        self,
        video: torch.Tensor,
        raw600: Optional[torch.Tensor] = None,
        aux_valid: Optional[torch.Tensor] = None,
        force_context_zero: bool = False,
        n_rounds: Optional[int] = None,
    ) -> Dict[str, object]:
        self._validate_inputs(video, raw600)
        T = video.shape[0]
        rounds = self.n_rounds if n_rounds is None else int(n_rounds)
        if rounds < 1:
            raise ValueError("n_rounds must be >= 1")

        if aux_valid is None:
            aux_valid = torch.ones(T, dtype=torch.bool, device=video.device)
        else:
            if aux_valid.shape != (T,):
                raise ValueError("aux_valid must be [T]")
            aux_valid = aux_valid.bool()

        # Matched no-auxiliary path: same classifier shape, exact zero RAW600 context.
        if raw600 is None:
            zero_context = video.new_zeros((T, self.imu_dim))
            logits, fused = self.classify(video, zero_context)
            return {
                "logits": logits,
                "round_logits": [logits],
                "round_contexts": [zero_context],
                "round_fused": [fused],
                "round_predictions": [self.final_prediction(logits)],
                "round_diagnostics": [],
            }

        round_logits: List[torch.Tensor] = []
        round_contexts: List[torch.Tensor] = []
        round_fused: List[torch.Tensor] = []
        round_predictions: List[torch.Tensor] = []
        round_diagnostics: List[Dict[str, torch.Tensor]] = []

        previous_prediction: Optional[torch.Tensor] = None

        for round_idx in range(rounds):
            if round_idx == 0:
                diag = self.fixed_window_attention(video, raw600, aux_valid)
            else:
                assert previous_prediction is not None
                # HARD RULE: the dynamic seed comes only from the previous round's
                # detached MS-TCN prediction. No GT, no preliminary Conv1D teacher.
                with torch.no_grad():
                    seed_start, seed_end = self.labels_to_bounds(previous_prediction.detach())
                diag = self.segment_seed_expand_attention(
                    video, raw600, seed_start, seed_end, aux_valid
                )

            context = diag["context"]
            used_context = torch.zeros_like(context) if force_context_zero else context
            logits, fused = self.classify(video, used_context)
            prediction = self.final_prediction(logits).detach()

            round_logits.append(logits)
            round_contexts.append(context)
            round_fused.append(fused)
            round_predictions.append(prediction)
            round_diagnostics.append(diag)
            previous_prediction = prediction

        return {
            "logits": round_logits[-1],
            "round_logits": round_logits,
            "round_contexts": round_contexts,
            "round_fused": round_fused,
            "round_predictions": round_predictions,
            "round_diagnostics": round_diagnostics,
            "temperature": self.effective_temperature(),
        }
