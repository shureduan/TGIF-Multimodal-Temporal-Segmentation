"""Round-1 DWA with the final fixed boundary-uncertainty support."""

from __future__ import annotations

from typing import Dict

import torch

from .iterative_dwa import IterativeRawDWA


class BoundaryUncertaintyDWA(IterativeRawDWA):
    """Use a soft prior on a fixed margin around each predicted segment.

    Parent cosine scores and the learned temperature are unchanged. The seed
    segment has prior 1.0 and each valid margin token has prior 0.5 by default;
    tokens outside the support are not candidates.
    """

    def __init__(
        self,
        *args,
        uncertainty_seconds: float = 1.0,
        margin_prior: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.uncertainty_seconds = float(uncertainty_seconds)
        self.margin_prior = float(margin_prior)
        steps = self.uncertainty_seconds / self.window.feature_stride_seconds
        if abs(steps - round(steps)) > 1e-9:
            raise ValueError("uncertainty_seconds must align to feature stride")
        self.uncertainty_steps = int(round(steps))
        if not 0 < self.margin_prior <= 1:
            raise ValueError("margin_prior must be in (0,1]")

    def segment_seed_expand_attention(
        self,
        video: torch.Tensor,
        raw600: torch.Tensor,
        seed_start: torch.Tensor,
        seed_end: torch.Tensor,
        aux_valid: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        # Preserve the exact parent path when no uncertainty margin is requested.
        if self.uncertainty_steps == 0:
            return super().segment_seed_expand_attention(
                video, raw600, seed_start, seed_end, aux_valid
            )

        length = video.shape[0]
        if seed_start.shape != (length,) or seed_end.shape != (length,):
            raise ValueError("seed_start/seed_end must be [T]")

        q_all, k_all = self._project_qk(video, raw600)
        context = video.new_zeros((length, self.imu_dim))
        entropy = video.new_zeros(length)
        max_weight = video.new_zeros(length)
        core_mass = video.new_zeros(length)
        left_mass = video.new_zeros(length)
        right_mass = video.new_zeros(length)
        cosine_std = video.new_zeros(length)
        window_start = torch.zeros(length, dtype=torch.long, device=video.device)
        window_end = torch.zeros_like(window_start)

        cursor = 0
        while cursor < length:
            start = int(seed_start[cursor])
            end = int(seed_end[cursor])
            if start != cursor or end < start or end >= length:
                raise ValueError(f"invalid contiguous seed [{start},{end}] at {cursor}")

            left = max(0, start - self.uncertainty_steps)
            right = min(length - 1, end + self.uncertainty_steps)
            candidate_indices = torch.arange(left, right + 1, device=video.device)
            valid = aux_valid[left:right + 1].bool()
            queries = q_all[start:end + 1]
            keys = k_all[left:right + 1]
            values = raw600[left:right + 1]

            cosine = queries @ keys.T
            base_scores = self.effective_temperature() * cosine
            prior = torch.where(
                (candidate_indices >= start) & (candidate_indices <= end),
                torch.ones_like(candidate_indices, dtype=video.dtype),
                torch.full_like(
                    candidate_indices, self.margin_prior, dtype=video.dtype
                ),
            )
            scores = base_scores + torch.log(prior + 1e-6)
            attention = self._safe_attention(
                scores, valid[None, :].expand(len(queries), -1)
            )
            segment_context = attention @ values

            query_indices = torch.arange(start, end + 1, device=video.device)
            context[query_indices] = segment_context
            entropy[query_indices] = -(
                attention.clamp_min(1e-8).log() * attention
            ).sum(-1)
            max_weight[query_indices] = attention.max(-1).values
            core = (candidate_indices >= start) & (candidate_indices <= end)
            left_margin = candidate_indices < start
            right_margin = candidate_indices > end
            core_mass[query_indices] = attention[:, core].sum(-1)
            left_mass[query_indices] = (
                attention[:, left_margin].sum(-1) if left_margin.any() else 0
            )
            right_mass[query_indices] = (
                attention[:, right_margin].sum(-1) if right_margin.any() else 0
            )
            cosine_std[query_indices] = cosine.std(-1, unbiased=False)
            window_start[query_indices] = left
            window_end[query_indices] = right
            cursor = end + 1

        duration = (
            (seed_end - seed_start + 1).to(video.dtype)
            * self.window.feature_stride_seconds
        )
        return {
            "context": context,
            "attention_entropy": entropy,
            "max_attention": max_weight,
            "selected_escape_seconds": torch.full(
                (length,),
                self.uncertainty_seconds,
                device=video.device,
                dtype=video.dtype,
            ),
            "selected_window_start": window_start,
            "selected_window_end": window_end,
            "seed_segment_start": seed_start,
            "seed_segment_end": seed_end,
            "seed_segment_duration_seconds": duration,
            "core_attention_mass": core_mass,
            "left_margin_attention_mass": left_mass,
            "right_margin_attention_mass": right_mass,
            "margin_attention_mass": left_mass + right_mass,
            "qk_cosine_std": cosine_std,
        }
