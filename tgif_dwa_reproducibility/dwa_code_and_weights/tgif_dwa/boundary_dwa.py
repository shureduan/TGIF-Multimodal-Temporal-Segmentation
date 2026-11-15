"""Round-1 DWA with the final fixed boundary-uncertainty support."""

from __future__ import annotations

import math
from typing import Dict

import torch

from .iterative_dwa import IterativeRawDWA, WindowConfig


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
        # Fixed-margin DWA must never silently enable the legacy saturation
        # expansion, including when its margin is set to zero for an ablation.
        kwargs.setdefault("window", WindowConfig(max_escape_seconds=0.0))
        super().__init__(*args, **kwargs)
        if self.window.max_escape_seconds != 0:
            raise ValueError("fixed-margin DWA requires max_escape_seconds=0")
        self.uncertainty_seconds = float(uncertainty_seconds)
        self.margin_prior = float(margin_prior)
        if not math.isfinite(self.uncertainty_seconds) or self.uncertainty_seconds < 0:
            raise ValueError("uncertainty_seconds must be finite and nonnegative")
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
        return_diagnostics: bool = True,
    ) -> Dict[str, torch.Tensor]:
        length = video.shape[0]
        if seed_start.shape != (length,) or seed_end.shape != (length,):
            raise ValueError("seed_start/seed_end must be [T]")

        q_all, k_all = self._project_qk(video, raw600)
        starts = seed_start.detach().cpu().tolist()
        ends = seed_end.detach().cpu().tolist()
        # Group similarly sized segments into padded batches. Every query still
        # reads exactly its own segment+margin; padding is strictly masked.
        # Long segments are split along the query axis only, never the key axis.
        # This preserves the complete softmax support while bounding memory.
        groups = {}
        long_segments = []
        cursor = 0
        max_cells = 1 << 20
        while cursor < length:
            start, end = starts[cursor], ends[cursor]
            if start != cursor or end < start or end >= length:
                raise ValueError(f"invalid contiguous seed [{start},{end}] at {cursor}")
            left = max(0, start - self.uncertainty_steps)
            right = min(length - 1, end + self.uncertainty_steps)
            key_length = right - left + 1
            if key_length > 256:
                # Preserve views for long supports. Padding/gathering the same
                # large Value tensor for every query chunk wastes memory.
                long_segments.append((start, end, left, right))
                cursor = end + 1
                continue
            key_width = 1 << (key_length - 1).bit_length()
            query_limit = min(256, max(1, max_cells // key_width))
            for query_start in range(start, end + 1, query_limit):
                query_length = min(query_limit, end - query_start + 1)
                query_width = 1 << (query_length - 1).bit_length()
                groups.setdefault((query_width, key_width), []).append(
                    (query_start, query_length, left, key_length, start, end))
            cursor = end + 1

        indices, contexts = [], []
        diagnostics = {name: [] for name in (
            "attention_entropy", "max_attention", "core_attention_mass",
            "left_margin_attention_mass", "right_margin_attention_mass",
            "qk_cosine_std", "selected_window_start", "selected_window_end",
        )}
        temperature = self.effective_temperature()
        for start, end, left, right in long_segments:
            key_index = torch.arange(left, right + 1, device=video.device)
            keys, values = k_all[left:right + 1], raw600[left:right + 1]
            valid = aux_valid[left:right + 1].bool()
            core = (key_index >= start) & (key_index <= end)
            prior = torch.where(core, video.new_tensor(1.0), video.new_tensor(self.margin_prior))
            chunk_size = min(256, max(1, max_cells // len(key_index)))
            for query_start in range(start, end + 1, chunk_size):
                query_end = min(end + 1, query_start + chunk_size)
                cosine = q_all[query_start:query_end] @ keys.T
                scores = temperature * cosine + torch.log(prior[None, :] + 1e-6)
                attention = self._safe_attention(scores, valid[None, :])
                indices.append(torch.arange(query_start, query_end, device=video.device))
                contexts.append(attention @ values)
                if return_diagnostics:
                    with torch.no_grad():
                        diagnostics["attention_entropy"].append(
                            -(attention.clamp_min(1e-8).log() * attention).sum(-1))
                        diagnostics["max_attention"].append(attention.max(-1).values)
                        for name, mask in (
                            ("core_attention_mass", core),
                            ("left_margin_attention_mass", key_index < start),
                            ("right_margin_attention_mass", key_index > end),
                        ):
                            diagnostics[name].append((attention * mask[None, :]).sum(-1))
                        diagnostics["qk_cosine_std"].append(cosine.std(-1, unbiased=False))
                        for name, boundary in (("selected_window_start", left), ("selected_window_end", right)):
                            diagnostics[name].append(torch.full(
                                (query_end - query_start,), boundary, dtype=torch.long, device=video.device))

        for (query_width, key_width), jobs in groups.items():
            batch_limit = max(1, min(256, max_cells // (query_width * key_width),
                                     (1 << 15) // key_width))
            q_offsets = torch.arange(query_width, device=video.device)[None, :]
            k_offsets = torch.arange(key_width, device=video.device)[None, :]
            for offset in range(0, len(jobs), batch_limit):
                spec = torch.tensor(jobs[offset:offset + batch_limit],
                                    dtype=torch.long, device=video.device)
                query_index = spec[:, 0:1] + q_offsets
                query_valid = q_offsets < spec[:, 1:2]
                key_index = spec[:, 2:3] + k_offsets
                key_slot = k_offsets < spec[:, 3:4]
                safe_key_index = key_index.clamp_max(length - 1)
                valid = key_slot & aux_valid[safe_key_index].bool()
                queries = q_all[query_index.clamp_max(length - 1)]
                keys = k_all[safe_key_index]
                values = raw600[safe_key_index]
                cosine = torch.bmm(queries, keys.transpose(1, 2))
                core = (key_index >= spec[:, 4:5]) & (key_index <= spec[:, 5:6])
                prior = torch.where(core, video.new_tensor(1.0),
                                    video.new_tensor(self.margin_prior))
                scores = temperature * cosine + torch.log(prior[:, None, :] + 1e-6)
                attention = self._safe_attention(scores, valid[:, None, :])
                context = torch.bmm(attention, values)
                indices.append(query_index[query_valid])
                contexts.append(context[query_valid])

                if return_diagnostics:
                    with torch.no_grad():
                        diagnostics["attention_entropy"].append(
                            (-(attention.clamp_min(1e-8).log() * attention).sum(-1))[query_valid])
                        diagnostics["max_attention"].append(attention.max(-1).values[query_valid])
                        for name, mask in (
                            ("core_attention_mass", core),
                            ("left_margin_attention_mass", key_index < spec[:, 4:5]),
                            ("right_margin_attention_mass", key_index > spec[:, 5:6]),
                        ):
                            diagnostics[name].append(
                                (attention * mask[:, None, :]).sum(-1)[query_valid])
                        count = spec[:, 3:4].to(cosine.dtype)
                        average = (cosine * key_slot[:, None, :]).sum(-1) / count
                        variance = ((cosine - average[..., None]).square()
                                    * key_slot[:, None, :]).sum(-1) / count
                        diagnostics["qk_cosine_std"].append(variance.sqrt()[query_valid])
                        diagnostics["selected_window_start"].append(
                            spec[:, 2:3].expand(-1, query_width)[query_valid])
                        diagnostics["selected_window_end"].append(
                            (spec[:, 2:3] + spec[:, 3:4] - 1).expand(-1, query_width)[query_valid])

        order = torch.cat(indices)
        result = {"context": video.new_zeros((length, self.imu_dim)).index_copy(
            0, order, torch.cat(contexts))}
        if not return_diagnostics:
            return result
        for name, values in diagnostics.items():
            concatenated = torch.cat(values)
            result[name] = concatenated.new_zeros(length).index_copy(0, order, concatenated)
        result.update({
            "selected_escape_seconds": torch.full(
                (length,), self.uncertainty_seconds, device=video.device, dtype=video.dtype),
            "seed_segment_start": seed_start,
            "seed_segment_end": seed_end,
            "seed_segment_duration_seconds": (
                (seed_end - seed_start + 1).to(video.dtype) * self.window.feature_stride_seconds),
            "margin_attention_mass": result["left_margin_attention_mass"]
                                     + result["right_margin_attention_mass"],
        })
        return result
