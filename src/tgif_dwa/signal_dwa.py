"""Experimental sensor-driven resizing BEFORE video-query attention.

Statistics are engineering proxies, not estimates of semantic information.
Calibration receives only normalized training-fold IMU arrays. Video, class
labels, attention weights and test scores are not window-controller inputs.
"""
from dataclasses import dataclass, asdict

import numpy as np
import torch

from .iterative_dwa import IterativeRawDWA, WindowConfig


@dataclass(frozen=True)
class SignalAblationConfig:
    resize: bool = True
    contract: bool = True
    expand: bool = True
    learned_attention: bool = True


@dataclass(frozen=True)
class SignalWindowConfig:
    channels: int = 12
    samples_per_token: int = 50
    sample_rate_hz: float = 50.0
    stride_seconds: float = 0.5
    boundary_span_steps: int = 2
    min_tokens: int = 3
    max_expansion_steps: int = 12
    boundary_quantile: float = 0.98
    stability_quantile: float = 0.50
    equivalence_quantile: float = 0.50
    calibration_rows_per_subject: int = 512

    def __post_init__(self):
        for key in ('channels', 'samples_per_token', 'boundary_span_steps', 'min_tokens',
                    'calibration_rows_per_subject'):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f'{key} must be a positive integer')
        if type(self.max_expansion_steps) is not int or self.max_expansion_steps < 0:
            raise ValueError('max_expansion_steps must be a nonnegative integer')
        if self.min_tokens < 3 or self.samples_per_token < 4:
            raise ValueError('need at least three temporal tokens and four raw samples')
        if not np.isfinite(self.sample_rate_hz) or self.sample_rate_hz <= 0:
            raise ValueError('sample_rate_hz must be positive and finite')
        if not np.isfinite(self.stride_seconds) or self.stride_seconds <= 0:
            raise ValueError('stride_seconds must be positive and finite')
        for key in ('boundary_quantile', 'stability_quantile', 'equivalence_quantile'):
            if not 0 < getattr(self, key) < 1:
                raise ValueError(f'{key} must be in (0,1)')


def descriptors(raw, config):
    """Per-channel mean, log energy and four log spectral-band powers.

    These 6*C statistics control support only. Attention V remains full RAW600.
    Band cutoffs are fractions of Nyquist: .08/.20/.48/1 (2/5/12/25 Hz at 50 Hz).
    """
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != config.channels * config.samples_per_token:
        raise ValueError('sensor array does not match the configured raw layout')
    if not len(raw) or not np.isfinite(raw).all():
        raise ValueError('sensor array must be nonempty and finite after masking')
    x = raw.reshape(-1, config.channels, config.samples_per_token)
    mean = x.mean(-1)
    energy = np.log1p(np.mean(x * x, -1))
    centered = x - mean[..., None]
    fft = np.fft.rfft(centered, axis=-1)
    power = abs(fft) ** 2 / config.samples_per_token ** 2
    frequencies = np.fft.rfftfreq(config.samples_per_token, d=1 / config.sample_rate_hz)
    nyquist = config.sample_rate_hz / 2
    bands = []
    for low, high in zip((0, .08, .20, .48), (.08, .20, .48, 1.0)):
        mask = (frequencies > low * nyquist) & (frequencies <= high * nyquist)
        bands.append(np.log1p(power[..., mask].sum(-1)))
    return np.stack([mean, energy, *bands], axis=-1).reshape(len(x), -1)


class _Moments:
    def __init__(self, x):
        self.first = np.vstack([np.zeros((1, x.shape[1])), np.cumsum(x, axis=0)])
        self.second = np.vstack([np.zeros((1, x.shape[1])), np.cumsum(x * x, axis=0)])

    def at(self, left, right):
        count = (right - left + 1)[:, None]
        mean = (self.first[right + 1] - self.first[left]) / count
        variance = np.maximum((self.second[right + 1] - self.second[left]) / count - mean ** 2, 0)
        return np.concatenate([mean, np.sqrt(variance)], axis=-1)

    def stability(self, left, right):
        middle = (left + right) // 2
        result = distance(self.at(left, middle), self.at(np.minimum(middle + 1, right), right))
        return np.where(right - left + 1 >= 3, result, np.inf)


def distance(a, b):
    return np.sqrt(np.mean((a - b) ** 2, axis=-1))


def boundary_scores(x, valid, span):
    """Sensor change score at the cut before token i; no label boundaries."""
    total = len(x)
    scores = np.zeros(total)
    if total >= 2 * span:
        positions = np.arange(span, total - span + 1)
        moments = _Moments(x)
        values = distance(moments.at(positions - span, positions - 1),
                          moments.at(positions, positions + span - 1))
        counts = np.r_[0, np.cumsum(valid)]
        complete = counts[positions + span] - counts[positions - span] == 2 * span
        scores[positions] = np.where(complete, values, 0.0)
    return scores


class SignalAdaptiveDWA(IterativeRawDWA):
    """Predicted seed -> sensor-only resize -> one-way Q_video/KV_sensor.

    Round 0 starts at +/-2 s and is resized too. Round 1 starts at the detached
    predicted segment. Decisions are discrete and deterministic, not learned
    scalar radii. Q/K, temperature and MS-TCN are trained through final contexts.
    """
    def __init__(self, *, signal_config=SignalWindowConfig(), ablation_config=SignalAblationConfig(), **kwargs):
        self.signal_config = signal_config
        self.ablation_config = ablation_config
        kwargs.setdefault('imu_dim', signal_config.channels * signal_config.samples_per_token)
        kwargs.setdefault('window', WindowConfig(feature_stride_seconds=signal_config.stride_seconds,
                                                 max_escape_seconds=0.0))
        super().__init__(**kwargs)
        if self.imu_dim != signal_config.channels * signal_config.samples_per_token:
            raise ValueError('IMU dimension does not match signal descriptor layout')
        if self.window.feature_stride_seconds != signal_config.stride_seconds:
            raise ValueError('window and signal feature strides differ')
        self.register_buffer('signal_center', torch.zeros(signal_config.channels * 6))
        self.register_buffer('signal_scale', torch.ones(signal_config.channels * 6))
        self.register_buffer('signal_thresholds', torch.ones(3))
        self.register_buffer('signal_calibrated', torch.tensor(False))

    def fit_signal_statistics(self, training_inertial):
        """Fit frozen descriptor scales and decision thresholds without labels."""
        config = self.signal_config
        arrays = [descriptors(x, config) for x in training_inertial]
        if not arrays:
            raise ValueError('training sensor arrays are required')
        def sample(x):
            return x[np.linspace(0, len(x) - 1, min(len(x), config.calibration_rows_per_subject), dtype=int)]
        sampled = np.concatenate([sample(x) for x in arrays])
        center = np.median(sampled, axis=0)
        scale = np.maximum(np.percentile(sampled, 75, axis=0) - np.percentile(sampled, 25, axis=0), .05)
        changes, stability, equivalence = [], [], []
        for raw in arrays:
            x = (raw - center) / scale
            n = len(x); t = np.arange(n); moments = _Moments(x)
            lo, hi = np.maximum(0, t - 4), np.minimum(n - 1, t + 4)
            small_lo, small_hi = np.maximum(0, t - 2), np.minimum(n - 1, t + 2)
            changes.append(sample(boundary_scores(x, np.ones(n, bool), config.boundary_span_steps)))
            s = moments.stability(lo, hi)
            if np.isfinite(s).any():
                stability.append(sample(s[np.isfinite(s)]))
            equivalence.append(sample(distance(moments.at(lo, hi), moments.at(small_lo, small_hi))))
        if not stability:
            raise ValueError('calibration requires a sequence of at least three tokens')
        thresholds = [max(float(np.quantile(np.concatenate(values), q)), 1e-3)
                      for values, q in zip((changes, stability, equivalence),
                                          (config.boundary_quantile, config.stability_quantile,
                                           config.equivalence_quantile))]
        with torch.no_grad():
            for buffer, value in ((self.signal_center, center), (self.signal_scale, scale),
                                  (self.signal_thresholds, thresholds)):
                buffer.copy_(torch.as_tensor(value, device=buffer.device, dtype=buffer.dtype))
            self.signal_calibrated.fill_(True)
        return {'descriptor': 'channel_mean_log_energy_four_log_band_powers',
                'config': asdict(config), 'training_sequences': len(arrays),
                'boundary_threshold': thresholds[0], 'stability_threshold': thresholds[1],
                'equivalence_threshold': thresholds[2]}

    @torch.no_grad()
    def select_sensor_windows(self, raw600, seed_start, seed_end, aux_valid):
        """Inspect and resize support using sensor statistics before Q/K scoring."""
        if not bool(self.signal_calibrated):
            raise ValueError('fit training-fold signal statistics before using adaptive windows')
        config = self.signal_config
        valid = aux_valid.detach().cpu().numpy().astype(bool)
        raw = raw600.detach().cpu().numpy().copy()
        raw[~valid] = 0
        x = (descriptors(raw, config) - self.signal_center.cpu().numpy()) / self.signal_scale.cpu().numpy()
        n = len(x); positions = np.arange(n)
        left = seed_start.detach().cpu().numpy().astype(np.int64).copy()
        right = seed_end.detach().cpu().numpy().astype(np.int64).copy()
        if (left.shape != (n,) or right.shape != (n,) or (left < 0).any() or (right >= n).any()
                or (left > positions).any() or (right < positions).any()):
            raise ValueError('every seed must be in range and contain its query')
        original_left, original_right = left.copy(), right.copy()
        boundary_tau, stable_tau, equivalent_tau = self.signal_thresholds.cpu().numpy()
        score = boundary_scores(x, valid, config.boundary_span_steps)
        # Local maxima suppress multiple cuts around one transition. Missing data
        # creates hard barriers, so it cannot alter neighboring valid contexts.
        cut = (score > boundary_tau) & (score >= np.r_[score[0], score[:-1]]) & (score > np.r_[score[1:], score[-1]])
        if not self.ablation_config.resize:
            cut[:] = False
        cut[0] = True
        cut[1:] |= ~valid[:-1] | ~valid[1:]
        region_left = np.maximum.accumulate(np.where(cut, positions, 0))
        ends = np.r_[cut[1:], True]
        region_right = np.minimum.accumulate(np.where(ends, positions, n - 1)[::-1])[::-1]
        if self.ablation_config.resize and self.ablation_config.contract:
            clip_left, clip_right = region_left, region_right
        else:
            # Missing data remains a hard safety boundary in every variant.
            # Disabling contraction removes BOTH signal-boundary clipping and
            # redundancy trimming, not merely one source of window shrinkage.
            missing_cut = np.r_[True, ~valid[:-1] | ~valid[1:]]
            clip_left = np.maximum.accumulate(np.where(missing_cut, positions, 0))
            clip_right = np.minimum.accumulate(np.where(np.r_[missing_cut[1:], True], positions, n - 1)[::-1])[::-1]
        left = np.maximum(left, clip_left); right = np.minimum(right, clip_right)
        clipped_left, clipped_right = left.copy(), right.copy()
        moments = _Moments(x)
        reference = moments.at(left, right)
        # Contract only when the smaller current window preserves the reference
        # moments AND its temporal halves are stable. Compare to the same anchor
        # throughout, preventing accumulated small changes from drifting away.
        while self.ablation_config.resize and self.ablation_config.contract:
            candidate_left = (left + positions + 1) // 2
            candidate_right = (right + positions) // 2
            accept = (valid & (candidate_right - candidate_left + 1 >= config.min_tokens)
                      & ((candidate_left != left) | (candidate_right != right))
                      & (distance(moments.at(candidate_left, candidate_right), reference) <= equivalent_tau)
                      & (moments.stability(candidate_left, candidate_right) <= stable_tau))
            if not accept.any():
                break
            left[accept] = candidate_left[accept]; right[accept] = candidate_right[accept]
        contracted_left, contracted_right = left.copy(), right.copy()
        # Expand unstable/undersampled windows one feature step at a time, with
        # independent endpoint clipping at sensor transitions and missing data.
        expansion_budget = config.max_expansion_steps if self.ablation_config.resize and self.ablation_config.expand else 0
        for _ in range(expansion_budget):
            active = valid & (((right - left + 1) < config.min_tokens)
                              | (moments.stability(left, right) > stable_tau))
            candidate_left = np.minimum(left, np.maximum(region_left, left - 1))
            candidate_right = np.maximum(right, np.minimum(region_right, right + 1))
            active &= (candidate_left != left) | (candidate_right != right)
            if not active.any():
                break
            left[active] = candidate_left[active]; right[active] = candidate_right[active]
        left[~valid] = positions[~valid]; right[~valid] = positions[~valid]
        arrays = {
            'selected_window_start': left, 'selected_window_end': right,
            'seed_segment_start': original_left, 'seed_segment_end': original_right,
            'boundary_clipped_tokens': (original_right - original_left) - (clipped_right - clipped_left),
            'redundancy_trimmed_tokens': (clipped_right - clipped_left) - (contracted_right - contracted_left),
            'expanded_tokens': (right - left) - (contracted_right - contracted_left),
            'sensor_boundary_score': score,
            'sensor_statistics_stable': (moments.stability(left, right) <= stable_tau) & valid,
        }
        # MPS does not support float64 diagnostic tensors. Selection itself
        # remains the same NumPy float64 calculation on every device.
        return {key: torch.as_tensor(value, device=raw600.device,
                    dtype=torch.float32 if raw600.device.type == 'mps' and value.dtype.kind == 'f' else None)
                for key, value in arrays.items()}

    def _sensor_attention(self, video, raw600, seed_start, seed_end, aux_valid, return_diagnostics):
        # Deliberate ordering: no video feature/projection or attention context
        # is available to the sensor-only support selector.
        raw600 = raw600.masked_fill(~aux_valid[:, None], 0.0)
        diagnostics = self.select_sensor_windows(raw600, seed_start, seed_end, aux_valid)
        left = diagnostics['selected_window_start']; right = diagnostics['selected_window_end']
        learned = self.ablation_config.learned_attention
        q, k = self._project_qk(video, raw600) if learned else (None, None)
        length = right - left + 1
        widths = 2 ** torch.ceil(torch.log2(length.float())).long()
        contexts, entropies, maxima, indices = [], [], [], []
        # Reuse one K/V view for shared long supports instead of duplicating
        # RAW600 for every query. Query chunking never clips the key support.
        long_groups = {}
        left_list, right_list = left.cpu().tolist(), right.cpu().tolist()
        for query in torch.nonzero(length > 128).flatten().cpu().tolist():
            pair = (left_list[query], right_list[query])
            long_groups.setdefault(pair, []).append(query)
        for (lo, hi), group in long_groups.items():
            keys, values = k[lo:hi + 1] if learned else None, raw600[lo:hi + 1]
            selected = torch.tensor(group, device=video.device)
            for subset in selected.split(min(256, max(1, (1 << 20) // (hi - lo + 1)))):
                mask = aux_valid[lo:hi + 1][None, :] & aux_valid[subset, None]
                scores = (self.effective_temperature() * (q[subset] @ keys.T) if learned
                          else video.new_zeros((len(subset), hi - lo + 1)))
                attention = self._safe_attention(scores, mask)
                contexts.append(attention @ values); indices.append(subset)
                if return_diagnostics:
                    entropies.append(-(attention.clamp_min(1e-8).log() * attention).sum(-1))
                    maxima.append(attention.max(-1).values)
        for width in widths[length <= 128].unique().tolist():

            queries = torch.nonzero(widths == width).flatten()
            # Limit gathered RAW value memory, not the selected temporal support.
            query_chunk = max(1, min(256, (1 << 20) // (width * self.imu_dim)))
            offsets = torch.arange(width, device=video.device)
            for subset in queries.split(query_chunk):
                key_index = left[subset, None] + offsets
                mask = (key_index <= right[subset, None]) & (key_index < len(video))
                safe = key_index.clamp(max=len(video) - 1)
                mask &= aux_valid[safe] & aux_valid[subset, None]
                scores = (self.effective_temperature() * torch.einsum('qd,qkd->qk', q[subset], k[safe]) if learned
                          else video.new_zeros((len(subset), width)))
                attention = self._safe_attention(scores, mask)
                contexts.append(torch.einsum('qk,qkd->qd', attention, raw600[safe]))
                indices.append(subset)
                if return_diagnostics:
                    entropies.append(-(attention.clamp_min(1e-8).log() * attention).sum(-1))
                    maxima.append(attention.max(-1).values)
        order = torch.argsort(torch.cat(indices))
        result = {'context': torch.cat(contexts)[order]}
        if return_diagnostics:
            result.update(diagnostics)
            result.update(attention_entropy=torch.cat(entropies)[order], max_attention=torch.cat(maxima)[order])
        return result

    def fixed_window_attention(self, video, raw600, aux_valid, return_diagnostics=True):
        positions = torch.arange(len(video), device=video.device)
        return self._sensor_attention(video, raw600, (positions - self.initial_radius_steps).clamp_min(0),
                                      (positions + self.initial_radius_steps).clamp_max(len(video) - 1),
                                      aux_valid, return_diagnostics)

    def segment_seed_expand_attention(self, video, raw600, seed_start, seed_end, aux_valid,
                                     return_diagnostics=True):
        return self._sensor_attention(video, raw600, seed_start, seed_end, aux_valid, return_diagnostics)
