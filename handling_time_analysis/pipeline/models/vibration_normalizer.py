"""Robust vibration normalizer (README_VIBRATION_NORMALIZATION §2).

Per-station median/IQR fit on TRAIN only; val/test read the stored stats; an
unseen station falls back to the global-train stats. This is the FIRST of the
two scale controls — it makes the two raw vib features (peak_energy, psd_20_40,
which live on very different linear scales) comparable to each other. The SECOND
(scale-match to the VIDEO feature) is done downstream by L2-normalized Q/K and
LayerNorm'd residual fusion (README §4-§5)."""
import numpy as np


class RobustVibrationNormalizer:
    def __init__(self, eps=1e-3):
        self.eps = eps
        self.stats = {}

    def fit(self, features, station_ids):
        """features: [N,D] raw window features; station_ids: [N] station per row."""
        features = np.asarray(features, np.float64)
        station_ids = np.asarray(station_ids)
        for st in set(station_ids.tolist()):
            x = features[station_ids == st]
            iqr = np.quantile(x, .75, 0) - np.quantile(x, .25, 0)
            self.stats[st] = (np.median(x, 0), np.maximum(iqr, self.eps))
        giqr = np.quantile(features, .75, 0) - np.quantile(features, .25, 0)
        self.stats["_global"] = (np.median(features, 0), np.maximum(giqr, self.eps))
        return self

    def transform(self, features, station_id):
        med, iqr = self.stats.get(station_id, self.stats["_global"])
        return ((np.asarray(features, np.float64) - med) / iqr).astype(np.float32)
