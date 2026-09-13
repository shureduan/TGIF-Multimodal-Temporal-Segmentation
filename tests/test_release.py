import hashlib
import os
import unittest
from pathlib import Path

import numpy as np
import torch

from tgif_dwa.boundary_dwa import BoundaryUncertaintyDWA
from tgif_dwa.iterative_dwa import IterativeRawDWA, WindowConfig
from tgif_dwa.segmentation_metrics import core_metrics
from tgif_dwa.wear_data import official_feature_timestamps, orient_feature
from tgif_dwa.wear_metrics import official_wear_concat_metrics, tal_map
from tgif_dwa.wear_model import load_wear_final, normalize_wear_inertial


ROOT = Path(__file__).resolve().parents[1]


def small_model(model_type=IterativeRawDWA):
    kwargs = dict(
        video_dim=4,
        imu_dim=2,
        attn_dim=2,
        n_classes=3,
        n_rounds=2,
        n_stages=2,
        n_layers=1,
        ch=4,
        window=WindowConfig(
            feature_stride_seconds=0.5,
            initial_radius_seconds=0.5,
            expansion_step_seconds=0.5,
            max_escape_seconds=0.0,
        ),
    )
    if model_type is BoundaryUncertaintyDWA:
        kwargs.update(uncertainty_seconds=0.5, margin_prior=0.5)
    return model_type(**kwargs).eval()


class ReleaseTests(unittest.TestCase):
    def test_tgif_original_assets_are_bit_exact(self):
        expected = {
            "assets/tgif_video_vs_multimodal.png": (
                "6393fc8d0157c2ea5ddd51d92db2e841291468c7d98953dabadc56d3fdcf3421"
            ),
            "assets/tgif_timeline_seed43.png": (
                "271b851ddfba48a06c99d42134b2e68dbe6e7c01756835c487c40bd934740f4f"
            ),
        }
        for relative_path, digest in expected.items():
            observed = hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()
            self.assertEqual(observed, digest)

    def test_two_round_shapes_and_prediction_seed(self):
        torch.manual_seed(0)
        model = small_model()
        output = model(torch.randn(8, 4), torch.randn(8, 2))
        self.assertEqual(tuple(output["logits"].shape), (2, 1, 3, 8))
        diagnostics = output["round_diagnostics"][1]
        expected_start = model.labels_to_bounds(output["round_predictions"][0])[0]
        self.assertTrue(torch.equal(diagnostics["seed_segment_start"], expected_start))

    def test_missing_auxiliary_uses_exact_zero_context(self):
        output = small_model()(torch.randn(7, 4), raw600=None)
        self.assertEqual(int(torch.count_nonzero(output["round_contexts"][0])), 0)

    def test_boundary_margin_support_and_prior_mass(self):
        torch.manual_seed(1)
        model = small_model(BoundaryUncertaintyDWA)
        video, inertial = torch.randn(6, 4), torch.randn(6, 2)
        start, end = model.labels_to_bounds(torch.tensor([0, 0, 1, 1, 1, 2]))
        diagnostics = model.segment_seed_expand_attention(
            video, inertial, start, end, torch.ones(6, dtype=torch.bool)
        )
        self.assertEqual(
            diagnostics["selected_window_start"].tolist(), [0, 0, 1, 1, 1, 4]
        )
        self.assertEqual(
            diagnostics["selected_window_end"].tolist(), [2, 2, 5, 5, 5, 5]
        )
        self.assertTrue(
            torch.allclose(
                diagnostics["core_attention_mass"]
                + diagnostics["margin_attention_mass"],
                torch.ones(6),
                atol=1e-6,
            )
        )

    def test_tgif_perfect_prediction_metrics(self):
        labels = np.array([0, 0, 1, 1, 2, 2])
        self.assertTrue(all(value == 100.0 for value in core_metrics(labels, labels).values()))

    def test_wear_perfect_prediction_metrics(self):
        labels = np.repeat(np.arange(19), 2)
        probabilities = np.eye(19, dtype=np.float32)[labels]
        frame = official_wear_concat_metrics([labels], [labels])
        temporal = tal_map(
            [{"id": "fixture", "true": labels, "pred": labels,
              "probabilities": probabilities}]
        )
        self.assertEqual(frame["macro_precision"], 1.0)
        self.assertEqual(frame["macro_recall"], 1.0)
        self.assertEqual(frame["macro_f1"], 1.0)
        self.assertEqual(temporal["map_at_0.5"], 1.0)

    def test_wear_orientation_timestamps_and_normalization(self):
        self.assertEqual(official_feature_timestamps(3).tolist(), [0.5, 1.0, 1.5])
        array = np.arange(3 * 600, dtype=np.float32).reshape(600, 3)
        oriented = orient_feature(array, 600)
        self.assertEqual(oriented.shape, (3, 600))
        normalized = normalize_wear_inertial(
            oriented, np.zeros(12, dtype=np.float32), np.ones(12, dtype=np.float32)
        )
        self.assertTrue(np.array_equal(normalized, oriented))

    @unittest.skipUnless(
        os.environ.get("WEAR_PARENT_CKPT") and os.environ.get("WEAR_PROBE_CKPT"),
        "set WEAR_PARENT_CKPT and WEAR_PROBE_CKPT for private-weight validation",
    )
    def test_private_wear_pair_strictly_loads(self):
        model, mean, std = load_wear_final(
            os.environ["WEAR_PARENT_CKPT"], os.environ["WEAR_PROBE_CKPT"]
        )
        self.assertEqual(mean.shape, (12,))
        self.assertEqual(std.shape, (12,))
        with torch.inference_mode():
            output = model(torch.randn(5, 2048), torch.randn(5, 600))
        self.assertEqual(tuple(output["probabilities"].shape), (5, 19))


if __name__ == "__main__":
    unittest.main()
