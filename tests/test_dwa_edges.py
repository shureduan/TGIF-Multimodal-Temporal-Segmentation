"""Behavioral checks for the maintained attention rule, including gradients."""

import unittest

import torch

from tgif_dwa.boundary_dwa import BoundaryUncertaintyDWA
from tgif_dwa.iterative_dwa import WindowConfig
from tgif_dwa.wear_model import WearFinalModel


def small_parent(**kwargs):
    return BoundaryUncertaintyDWA(video_dim=4, imu_dim=2, attn_dim=2, n_classes=3,
                                  n_stages=2, n_layers=1, ch=4, **kwargs).eval()


class DwaEdgeTests(unittest.TestCase):
    def test_long_segment_query_chunks_keep_the_complete_key_support(self):
        model = small_parent(uncertainty_seconds=0.5)
        with torch.no_grad():
            model.q_proj.weight.zero_()
            model.k_proj.weight.zero_()
        sensor = torch.arange(1200, dtype=torch.float32).reshape(600, 2)
        start, end = model.labels_to_bounds(torch.tensor([0] * 300 + [1] * 300))
        out = model.segment_seed_expand_attention(
            torch.zeros(600, 4), sensor, start, end, torch.ones(600, dtype=torch.bool))
        first = (sensor[:300].sum(0) + 0.5 * sensor[300]) / 300.5
        second = (sensor[300:].sum(0) + 0.5 * sensor[299]) / 300.5
        torch.testing.assert_close(out['context'][:300], first.expand(300, -1))
        torch.testing.assert_close(out['context'][300:], second.expand(300, -1))
        self.assertEqual(out['selected_window_end'][:300].unique().tolist(), [300])
        self.assertEqual(out['selected_window_start'][300:].unique().tolist(), [299])

    def test_fast_attention_preserves_outputs_and_gradients(self):
        torch.manual_seed(71)
        model = small_parent().double()
        video = torch.randn(10, 4, dtype=torch.float64)
        sensor = torch.randn(10, 2, dtype=torch.float64)
        start, end = model.labels_to_bounds(torch.tensor([0, 0, 1, 0, 1, 1, 1, 2, 1, 1]))
        self.assertEqual(start.tolist(), [0, 0, 2, 3, 4, 4, 4, 7, 8, 8])
        self.assertEqual(end.tolist(), [1, 1, 2, 3, 6, 6, 6, 7, 9, 9])
        valid = torch.ones(10, dtype=torch.bool)
        reference = model.segment_seed_expand_attention(video, sensor, start, end, valid)
        reference['context'].square().sum().backward()
        gradients = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
        model.zero_grad(set_to_none=True)
        fast = model.segment_seed_expand_attention(
            video, sensor, start, end, valid, return_diagnostics=False)
        self.assertEqual(set(fast), {'context'})
        torch.testing.assert_close(fast['context'], reference['context'], rtol=0, atol=0)
        fast['context'].square().sum().backward()
        for name, parameter in model.named_parameters():
            if name in gradients:
                torch.testing.assert_close(parameter.grad, gradients[name], rtol=0, atol=0)

    def test_equal_scores_follow_core_margin_prior_and_exclude_invalid_keys(self):
        model = small_parent(uncertainty_seconds=0.5, margin_prior=0.5)
        with torch.no_grad():
            model.q_proj.weight.zero_()
            model.k_proj.weight.zero_()
        video = torch.zeros(4, 4)
        sensor = torch.tensor([[2., 4.], [6., 8.], [10., 12.], [100., 100.]])
        start, end = model.labels_to_bounds(torch.tensor([0, 1, 1, 2]))
        valid = torch.tensor([True, True, True, False])
        diag = model.segment_seed_expand_attention(video, sensor, start, end, valid)
        # Middle queries read one valid margin token plus two core tokens.
        expected = (0.5 * sensor[0] + sensor[1] + sensor[2]) / 2.5
        torch.testing.assert_close(diag["context"][1:3], expected.expand(2, -1))
        torch.testing.assert_close(diag["core_attention_mass"][1:3], torch.full((2,), 0.8))
        torch.testing.assert_close(diag["right_margin_attention_mass"][1:3], torch.zeros(2))

    def test_zero_margin_never_enables_saturation_expansion(self):
        model = small_parent(uncertainty_seconds=0)
        start, end = model.labels_to_bounds(torch.tensor([0, 0, 1, 1]))
        diag = model.segment_seed_expand_attention(
            torch.randn(4, 4), torch.randn(4, 2), start, end, torch.ones(4, dtype=torch.bool))
        self.assertTrue(torch.equal(diag["selected_window_start"], start))
        self.assertTrue(torch.equal(diag["selected_window_end"], end))
        torch.testing.assert_close(diag["margin_attention_mass"], torch.zeros(4))
        for margin in (-0.5, float("nan"), float("inf")):
            with self.subTest(margin=margin), self.assertRaises(ValueError):
                small_parent(uncertainty_seconds=margin)
        with self.assertRaisesRegex(ValueError, "max_escape_seconds"):
            small_parent(window=WindowConfig(max_escape_seconds=1.0))

    def test_all_masked_attention_has_zero_context_and_finite_backward(self):
        model = small_parent()
        video = torch.randn(4, 4, requires_grad=True)
        sensor = torch.full((4, 2), float("nan"), requires_grad=True)
        out = model(video, sensor, aux_valid=torch.zeros(4, dtype=torch.bool))
        for context in out["round_contexts"]:
            torch.testing.assert_close(context, torch.zeros_like(context))
        sum(logits.square().mean() for logits in out["round_logits"]).backward()
        self.assertTrue(torch.isfinite(video.grad).all())
        self.assertTrue(torch.isfinite(sensor.grad).all())
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_missing_imu_cannot_influence_parent_or_probe(self):
        torch.manual_seed(1)
        model = WearFinalModel().eval()
        video, sensor = torch.randn(5, 2048), torch.randn(5, 600)
        valid = torch.tensor([True, False, True, False, True])
        corrupted = sensor.clone()
        corrupted[~valid] = float("nan")
        with torch.inference_mode():
            a = model(video, sensor, aux_valid=valid)
            b = model(video, corrupted, aux_valid=valid)
        torch.testing.assert_close(a["probabilities"], b["probabilities"], rtol=0, atol=0)
        parent_prob = b["parent"]["logits"][-1, 0].T.softmax(-1)
        torch.testing.assert_close(b["probabilities"][~valid], parent_prob[~valid])
        torch.testing.assert_close(b["p_background"][~valid], torch.full((2,), 0.5))


if __name__ == "__main__":
    unittest.main()
