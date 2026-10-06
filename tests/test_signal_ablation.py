from dataclasses import asdict
import copy
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from tgif_dwa.signal_dwa import SignalAdaptiveDWA, SignalWindowConfig
from tgif_dwa.signal_wear import VARIANTS, METHOD, load_signal_wear
from tgif_dwa import signal_training


def toy(method):
    model = SignalAdaptiveDWA(signal_config=SignalWindowConfig(channels=1),
        ablation_config=VARIANTS[method], video_dim=4, imu_dim=50, attn_dim=4,
        n_classes=3, n_stages=1, n_layers=1, ch=4).eval()
    model.signal_calibrated.fill_(True)
    model.signal_thresholds.copy_(torch.tensor([.02, .05, .05]))
    return model


class AblationTests(unittest.TestCase):
    def test_no_resize_preserves_seed_and_never_uses_signal_barriers(self):
        model = toy('NO_SENSOR_RESIZE'); n = 31
        x = torch.zeros(n, 50); x[15:] = 5
        t = torch.arange(n); lo = (t - 4).clamp_min(0); hi = (t + 4).clamp_max(n - 1)
        d = model.select_sensor_windows(x, lo, hi, torch.ones(n, dtype=torch.bool))
        torch.testing.assert_close(d['selected_window_start'], lo)
        torch.testing.assert_close(d['selected_window_end'], hi)
        self.assertEqual(int(d['expanded_tokens'].sum() + d['redundancy_trimmed_tokens'].sum() + d['boundary_clipped_tokens'].sum()), 0)

    def test_no_contraction_never_shrinks_even_across_signal_changes(self):
        m = toy('NO_CONTRACTION'); x = torch.zeros(31, 50); x[15:] = 5
        t = torch.arange(31); lo = (t - 8).clamp_min(0); hi = (t + 8).clamp_max(30)
        d = m.select_sensor_windows(x, lo, hi, torch.ones(31, dtype=torch.bool))
        self.assertTrue((d['selected_window_start'] <= lo).all())
        self.assertTrue((d['selected_window_end'] >= hi).all())
        self.assertEqual(int(d['redundancy_trimmed_tokens'].sum() + d['boundary_clipped_tokens'].sum()), 0)

    def test_no_expansion_never_grows_the_seed(self):
        m = toy('NO_EXPANSION'); x = torch.zeros(31, 50); x[15] = 5
        t = torch.arange(31); lo = (t - 1).clamp_min(0); hi = (t + 1).clamp_max(30)
        d = m.select_sensor_windows(x, lo, hi, torch.ones(31, dtype=torch.bool))
        self.assertTrue((d['selected_window_start'] >= lo).all())
        self.assertTrue((d['selected_window_end'] <= hi).all())
        self.assertEqual(int(d['expanded_tokens'].sum()), 0)

    def test_uniform_pool_uses_selected_raw_mean_without_qk(self):
        m = toy('UNIFORM_POOL'); raw = torch.randn(20, 50, requires_grad=True)
        with patch.object(m, '_project_qk', side_effect=AssertionError('Q/K must be disabled')):
            d = m.fixed_window_attention(torch.randn(20, 4), raw, torch.ones(20, dtype=torch.bool))
        for i in range(20):
            lo, hi = int(d['selected_window_start'][i]), int(d['selected_window_end'][i])
            torch.testing.assert_close(d['context'][i], raw[lo:hi + 1].mean(0))
        d['context'].square().mean().backward()
        self.assertTrue(torch.isfinite(raw.grad).all())
        self.assertIsNone(m.q_proj.weight.grad)

    def test_missing_sensor_barriers_apply_to_every_variant(self):
        raw = torch.randn(9, 50); raw[4] = float('nan')
        valid = torch.ones(9, dtype=torch.bool); valid[4] = False
        for method in VARIANTS:
            m = toy(method)
            d = m.segment_seed_expand_attention(torch.randn(9, 4), raw, torch.zeros(9, dtype=torch.long), torch.full((9,), 8), valid)
            self.assertLess(int(d['selected_window_end'][2]), 4)
            self.assertGreater(int(d['selected_window_start'][6]), 4)
            self.assertTrue(torch.isfinite(d['context']).all())
            self.assertEqual(float(d['context'][4].abs().sum()), 0)

    def test_retrained_variant_roundtrip_and_identity_validation(self):
        def load(root, subject):
            self.assertNotEqual(subject, 'sbj_0')
            rng = np.random.default_rng(5)
            return {'video': rng.normal(size=(6, 2048)).astype('float32'),
                    'inertial': rng.normal(size=(6, 600)).astype('float32'),
                    'labels': np.array([0, 0, 1, 1, 18, 18])}
        with tempfile.TemporaryDirectory() as output, patch.object(signal_training, 'load_sequence', side_effect=load):
            folder = signal_training.run_signal_training(data_root='unused', fold=1, output=output,
                parent_epochs=1, probe_epochs=1, method='NO_EXPANSION')
            model, _, _ = load_signal_wear(folder / 'parent.pt', folder / 'background_probe.pt')
            self.assertFalse(model.parent.ablation_config.expand)
            self.assertEqual(model.run_metadata['ablation_config'], asdict(VARIANTS['NO_EXPANSION']))
            payload = torch.load(folder / 'parent.pt', weights_only=False)
            payload['run_metadata']['method'] = METHOD
            torch.save(payload, folder / 'bad.pt')
            with self.assertRaisesRegex(ValueError, 'ablation configuration'):
                load_signal_wear(folder / 'bad.pt', folder / 'background_probe.pt')


if __name__ == '__main__':
    unittest.main()
