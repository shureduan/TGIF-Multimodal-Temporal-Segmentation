"""Candidate tests: sensor-dependent resizing occurs before one-way attention."""
from dataclasses import asdict
import copy
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from tgif_dwa.signal_dwa import SignalAdaptiveDWA, SignalWindowConfig, descriptors
from tgif_dwa.signal_wear import SignalWearModel, load_signal_wear, PROTOCOL
from tgif_dwa import signal_training
from tgif_dwa.wear_model import WearFinalModel


def small():
    model = SignalAdaptiveDWA(signal_config=SignalWindowConfig(channels=1), video_dim=4,
        imu_dim=50, attn_dim=4, n_classes=3, n_stages=2, n_layers=1, ch=4).eval()
    with torch.no_grad():
        model.signal_calibrated.fill_(True)
        model.signal_thresholds.copy_(torch.tensor([1e6, .05, .05]))
    return model


def seeds(n, radius):
    t = torch.arange(n)
    return (t - radius).clamp_min(0), (t + radius).clamp_max(n - 1)


def sequence(subject):
    rng = np.random.default_rng(7)
    return {'sequence_id': subject, 'video': rng.normal(size=(8, 2048)).astype('float32'),
            'inertial': rng.normal(size=(8, 600)).astype('float32'),
            'labels': np.array([0, 0, 1, 1, 18, 18, 1, 0])}


class SignalDwaTests(unittest.TestCase):
    def test_constant_redundant_window_contracts_and_impulse_expands(self):
        model = small(); n = 31; valid = torch.ones(n, dtype=torch.bool)
        start, end = seeds(n, 10)
        constant = torch.zeros(n, 50)
        shrunk = model.select_sensor_windows(constant, start, end, valid)
        self.assertGreater(int(shrunk['redundancy_trimmed_tokens'][15]), 0)
        self.assertLess(int(shrunk['selected_window_end'][15] - shrunk['selected_window_start'][15]), 20)
        start, end = seeds(n, 1)
        impulse = constant.clone(); impulse[15] = 2
        expanded = model.select_sensor_windows(impulse, start, end, valid)
        unchanged = model.select_sensor_windows(constant, start, end, valid)
        self.assertGreater(int(expanded['expanded_tokens'][15]), 0)
        self.assertEqual(int(unchanged['expanded_tokens'][15]), 0)
        self.assertFalse(torch.equal(expanded['selected_window_end'], unchanged['selected_window_end']))

    def test_sensor_state_change_clips_a_seed_crossing_the_transition(self):
        model = small(); n = 30
        model.signal_thresholds[0] = .5
        raw = torch.zeros(n, 50); raw[15:] = 4
        result = model.select_sensor_windows(raw, torch.zeros(n, dtype=torch.long),
                                             torch.full((n,), n - 1), torch.ones(n, dtype=torch.bool))
        self.assertLess(int(result['selected_window_end'][5]), 15)
        self.assertGreaterEqual(int(result['selected_window_start'][25]), 15)
        self.assertGreater(int(result['boundary_clipped_tokens'][5]), 0)

    def test_equal_energy_frequency_change_can_change_sensor_support(self):
        model = small(); n = 30
        model.signal_thresholds[0] = .02
        time = np.arange(50) / 50
        low = np.sin(2 * np.pi * 3 * time)
        high = np.sin(2 * np.pi * 10 * time)
        self.assertAlmostEqual(float(np.mean(low ** 2)), float(np.mean(high ** 2)))
        stationary = torch.tensor(np.tile(low, (n, 1)), dtype=torch.float32)
        changed = stationary.clone(); changed[15:] = torch.tensor(high, dtype=torch.float32)
        start = torch.zeros(n, dtype=torch.long); end = torch.full((n,), n - 1)
        valid = torch.ones(n, dtype=torch.bool)
        a = model.select_sensor_windows(stationary, start, end, valid)
        b = model.select_sensor_windows(changed, start, end, valid)
        self.assertEqual(int(a['boundary_clipped_tokens'][5]), 0)
        self.assertGreater(int(b['boundary_clipped_tokens'][5]), 0)
        self.assertLess(int(b['selected_window_end'][5]), 15)
        self.assertGreaterEqual(int(b['selected_window_start'][25]), 15)

    def test_video_and_qk_changes_cannot_change_the_same_seed_windows(self):
        model = small(); raw = torch.randn(15, 50); valid = torch.ones(15, dtype=torch.bool)
        start, end = seeds(15, 3)
        a = model.segment_seed_expand_attention(torch.randn(15, 4), raw, start, end, valid)
        with torch.no_grad():
            model.q_proj.weight.fill_(10); model.k_proj.weight.zero_()
        b = model.segment_seed_expand_attention(torch.randn(15, 4) * 100, raw, start, end, valid)
        for key in ('selected_window_start', 'selected_window_end'):
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)

    def test_selection_precedes_projection_and_context_is_raw_value_mean(self):
        model = small(); raw = torch.randn(13, 50); video = torch.randn(13, 4)
        valid = torch.ones(13, dtype=torch.bool); start, end = seeds(13, 4); events = []
        select = model.select_sensor_windows; project = model._project_qk
        def choose(*args):
            events.append('select'); return select(*args)
        def qk(*args):
            events.append('qk'); return project(*args)
        with torch.no_grad():
            model.q_proj.weight.zero_(); model.k_proj.weight.zero_()
        with patch.object(model, 'select_sensor_windows', side_effect=choose), patch.object(model, '_project_qk', side_effect=qk):
            result = model.segment_seed_expand_attention(video, raw, start, end, valid)
        self.assertEqual(events, ['select', 'qk'])
        for i in range(13):
            lo, hi = int(result['selected_window_start'][i]), int(result['selected_window_end'][i])
            torch.testing.assert_close(result['context'][i], raw[lo:hi + 1].mean(0))

    def test_gradients_fast_path_and_missing_sensor_are_finite(self):
        torch.manual_seed(3); model = small()
        raw = torch.randn(13, 50, requires_grad=True); video = torch.randn(13, 4, requires_grad=True)
        full = model(video, raw)
        fast = model(video, raw, return_diagnostics=False)
        torch.testing.assert_close(full['logits'], fast['logits'], rtol=0, atol=0)
        sum(x.square().mean() for x in full['round_logits']).backward()
        self.assertTrue(torch.isfinite(raw.grad).all() and torch.isfinite(video.grad).all())
        for key in ('q_proj.weight', 'k_proj.weight', 'raw_temperature'):
            grad = dict(model.named_parameters())[key].grad
            self.assertIsNotNone(grad); self.assertTrue(torch.isfinite(grad).all())
        model.zero_grad(set_to_none=True)
        missing = torch.full((13, 50), float('nan'), requires_grad=True)
        result = model(video.detach(), missing, aux_valid=torch.zeros(13, dtype=torch.bool))
        for context in result['round_contexts']:
            torch.testing.assert_close(context, torch.zeros_like(context))
        result['logits'].sum().backward()
        self.assertTrue(torch.isfinite(missing.grad).all())

    def test_invalid_rows_are_barriers_and_do_not_change_valid_support(self):
        model = small(); a = torch.randn(11, 50); valid = torch.ones(11, dtype=torch.bool); valid[5] = False
        b = a.clone(); b[5] = float('nan'); start, end = seeds(11, 10)
        x = model.select_sensor_windows(a, start, end, valid)
        y = model.select_sensor_windows(b, start, end, valid)
        for key in ('selected_window_start', 'selected_window_end'):
            torch.testing.assert_close(x[key], y[key], rtol=0, atol=0)
        self.assertLess(int(y['selected_window_end'][3]), 5)
        self.assertGreater(int(y['selected_window_start'][7]), 5)

    def test_long_support_keeps_all_keys_and_raw_value_gradients(self):
        model = small()
        with torch.no_grad():
            model.q_proj.weight.zero_(); model.k_proj.weight.zero_()
        raw = torch.randn(300, 50, requires_grad=True)
        left = torch.zeros(300, dtype=torch.long); right = torch.full((300,), 299)
        with patch.object(model, 'select_sensor_windows', return_value={
                'selected_window_start': left, 'selected_window_end': right}):
            result = model.segment_seed_expand_attention(torch.randn(300, 4), raw,
                        left, right, torch.ones(300, dtype=torch.bool))
        torch.testing.assert_close(result['context'], raw.mean(0).expand(300, -1))
        result['context'].sum().backward()
        torch.testing.assert_close(raw.grad, torch.ones_like(raw))

    def test_signal_calibration_is_frozen_serializable_and_required(self):
        config = SignalWindowConfig(channels=1)
        model = SignalAdaptiveDWA(signal_config=config, video_dim=4, imu_dim=50, n_classes=3,
                                  n_stages=1, n_layers=1, ch=4)
        with self.assertRaisesRegex(ValueError, 'fit training-fold'):
            model(torch.randn(6, 4), torch.randn(6, 50))
        rng = np.random.default_rng(5)
        summary = model.fit_signal_statistics([rng.normal(size=(30, 50)), rng.normal(size=(40, 50))])
        self.assertEqual(summary['training_sequences'], 2)
        self.assertTrue(torch.isfinite(model.signal_thresholds).all())
        before = copy.deepcopy(model.state_dict()); model.eval()
        model(torch.randn(6, 4), torch.randn(6, 50) * 100)
        for name in ('signal_center', 'signal_scale', 'signal_thresholds', 'signal_calibrated'):
            torch.testing.assert_close(before[name], model.state_dict()[name], rtol=0, atol=0)

    def test_single_token_and_signal_layout(self):
        model = small(); result = model(torch.randn(1, 4), torch.zeros(1, 50))
        self.assertTrue(torch.isfinite(result['logits']).all())
        for diag in result['round_diagnostics']:
            self.assertEqual(diag['selected_window_start'].tolist(), [0])
            self.assertEqual(diag['selected_window_end'].tolist(), [0])
        with self.assertRaises(ValueError):
            descriptors(np.zeros((5, 49)), SignalWindowConfig(channels=1))

    def test_real_training_never_reads_outer_subject_and_loader_preserves_support(self):
        loaded = []
        def load(root, subject):
            self.assertNotEqual(subject, 'sbj_0'); loaded.append(subject); return sequence(subject)
        with tempfile.TemporaryDirectory() as directory, patch.object(signal_training, 'load_sequence', side_effect=load):
            folder = signal_training.run_signal_training(data_root='unused', fold=1, output=directory,
                                                          parent_epochs=1, probe_epochs=1, seed=47)
            self.assertEqual(set(loaded), {f'sbj_{i}' for i in range(1, 18)})
            model, mean, std = load_signal_wear(folder / 'parent.pt', folder / 'background_probe.pt')
            payload = torch.load(folder / 'parent.pt', weights_only=False)
            parent = signal_training.build_signal_parent().eval(); parent.load_state_dict(payload['model'])
            v, raw = torch.randn(8, 2048), torch.randn(8, 600)
            with torch.no_grad():
                a = parent(v, raw); b = model(v, raw)['parent']
            torch.testing.assert_close(a['logits'], b['logits'], rtol=0, atol=0)
            for i in range(2):
                for key in ('selected_window_start', 'selected_window_end'):
                    torch.testing.assert_close(a['round_diagnostics'][i][key], b['round_diagnostics'][i][key], rtol=0, atol=0)
            self.assertEqual(model.run_metadata['protocol'], PROTOCOL)
            probe = torch.load(folder / 'background_probe.pt', weights_only=False)
            probe['run_metadata']['run_id'] = 'mismatched'; torch.save(probe, folder / 'bad.pt')
            with self.assertRaisesRegex(ValueError, 'metadata differ'):
                load_signal_wear(folder / 'parent.pt', folder / 'bad.pt')
            with self.assertRaises(RuntimeError):
                WearFinalModel().parent.load_state_dict(payload['model'], strict=True)


if __name__ == '__main__':
    unittest.main()
