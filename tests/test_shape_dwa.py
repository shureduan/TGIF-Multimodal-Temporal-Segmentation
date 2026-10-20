"""Geometry, missing data, pairing and train-only calibration for the shape pilot."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from tgif_dwa.shape_dwa import EnvelopeShapeDWA, ShapeWindowConfig, log_ac_envelope, local_shape_features
from tgif_dwa.shape_wear import build_shape_parent, load_shape_wear
from tgif_dwa.signal_wear import build_signal_parent
from tgif_dwa import shape_training


def small():
    model=EnvelopeShapeDWA(shape_config=ShapeWindowConfig(channels=1),video_dim=4,imu_dim=50,
            attn_dim=4,n_classes=3,n_stages=2,n_layers=1,ch=4).eval()
    model.shape_calibrated.fill_(True);model.shape_threshold.fill_(.1)
    model.slope_deadzone.fill_(.01)
    return model


def data(n=101):
    raw=np.zeros((n,50),np.float32)
    wave=np.sin(2*np.pi*8*np.arange(50)/50).astype(np.float32)
    raw[30:71]=4*wave
    return torch.from_numpy(raw)


class ShapeDwaTests(unittest.TestCase):
    def test_vibration_island_is_separated_from_flat_surroundings(self):
        model=small();raw=data();n=len(raw)
        selected=model.select_sensor_windows(raw,torch.zeros(n,dtype=torch.long),torch.full((n,),n-1),torch.ones(n,dtype=torch.bool))
        self.assertGreaterEqual(int(selected['selected_window_start'][50]),28)
        self.assertLessEqual(int(selected['selected_window_end'][50]),72)
        self.assertLess(int(selected['selected_window_end'][10]),30)
        self.assertGreater(int(selected['selected_window_start'][90]),70)
        self.assertGreater(int(selected['selected_window_end'][50]-selected['selected_window_start'][50]),20)

    def test_same_shape_grows_beyond_seed_and_respects_growth_limit(self):
        model=small();raw=torch.zeros(101,50);t=torch.arange(101)
        result=model.select_sensor_windows(raw,(t-1).clamp_min(0),(t+1).clamp_max(100),torch.ones(101,dtype=torch.bool))
        self.assertEqual(int(result['selected_window_start'][50]),37)
        self.assertEqual(int(result['selected_window_end'][50]),63)
        self.assertGreater(int(result['expanded_tokens'][50]),0)

    def test_different_envelope_is_detected_even_with_equal_raw_dc_level(self):
        config=ShapeWindowConfig(channels=1);raw=data()
        envelope=log_ac_envelope(raw.numpy(),config)
        self.assertAlmostEqual(float(raw[:30].mean()),float(raw[30:71].mean()),places=5)
        self.assertGreater(float(envelope[50,0]),float(envelope[10,0])+1)

    def test_temporal_order_is_retained(self):
        config=ShapeWindowConfig(channels=1,local_radius_steps=2)
        a=np.array([0,0,1,1,0,0,1,1,0],float)[:,None]
        b=np.array([0,1,0,1,0,1,0,1,0],float)[:,None]
        x=local_shape_features(a,np.ones(9,bool),np.array([.01]),config)
        y=local_shape_features(b,np.ones(9,bool),np.array([.01]),config)
        self.assertFalse(np.allclose(x[4],y[4]))

    def test_missing_data_is_a_barrier_and_its_values_are_irrelevant(self):
        model=small();raw=torch.zeros(21,50);valid=torch.ones(21,dtype=torch.bool);valid[10]=False
        t=torch.arange(21);lo=torch.zeros_like(t);hi=torch.full_like(t,20)
        a=model.select_sensor_windows(raw,lo,hi,valid)
        raw[10]=float('nan');b=model.select_sensor_windows(raw,lo,hi,valid)
        for key in ('selected_window_start','selected_window_end'):
            torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
        self.assertLess(int(b['selected_window_end'][5]),10)
        self.assertGreater(int(b['selected_window_start'][15]),10)
        output=model(torch.randn(21,4),raw,valid)
        for context in output['round_contexts']:torch.testing.assert_close(context[10],torch.zeros(50))
        self.assertTrue(torch.isfinite(output['logits']).all())

    def test_fixed_seed_support_is_independent_of_video_and_qk(self):
        model=small();raw=data(101);valid=torch.ones(101,dtype=torch.bool)
        lo=torch.zeros(101,dtype=torch.long);hi=torch.full_like(lo,100)
        a=model.segment_seed_expand_attention(torch.randn(101,4),raw,lo,hi,valid)
        model.q_proj.weight.data.fill_(10);model.k_proj.weight.data.zero_()
        b=model.segment_seed_expand_attention(torch.randn(101,4)*100,raw,lo,hi,valid)
        for key in ('selected_window_start','selected_window_end'):
            torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)

    def test_controller_runs_before_attention_and_fast_path_has_finite_gradients(self):
        model=small();events=[];select=model.select_sensor_windows;project=model._project_qk
        def choose(*args):events.append('select');return select(*args)
        def project_qk(*args):events.append('qk');return project(*args)
        video=torch.randn(13,4,requires_grad=True);raw=torch.randn(13,50,requires_grad=True)
        with patch.object(model,'select_sensor_windows',side_effect=choose),patch.object(model,'_project_qk',side_effect=project_qk):
            output=model(video,raw)
        self.assertEqual(events,['select','qk','select','qk'])
        fast=model(video,raw,return_diagnostics=False)
        torch.testing.assert_close(output['logits'],fast['logits'],rtol=0,atol=0)
        sum(x.square().mean() for x in fast['round_logits']).backward()
        for p in (raw,video,model.q_proj.weight,model.k_proj.weight,model.raw_temperature):
            self.assertIsNotNone(p.grad);self.assertTrue(torch.isfinite(p.grad).all())

    def test_long_homogeneous_support_is_not_truncated(self):
        model=small();n=300
        raw=torch.arange(n,dtype=torch.float32)[:,None].expand(n,50).clone().requires_grad_(True)
        # DC-only rows all have the same zero AC envelope, so full support stays.
        lo=torch.zeros(n,dtype=torch.long);hi=torch.full_like(lo,n-1)
        model.q_proj.weight.data.zero_();model.k_proj.weight.data.zero_()
        output=model.segment_seed_expand_attention(torch.randn(n,4),raw,lo,hi,torch.ones(n,dtype=torch.bool))
        torch.testing.assert_close(output['context'],raw.mean(0).expand(n,-1),rtol=1e-5,atol=1e-4)
        output['context'].sum().backward()
        torch.testing.assert_close(raw.grad,torch.ones_like(raw),rtol=1e-5,atol=1e-4)
        self.assertEqual(int(output['selected_window_end'][150]),n-1)

    def test_learned_parameters_start_identically_to_v3(self):
        torch.manual_seed(47);old=build_signal_parent()
        torch.manual_seed(47);new=build_shape_parent()
        a=dict(old.named_parameters());b=dict(new.named_parameters())
        self.assertEqual(a.keys(),b.keys())
        for key in a:torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)

    def test_calibration_and_checkpoint_roundtrip_use_training_subjects_only(self):
        reads=[]
        def sequence(root,subject):
            reads.append(subject);rng=np.random.default_rng(int(subject.split('_')[-1])+7)
            return {'sequence_id':subject,'video':rng.normal(size=(8,2048)).astype(np.float32),
                    'inertial':rng.normal(size=(8,600)).astype(np.float32),
                    'labels':np.array([0,0,1,1,18,18,1,0])}
        with tempfile.TemporaryDirectory() as temp,patch.object(shape_training,'load_sequence',side_effect=sequence):
            folder=shape_training.run_shape_training(data_root=Path(temp),output=Path(temp)/'train',fold=2,seed=47,parent_epochs=1,probe_epochs=1,device='cpu')
            self.assertEqual(len(reads),17);self.assertNotIn('sbj_1',reads)
            model,mean,std=load_shape_wear(folder/'parent.pt',folder/'background_probe.pt')
            self.assertTrue(bool(model.parent.shape_calibrated))
            self.assertEqual(mean.shape,(12,))
            output=model(torch.randn(8,2048),torch.randn(8,600))
            self.assertEqual(tuple(output['probabilities'].shape),(8,19))
            payload=torch.load(folder/'background_probe.pt',weights_only=False)
            payload['run_metadata']['seed']=53;torch.save(payload,folder/'wrong_probe.pt')
            with self.assertRaisesRegex(ValueError,'seed|metadata'):
                load_shape_wear(folder/'parent.pt',folder/'wrong_probe.pt')


if __name__=='__main__':unittest.main()
