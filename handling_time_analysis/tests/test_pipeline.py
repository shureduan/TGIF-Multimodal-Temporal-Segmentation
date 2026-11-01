import csv,json,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from pipeline.common import validate_timestamps
from pipeline.visual import validate_roi,extract
from pipeline.vibration import features,dynamic_signal,minute_grid
from pipeline.handling import summarize
from pipeline.inference import infer_arrays
from pipeline.models.dynamic_window_multimodal_mstcn import DynamicWindowMultimodalMSTCN
from pipeline.models.risk_gate import binary_entropy
from pipeline.models.js_conflict import js_conflict

torch.set_num_threads(1)

class Contracts(unittest.TestCase):
    def test_handling_duration_half_open_including_last_sample(self):
        runs,s=summarize(np.array([2,2,1,0,2]),np.arange(5)*.5,2.5)
        self.assertAlmostEqual(s['handling_seconds'],1.5)
        self.assertAlmostEqual(sum(r['duration_sec'] for r in runs),2.5)
        self.assertEqual(s['classes']['handling']['n_segments'],2)
        self.assertEqual(runs[-1]['end_sec'],2.5)
    def test_nonzero_time_origin(self):
        _,s=summarize(np.array([2,1]),np.array([10.,10.5]),11.)
        self.assertEqual(s['handling_seconds'],.5);self.assertEqual(s['analyzed_seconds'],1.)
    def test_video_gaps_are_not_counted_as_work(self):
        with self.assertRaises(ValueError):summarize(np.array([2,1,2,2]),np.array([0.,.1,.2,10.]),10.1)
    def test_duplicate_and_nonfinite_timestamps_rejected(self):
        for t in [[0,0],[1,0],[0,np.nan],[]]:
            with self.assertRaises(ValueError):validate_timestamps(t)
    def test_invalid_roi_rejected(self):
        for roi in [[1,1,1,2],[-1,0,10,10],[0,0,101,10]]:
            with self.assertRaises(ValueError):validate_roi(roi,100,100)
    def test_signal_band_and_missing_validity(self):
        t=np.arange(2500)/500;signal=.05*np.sin(2*np.pi*30*t)
        x,valid,_=features(signal,np.array([.1,1.,2.,3.,4.8]),0.)
        self.assertFalse(valid[0]);self.assertFalse(valid[-1])
        self.assertTrue(valid[2]);self.assertGreater(x[2,3],.99)
        self.assertTrue(np.isfinite(x).all())
    def test_all_missing_does_not_fabricate_features(self):
        x,valid,_=features(np.full(2000,np.nan),np.array([1.,2.]),0.)
        self.assertFalse(valid.any());self.assertTrue(np.isnan(x).all())
    def test_clock_grid_preserves_gap_and_scale(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'imu1_2026-05-21_12-00.csv'
            p.write_text('timestamp,ax,ay,az\n0,0,0,16384\n2,0,0,16384\n8,0,0,16384\n')
            grid,n=minute_grid(temp,'2026-05-21T12:00:00',.02)
            self.assertEqual(n,1);self.assertEqual(grid[0],1.);self.assertTrue(np.isnan(grid[2]));self.assertEqual(grid[4],1.)
    def test_saturated_binary_heads_are_finite(self):
        p=torch.tensor([0.,1.,0.,1.]);q=torch.tensor([0.,1.,1.,0.])
        self.assertTrue(torch.isfinite(binary_entropy(p)).all())
        self.assertTrue(torch.isfinite(js_conflict(p,q)).all())
        self.assertTrue((js_conflict(p,q)<=1).all())

class ModelContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model=DynamicWindowMultimodalMSTCN(n_stages=2,n_layers=2,ch=8).eval()
        self.video=np.random.default_rng(2).normal(size=(40,1536)).astype(np.float32)
        self.stats=dict(video_mean=np.zeros(1536,np.float32),video_std=np.ones(1536,np.float32),
                        vibration_median=np.zeros((1,2)),vibration_iqr=np.ones((1,2)))
        self.meta=dict(station_aliases={},eligible_vibration_stations=['station3'],vibration_stations=['_global'],
                       segment_config=dict(smoothing_seconds=1.,min_segment_seconds=2.,max_segment_seconds=20.))
    def run_model(self,v=None,mask=None,station='station3',chunk=2247):
        return infer_arrays(self.model,self.video,v,mask,station,self.stats,self.meta,torch.device('cpu'),chunk)
    def test_missing_sensor_exact_video_path(self):
        pure=self.run_model()[0]
        missing,used,_=self.run_model(np.zeros((40,4),np.float32),np.zeros(40,bool))
        np.testing.assert_array_equal(pure,missing);self.assertFalse(used.any())
    def test_disabled_station_exact_video_path(self):
        pure=self.run_model()[0]
        other,used,_=self.run_model(np.ones((40,4),np.float32),np.ones(40,bool),station='station1')
        np.testing.assert_array_equal(pure,other);self.assertFalse(used.any())
    def test_remainder_chunk_is_classified(self):
        prob,_,_=self.run_model(chunk=17)
        self.assertEqual(prob.shape,(40,3));np.testing.assert_allclose(prob.sum(1),1,atol=1e-6)
    def test_variable_windows_one_way_attention_and_inference(self):
        from pipeline.models.dynamic_window_manager import build_segment_tokens
        wins,mask,_=build_segment_tokens([(0,10),(10,40)],40,7.4933,2.,0)
        self.assertNotEqual(wins[0][1]-wins[0][0],wins[1][1]-wins[1][0])
        x=torch.tensor(self.video)[None];v=torch.zeros(1,2,2)
        with torch.no_grad():result=self.model(x,v,wins,allow_mask=mask)
        self.assertEqual(tuple(result['attention_weights'].shape),(1,40,2))
        blocked=~torch.tensor(mask)[None]
        self.assertTrue((result['attention_weights'][blocked]==0).all())
    def test_nonfinite_or_misaligned_features_rejected(self):
        with self.assertRaises(ValueError):self.run_model(np.ones((39,4)),np.ones(40,bool))
        self.video[0,0]=np.nan
        with self.assertRaises(ValueError):self.run_model()

class VisualProtocol(unittest.TestCase):
    def test_small_video_preserves_grid_and_two_streams(self):
        import cv2
        class FakeEncoder:
            def encode(self,clips):
                return np.asarray([np.full(768,float(np.asarray(c[0]).mean())) for c in clips],np.float32)
        with tempfile.TemporaryDirectory() as temp:
            video=Path(temp)/'synthetic.avi';writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),30,(80,60))
            for i in range(40):writer.write(np.full((60,80,3),i*3,np.uint8))
            writer.release();out=Path(temp)/'visual'
            meta=extract(video,[10,10,60,50],out,batch_size=3,encoder=FakeEncoder())
            self.assertEqual(np.load(out/'global_features.npy').shape,(10,768))
            np.testing.assert_allclose(np.load(out/'timestamps_sec.npy'),np.arange(10)*4/30)
            self.assertEqual(meta['clip_frame_offsets'],[-30,-26,-22,-18,-14,-10,-6,-2,2,6,10,14,18,22,26,30])


class BundleIntegrity(unittest.TestCase):
    def make_bundle(self,directory):
        from pipeline.bundle import SCHEMA
        from pipeline.common import write_json,sha256
        config=dict(n_stages=2,n_layers=2,ch=8)
        model=DynamicWindowMultimodalMSTCN(**config).eval()
        torch.save(dict(model=model.state_dict(),seed=43),directory/'model.pt')
        np.savez(directory/'normalization.npz',video_mean=np.zeros(1536),video_std=np.ones(1536),
                 vibration_median=np.zeros((1,2)),vibration_iqr=np.ones((1,2)))
        metadata=dict(schema=SCHEMA,split_seed=43,model_config=config,vibration_stations=['_global'])
        write_json(directory/'metadata.json',metadata)
        write_json(directory/'sha256.json',{n:sha256(directory/n) for n in ['model.pt','normalization.npz','metadata.json']})
    def test_strict_bundle_load_and_checksum_failure(self):
        from pipeline.bundle import load_bundle
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp);self.make_bundle(directory)
            model,stats,meta=load_bundle(directory,torch.device('cpu'))
            self.assertEqual(meta['split_seed'],43)
            with open(directory/'normalization.npz','ab') as f:f.write(b'modified')
            with self.assertRaisesRegex(ValueError,'checksum'):load_bundle(directory,torch.device('cpu'))
    def test_seed_pair_mismatch_rejected(self):
        from pipeline.bundle import load_bundle
        from pipeline.common import write_json,sha256
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp);self.make_bundle(directory)
            meta=json.loads((directory/'metadata.json').read_text());meta['split_seed']=41
            write_json(directory/'metadata.json',meta)
            sums=json.loads((directory/'sha256.json').read_text());sums['metadata.json']=sha256(directory/'metadata.json')
            write_json(directory/'sha256.json',sums)
            with self.assertRaisesRegex(ValueError,'seed mismatch'):load_bundle(directory,torch.device('cpu'))
    def test_archive_path_traversal_rejected(self):
        import zipfile
        from download_models import unpack
        with tempfile.TemporaryDirectory() as temp:
            archive=Path(temp)/'bad.zip'
            with zipfile.ZipFile(archive,'w') as z:z.writestr('../outside','bad')
            with self.assertRaises(ValueError):unpack(archive,temp,43)

if __name__=='__main__':unittest.main()
