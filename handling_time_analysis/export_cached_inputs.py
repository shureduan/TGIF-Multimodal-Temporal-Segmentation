"""Use existing laboratory features without re-running the expensive encoders.

This only creates local input files. Labels are copied only with the explicit
--include-evaluation-labels option and are not part of the inference input.
"""
import argparse,json,shutil
from pathlib import Path
import numpy as np
from pipeline.common import write_json

def export(root,recording,output,include_labels=False,max_frames=None):
    root=Path(root);output=Path(output);visual=output/'visual';visual.mkdir(parents=True,exist_ok=True)
    source=root/'videomae_pipeline/outputs/extracted_features'/recording
    n=len(np.load(source/'timestamps_sec.npy',allow_pickle=False))
    if max_frames is not None:n=min(n,max_frames)
    if n<1:raise ValueError('At least one feature frame is required')
    for name in ['global_features.npy','operator_features.npy','timestamps_sec.npy']:
        x=np.load(source/name,mmap_mode='r',allow_pickle=False)[:n]
        np.save(visual/name,x,allow_pickle=False)
    t=np.load(visual/'timestamps_sec.npy',allow_pickle=False)
    dt=float(np.median(np.diff(t))) if len(t)>1 else .133467
    write_json(visual/'metadata.json',dict(feature_dt=dt,analysis_start_sec=float(t[0]),
                analysis_end_sec=float(t[-1]+dt),source='Existing private laboratory feature cache',encoder_qkv_bias=False))
    file=root/'multimodal/video_prior_fusion/data_store/vibration_features'/f'{recording}_vib.npy'
    if file.exists():
        x=np.load(file,allow_pickle=False)[:n]
        valid=np.isfinite(x).all(axis=1)
        qc=root/'labeling/signal_qc/signal_usability.json'
        if qc.exists():
            entry=json.loads(qc.read_text()).get(recording,{})
            for a,b in entry.get('dead_segments_s',[]):valid[(t>=a)&(t<b)]=False
        x=np.nan_to_num(x).astype(np.float32)
        np.savez_compressed(output/'vibration.npz',features=x,valid=valid,timestamps_sec=t,
                            feature_names=np.array(['rms_dyn','peak_to_peak','log_e_20_40','ratio_20_40']))
    if include_labels:np.save(output/'evaluation_labels.npy',np.load(source/'labels.npy',allow_pickle=False)[:n],allow_pickle=False)
    return output

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--legacy-root',required=True);p.add_argument('--recording',required=True)
    p.add_argument('--output',required=True);p.add_argument('--include-evaluation-labels',action='store_true');p.add_argument('--max-frames',type=int)
    a=p.parse_args();print(export(a.legacy_root,a.recording,a.output,a.include_evaluation_labels,a.max_frames))
