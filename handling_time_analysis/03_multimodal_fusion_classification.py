"""Paper subsection 3: fusion/classification and predicted handling durations."""
import argparse,json
from pathlib import Path
import numpy as np
from pipeline.inference import infer
from pipeline.dwa_inference import DEFAULT_DWA_DIR,choose_device,infer_arrays
from pipeline.handling import write_outputs

def run_dwa(video_features,vibration_path,output,seed,checkpoint,video_norm,device,labels_path=None):
    visual=Path(video_features)
    with np.load(vibration_path,allow_pickle=False) as vib:
        if 'features' not in vib:
            raise ValueError('vibration NPZ must contain a features array [T,6]')
        features=np.asarray(vib['features'],dtype=np.float32)
        mask=np.asarray(vib['valid'],dtype=bool) if 'valid' in vib else None
        if 'timestamps_sec' in vib:
            vibration_times=np.asarray(vib['timestamps_sec'],dtype=np.float64)
        else:
            vibration_times=None
    timestamps=np.load(visual/'timestamps_sec.npy',allow_pickle=False).astype(np.float64)
    if vibration_times is not None and (vibration_times.shape!=timestamps.shape or not np.allclose(vibration_times,timestamps,atol=1e-6)):
        raise ValueError('video and vibration timestamps must match exactly')
    meta_path=visual/'metadata.json'
    visual_meta=json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if len(timestamps)>1:
        dt=float(np.median(np.diff(timestamps)))
    else:
        dt=1./7.4933
    end_sec=float(visual_meta.get('analysis_end_sec',timestamps[-1]+dt))
    result=infer_arrays(
        np.load(visual/'global_features.npy',allow_pickle=False),
        np.load(visual/'operator_features.npy',allow_pickle=False),
        features,checkpoint_path=checkpoint,video_norm_path=video_norm,
        valid_mask=mask,device=device,
    )
    if result['split_seed']!=seed:
        raise ValueError(f"requested split seed {seed}, but checkpoint is seed {result['split_seed']}")
    labels=np.load(labels_path,allow_pickle=False) if labels_path else None
    run_meta={'model':'TGIF signal-driven DWA','split_seed':result['split_seed'],
              'best_epoch':result['best_epoch'],'checkpoint':str(checkpoint),
              'feature_dimension':6,'device':str(device)}
    summary=write_outputs(output,result['predictions'],result['probabilities'],timestamps,end_sec,
                          result['sensor_valid'],run_meta,labels=labels)
    print(json.dumps(summary,indent=2))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=('handling','tgif-dwa'),default='handling',
                   help='handling preserves the original 4-feature model; tgif-dwa uses the six-feature DWA release')
    p.add_argument('--video-features',required=True);p.add_argument('--vibration')
    p.add_argument('--bundle');p.add_argument('--station-id');p.add_argument('--seed',type=int,default=41,choices=(41,11,43,17,53))
    p.add_argument('--checkpoint',help='Optional TGIF DWA checkpoint override')
    p.add_argument('--video-normalization',help='Optional 1536-D normalization override')
    p.add_argument('--output',required=True)
    p.add_argument('--device',choices=['cpu','cuda','mps','auto'],default='cpu')
    p.add_argument('--chunk-frames',type=int,default=2247)
    p.add_argument('--labels',help='Optional .npy evaluation labels; never used to generate predictions')
    a=p.parse_args()
    if a.model=='tgif-dwa':
        if not a.vibration:
            p.error('--vibration is required with --model tgif-dwa and must contain the six-feature output from stage 2')
        checkpoint=Path(a.checkpoint) if a.checkpoint else DEFAULT_DWA_DIR/'weights'/f'seed_{a.seed}_best.pt'
        video_norm=Path(a.video_normalization) if a.video_normalization else DEFAULT_DWA_DIR/'weights'/'video_norm.npz'
        run_dwa(a.video_features,a.vibration,a.output,a.seed,checkpoint,video_norm,
                choose_device(a.device),a.labels)
        return
    if not a.bundle or not a.station_id:
        p.error('--bundle and --station-id are required with --model handling')
    print(json.dumps(infer(a.video_features,a.bundle,a.station_id,a.output,vibration=a.vibration,
                          device=a.device,chunk_frames=a.chunk_frames,labels=a.labels),indent=2))
if __name__=='__main__':main()
