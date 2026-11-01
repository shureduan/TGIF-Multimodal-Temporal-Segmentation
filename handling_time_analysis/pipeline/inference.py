"""3. Predicted sensor windows -> one-way attention -> shared MS-TCN head."""
from __future__ import annotations
from pathlib import Path
import time
import numpy as np
from scipy.signal import find_peaks
from .common import FEATURE_RATE,load_video_features,device_for,synchronize
from .bundle import load_bundle
from .models.dynamic_window_manager import predict_segments,build_segment_tokens
from .handling import write_outputs

def window_features(vib,windows):
    rms=np.nan_to_num(vib[:,0]);peaks=np.zeros_like(rms)
    indices,_=find_peaks(rms,prominence=(np.std(rms)*.3 or 1e-9))
    peaks[indices]=rms[indices]
    out=[]
    for a,b in windows:
        duration=max((b-a)/FEATURE_RATE,1e-6)
        energy=float(np.nan_to_num(10.**vib[a:b,2]).mean()) if b>a else 0.
        out.append([float(peaks[a:b].sum())/duration,energy])
    return np.asarray(out,np.float32)

def infer_arrays(model,video,vibration,valid,station,stats,meta,device,chunk_frames=2247):
    import torch
    if chunk_frames<1: raise ValueError('chunk_frames must be positive')
    video=np.asarray(video,np.float32)
    if video.ndim!=2 or video.shape[1]!=1536 or not len(video) or not np.isfinite(video).all():
        raise ValueError('video must be finite [T,1536]')
    n=len(video)
    if vibration is not None:
        vibration=np.asarray(vibration,np.float32)
        if vibration.shape!=(n,4) or not np.isfinite(vibration).all(): raise ValueError('vibration must be finite [T,4]')
        valid=np.asarray(valid)
        if valid.shape!=(n,) or not np.isin(valid,[False,True]).all():
            raise ValueError('sensor validity must contain boolean values with shape [T]')
        valid=valid.astype(bool)
    normalized=(video-stats['video_mean'])/stats['video_std']
    if not np.isfinite(normalized).all(): raise ValueError('Normalization overflow')
    station=meta['station_aliases'].get(station,station)
    enabled=vibration is not None and station in meta['eligible_vibration_stations']
    station_index=meta['vibration_stations'].index(station) if station in meta['vibration_stations'] else meta['vibration_stations'].index('_global')
    probabilities=[];used=np.zeros(n,bool);window_lengths=[]
    with torch.inference_mode():
        for start in range(0,n,chunk_frames):
            stop=min(start+chunk_frames,n)
            x=torch.as_tensor(normalized[start:stop],device=device).unsqueeze(0)
            if not enabled or not valid[start:stop].any():
                out=model(x,has_vib=False);prob=out['logits'][-1,0].softmax(dim=0).T
            else:
                pre=model.pre(model.w_video(x)).argmax(-1)[0].cpu().numpy()
                segments=predict_segments(pre,FEATURE_RATE,**meta['segment_config']) or [(0,stop-start)]
                windows,allow,_=build_segment_tokens(segments,stop-start,FEATURE_RATE,min_seconds=2.,neighbors=0)
                token_valid=np.asarray([valid[start+a:start+b].mean()>=.8 for a,b in windows])
                allow=allow & token_valid[None,:]
                raw=window_features(vibration[start:stop],windows)
                z=(raw-stats['vibration_median'][station_index])/stats['vibration_iqr'][station_index]
                if not np.isfinite(z).all(): raise ValueError('Non-finite normalized sensor tokens')
                z=torch.as_tensor(z,dtype=torch.float32,device=device).unsqueeze(0)
                out=model(x,z,windows,valid_mask=None,has_vib=True,allow_mask=allow)
                active=allow.any(axis=1)
                # Rows with no usable sensor token must take the exact no-sensor path.
                if not active.all():
                    pure=model(x,has_vib=False)['fused_features']
                    fused=torch.where(torch.as_tensor(active,device=device)[None,:,None],out['fused_features'],pure)
                    logits=model._classify(fused)
                else: logits=out['logits']
                prob=logits[-1,0].softmax(dim=0).T
                used[start:stop]=active
                window_lengths.extend([(b-a)/FEATURE_RATE for a,b in windows])
            probabilities.append(prob.cpu().numpy())
    prob=np.concatenate(probabilities).astype(np.float32)
    if not np.isfinite(prob).all(): raise ValueError('Model produced non-finite probabilities')
    return prob,used,window_lengths

def infer(video_features,bundle,station,output,*,vibration=None,device='cpu',chunk_frames=2247,labels=None):
    device=device_for(device);begin=time.perf_counter()
    model,stats,meta=load_bundle(bundle,device)
    model_load_seconds=time.perf_counter()-begin
    video,timestamps,visual_meta=load_video_features(video_features)
    for name,expected in [('encoder',meta['video_encoder']),('revision',meta['video_encoder_revision']),('encoder_qkv_bias',False)]:
        if name in visual_meta and visual_meta[name]!=expected:
            raise ValueError(f'Video feature protocol mismatch: {name}')
    if len(timestamps)>1:
        dt=float(np.median(np.diff(timestamps)))
        if not np.isclose(dt,1/FEATURE_RATE,rtol=.01): raise ValueError('This checkpoint expects approximately 7.4933 video features per second; use stage 1 sampling')
    else: dt=float(visual_meta.get('feature_dt',1/FEATURE_RATE))
    vib,valid=None,None
    if vibration:
        with np.load(vibration,allow_pickle=False) as z:
            from .vibration import FEATURE_NAMES
            if 'feature_names' not in z or z['feature_names'].tolist()!=list(FEATURE_NAMES):
                raise ValueError('Wrong vibration feature names/order; use stage 2 or export_cached_inputs.py')
            vib=z['features'];valid=z['valid'];vt=z['timestamps_sec']
            if vt.shape!=timestamps.shape or not np.allclose(vt,timestamps,rtol=0,atol=1e-6):
                raise ValueError('Vibration timestamps do not match video; rerun stage 2')
    synchronize(device);begin=time.perf_counter()
    prob,used,lengths=infer_arrays(model,video,vib,valid,station,stats,meta,device,chunk_frames)
    synchronize(device);elapsed=time.perf_counter()-begin
    pred=prob.argmax(axis=1)
    end=float(visual_meta.get('analysis_end_sec',timestamps[-1]+dt))
    # Ground truth is loaded only after predictions exist, for optional evaluation.
    gt=np.load(labels,allow_pickle=False) if labels else None
    run=dict(schema=meta['schema'],split_seed=meta['split_seed'],station_id=station,
             model_sha256=meta['source_checkpoint_sha256'],device=str(device),chunk_frames=chunk_frames,
             model_load_seconds=model_load_seconds,inference_seconds=elapsed,n_features=len(pred),
             used_vibration_frames=int(used.sum()),n_sensor_windows=len(lengths),
             mean_sensor_window_seconds=float(np.mean(lengths)) if lengths else 0.,
             timing_scope='normalization, predicted segments, sensor pooling, attention, gate, MS-TCN; excludes extraction, model/data loading and output writes',
             inference_label_use='none')
    return write_outputs(output,pred,prob,timestamps,end,used,run,gt)
