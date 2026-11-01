"""2. Label-free extraction of four aligned 500-Hz vibration features."""
from __future__ import annotations
from datetime import datetime
from pathlib import Path
import re,time
import numpy as np
import pandas as pd
from scipy.signal.windows import hann
from .common import validate_timestamps,write_json

FEATURE_NAMES=('rms_dyn','peak_to_peak','log_e_20_40','ratio_20_40')
FS=500.

def minute_grid(folder, video_start, duration, accel_scale=16384.):
    """Legacy clock contract: anchor each minute independently; preserve gaps."""
    start=datetime.fromisoformat(video_start)
    if start.tzinfo is not None: raise ValueError('Use the same naive local clock as the CSV filenames')
    if not np.isfinite(duration) or duration<=0 or not np.isfinite(accel_scale) or accel_scale<=0:
        raise ValueError('duration and accel_scale must be positive and finite')
    grid=np.full(int(np.ceil(duration*FS)),np.nan,dtype=np.float64)
    files=sorted(Path(folder).glob('*.csv'));used=0
    for file in files:
        m=re.search(r'_(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})\.csv$',file.name)
        if not m: continue
        clock=datetime.fromisoformat(f'{m[1]}T{m[2]}:{m[3]}:00')
        delta=(clock-start).total_seconds()
        if not (-60 <= delta < duration): continue
        frame=pd.read_csv(file)
        required=('timestamp','ax','ay','az')
        if not set(required)<=set(frame):
            raise ValueError(f'{file.name}: require timestamp, ax, ay, az')
        if not len(frame): continue
        x=frame[list(required)].apply(pd.to_numeric,errors='raise').to_numpy(dtype=np.float64)
        if not np.isfinite(x).all(): raise ValueError(f'{file.name}: non-finite raw input')
        if np.any(np.diff(x[:,0])<0): raise ValueError(f'{file.name}: timestamp must not decrease')
        times=delta+(x[:,0]-x[0,0])/1000.
        indices=np.rint(times*FS).astype(np.int64)
        keep=(indices>=0)&(indices<len(grid))
        magnitude=np.linalg.norm(x[:,1:]/accel_scale,axis=1)
        grid[indices[keep]]=magnitude[keep];used+=1
    if not used: raise ValueError('No minute CSV files overlap the requested video clock interval')
    return grid,used

def dynamic_signal(grid):
    s=pd.Series(np.asarray(grid,dtype=np.float64))
    return (s-s.rolling(1000,center=True,min_periods=250).mean()).to_numpy()

def features(dyn,timestamps,offset_seconds):
    """Same 2-second Hann FFT and four feature definitions as the TGIF model.

    offset_seconds is explicitly supplied, never estimated from action labels.
    Interpolated values retain a false validity flag and cannot become evidence.
    """
    timestamps=validate_timestamps(timestamps)
    if not np.isfinite(offset_seconds): raise ValueError('offset_seconds must be finite')
    n=1000;freq=np.fft.rfftfreq(n,d=1/FS);window=hann(n)
    total=(freq>=.5)&(freq<=150);band=(freq>=20)&(freq<=40)
    result=np.full((len(timestamps),4),np.nan,np.float32)
    valid=np.zeros(len(timestamps),bool);fraction=np.zeros(len(timestamps),np.float32)
    shift=int(round(offset_seconds*FS))
    for i,t in enumerate(timestamps):
        center=shift+int(t*FS);a=center-n//2;b=a+n
        if a<0 or b>len(dyn): continue
        seg=dyn[a:b];mask=np.isfinite(seg);fraction[i]=mask.mean()
        if fraction[i]<.3: continue
        z=np.where(mask,seg,0.)
        power=np.abs(np.fft.rfft(z*window))**2/n
        energy=power[band].sum();all_energy=power[total].sum()+1e-12
        result[i]=[np.sqrt(np.mean(z*z)),np.nanmax(seg)-np.nanmin(seg),
                   np.log10(energy+1e-12),energy/all_energy]
        valid[i]=True
    for j in range(4):
        good=np.flatnonzero(np.isfinite(result[:,j]))
        if len(good):
            missing=~np.isfinite(result[:,j])
            result[missing,j]=np.interp(np.flatnonzero(missing),good,result[good,j])
    return result,valid,fraction

def extract(folder,video_features,video_start,offset_seconds,output,*,accel_scale=16384.,allow_missing=False):
    video_features=Path(video_features)
    ts=validate_timestamps(np.load(video_features/'timestamps_sec.npy',allow_pickle=False))
    meta_path=video_features/'metadata.json'
    import json
    visual_meta=json.loads(meta_path.read_text()) if meta_path.exists() else {}
    dt=float(visual_meta.get('feature_dt',np.median(np.diff(ts)) if len(ts)>1 else 1/7.4933))
    end=float(visual_meta.get('analysis_end_sec',ts[-1]+dt))
    before=time.perf_counter();grid,nfiles=minute_grid(folder,video_start,end,accel_scale)
    dyn=dynamic_signal(grid)
    x,valid,fraction=features(dyn,ts,offset_seconds)
    if not valid.any():
        if not allow_missing: raise ValueError('No usable aligned vibration; inspect clock/offset or pass --allow-missing')
        x=np.zeros((len(ts),4),np.float32)
    if not np.isfinite(x).all(): raise ValueError('Vibration feature extraction produced non-finite values')
    elapsed=time.perf_counter()-before
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,features=x,valid=valid,valid_fraction=fraction,timestamps_sec=ts,
                        feature_names=np.asarray(FEATURE_NAMES))
    metadata=dict(stage='vibration_feature_extraction',sample_rate_hz=FS,accel_scale=accel_scale,
                  window_seconds=2.,rolling_baseline_seconds=2.,offset_seconds=offset_seconds,
                  offset_convention='sensor grid center = video seconds + offset_seconds',
                  video_start=video_start,n_csv_files=nfiles,n_features=len(ts),
                  n_usable_features=int(valid.sum()),availability=float(np.isfinite(grid).mean()),
                  feature_names=list(FEATURE_NAMES),extraction_seconds=elapsed,
                  timing_scope='CSV reads, clock grid, rolling baseline and four features; excludes output writes',
                  label_use='none; clock offset supplied by the caller')
    write_json(output.with_suffix('.json'),metadata)
    return metadata
