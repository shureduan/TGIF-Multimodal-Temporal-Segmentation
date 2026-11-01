"""Convert model labels to half-open activity intervals and handling durations."""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
from .common import CLASSES,validate_timestamps,write_json

def intervals(prediction,timestamps,end_sec):
    t=validate_timestamps(timestamps);p=np.asarray(prediction)
    if p.shape!=t.shape or not np.isin(p,[0,1,2]).all():
        raise ValueError('Predictions must be idle=0/sewing=1/handling=2 on the timestamp grid')
    if not np.isfinite(end_sec) or end_sec<=t[-1]:
        raise ValueError('analysis_end_sec must be finite and after the last feature timestamp')
    if len(t)>1:
        dt=np.median(np.diff(t))
        if np.any(np.diff(t)>1.5*dt) or end_sec-t[-1]>1.5*dt:
            raise ValueError('Video timestamp gaps require explicit splitting; refusing to count unobserved time')
    boundaries=np.r_[0,np.flatnonzero(p[1:]!=p[:-1])+1,len(p)]
    result=[]
    for a,b in zip(boundaries[:-1],boundaries[1:]):
        start=float(t[a]);end=float(t[b]) if b<len(t) else float(end_sec)
        result.append(dict(start_sec=start,end_sec=end,duration_sec=end-start,
                           class_id=int(p[a]),activity=CLASSES[int(p[a])]))
    return result

def summarize(prediction,timestamps,end_sec):
    runs=intervals(prediction,timestamps,end_sec)
    total=float(end_sec-timestamps[0]);by_class={}
    for name in CLASSES:
        durations=[r['duration_sec'] for r in runs if r['activity']==name]
        seconds=float(sum(durations))
        by_class[name]=dict(total_seconds=seconds,total_minutes=seconds/60,
                            fraction_of_analyzed_time=seconds/total,n_segments=len(durations),
                            mean_segment_seconds=float(np.mean(durations)) if durations else 0.)
    return runs,dict(analysis_start_sec=float(timestamps[0]),analysis_end_sec=float(end_sec),
                     analyzed_seconds=total,classes=by_class,
                     handling_seconds=by_class['handling']['total_seconds'],
                     definition='Sum of half-open intervals predicted as handling (class 2); not computation time')

def write_outputs(output,pred,prob,timestamps,end_sec,used_vibration,run_meta,labels=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    runs,summary=summarize(pred,timestamps,end_sec)
    ends=np.r_[timestamps[1:],end_sec]
    with open(output/'predictions.csv','w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['start_sec','end_sec','class_id','activity','confidence','p_idle','p_sewing','p_handling','used_vibration'])
        for i,c in enumerate(pred):
            writer.writerow([float(timestamps[i]),float(ends[i]),int(c),CLASSES[int(c)],float(prob[i,int(c)]),*map(float,prob[i]),bool(used_vibration[i])])
    with open(output/'activity_segments.csv','w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['start_sec','end_sec','duration_sec','class_id','activity'])
        writer.writeheader();writer.writerows(runs)
    summary['run']=run_meta
    if labels is not None:
        from sklearn.metrics import f1_score
        y=np.asarray(labels)
        if y.shape!=pred.shape or not np.isin(y,[-100,0,1,2]).all():
            raise ValueError('Evaluation labels must have shape [T], values -100/0/1/2')
        valid=y>=0
        if valid.any():
            summary['evaluation']=dict(n_labeled=int(valid.sum()),accuracy=float(np.mean(pred[valid]==y[valid])),
                  macro_f1=float(f1_score(y[valid],pred[valid],labels=[0,1,2],average='macro',zero_division=0)))
        else: summary['evaluation']=dict(n_labeled=0)
    write_json(output/'handling_time_summary.json',summary)
    np.savez_compressed(output/'predictions.npz',prediction=pred,probabilities=prob,timestamps_sec=timestamps,used_vibration=used_vibration)
    return summary
