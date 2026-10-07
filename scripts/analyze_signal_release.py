#!/usr/bin/env python3
"""Recompute the released 18-subject comparison and matched nine-subject controls.

Inputs are inference .npz files, never precomputed scores. The release cohort
is explicit and fixed; incomplete/unmatched extra runs cannot enter its means.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy import stats
from tgif_dwa.signal_wear import METHOD, PROTOCOL
from tgif_dwa.signal_study import BASELINES, DIAGNOSTICS, METRICS, PRIMARY, measure
from tgif_dwa.wear_metrics import official_wear_concat_metrics
from tgif_dwa.wear_statistics import holm_adjust

REPO=Path(__file__).resolve().parents[1]
SEEDS=[41,47,53]; FOLDS=list(range(1,19)); ODD=list(range(1,19,2))
CONTROLS=['NO_SENSOR_RESIZE','NO_CONTRACTION']

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def dump(p,v):
    Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')

def csv_write(p,rows):
    with Path(p).open('w',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]),lineterminator='\n')
        w.writeheader();w.writerows(rows)

def comparisons(index,family,comparators,folds,family_size):
    rows=[]
    for comparator in comparators:
        for metric in PRIMARY:
            d=np.array([[index[(METHOD,s,f)][metric]-index[(comparator,s,f)][metric] for s in SEEDS] for f in folds])
            x=d.mean(1);mean=float(x.mean());sd=float(x.std(ddof=1));n=len(x)
            half=float(stats.t.ppf(.975,n-1)*sd/np.sqrt(n))
            if sd:res=stats.ttest_1samp(x,0);t,p=float(res.statistic),float(res.pvalue)
            else:t,p=(0.,1.) if mean==0 else (None,0.)
            rows.append({'family':family,'comparator':comparator,'metric':metric,'n_subjects':n,'n_seeds':3,
                'mean_difference':mean,'ci95_low':mean-half,'ci95_high':mean+half,'t_statistic':t,'p_two_sided':p,
                'wins':int((x>0).sum()),'ties':int((x==0).sum()),'losses':int((x<0).sum()),
                'per_subject_difference':dict(zip([f'sbj_{f-1}' for f in folds],x.tolist())),
                'per_seed_difference':dict(zip(map(str,SEEDS),d.mean(0).tolist()))})
    adjusted=holm_adjust([r['p_two_sided'] for r in rows]+[1.]*(family_size-len(rows)))
    for r,p in zip(rows,adjusted):r.update(p_holm_family=p,family_test_count=family_size)
    return rows

def analyze(baseline, signal, source):
    source.mkdir(parents=True,exist_ok=True)
    jobs=[(METHOD,s,f) for s in SEEDS for f in FOLDS]+[(m,s,f) for m in CONTROLS for s in SEEDS for f in ODD]
    rows=[];classes=[];windows=[];sources=[];truths={};tracks={}
    selected=[(m,s,f) for m in BASELINES for s in SEEDS for f in FOLDS]+jobs
    for m,s,f in selected:
        origin=baseline if m in BASELINES else signal
        path=origin/m/f'seed_{s}/split_{f:02d}.npz'
        with np.load(path,allow_pickle=False) as z:p={k:z[k].copy() for k in z.files}
        expected_protocol='fixed_epoch_loso_v2' if m in BASELINES else PROTOCOL
        assert (p['id'].item(),int(p['seed']),int(p['fold']),p['method'].item(),p['protocol'].item())==(f'sbj_{f-1}',s,f,m,expected_protocol)
        assert np.array_equal(p['pred'],p['probabilities'].argmax(-1))
        if (s,f) in truths:assert np.array_equal(truths[(s,f)],p['true'])
        truths[(s,f)]=p['true']
        fields=[(m,'probabilities')]+([('PARENT_NO_PROBE','parent_probabilities'),('ROUND0','round0_probabilities')] if m==METHOD else [])
        for display,field in fields:
            values,pc=measure(p['true'],p[field],p['id'].item())
            identity={'method':display,'source_method':m,'subject':f'sbj_{f-1}','seed':s,'fold':f,'protocol':expected_protocol,'run_id':p['run_id'].item()}
            rows.append({**identity,**values});classes.extend({**identity,**c} for c in pc)
            tracks[(display,s,f)]=p[field].argmax(-1)
        sources.append({'method':m,'seed':s,'fold':f,'path':str(path.relative_to(origin)),'sha256':sha(path)})
        if m not in BASELINES:
            for r in [0,1]:
                g=lambda name:p[f'round{r}_{name}']
                lo,hi=g('selected_window_start'),g('selected_window_end');a,b=g('seed_segment_start'),g('seed_segment_end')
                initial=b-a+1;selected_width=hi-lo+1;positions=np.arange(len(lo))
                assert (lo<=positions).all() and (hi>=positions).all() and (lo>=0).all() and (hi<len(lo)).all()
                assert np.array_equal(selected_width,initial-g('boundary_clipped_tokens')-g('redundancy_trimmed_tokens')+g('expanded_tokens'))
                if m=='NO_SENSOR_RESIZE':assert np.array_equal(lo,a) and np.array_equal(hi,b)
                if m=='NO_CONTRACTION':assert (lo<=a).all() and (hi>=b).all()
                windows.append({'method':m,'seed':s,'fold':f,'subject':f'sbj_{f-1}','round':r,
                    'initial_mean_tokens':float(initial.mean()),'selected_mean_tokens':float(selected_width.mean()),
                    'shrink_fraction':float(np.mean(selected_width<initial)),'expand_fraction':float(np.mean(selected_width>initial)),
                    'same_length_fraction':float(np.mean(selected_width==initial)),'stable_fraction':float(g('sensor_statistics_stable').mean())})
    index={(r['method'],r['seed'],r['fold']):r for r in rows}
    assert len(index)==len(rows)==432
    csv_write(source/'per_subject_seed_metrics.csv',rows);csv_write(source/'per_class_metrics.csv',classes)
    csv_write(source/'window_behavior.csv',windows)
    dump(source/'prediction_sources.json',sources)
    def aggregate(methods,folds):
        subjects=[];seeds=[];summary=[]
        for m in methods:
            for f in folds:
                subjects.append({'method':m,'fold':f,'subject':f'sbj_{f-1}',**{k:float(np.mean([index[(m,s,f)][k] for s in SEEDS])) for k in METRICS}})
            for s in SEEDS:
                pooled=official_wear_concat_metrics([truths[(s,f)] for f in folds],[tracks[(m,s,f)] for f in folds])['macro_f1']
                seeds.append({'method':m,'seed':s,'concat_macro_f1':pooled,**{k:float(np.mean([index[(m,s,f)][k] for f in folds])) for k in METRICS}})
            ss=[r for r in subjects if r['method']==m]
            summary.append({'method':m,'n_subjects':len(folds),'n_seeds':3,'concat_macro_f1':float(np.mean([r['concat_macro_f1'] for r in seeds if r['method']==m])),
                **{k:float(np.mean([r[k] for r in ss])) for k in METRICS},**{k+'_subject_sd':float(np.std([r[k] for r in ss],ddof=1)) for k in METRICS}})
        return subjects,seeds,summary
    subjects,seeds,summary=aggregate([*BASELINES,METHOD,*DIAGNOSTICS],FOLDS)
    csv_write(source/'subject_means.csv',subjects);csv_write(source/'seed_means.csv',seeds);csv_write(source/'aggregate_metrics.csv',summary)
    sub,_,summarysub=aggregate([METHOD,*CONTROLS],ODD)
    csv_write(source/'ablation_subject_means.csv',sub);csv_write(source/'ablation_aggregate_metrics.csv',summarysub)
    comparisons_all=[]
    for family,comps,folds,size in [('baseline',BASELINES,FOLDS,8),('retrained_components',CONTROLS,ODD,8),('frozen_components',DIAGNOSTICS,FOLDS,4),
        ('baseline_excluding_pilot',BASELINES,FOLDS[1:],8),('components_excluding_pilot',CONTROLS,ODD[1:],8)]:
        comparisons_all+=comparisons(index,family,comps,folds,size)
    dump(source/'paired_statistics.json',{'interim':True,'unit':'subject after averaging three matched seed differences','comparisons':comparisons_all})
    csv_write(source/'paired_comparisons.csv',[{k:v for k,v in r.items() if not isinstance(v,dict)} for r in comparisons_all])
    for f in [2,10]:
        timeline=[{'time_seconds':(i+1)*.5,'track':'GT','label':int(y)} for i,y in enumerate(truths[(47,f)])]
        for m in [*BASELINES,METHOD]:timeline.extend({'time_seconds':(i+1)*.5,'track':m,'label':int(y)} for i,y in enumerate(tracks[(m,47,f)]))
        csv_write(source/f'timeline_sbj_{f-1}_seed47.csv',timeline)
    manifest=json.loads((REPO/'results/wear_signal_v3/manifest.json').read_text())
    manifest.pop('training_source_sha256',None)
    manifest.pop('data_sha256',None)
    manifest['sha256']={p.name:sha(p) for p in sorted(source.iterdir()) if p.suffix in ('.csv','.json') and p.name!='manifest.json'}
    manifest['origin']='Recomputed from the supplied inference files; prediction hashes are in prediction_sources.json.'
    dump(source/'manifest.json',manifest)
    print(f'Validated {len(rows)} records; exported tables and all 28 paired comparisons to {source}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--v2-predictions',type=Path,required=True,help='directory containing METHOD/seed_N/split_FF.npz')
    parser.add_argument('--v3-predictions',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True,help='new numerical-source directory')
    args=parser.parse_args()
    analyze(args.v2_predictions.resolve(),args.v3_predictions.resolve(),args.output.resolve())

if __name__=='__main__':
    main()
