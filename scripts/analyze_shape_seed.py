#!/usr/bin/env python3
"""Score all completed one-seed folds without changing or selecting the model."""
import argparse
import csv
import json
from pathlib import Path
import time
import numpy as np
from scipy import stats
from tgif_dwa.shape_wear import METHOD,PROTOCOL,load_shape_wear
from tgif_dwa.signal_study import METRICS,measure
from tgif_dwa.signal_wear import load_signal_wear
from tgif_dwa.wear_metrics import official_wear_concat_metrics
from tgif_dwa.wear_statistics import holm_adjust
from tgif_dwa.wear_data import load_sequence
from tgif_dwa.wear_model import normalize_wear_inertial
from run_shape_seed import REFERENCES,sha,dump


def csv_write(path,rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--benchmark',type=Path,required=True)
    parser.add_argument('--skip-timing',action='store_true',help='omit forward latency measurement for synthetic pipeline checks')
    args=parser.parse_args();root=args.benchmark.resolve();plan=json.loads((root/'benchmark_plan.json').read_text());seed=plan['seed']
    source=root/'source_data';source.mkdir(exist_ok=True)
    rows=[];classes=[];tracks={};truths={};windows=[];prediction_hashes={}
    methods=[*REFERENCES,METHOD];diagnostics=['SHAPE_PARENT_NO_PROBE','SHAPE_ROUND0']
    for method in methods:
        for fold in plan['folds']:
            if method==METHOD:
                path=root/'predictions'/METHOD/f'seed_{seed}/split_{fold:02d}.npz'
                protocol=PROTOCOL
            else:
                path=Path(plan['reference_predictions'][f'{method}/{fold}'])
                if sha(path)!=plan['reference_artifact_sha256'][str(path)]:raise ValueError('Reference prediction changed')
                protocol='signal_adaptive_loso_v3_candidate' if method=='SIGNAL_ADAPTIVE_DWA' else 'fixed_epoch_loso_v2'
            with np.load(path,allow_pickle=False) as saved:p={k:saved[k].copy() for k in saved.files}
            identity=(p['method'].item(),p['protocol'].item(),int(p['seed']),int(p['fold']),p['id'].item())
            if identity!=(method,protocol,seed,fold,f'sbj_{fold-1}'):raise ValueError('Prediction identity differs')
            if not np.array_equal(p['pred'],p['probabilities'].argmax(-1)):raise ValueError('Prediction/probability mismatch')
            if fold in truths and not np.array_equal(truths[fold],p['true']):raise ValueError('Ground truths differ')
            truths[fold]=p['true'];prediction_hashes[str(path)]=sha(path)
            fields=[(method,'probabilities')]
            if method==METHOD:fields.extend(zip(diagnostics,['parent_probabilities','round0_probabilities']))
            for display,field in fields:
                values,per_class=measure(p['true'],p[field],p['id'].item())
                info={'method':display,'source_method':method,'subject':f'sbj_{fold-1}','seed':seed,'fold':fold,
                      'protocol':protocol,'run_id':p['run_id'].item()}
                rows.append({**info,**values});classes.extend({**info,**c} for c in per_class)
                tracks[display,fold]=p[field].argmax(-1)
            if method in ('SIGNAL_ADAPTIVE_DWA',METHOD):
                for r in (0,1):
                    lo=p[f'round{r}_selected_window_start'];hi=p[f'round{r}_selected_window_end']
                    a=p[f'round{r}_seed_segment_start'];b=p[f'round{r}_seed_segment_end'];t=np.arange(len(lo))
                    if not ((lo<=t).all() and (hi>=t).all() and (lo>=0).all() and (hi<len(lo)).all()):
                        raise ValueError('Invalid window geometry')
                    width=hi-lo+1;initial=b-a+1
                    ledger=initial-p[f'round{r}_boundary_clipped_tokens']-p[f'round{r}_redundancy_trimmed_tokens']+p[f'round{r}_expanded_tokens']
                    if not np.array_equal(width,ledger):raise ValueError('Window bookkeeping mismatch')
                    windows.append({'method':method,'seed':seed,'fold':fold,'subject':f'sbj_{fold-1}','round':r,
                                    'mean_initial_tokens':float(initial.mean()),'mean_selected_tokens':float(width.mean()),
                                    'shrink_fraction':float(np.mean(width<initial)),'same_length_fraction':float(np.mean(width==initial)),
                                    'expand_fraction':float(np.mean(width>initial))})
    index={(r['method'],r['fold']):r for r in rows}
    if len(rows)!=144 or len(index)!=144:raise ValueError('Incomplete or duplicate records')
    csv_write(source/'per_subject_metrics.csv',rows);csv_write(source/'per_class_metrics.csv',classes)
    csv_write(source/'window_behavior.csv',windows)
    summary=[]
    for method in [*methods,*diagnostics]:
        records=[index[method,f] for f in plan['folds']]
        pooled=official_wear_concat_metrics([truths[f] for f in plan['folds']],[tracks[method,f] for f in plan['folds']])['macro_f1']
        summary.append({'method':method,'n_subjects':18,'n_seeds':1,'seed':seed,'concat_macro_f1':pooled,
                        **{metric:float(np.mean([r[metric] for r in records])) for metric in METRICS},
                        **{metric+'_subject_sd':float(np.std([r[metric] for r in records],ddof=1)) for metric in METRICS}})
    csv_write(source/'aggregate_metrics.csv',summary)
    comparisons=[]
    for comparator in REFERENCES:
        for metric in ('macro_f1_19','map_at_0.5'):
            differences=np.array([index[METHOD,f][metric]-index[comparator,f][metric] for f in plan['folds']])
            mean=float(differences.mean());sd=float(differences.std(ddof=1));half=float(stats.t.ppf(.975,17)*sd/np.sqrt(18))
            if sd==0:t,p=(0.,1.) if mean==0 else (None,0.)
            else:
                result=stats.ttest_1samp(differences,0);t,p=float(result.statistic),float(result.pvalue)
            comparisons.append({'comparator':comparator,'metric':metric,'n_subjects':18,'n_seeds':1,'mean_difference':mean,
                                'ci95_low':mean-half,'ci95_high':mean+half,'t_statistic':t,'p_two_sided':p,
                                'wins':int((differences>0).sum()),'ties':int((differences==0).sum()),'losses':int((differences<0).sum()),
                                'per_subject_difference':dict(zip([f'sbj_{f-1}' for f in plan['folds']],differences.tolist()))})
    adjusted=holm_adjust([r['p_two_sided'] for r in comparisons])
    for r,p in zip(comparisons,adjusted):r['p_holm_family']=p;r['family_test_count']=10
    dump(source/'paired_statistics.json',{'unit':'paired held-out subject','seed':seed,
         'scope':'Exploratory single-seed comparison with overlapping LOSO training sets and prior method development on these subjects.',
         'comparisons':comparisons})
    csv_write(source/'paired_comparisons.csv',[{k:v for k,v in r.items() if not isinstance(v,dict)} for r in comparisons])
    for fold in (2,10):
        timeline=[{'time_seconds':(i+1)*.5,'track':'GT','label':int(y)} for i,y in enumerate(truths[fold])]
        for method in methods:
            timeline.extend({'time_seconds':(i+1)*.5,'track':method,'label':int(y)} for i,y in enumerate(tracks[method,fold]))
        csv_write(source/f'timeline_sbj_{fold-1}.csv',timeline)
    # Matched CPU forward timings are separate from mixed-device score evaluation.
    import torch
    torch.set_num_threads(1);timings=[]
    for fold in (() if args.skip_timing else (1,10,18)):
        parent_path=next(Path(p) for p in plan['reference_artifact_sha256'] if p.endswith(f'SIGNAL_ADAPTIVE_DWA/seed_{seed}/split_{fold:02d}/parent.pt'))
        old,mean,std=load_signal_wear(parent_path,parent_path.parent/'background_probe.pt','cpu')
        folder=root/'checkpoints'/METHOD/f'seed_{seed}/split_{fold:02d}'
        new,mean2,std2=load_shape_wear(folder/'parent.pt',folder/'background_probe.pt','cpu')
        if not np.array_equal(mean,mean2) or not np.array_equal(std,std2):raise ValueError('Normalization differs from matched v3')
        data=load_sequence(plan['data_root'],f'sbj_{fold-1}')
        video=torch.from_numpy(data['video']);raw=torch.from_numpy(normalize_wear_inertial(data['inertial'],mean,std))
        for method,model in [('SIGNAL_ADAPTIVE_DWA',old),(METHOD,new)]:
            with torch.inference_mode():
                model(video,raw)
                for repeat in range(5):
                    start=time.perf_counter();model(video,raw);elapsed=time.perf_counter()-start
                    timings.append({'method':method,'subject':f'sbj_{fold-1}','fold':fold,'repeat':repeat,'device':'cpu',
                                    'tokens':len(video),'forward_seconds':elapsed})
    if timings:csv_write(source/'inference_timing.csv',timings)
    dump(source/'manifest.json',{'study':plan['study'],'seed':seed,'n_subjects':18,'n_seeds':1,
         'aggregation':'Equal subject means; sample subject SD. Pooled Macro-F1 is separately labeled.',
         'prediction_sha256':prediction_hashes,'sha256':{p.name:sha(p) for p in sorted(source.iterdir()) if p.suffix in ('.csv','.json') and p.name!='manifest.json'}})
    new=next(r for r in summary if r['method']==METHOD);old=next(r for r in summary if r['method']=='SIGNAL_ADAPTIVE_DWA')
    difference={metric:new[metric]-old[metric] for metric in METRICS}
    conclusion=('Both primary descriptive means increased.' if all(difference[k]>0 for k in ('macro_f1_19','map_at_0.5'))
                else 'The two primary descriptive comparisons are mixed.' if any(difference[k]>0 for k in ('macro_f1_19','map_at_0.5'))
                else 'The envelope-shape pilot did not improve either primary descriptive mean.')
    lines=['# Envelope-shape DWA: seed '+str(seed),'','Complete: 18 LOSO folds, fixed parent/probe budgets 30/15; final checkpoints.','',
           'The candidate uses AC-RMS envelopes, three ordered temporal cells and coarse slope histograms. Local support grows through positions similar to a fixed query descriptor and stops at the first mismatch. All thresholds use training subjects only.','',
           '| Method | Subject Macro-F1 | Concat. F1 | Accuracy | mAP@0.5 | Avg mAP |','|---|---:|---:|---:|---:|---:|']
    for r in summary:lines.append('| '+r['method']+' | '+' | '.join(f"{r[k]:.4f}" for k in ('macro_f1_19','concat_macro_f1','accuracy','map_at_0.5','avg_map'))+' |')
    lines+=['',conclusion,'','Candidate minus current DWA: '+json.dumps(difference),
            '', 'This is an exploratory one-seed pilot, not an independent multi-seed confirmation. Whole-model reference rows use the same held-out subjects, seed, budgets and per-fold devices. Frozen output diagnostics share candidate weights.',
            '', 'All primary and secondary metrics, paired comparisons, window geometry and matched CPU timings are retained in source_data/. Figures are generated automatically after analysis. No automatic publication.']
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print('Scored 144 records, all 12 metrics and paired reference comparisons.',flush=True)

if __name__=='__main__':main()
