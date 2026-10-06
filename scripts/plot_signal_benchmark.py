#!/usr/bin/env python3
"""Render the complete local comparison pack from hash-verified study tables."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd

from tgif_dwa.signal_wear import METHOD, VARIANTS
from tgif_dwa.signal_study import BASELINES, DIAGNOSTICS, METRICS

MAIN = [*BASELINES,METHOD]
LABELS = {'VIDEO_ONLY':'Video only','EARLY_CONCAT':'Early concat.','FIXED_WINDOW_ATTENTION':'Fixed attention',
          'FINAL_MODEL':'Published fixed margin',METHOD:'Sensor-driven DWA',
          'NO_SENSOR_RESIZE':'No sensor resize','NO_CONTRACTION':'No contraction','NO_EXPANSION':'No expansion',
          'UNIFORM_POOL':'Uniform pooling','PARENT_NO_PROBE':'Parent without probe','ROUND0':'Round 0 output'}
COLORS = dict(zip(MAIN,['#4C78A8','#F28E2B','#59A14F','#8B79A9','#C84452']))
METRIC_LABELS = {'accuracy':'Frame accuracy','macro_precision_19':'Macro precision','macro_recall_19':'Macro recall',
    'macro_f1_19':'Macro-F1 (subject mean)','background_f1':'Background F1','action_macro_f1':'Action Macro-F1',
    'map_at_0.3':'mAP@0.3','map_at_0.4':'mAP@0.4','map_at_0.5':'mAP@0.5','map_at_0.6':'mAP@0.6',
    'map_at_0.7':'mAP@0.7','avg_map':'Average mAP','concat_macro_f1':'Macro-F1 (pooled)'}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--benchmark',type=Path,required=True)
    args=parser.parse_args();root=args.benchmark.resolve();source=root/'source_data'
    plan=json.loads((root/'benchmark_plan.json').read_text());manifest=json.loads((source/'manifest.json').read_text())
    for name,expected in manifest['sha256'].items():
        assert hashlib.sha256((source/name).read_bytes()).hexdigest()==expected,name
    aggregate=pd.read_csv(source/'aggregate_metrics.csv').set_index('method')
    subjects=pd.read_csv(source/'subject_means.csv');seeds=pd.read_csv(source/'seed_means.csv')
    comparisons=pd.read_csv(source/'paired_comparisons.csv');windows=pd.read_csv(source/'window_behavior.csv')
    jitter=pd.read_csv(source/'boundary_jitter.csv');classes=pd.read_csv(source/'per_class_metrics.csv')
    root.joinpath('figures').mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':11,'axes.labelsize':9,
        'legend.fontsize':8,'xtick.labelsize':8,'ytick.labelsize':8,'axes.spines.top':False,'axes.spines.right':False,
        'pdf.fonttype':42,'ps.fonttype':42})
    names=[];smoke=plan.get('smoke_test',False)
    pdf_path=root/'WEAR_sensor_DWA_full_comparison.pdf'
    with PdfPages(pdf_path) as pdf:
        def save(fig,name,subtitle,bottom=.065):
            names.append(name)
            if smoke:
                fig.suptitle('SYNTHETIC PIPELINE CHECK - NOT EXPERIMENTAL RESULTS',color='red',fontsize=14)
            fig.text(.5,.012,subtitle,ha='center',fontsize=8)
            fig.tight_layout(rect=(0,bottom,1,.95 if smoke else 1))
            fig.savefig(root/'figures'/f'{name}.png',dpi=200)
            pdf.savefig(fig);plt.close(fig)

        fig,axes=plt.subplots(1,2,figsize=(14.5,5.1),gridspec_kw={'width_ratios':[1.4,1]})
        keys=['macro_f1_19','concat_macro_f1','map_at_0.5','avg_map'];x=np.arange(len(keys));width=.15
        for i,m in enumerate(MAIN):
            axes[0].bar(x+(i-2)*width,aggregate.loc[m,keys].to_numpy(float),width,color=COLORS[m],label=LABELS[m])
        axes[0].set_xticks(x,['Subject\nMacro-F1','Pooled\nMacro-F1','mAP@0.5','Avg. mAP'])
        axes[0].set_ylim(0,1);axes[0].set_ylabel('Score');axes[0].set_title('Aggregate performance')
        axes[0].grid(axis='y',alpha=.2)
        for m in MAIN:
            axes[1].plot([.3,.4,.5,.6,.7],[aggregate.loc[m,f'map_at_{t:.1f}'] for t in (.3,.4,.5,.6,.7)],marker='o',color=COLORS[m],label=LABELS[m])
        axes[1].set(xlabel='Temporal IoU threshold',ylabel='mAP',ylim=(0,1),title='Localization strictness');axes[1].grid(alpha=.2)
        handles,labels=axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.06),ncol=5,frameon=False)
        save(fig,'01_main_comparison','18 subjects x seeds 41/47/53. Subject metrics average seeds then subjects; pooled F1 concatenates subjects within each seed.',bottom=.16)

        fig,axes=plt.subplots(3,4,figsize=(15,10))
        for ax,key in zip(axes.flat,METRICS):
            ax.bar(range(5),aggregate.loc[MAIN,key],color=[COLORS[m] for m in MAIN])
            ax.set(title=METRIC_LABELS[key],ylim=(0,1));ax.set_xticks(range(5),['Video','Concat','Fixed','Margin','DWA'],rotation=25)
            for i,m in enumerate(MAIN):ax.text(i,aggregate.loc[m,key]+.012,f'{aggregate.loc[m,key]:.3f}',ha='center',fontsize=7)
            ax.grid(axis='y',alpha=.15)
        save(fig,'02_all_metrics','All values are fractions on the 2 Hz feature grid. Background class 18; action F1 averages classes 0-17. TAL uses frame-derived segments.')

        fig,axes=plt.subplots(2,1,figsize=(14,7.3),sharex=True)
        for ax,key in zip(axes,['macro_f1_19','map_at_0.5']):
            full=subjects[subjects.method==METHOD].set_index('fold').sort_index()
            for i,m in enumerate(BASELINES):
                base=subjects[subjects.method==m].set_index('fold').sort_index()
                ax.bar(np.arange(18)+(i-1.5)*.19,full[key]-base[key],.19,label=LABELS[m],color=COLORS[m])
            ax.axhline(0,color='black',lw=.8);ax.set_ylabel('DWA - baseline');ax.set_title(METRIC_LABELS[key]);ax.grid(axis='y',alpha=.2)
        handles,labels=axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,ncol=4,loc='lower center',bbox_to_anchor=(.5,.05),frameon=False)
        axes[1].set_xticks(range(18),[f'sbj_{i}' for i in range(18)],rotation=45)
        save(fig,'03_subject_differences','Each pair uses the same held-out subject and seeds. All 18 subjects are shown, including negative differences.',bottom=.13)

        fig,axes=plt.subplots(1,2,figsize=(14.5,6))
        for ax,family in zip(axes,['baseline','retrained_components']):
            q=comparisons[comparisons.family==family].copy().reset_index(drop=True)
            for i,r in q.iterrows():
                color='#C84452' if r['metric']=='macro_f1_19' else '#007F85'
                ax.errorbar(r.mean_difference,i,xerr=[[r.mean_difference-r.ci95_low],[r.ci95_high-r.mean_difference]],fmt='o',color=color,capsize=3)
            ax.axvline(0,color='black',lw=.8)
            labels=[f"{LABELS[r['comparator']]} / {'F1' if r['metric']=='macro_f1_19' else 'mAP'}\nHolm p={r['p_holm_family']:.3g}" for _,r in q.iterrows()]
            ax.set_yticks(range(len(q)),labels);ax.invert_yaxis();ax.set_xlabel('Mean paired difference: full DWA - comparator')
            ax.set_title('Baseline comparisons' if family=='baseline' else 'Retrained component comparisons');ax.grid(axis='x',alpha=.2)
        save(fig,'04_paired_statistics','95% unadjusted paired t intervals; Holm p within each 8-test family. Unit = subject after seed averaging. Exploratory: LOSO training sets overlap.')

        fig,axes=plt.subplots(1,2,figsize=(14,5.8));abls=list(VARIANTS)
        for ax,key in zip(axes,['macro_f1_19','map_at_0.5']):
            means=aggregate.loc[abls,key].to_numpy();sd=aggregate.loc[abls,key+'_subject_sd'].to_numpy()
            ax.barh(range(5),means,color=['#C84452',*['#779BA5']*4]);ax.errorbar(means,range(5),xerr=sd,fmt='none',color='#333333',capsize=3)
            ax.set_yticks(range(5),[LABELS[m] for m in abls]);ax.invert_yaxis();ax.set_xlim(0,max(1,float((means+sd).max())+.02))
            ax.set_title(METRIC_LABELS[key]);ax.grid(axis='x',alpha=.2)
        save(fig,'05_retrained_ablations','Each of the five models is independently trained for 30 parent / 15 probe epochs, 18 folds x 3 seeds. Error bars: SD across 18 subject means, not CI.')

        fig,axes=plt.subplots(1,3,figsize=(15,5.3))
        methods=['ROUND0','PARENT_NO_PROBE',METHOD];x=np.arange(3)
        for ax,key in zip(axes[:2],['macro_f1_19','map_at_0.5']):
            ax.bar(x,aggregate.loc[methods,key],color=['#94AAB2','#568E9C','#C84452']);ax.set_ylim(0,1)
            ax.set_xticks(x,['Round 0','Parent\n(no probe)','Full']);ax.set_title(METRIC_LABELS[key]);ax.grid(axis='y',alpha=.2)
        grouped=jitter.groupby(['controller','jitter_seconds']).context_l2.mean()
        for i,c in enumerate(['Seed only','Sensor resize']):
            axes[2].bar(np.arange(2)+(i-.5)*.32,[grouped.loc[(c,t)] for t in (.5,1)],.32,label=c,color=['#94AAB2','#C84452'][i])
        axes[2].set_xticks([0,1],['+/-0.5 s','+/-1.0 s']);axes[2].set_ylabel('Mean context L2 change');axes[2].set_title('Boundary perturbation')
        axes[2].legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,-.13),ncol=2)
        save(fig,'06_frozen_parent_diagnostics','Inference diagnostics on trained full-model weights, not retrained ablations. Jitter controllers share the same weights and Round-0 seeds; lower L2 means less change.',bottom=.18)

        fig,axes=plt.subplots(1,3,figsize=(15,5.1));full=windows[windows.method==METHOD]
        grouped=full.groupby('round')[['shrink_fraction','expand_fraction','same_length_fraction']].mean()
        bottom=np.zeros(2)
        for key,label,color in [('shrink_fraction','Shrink','#4C78A8'),('expand_fraction','Expand','#F28E2B'),('same_length_fraction','Same length','#AAAAAA')]:
            values=grouped[key].to_numpy();axes[0].bar([0,1],values,bottom=bottom,label=label,color=color);bottom+=values
        axes[0].set_xticks([0,1],['Round 0','Round 1']);axes[0].set_ylim(0,1);axes[0].set_title('Sensor-driven changes')
        axes[0].legend(loc='upper center',bbox_to_anchor=(.5,-.13),ncol=3,frameon=False)
        for i,r in enumerate((0,1)):
            q=full[full['round']==r].groupby('fold')[['initial_mean_tokens','selected_mean_tokens']].mean()
            axes[1].scatter(q.initial_mean_tokens,q.selected_mean_tokens,label=f'Round {r}',s=24,alpha=.8)
        limit=max(full.initial_mean_tokens.max(),full.selected_mean_tokens.max())*1.04
        axes[1].plot([0,limit],[0,limit],color='gray',ls='--');axes[1].set(xlabel='Initial mean tokens',ylabel='Selected mean tokens',title='Per-subject support sizes');axes[1].legend()
        for r in (0,1):
            q=full[full['round']==r].groupby('fold').stable_fraction.mean().sort_index()
            axes[2].plot(range(18),q,marker='o',ms=3,label=f'Round {r}')
        axes[2].set(ylim=(0,1),ylabel='Statistics-stable fraction',title='Stopping condition reached');axes[2].set_xticks(range(0,18,2),[f's{i}' for i in range(0,18,2)]);axes[2].legend()
        save(fig,'07_window_behavior','Fractions average subjects/seeds equally. Same length may still have shifted endpoints. A budget- or boundary-limited window is not automatically statistically stable.',bottom=.18)

        palette=ListedColormap(plt.get_cmap('tab20')(np.linspace(0,1,19)))
        fig,axes=plt.subplots(2,1,figsize=(15,7))
        for ax,subject in zip(axes,[1,9]):
            data=pd.read_csv(source/f'timeline_sbj_{subject}_seed47.csv');order=['GT',*MAIN]
            matrix=np.stack([data[data.track==m].sort_values('time_seconds').label.to_numpy() for m in order])
            tmax=data.time_seconds.max();ax.imshow(matrix,aspect='auto',interpolation='nearest',cmap=palette,vmin=0,vmax=18,extent=(.5,tmax,5.5,-.5))
            ax.set_yticks(range(6),['Ground truth',*[LABELS[m] for m in MAIN]]);ax.set_xlabel('Time (seconds)');ax.set_title(f'sbj_{subject}, seed 47')
            for y in np.arange(.5,5.5,1):ax.axhline(y,color='white',lw=1)
        save(fig,'08_temporal_examples','Same subjects/seeds as the previous figures; not selected by new scores. Class IDs 0-18 have consistent colors; class 18 is background.')

        fig,axes=plt.subplots(1,2,figsize=(15,5),gridspec_kw={'width_ratios':[1.7,1]})
        values=classes[classes.method.isin(MAIN)].groupby(['method','class_id']).f1.mean().unstack().loc[MAIN]
        im=axes[0].imshow(values,aspect='auto',vmin=0,vmax=1,cmap='viridis');axes[0].set_xticks(range(19),[str(x) for x in range(19)])
        axes[0].set_yticks(range(5),[LABELS[m] for m in MAIN]);axes[0].set_xlabel('Class ID (18 = background)');axes[0].set_title('Per-class F1 (subject/seed mean)');fig.colorbar(im,ax=axes[0],fraction=.035,pad=.025)
        for m in MAIN:
            q=seeds[seeds.method==m].sort_values('seed');axes[1].plot(q.seed,q.macro_f1_19,marker='o',color=COLORS[m],label=LABELS[m])
        axes[1].set_xticks([41,47,53]);axes[1].set(ylim=(0,1),xlabel='Seed',ylabel='Mean-subject Macro-F1',title='Seed variability');axes[1].legend(loc='lower right');axes[1].grid(alpha=.2)
        save(fig,'09_classes_and_seeds','Per-class frame scores use all 19 labels and zero_division=0. Seed scores average all 18 subjects. All three predeclared seeds are included.')
    hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [pdf_path,*sorted((root/'figures').glob('*.png'))]}
    (root/'figure_outputs.json').write_text(json.dumps({'pages':len(names),'figure_names':names,'sha256':hashes},indent=2)+'\n')
    review=json.loads((root/'review_conclusions.json').read_text());review['figures_pending']=False
    review['pdf_pages']=len(names);review['visual_review']='Pending final human/agent inspection of rendered PDF pages'
    (root/'review_conclusions.json').write_text(json.dumps(review,indent=2)+'\n')
    renderer=shutil.which('pdftoppm')
    if renderer:
        qa=root/'pdf_render_check';qa.mkdir(exist_ok=True)
        subprocess.run([renderer,'-r','80','-png',str(pdf_path),str(qa/'page')],check=True)
        assert len(list(qa.glob('page-*.png')))==len(names)
    print(f'Generated {len(names)} figures and {pdf_path}',flush=True)


if __name__ == '__main__':
    main()
