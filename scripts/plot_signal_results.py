#!/usr/bin/env python3
"""Redraw the six sensor-driven WEAR figures from hash-verified release tables."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd

REPO=Path(__file__).resolve().parents[1]
METHODS=['VIDEO_ONLY','EARLY_CONCAT','FIXED_WINDOW_ATTENTION','FINAL_MODEL','SIGNAL_ADAPTIVE_DWA']
DWA=METHODS[-1]
LABELS=dict(zip(METHODS,['Video-only','Early concat.','Fixed attention','Segment-guided attention (SWA)','Sensor-driven DWA']))
COLORS=dict(zip(METHODS,['#4C78A8','#F58518','#54A24B','#9970AB','#D1495B']))
NAMES=['wear_main_results','wear_ablation_robustness','wear_temporal_segmentation','wear_gt_vs_final','wear_all_metrics','wear_class_and_seed']


def verify_source(source):
    manifest=json.loads((source/'manifest.json').read_text())
    if manifest['main_folds'] != list(range(1,19)) or manifest['ablation_folds'] != list(range(1,19,2)) or manifest['seeds'] != [41,47,53]:
        raise ValueError('Unexpected release cohorts')
    expected={'aggregate_metrics.csv','subject_means.csv','ablation_aggregate_metrics.csv','window_behavior.csv',
              'timeline_sbj_1_seed47.csv','timeline_sbj_9_seed47.csv','per_class_metrics.csv','seed_means.csv'}
    if not expected.issubset(manifest['sha256']):
        raise ValueError('Incomplete figure manifest')
    for name in expected:
        if hashlib.sha256((source/name).read_bytes()).hexdigest() != manifest['sha256'][name]:
            raise ValueError(f'Figure source hash mismatch: {name}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,default=REPO/'results/wear_signal_v3')
    parser.add_argument('--output',type=Path,default=REPO/'outputs/wear_signal_figures')
    args=parser.parse_args();source=args.data_root;out=args.output
    verify_source(source)
    (out/'figures').mkdir(parents=True,exist_ok=True);(out/'pdfs').mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'serif','font.size':9,'axes.titlesize':10,'axes.labelsize':9,
                         'legend.fontsize':7.5,'xtick.labelsize':8,'ytick.labelsize':8,
                         'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
    aggregate=pd.read_csv(source/'aggregate_metrics.csv').set_index('method')
    subjects=pd.read_csv(source/'subject_means.csv')
    pdfpath=out/'WEAR_sensor_DWA_results.pdf'
    with PdfPages(pdfpath) as pdf:
        def save(fig,name):
            fig.savefig(out/'figures'/f'{name}.png',dpi=300,bbox_inches='tight')
            fig.savefig(out/'pdfs'/f'{name}.pdf',bbox_inches='tight')
            pdf.savefig(fig,bbox_inches='tight');plt.close(fig)

        fig,axes=plt.subplots(1,3,figsize=(15,4.6),gridspec_kw={'width_ratios':[1.25,1.5,1]})
        metrics=['macro_f1_19','concat_macro_f1','map_at_0.5','avg_map'];x=np.arange(4)
        for i,m in enumerate(METHODS):
            axes[0].bar(x+(i-2)*.16,aggregate.loc[m,metrics].to_numpy(float),.16,color=COLORS[m],label=LABELS[m])
        axes[0].set_xticks(x,['Subject\nMacro-F1','Concat.\nMacro-F1','mAP@0.5','Avg. mAP'])
        axes[0].set_ylim(.5,.86);axes[0].set_ylabel('Score');axes[0].set_title('(a) Aggregate performance',fontweight='bold')
        full=subjects[subjects.method==DWA].sort_values('fold')
        for i,m in enumerate(METHODS[:-1]):
            base=subjects[subjects.method==m].sort_values('fold')
            if not np.array_equal(base.fold,full.fold):raise ValueError('Unpaired subject scores')
            axes[1].bar(np.arange(18)+(i-1.5)*.21,100*(full['map_at_0.5'].to_numpy()-base['map_at_0.5'].to_numpy()),.2,color=COLORS[m])
        axes[1].axhline(0,color='black',lw=.8);axes[1].set_xticks(np.arange(18),[f's{i}' for i in range(18)],rotation=90)
        axes[1].set_ylabel('DWA - baseline: mAP@0.5 (pp)');axes[1].set_title('(b) Cross-subject differences',fontweight='bold')
        thresholds=[.3,.4,.5,.6,.7]
        for m in METHODS:axes[2].plot(thresholds,[aggregate.loc[m,f'map_at_{t:.1f}'] for t in thresholds],marker='o',lw=1.8,color=COLORS[m])
        axes[2].set_xticks(thresholds);axes[2].set_xlabel('Temporal IoU threshold');axes[2].set_ylabel('mAP')
        axes[2].set_title('(c) Localization strictness',fontweight='bold')
        for ax in axes:ax.grid(axis='y',alpha=.22);ax.set_axisbelow(True)
        handles,labels=axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='lower center',ncol=5,frameon=False,bbox_to_anchor=(.5,.045))
        fig.text(.5,.01,'18 LOSO subjects, seeds 41/47/53 | Subject metrics average seeds within subject; concatenated F1 pools subjects per seed.',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.15,1,1));save(fig,NAMES[0])

        fig,axes=plt.subplots(1,3,figsize=(14,4.7))
        ablation=pd.read_csv(source/'ablation_aggregate_metrics.csv').set_index('method')
        controls=['NO_SENSOR_RESIZE','NO_CONTRACTION',DWA]
        for i,(metric,color,label) in enumerate([('macro_f1_19','#4C78A8','Macro-F1'),('map_at_0.5','#D1495B','mAP@0.5')]):
            bars=axes[0].bar(np.arange(3)+(i-.5)*.34,ablation.loc[controls,metric],.34,color=color,label=label)
            axes[0].bar_label(bars,fmt='%.3f',fontsize=7,padding=3)
        axes[0].set_xticks(range(3),['No resize','No contraction','Full DWA'],rotation=12)
        axes[0].set_ylim(.65,.82);axes[0].legend(frameon=False,loc='upper left',ncol=2)
        axes[0].set_title('(a) Retrained controls | 9 subjects',fontweight='bold');axes[0].set_ylabel('Subject-mean score')
        outputs=['ROUND0','PARENT_NO_PROBE',DWA]
        for metric,color,label in [('macro_f1_19','#4C78A8','Macro-F1'),('map_at_0.5','#D1495B','mAP@0.5')]:
            y=aggregate.loc[outputs,metric].to_numpy();axes[1].plot(range(3),y,marker='o',color=color,label=label)
            for i,value in enumerate(y):axes[1].annotate(f'{value:.4f}',(i,value),xytext=(0,7),textcoords='offset points',ha='center',fontsize=8)
        axes[1].set_xticks(range(3),['Round 0','Round 1','+ Probe']);axes[1].set_ylim(.70,.775)
        axes[1].legend(frameon=False,loc='upper left',ncol=2)
        axes[1].set_title('(b) Frozen outputs | 18 subjects',fontweight='bold');axes[1].set_ylabel('Subject-mean score')
        window=pd.read_csv(source/'window_behavior.csv');q=window[window.method==DWA].groupby('round').mean(numeric_only=True)
        bottom=np.zeros(2)
        for key,label,color in [('shrink_fraction','Shorter','#4C78A8'),('same_length_fraction','Same length','#BAB0AC'),('expand_fraction','Longer','#D1495B')]:
            values=q[key].to_numpy();axes[2].bar([0,1],values,bottom=bottom,color=color,label=label,width=.55)
            for i,value in enumerate(values):axes[2].text(i,bottom[i]+value/2,f'{100*value:.1f}%',ha='center',va='center',fontsize=9,color='white' if color!='#BAB0AC' else 'black')
            bottom+=values
        axes[2].set_xticks([0,1],['Round 0','Round 1']);axes[2].set_ylim(0,1.19);axes[2].set_ylabel('Fraction of query windows')
        axes[2].set_title('(c) Sensor-driven window resizing',fontweight='bold');axes[2].legend(frameon=False,loc='upper center',ncol=3)
        for ax in axes:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        fig.text(.5,.015,'(a) Same 9 subjects (sbj_0,2,...,16), 3 seeds, independently trained controls. (b,c) All 18 subjects, 3 seeds; (b) shares full-model weights.',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.085,1,1));save(fig,NAMES[1])

        data=pd.read_csv(source/'timeline_sbj_1_seed47.csv');tracks=['GT',*METHODS]
        array=np.stack([data[data.track==m].sort_values('time_seconds').label.to_numpy(int) for m in tracks])
        palette=plt.get_cmap('tab20')(np.linspace(0,1,19))
        fig,ax=plt.subplots(figsize=(12,3.2))
        ax.imshow(array,aspect='auto',interpolation='nearest',cmap=ListedColormap(palette),vmin=0,vmax=18)
        ax.set_yticks(range(6),['Ground truth',*[LABELS[m] for m in METHODS]])
        ax.set_xticks(np.linspace(0,array.shape[1]-1,7),[f'{x:.0f}' for x in np.linspace(.5,data.time_seconds.max(),7)])
        for y in np.arange(.5,6,1):ax.axhline(y,color='white',lw=1.2)
        ax.set_xlabel('Time (s)');ax.set_title('WEAR temporal segmentation | sbj_1, seed 47',fontweight='bold')
        fig.text(.5,.01,'Same subject, seed and 19-class palette as the previous release; a sensor-driven DWA track is added.',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.07,1,1));save(fig,NAMES[2])

        data=pd.read_csv(source/'timeline_sbj_9_seed47.csv')
        fig,ax=plt.subplots(figsize=(9,4.8))
        for m,label,color in [('GT','GT','#4C78A8'),('FINAL_MODEL','Segment-guided attention (SWA)','#9970AB'),(DWA,'Sensor-driven DWA','#D1495B')]:
            q=data[data.track==m];ax.plot(q.time_seconds,q.label,label=label,color=color,lw=1,alpha=.85)
        ax.set_title('WEAR sbj_9 (fold 10), seed 47 | GT and model predictions',fontweight='bold')
        ax.set_xlabel('Time (s)');ax.set_ylabel('Class ID');ax.set_yticks([0,3,6,9,12,15,18]);fig.legend(*ax.get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.045),ncol=3,frameon=False)
        fig.text(.5,.01,'The original subject and seed are retained; class 18 is background.',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.14,1,1));save(fig,NAMES[3])

        grid=[('accuracy','Accuracy'),('macro_precision_19','Macro precision'),('macro_recall_19','Macro recall'),('macro_f1_19','Macro-F1'),
              ('background_f1','Background F1'),('action_macro_f1','Action Macro-F1'),('map_at_0.3','mAP@0.3'),('map_at_0.4','mAP@0.4'),
              ('map_at_0.5','mAP@0.5'),('map_at_0.6','mAP@0.6'),('map_at_0.7','mAP@0.7'),('avg_map','Avg. mAP')]
        fig,axes=plt.subplots(3,4,figsize=(13.5,8.6))
        for ax,(metric,title) in zip(axes.flat,grid):
            bars=ax.bar(np.arange(5),aggregate.loc[METHODS,metric],color=[COLORS[m] for m in METHODS],width=.7)
            ax.errorbar(np.arange(5),aggregate.loc[METHODS,metric],yerr=aggregate.loc[METHODS,metric+'_subject_sd'],fmt='none',ecolor='#555555',elinewidth=.8,capsize=2)
            ax.bar_label(bars,fmt='%.3f',padding=2,fontsize=7,bbox={'facecolor':'white','edgecolor':'none','pad':.3})
            ax.set_title(title);ax.set_ylim(0,1.13);ax.set_xticks([]);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        fig.legend(handles,labels,loc='lower center',ncol=5,frameon=False,bbox_to_anchor=(.5,.028))
        fig.suptitle('WEAR complete metric comparison | 18 subjects, 3 seeds',fontweight='bold',y=.995)
        fig.text(.5,.01,'Bars: subject means after averaging seeds. Error bars: sample SD across 18 subject means. All methods use identical metric definitions.',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.08,1,.975));save(fig,NAMES[4])

        fig,axes=plt.subplots(1,2,figsize=(13,4.7),gridspec_kw={'width_ratios':[1.6,1]})
        classes=pd.read_csv(source/'per_class_metrics.csv')
        for m in METHODS:
            q=classes[classes.method==m].groupby('class_id').f1.mean()
            axes[0].plot(q.index,q.values,marker='.',color=COLORS[m],lw=1.4)
        axes[0].set_xticks(range(19),[str(x) if x<18 else 'BG' for x in range(19)])
        axes[0].set_xlabel('Class ID');axes[0].set_ylabel('Mean class F1');axes[0].set_ylim(0,1)
        axes[0].set_title('(a) Per-class F1 | all subjects and seeds',fontweight='bold')
        seeds=pd.read_csv(source/'seed_means.csv')
        for m in METHODS:
            q=seeds[seeds.method==m].sort_values('seed');axes[1].plot(q.seed,q.macro_f1_19,marker='o',color=COLORS[m])
        axes[1].set_xticks([41,47,53]);axes[1].set_xlabel('Training seed');axes[1].set_ylabel('Subject-mean Macro-F1')
        axes[1].set_title('(b) Seed-level comparison',fontweight='bold')
        for ax in axes:ax.grid(alpha=.2)
        fig.legend(handles,labels,loc='lower center',ncol=5,frameon=False,bbox_to_anchor=(.5,.01))
        fig.tight_layout(rect=(0,.09,1,1));save(fig,NAMES[5])
    hashes={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [pdfpath,*sorted((out/'figures').glob('*.png'))]}
    (out/'figure_outputs.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print(f'Generated six figures and {pdfpath}')


if __name__=='__main__':
    main()
