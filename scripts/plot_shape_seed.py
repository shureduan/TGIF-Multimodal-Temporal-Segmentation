#!/usr/bin/env python3
"""Create the one-seed pilot comparison figures from verified metric tables."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from tgif_dwa.shape_wear import METHOD
from run_shape_seed import REFERENCES,sha

LABELS={'VIDEO_ONLY':'Video-only','EARLY_CONCAT':'Early concat.','FIXED_WINDOW_ATTENTION':'Fixed attention',
        'FINAL_MODEL':'SWA','SIGNAL_ADAPTIVE_DWA':'Current DWA',METHOD:'Envelope-shape DWA'}
METHODS=[*REFERENCES,METHOD]
COLORS=dict(zip(METHODS,['#4C78A8','#F58518','#54A24B','#9970AB','#D1495B','#17A398']))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--benchmark',type=Path,required=True)
    root=parser.parse_args().benchmark.resolve();source=root/'source_data'
    manifest=json.loads((source/'manifest.json').read_text())
    for name,digest in manifest['sha256'].items():
        if sha(source/name)!=digest:raise ValueError('Numerical source changed: '+name)
    (root/'figures').mkdir(exist_ok=True);(root/'pdfs').mkdir(exist_ok=True)
    seed=manifest['seed'];aggregate=pd.read_csv(source/'aggregate_metrics.csv').set_index('method')
    subjects=pd.read_csv(source/'per_subject_metrics.csv')
    plt.rcParams.update({'font.family':'serif','font.size':9,'axes.titlesize':10,'axes.spines.top':False,
                         'axes.spines.right':False,'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42})
    paths=[];pdf_path=root/f'WEAR_envelope_shape_seed{seed}.pdf'
    with PdfPages(pdf_path) as pdf:
        def save(fig,name):
            path=root/'figures'/f'{name}.png';fig.savefig(path,dpi=250,bbox_inches='tight')
            fig.savefig(root/'pdfs'/f'{name}.pdf',bbox_inches='tight');pdf.savefig(fig,bbox_inches='tight')
            plt.close(fig);paths.append(path)
        fig,axes=plt.subplots(1,3,figsize=(15,4.6),gridspec_kw={'width_ratios':[1.2,1.5,1]})
        metrics=['macro_f1_19','map_at_0.5','avg_map'];x=np.arange(3)
        for i,m in enumerate(METHODS):
            axes[0].bar(x+(i-2.5)*.135,aggregate.loc[m,metrics].to_numpy(float),.13,label=LABELS[m],color=COLORS[m])
        axes[0].set_xticks(x,['Subject Macro-F1','mAP@0.5','Avg. mAP']);axes[0].set_ylim(0,1)
        axes[0].set_title('(a) Complete 18-subject comparison',fontweight='bold');axes[0].set_ylabel('Subject mean')
        full=subjects[subjects.method==METHOD].sort_values('fold')
        for m,offset in [('SIGNAL_ADAPTIVE_DWA',-.17),('FINAL_MODEL',.17)]:
            base=subjects[subjects.method==m].sort_values('fold')
            if not np.array_equal(base.fold,full.fold):raise ValueError('Unpaired subjects')
            axes[1].bar(np.arange(18)+offset,100*(full.macro_f1_19.to_numpy()-base.macro_f1_19.to_numpy()),.32,color=COLORS[m],label='vs '+LABELS[m])
        axes[1].axhline(0,color='black',lw=.8);axes[1].set_xticks(np.arange(18),[f's{i}' for i in range(18)],rotation=90)
        axes[1].set_ylabel('Shape DWA - reference: Macro-F1 (pp)');axes[1].set_title('(b) Paired subject differences',fontweight='bold')
        axes[1].legend(frameon=False)
        thresholds=[.3,.4,.5,.6,.7]
        for m in METHODS:axes[2].plot(thresholds,[aggregate.loc[m,f'map_at_{t:.1f}'] for t in thresholds],marker='o',color=COLORS[m])
        axes[2].set_xticks(thresholds);axes[2].set_xlabel('Temporal IoU');axes[2].set_ylabel('mAP')
        axes[2].set_title('(c) Localization thresholds',fontweight='bold')
        for ax in axes:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        handles,labels=axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.035),ncol=6,frameon=False)
        fig.text(.5,.005,f'18 LOSO subjects | seed {seed} | fixed parent/probe 30/15 epochs | all rows use the same seed and cohort.',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.14,1,1));save(fig,'01_main_comparison')

        grid=[('accuracy','Accuracy'),('macro_precision_19','Macro precision'),('macro_recall_19','Macro recall'),('macro_f1_19','Macro-F1'),
              ('background_f1','Background F1'),('action_macro_f1','Action Macro-F1'),('map_at_0.3','mAP@0.3'),('map_at_0.4','mAP@0.4'),
              ('map_at_0.5','mAP@0.5'),('map_at_0.6','mAP@0.6'),('map_at_0.7','mAP@0.7'),('avg_map','Avg. mAP')]
        fig,axes=plt.subplots(3,4,figsize=(14,8.7))
        for ax,(metric,title) in zip(axes.flat,grid):
            bars=ax.bar(range(6),aggregate.loc[METHODS,metric],color=[COLORS[m] for m in METHODS],width=.7)
            ax.errorbar(range(6),aggregate.loc[METHODS,metric],yerr=aggregate.loc[METHODS,metric+'_subject_sd'],fmt='none',ecolor='#555',elinewidth=.8,capsize=2)
            ax.bar_label(bars,fmt='%.3f',fontsize=6.5,padding=2,bbox={'facecolor':'white','edgecolor':'none','pad':.2})
            ax.set_ylim(0,1.13);ax.set_xticks([]);ax.set_title(title);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.025),ncol=6,frameon=False)
        fig.suptitle(f'All metrics | 18 subjects, seed {seed}',fontweight='bold',y=.997)
        fig.text(.5,.005,'Bars: equal subject means. Error bars: sample SD across 18 subjects; no seed averaging in this pilot.',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.08,1,.975));save(fig,'02_all_metrics')

        window=pd.read_csv(source/'window_behavior.csv');q=window.groupby(['method','round']).mean(numeric_only=True)
        fig,axes=plt.subplots(1,3,figsize=(14,4.5))
        for i,m in enumerate(['SIGNAL_ADAPTIVE_DWA',METHOD]):
            axes[0].bar(np.arange(2)+(i-.5)*.32,q.loc[m,'mean_selected_tokens'].to_numpy(),.32,color=COLORS[m],label=LABELS[m])
        axes[0].set_xticks([0,1],['Round 0','Round 1']);axes[0].set_ylabel('Mean selected tokens');axes[0].legend(frameon=False)
        axes[0].set_ylim(0,max(1.,float(q['mean_selected_tokens'].max()))*1.3)
        axes[0].set_title('(a) Selected support length',fontweight='bold')
        bottom=np.zeros(4);keys=[('SIGNAL_ADAPTIVE_DWA',0),('SIGNAL_ADAPTIVE_DWA',1),(METHOD,0),(METHOD,1)]
        for name,label,color in [('shrink_fraction','Shorter','#4C78A8'),('same_length_fraction','Same length','#BAB0AC'),('expand_fraction','Longer','#D1495B')]:
            values=q.loc[keys,name].to_numpy();axes[1].bar(range(4),values,bottom=bottom,color=color,label=label)
            for i,value in enumerate(values):
                if value>.07:axes[1].text(i,bottom[i]+value/2,f'{100*value:.0f}%',ha='center',va='center',fontsize=8)
            bottom+=values
        axes[1].set_xticks(range(4),['Current\nR0','Current\nR1','Shape\nR0','Shape\nR1']);axes[1].set_ylim(0,1.17)
        axes[1].legend(frameon=False,ncol=3,loc='upper center');axes[1].set_title('(b) Change relative to seed support',fontweight='bold')
        outputs=['SHAPE_ROUND0','SHAPE_PARENT_NO_PROBE',METHOD]
        for metric,color,label in [('macro_f1_19','#4C78A8','Macro-F1'),('map_at_0.5','#D1495B','mAP@0.5')]:
            y=aggregate.loc[outputs,metric].to_numpy();axes[2].plot(range(3),y,marker='o',color=color,label=label)
            for i,value in enumerate(y):axes[2].annotate(f'{value:.4f}',(i,value),xytext=(0,7),textcoords='offset points',ha='center',fontsize=8)
        axes[2].set_xticks(range(3),['Round 0','Round 1','+ Probe']);axes[2].legend(frameon=False);axes[2].set_ylim(0,1)
        axes[2].set_title('(c) Frozen shape-model outputs',fontweight='bold')
        for ax in axes:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        fig.text(.5,.005,'All 18 subjects, one seed. (c) shares trained weights; it is an output diagnostic, not a retrained ablation.',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.06,1,1));save(fig,'03_windows_and_outputs')

        fig,axes=plt.subplots(2,1,figsize=(14,6.4))
        palette=plt.get_cmap('tab20')(np.linspace(0,1,19))
        for ax,subject in zip(axes,[1,9]):
            data=pd.read_csv(source/f'timeline_sbj_{subject}.csv');tracks=['GT',*METHODS]
            array=np.stack([data[data.track==m].sort_values('time_seconds').label.to_numpy(int) for m in tracks])
            ax.imshow(array,aspect='auto',interpolation='nearest',cmap=ListedColormap(palette),vmin=0,vmax=18)
            ax.set_yticks(range(7),['GT',*[LABELS[m] for m in METHODS]])
            ax.set_xticks(np.linspace(0,array.shape[1]-1,7),[f'{t:.0f}' for t in np.linspace(.5,data.time_seconds.max(),7)])
            for y in np.arange(.5,7,1):ax.axhline(y,color='white',lw=1)
            ax.set_xlabel('Time (s)');ax.set_title(f'sbj_{subject}, seed {seed}',fontweight='bold')
        fig.text(.5,.005,'The two qualitative subjects are retained from the published examples; neither was selected using pilot scores.',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.04,1,1));save(fig,'04_temporal_examples')
    (root/'figure_outputs.json').write_text(json.dumps({str(p.relative_to(root)):sha(p) for p in [pdf_path,*paths]},indent=2)+'\n')
    print('Created four PNGs, four PDFs and',pdf_path,flush=True)

if __name__=='__main__':main()
