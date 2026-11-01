"""Export original TGIF trained weights with their paired preprocessing statistics.

Run inside the laboratory environment once. Recipients use download_models.py.
No raw video, sensor CSV, annotations, or ROI coordinates enter the bundle.
"""
import argparse,json,random
from pathlib import Path
import numpy as np
import torch
from pipeline.common import write_json,sha256,ENCODER_ID,ENCODER_REVISION
from pipeline.bundle import SCHEMA
from pipeline.models.vibration_normalizer import RobustVibrationNormalizer
from pipeline.models.dynamic_window_manager import build_fixed_windows
from pipeline.models.dynamic_window_multimodal_mstcn import DynamicWindowMultimodalMSTCN
from pipeline.inference import window_features

VIDEOS=['station1','station2','station2_clip2','station3','station3_clip2','station4','station1_0_30','station1_90_120','station2_40_70','station3_40_70','station4_40_70']
VIB={'station1','station1_0_30','station2','station3','station3_clip2','station4'}
GOOD={'station2','station3','station4'}

def records(root):
    result=[]
    for station in VIDEOS:
        labels=np.load(root/'videomae_pipeline/outputs/extracted_features'/station/'labels.npy',allow_pickle=False)
        for i in range(len(labels)//2247):
            y=labels[i*2247:(i+1)*2247]
            result.append(dict(sequence_id=f'{station}_seg{i:02d}',station_id=station,label=y,
                               has_vib=station in VIB,start=i*2247,stop=(i+1)*2247))
    return result

def split(records,seed):
    def pick(indices,k,s):
        rng=random.Random(s)
        ranked=sorted(indices,key=lambda i:float((records[i]['label']==0).sum())/max(int((records[i]['label']>=0).sum()),1))
        bucket=len(ranked)/k
        return [rng.choice(ranked[int(b*bucket):int((b+1)*bucket)]) for b in range(k)]
    val=set(pick([i for i,s in enumerate(records) if s['has_vib']],6,seed*100003))
    val|=set(pick([i for i,s in enumerate(records) if not s['has_vib']],6,seed*100003+1))
    return [s for i,s in enumerate(records) if i not in val],[s for i,s in enumerate(records) if i in val]

def export(root,output,seeds):
    root=Path(root);output=Path(output);seqs=records(root)
    with np.load(root/'multimodal/video_prior_fusion/data_store/norm.npz',allow_pickle=False) as n:
        mean=n['mean'].astype(np.float32).reshape(-1);std=n['std'].astype(np.float32).reshape(-1)
    for seed in seeds:
        train,val=split(seqs,seed);xs=[];stations=[]
        for s in train:
            if s['station_id'] not in GOOD or not s['has_vib']:continue
            v=np.load(root/'multimodal/video_prior_fusion/data_store/vibration_features'/f"{s['station_id']}_vib.npy",allow_pickle=False)[s['start']:s['stop']]
            windows=build_fixed_windows(len(v),7.4933,6.,3.)
            xs.append(window_features(v,windows));stations.extend([s['station_id']]*len(windows))
        norm=RobustVibrationNormalizer().fit(np.concatenate(xs),np.array(stations))
        source=root/'multimodal/dynamic_window_fusion/outputs/checkpoints'/f'attn_masked_seed{seed}.pt'
        state=torch.load(source,map_location='cpu',weights_only=True)
        model=DynamicWindowMultimodalMSTCN();model.load_state_dict(state['model'],strict=True)
        dest=output/f'seed_{seed}';dest.mkdir(parents=True,exist_ok=True)
        torch.save(dict(model=state['model'],seed=seed,schema=SCHEMA),dest/'model.pt')
        keys=sorted(norm.stats)
        np.savez(dest/'normalization.npz',video_mean=mean,video_std=std,
                 vibration_median=np.stack([norm.stats[k][0] for k in keys]),
                 vibration_iqr=np.stack([norm.stats[k][1] for k in keys]))
        meta=dict(schema=SCHEMA,split_seed=seed,source_checkpoint_sha256=sha256(source),
                  model_config=dict(in_dim=1536,video_dim=256,token_dim=64,machine_dim=32,
                    attention_dim=64,num_heads=2,neighbors=1,dropout=.10,n_stages=4,n_layers=8,ch=64,n_classes=3),
                  segment_config=dict(smoothing_seconds=1.,min_segment_seconds=2.,max_segment_seconds=20.),
                  eligible_vibration_stations=sorted(GOOD),vibration_stations=keys,station_aliases={},
                  train_sequence_ids=[s['sequence_id'] for s in train],validation_sequence_ids=[s['sequence_id'] for s in val],
                  video_normalization_source='Frozen original VideoMAE two-ROI normalization file; not refitted at inference',
                  vibration_normalization_source='Median/IQR reconstructed on this historical training split only',
                  video_encoder=ENCODER_ID,video_encoder_revision=ENCODER_REVISION,video_encoder_qkv_bias=False,
                  protocol='Offline centered clips; adaptive predicted-segment tokens; label-free inference; sensor validity mask',
                  labels=dict(idle=0,sewing=1,handling=2))
        write_json(dest/'metadata.json',meta)
        write_json(dest/'sha256.json',{name:sha256(dest/name) for name in ['metadata.json','normalization.npz','model.pt']})
        print(dest)
    return output

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--legacy-root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--seeds',nargs='+',type=int,default=[43]);a=p.parse_args();export(a.legacy_root,a.output,a.seeds)
