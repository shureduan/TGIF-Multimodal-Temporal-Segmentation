"""Frozen, checksummed TGIF weights and preprocessing statistics."""
from pathlib import Path
import json
import numpy as np
from .common import sha256

SCHEMA='tgif_handling_attention_v1'

def load_bundle(folder,device):
    import torch
    from .models.dynamic_window_multimodal_mstcn import DynamicWindowMultimodalMSTCN
    folder=Path(folder)
    meta=json.loads((folder/'metadata.json').read_text())
    hashes=json.loads((folder/'sha256.json').read_text())
    if meta.get('schema')!=SCHEMA: raise ValueError('Unsupported TGIF bundle schema')
    for name in ['metadata.json','normalization.npz','model.pt']:
        if sha256(folder/name)!=hashes.get(name): raise ValueError(f'Bundle checksum mismatch: {name}')
    with np.load(folder/'normalization.npz',allow_pickle=False) as z:
        stats={k:z[k].copy() for k in z.files}
    if stats['video_mean'].shape!=(1536,) or stats['video_std'].shape!=(1536,):
        raise ValueError('Video normalization must have 1536 components')
    if not all(np.isfinite(x).all() for x in stats.values()): raise ValueError('Non-finite normalization')
    if (stats['video_std']<=0).any() or (stats['vibration_iqr']<=0).any():
        raise ValueError('Normalization scales must be positive')
    if stats['vibration_median'].shape!=(len(meta['vibration_stations']),2) or stats['vibration_iqr'].shape!=stats['vibration_median'].shape:
        raise ValueError('Vibration statistics and station IDs do not match')
    model=DynamicWindowMultimodalMSTCN(**meta['model_config']).to(device)
    state=torch.load(folder/'model.pt',map_location='cpu',weights_only=True)
    if int(state['seed'])!=int(meta['split_seed']): raise ValueError('Weights/statistics seed mismatch')
    model.load_state_dict(state['model'],strict=True)
    model.eval()
    return model,stats,meta
