"""Envelope and ordered local shape control for a separately versioned pilot.

A compact PAA/envelope plus coarse slope-histogram descriptor replaces the v3
mean/energy/spectrum controller. It is inspired by PAA and HOG-1D, not an
implementation of the full published HOG-1D classifier. Window decisions have
no video, attention, labels, or test-score input once seed endpoints are given.
"""
from dataclasses import dataclass, asdict
import numpy as np
import torch
from .iterative_dwa import IterativeRawDWA, WindowConfig
from .signal_dwa import SignalAdaptiveDWA, SignalAblationConfig


@dataclass(frozen=True)
class ShapeWindowConfig:
    channels: int = 12
    samples_per_token: int = 50
    stride_seconds: float = .5
    local_radius_steps: int = 2
    temporal_cells: int = 3
    max_expansion_steps: int = 12
    amplitude_weight: float = .75
    similarity_quantile: float = .90
    slope_deadzone_quantile: float = .50
    calibration_rows_per_subject: int = 512

    def __post_init__(self):
        for name in ('channels','samples_per_token','temporal_cells','calibration_rows_per_subject'):
            if type(getattr(self,name)) is not int or getattr(self,name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        for name in ('local_radius_steps','max_expansion_steps'):
            if type(getattr(self,name)) is not int or getattr(self,name) < 0:
                raise ValueError(f'{name} must be a nonnegative integer')
        if self.samples_per_token < 4 or not np.isfinite(self.stride_seconds) or self.stride_seconds <= 0:
            raise ValueError('invalid sampling geometry')
        for name in ('amplitude_weight','similarity_quantile','slope_deadzone_quantile'):
            if not 0 < getattr(self,name) < 1:
                raise ValueError(f'{name} must be between zero and one')


def log_ac_envelope(raw, config):
    """Per-token log(1+AC RMS). Remove each raw channel's local DC level."""
    raw=np.asarray(raw,dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != config.channels*config.samples_per_token:
        raise ValueError('raw sensor array does not match shape geometry')
    if not len(raw) or not np.isfinite(raw).all():
        raise ValueError('shape input must be nonempty and finite after masking')
    x=raw.reshape(-1,config.channels,config.samples_per_token)
    return np.log1p(np.sqrt(np.mean((x-x.mean(-1,keepdims=True))**2,axis=-1)))


def valid_regions(valid):
    n=len(valid);t=np.arange(n)
    cuts=np.r_[True,~valid[:-1] | ~valid[1:]]
    left=np.maximum.accumulate(np.where(cuts,t,0))
    right=np.minimum.accumulate(np.where(np.r_[cuts[1:],True],t,n-1)[::-1])[::-1]
    return left,right


def local_shape_features(levels,valid,deadzone,config):
    """Three ordered temporal cells; mean envelope and 3-bin slope histogram.

    No per-window variance normalization. Levels retain amplitude relative to
    frozen training-fold scales. Missing data bounds each descriptor's support.
    """
    n,channels=levels.shape;t=np.arange(n);region_left,region_right=valid_regions(valid)
    left=np.maximum(region_left,t-config.local_radius_steps)
    right=np.minimum(region_right,t+config.local_radius_steps)
    delta=np.vstack([np.zeros((1,channels)),np.diff(levels,axis=0)])
    delta[np.r_[True,~valid[:-1] | ~valid[1:]]]=0
    code=np.where(delta < -deadzone,0,np.where(delta > deadzone,2,1))
    slopes=np.eye(3,dtype=np.float64)[code]*valid[:,None,None]
    level_prefix=np.vstack([np.zeros((1,channels)),np.cumsum(levels,axis=0)])
    slope_prefix=np.concatenate([np.zeros((1,channels,3)),np.cumsum(slopes,axis=0)])
    cells=[];histograms=[];width=right-left+1
    for cell in range(config.temporal_cells):
        lo=left+(cell*width)//config.temporal_cells
        hi=left+((cell+1)*width)//config.temporal_cells-1
        hi=np.maximum(lo,hi);hi=np.minimum(hi,right);lo=np.minimum(lo,right)
        count=hi-lo+1
        cells.append((level_prefix[hi+1]-level_prefix[lo])/count[:,None])
        histograms.append((slope_prefix[hi+1]-slope_prefix[lo])/count[:,None,None])
    amplitude=np.stack(cells,axis=1).reshape(n,-1)
    histogram=np.stack(histograms,axis=1).reshape(n,-1)
    features=np.concatenate([amplitude*np.sqrt(config.amplitude_weight/amplitude.shape[1]),
                             histogram*np.sqrt((1-config.amplitude_weight)/histogram.shape[1])],axis=1)
    features[~valid]=0
    return features


class EnvelopeShapeDWA(IterativeRawDWA):
    # Share exactly the existing learned Q/K attention, RAW600 values, and the
    # two-round forward. The only replaced component is sensor support control.
    _sensor_attention=SignalAdaptiveDWA._sensor_attention
    fixed_window_attention=SignalAdaptiveDWA.fixed_window_attention
    segment_seed_expand_attention=SignalAdaptiveDWA.segment_seed_expand_attention

    def __init__(self,*,shape_config=ShapeWindowConfig(),**kwargs):
        self.shape_config=shape_config
        self.ablation_config=SignalAblationConfig()
        kwargs.setdefault('imu_dim',shape_config.channels*shape_config.samples_per_token)
        kwargs.setdefault('window',WindowConfig(feature_stride_seconds=shape_config.stride_seconds,max_escape_seconds=0.0))
        super().__init__(**kwargs)
        if self.imu_dim != shape_config.channels*shape_config.samples_per_token:
            raise ValueError('raw dimensions differ from shape config')
        if self.window.feature_stride_seconds != shape_config.stride_seconds:
            raise ValueError('feature stride differs from shape config')
        self.register_buffer('envelope_center',torch.zeros(shape_config.channels))
        self.register_buffer('envelope_scale',torch.ones(shape_config.channels))
        self.register_buffer('slope_deadzone',torch.ones(shape_config.channels))
        self.register_buffer('shape_threshold',torch.ones(()))
        self.register_buffer('shape_calibrated',torch.tensor(False))

    def fit_signal_statistics(self,training_inertial):
        config=self.shape_config
        arrays=[log_ac_envelope(raw,config) for raw in training_inertial]
        if not arrays or any(len(x)<3 for x in arrays):
            raise ValueError('training sequences of at least three tokens are required')
        def sample(x):
            return x[np.linspace(0,len(x)-1,min(len(x),config.calibration_rows_per_subject),dtype=int)]
        pooled=np.concatenate([sample(x) for x in arrays])
        center=np.median(pooled,axis=0)
        scale=np.maximum(np.percentile(pooled,75,axis=0)-np.percentile(pooled,25,axis=0),.05)
        levels=[(x-center)/scale for x in arrays]
        differences=np.concatenate([sample(np.abs(np.diff(x,axis=0))) for x in levels])
        deadzone=np.maximum(np.quantile(differences,config.slope_deadzone_quantile,axis=0),.01)
        distances=[]
        for x in levels:
            features=local_shape_features(x,np.ones(len(x),bool),deadzone,config)
            for lag in (1,2):
                distances.append(sample(np.linalg.norm(features[lag:]-features[:-lag],axis=1)))
        threshold=max(float(np.quantile(np.concatenate(distances),config.similarity_quantile)),.05)
        with torch.no_grad():
            for buffer,value in ((self.envelope_center,center),(self.envelope_scale,scale),
                                 (self.slope_deadzone,deadzone),(self.shape_threshold,threshold)):
                buffer.copy_(torch.as_tensor(value,device=buffer.device,dtype=buffer.dtype))
            self.shape_calibrated.fill_(True)
        return {'descriptor':'log_AC_RMS_envelope_PAA_cells_and_coarse_slope_histograms',
                'config':asdict(config),'training_sequences':len(arrays),'similarity_threshold':threshold,
                'slope_deadzone':deadzone.tolist(),'per_window_amplitude_normalization':False}

    @torch.no_grad()
    def select_sensor_windows(self,raw600,seed_start,seed_end,aux_valid):
        if not bool(self.shape_calibrated):
            raise ValueError('fit training-fold shape statistics before window selection')
        config=self.shape_config;valid=aux_valid.detach().cpu().numpy().astype(bool)
        raw=raw600.detach().cpu().numpy().copy();raw[~valid]=0
        envelope=log_ac_envelope(raw,config)
        levels=(envelope-self.envelope_center.cpu().numpy())/self.envelope_scale.cpu().numpy()
        features=local_shape_features(levels,valid,self.slope_deadzone.cpu().numpy(),config)
        n=len(raw);positions=np.arange(n)
        initial_left=seed_start.detach().cpu().numpy().astype(np.int64)
        initial_right=seed_end.detach().cpu().numpy().astype(np.int64)
        if (initial_left.shape!=(n,) or initial_right.shape!=(n,) or (initial_left<0).any()
            or (initial_right>=n).any() or (initial_left>positions).any() or (initial_right<positions).any()):
            raise ValueError('each seed must contain its query and remain within the sequence')
        region_left,region_right=valid_regions(valid)
        allowed_left=np.maximum(initial_left-config.max_expansion_steps,region_left)
        allowed_right=np.minimum(initial_right+config.max_expansion_steps,region_right)
        widths=allowed_right-allowed_left+1
        norm=np.sum(features*features,axis=1);threshold2=float(self.shape_threshold.cpu())**2
        selected_left=np.empty(n,np.int64);selected_right=np.empty(n,np.int64)
        max_dist=np.zeros(n)
        def finish(queries,keys,squared,mask):
            compatible=(squared<=threshold2)&mask&valid[queries,None]&valid[keys]
            compatible |= keys==queries[:,None]
            bad=(~compatible)&mask
            selected_left[queries]=np.maximum(allowed_left[queries],np.max(np.where(bad&(keys<queries[:,None]),keys,-1),axis=1)+1)
            selected_right[queries]=np.minimum(allowed_right[queries],np.min(np.where(bad&(keys>queries[:,None]),keys,n),axis=1)-1)
            kept=mask&(keys>=selected_left[queries,None])&(keys<=selected_right[queries,None])
            max_dist[queries]=np.sqrt(np.max(np.where(kept,squared,0),axis=1))
        # For long predicted segments, share the same key descriptor matrix.
        groups={}
        for query in np.flatnonzero(widths>128):
            groups.setdefault((int(allowed_left[query]),int(allowed_right[query])),[]).append(int(query))
        for (lo,hi),group in groups.items():
            key=np.arange(lo,hi+1);key_features=features[key]
            for start in range(0,len(group),128):
                q=np.asarray(group[start:start+128]);keys=np.broadcast_to(key,(len(q),len(key)))
                squared=np.maximum(norm[q,None]+norm[key][None,:]-2*(features[q]@key_features.T),0)
                finish(q,keys,squared,np.ones(keys.shape,bool))
        small=np.flatnonzero(widths<=128)
        for width in np.unique(2**np.ceil(np.log2(widths[small])).astype(int)):
            queries=small[(2**np.ceil(np.log2(widths[small])).astype(int))==width]
            chunk=max(1,min(256,(1<<20)//(width*features.shape[1])))
            offsets=np.arange(width)
            for start in range(0,len(queries),chunk):
                q=queries[start:start+chunk];keys=allowed_left[q,None]+offsets
                mask=(keys<=allowed_right[q,None])&(keys<n);keys=np.minimum(keys,n-1)
                squared=np.maximum(norm[q,None]+norm[keys]-2*np.einsum('qd,qkd->qk',features[q],features[keys]),0)
                finish(q,keys,squared,mask)
        selected_left[~valid]=positions[~valid];selected_right[~valid]=positions[~valid]
        overlap=np.maximum(0,np.minimum(selected_right,initial_right)-np.maximum(selected_left,initial_left)+1)
        initial_width=initial_right-initial_left+1;selected_width=selected_right-selected_left+1
        arrays={'selected_window_start':selected_left,'selected_window_end':selected_right,
                'seed_segment_start':initial_left,'seed_segment_end':initial_right,
                'boundary_clipped_tokens':initial_width-overlap,'redundancy_trimmed_tokens':np.zeros(n,np.int64),
                'expanded_tokens':selected_width-overlap,'sensor_statistics_stable':valid,
                'shape_homogeneous':valid,'shape_max_distance':max_dist,
                'sensor_boundary_score':np.r_[0,np.linalg.norm(np.diff(features,axis=0),axis=1)]}
        return {key:torch.as_tensor(value,device=raw600.device,
                dtype=torch.float32 if raw600.device.type=='mps' and value.dtype.kind=='f' else None)
                for key,value in arrays.items()}
