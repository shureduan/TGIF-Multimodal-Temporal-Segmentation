"""1. Frozen global/operator-ROI VideoMAE feature extraction."""
from __future__ import annotations
from collections import deque
from pathlib import Path
import time
import numpy as np
from .common import ENCODER_ID, ENCODER_REVISION, FEATURE_RATE, OFFSETS, device_for, synchronize, write_json

class VideoEncoder:
    def __init__(self, device, local_files_only=False):
        import torch
        from transformers import AutoImageProcessor, VideoMAEConfig, VideoMAEModel
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(
            ENCODER_ID, revision=ENCODER_REVISION, local_files_only=local_files_only)
        config = VideoMAEConfig.from_pretrained(
            ENCODER_ID, revision=ENCODER_REVISION, local_files_only=local_files_only)
        # Existing TGIF feature caches used the zero-bias encoder behavior.
        # Make that behavior explicit, avoiding silently initialized bias keys
        # when new Transformers versions read the old pretraining checkpoint.
        config.qkv_bias = False
        self.model, loading = VideoMAEModel.from_pretrained(
            ENCODER_ID, revision=ENCODER_REVISION, config=config,
            local_files_only=local_files_only, output_loading_info=True)
        if loading.get('missing_keys') or loading.get('mismatched_keys'):
            raise RuntimeError(f'Incomplete frozen VideoMAE load: {loading}')
        self.model = self.model.to(device).eval()
        for parameter in self.model.parameters(): parameter.requires_grad_(False)

    def encode(self, clips):
        import torch
        with torch.inference_mode():
            pixels = self.processor(clips, return_tensors='pt')['pixel_values'].to(self.device)
            patch = self.model.embeddings(pixels, bool_masked_pos=None)
            encoded = self.model.encoder(patch).last_hidden_state
            return self.model.layernorm(encoded).mean(dim=1).cpu().float().numpy()

def validate_roi(roi, width, height):
    if len(roi) != 4 or any(int(v) != v for v in roi):
        raise ValueError('operator ROI must contain four integer pixel coordinates')
    x1, y1, x2, y2 = map(int, roi)
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f'ROI {roi} is outside the {width}x{height} video')
    return x1, y1, x2, y2

def crop(frame, roi):
    import cv2
    from PIL import Image
    x1,y1,x2,y2 = roi
    rgb = cv2.cvtColor(frame[y1:y2,x1:x2], cv2.COLOR_BGR2RGB)
    return Image.fromarray(cv2.resize(rgb,(224,224),interpolation=cv2.INTER_LINEAR))

def extract(video, roi, output, *, start=0., end=None, batch_size=4,
            device='cpu', local_files_only=False, max_anchors=None, encoder=None):
    import cv2
    import torch
    if batch_size < 1 or not np.isfinite(start) or start < 0:
        raise ValueError('batch_size must be positive and start must be nonnegative')
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened(): raise ValueError(f'Cannot open video: {video}')
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width,height = int(cap.get(3)),int(cap.get(4))
        if not np.isfinite(fps) or fps <= 0 or frames < 1:
            raise ValueError('Video has invalid frame rate or duration')
        if not np.isclose(fps/4,FEATURE_RATE,rtol=.01):
            raise ValueError('This TGIF feature protocol expects approximately 30-fps input; resample the video before extracting')
        roi = validate_roi(roi,width,height)
        duration = frames/fps
        end = duration if end is None else float(end)
        if not np.isfinite(end) or not start < end <= duration+1/fps:
            raise ValueError(f'Require 0 <= start < end <= video duration ({duration:.6f}s)')
        start_frame,end_frame = int(start*fps),min(frames,int(end*fps))
        n = (end_frame-start_frame)//4
        if max_anchors is not None:
            if max_anchors < 1: raise ValueError('max_anchors must be positive')
            n = min(n,max_anchors)
        if n < 1: raise ValueError('At least four source frames are needed')
        device = device_for(device)
        encoder = encoder or VideoEncoder(device,local_files_only)
        output = Path(output); output.mkdir(parents=True,exist_ok=True)
        gl,op = np.empty((n,768),np.float32),np.empty((n,768),np.float32)
        timestamps = start + np.arange(n,dtype=np.float64)*4/fps
        cap.set(cv2.CAP_PROP_POS_FRAMES,start_frame)
        ok,first = cap.read()
        if not ok: raise ValueError('Cannot decode first requested frame')
        queue=deque([first]*61,maxlen=61); current=0
        ids,global_batch,operator_batch=[],[],[]
        forward_seconds=0.
        synchronize(device); begin=time.perf_counter()
        def flush():
            nonlocal forward_seconds
            if not ids: return
            synchronize(device); before=time.perf_counter()
            global_z=encoder.encode(global_batch);operator_z=encoder.encode(operator_batch)
            synchronize(device);forward_seconds+=time.perf_counter()-before
            gl[ids],op[ids]=global_z,operator_z
            ids.clear();global_batch.clear();operator_batch.clear()
        for index in range(n):
            while current < index*4+30:
                ok,frame=cap.read()
                if not ok and start_frame+current+1 < frames:
                    raise ValueError('Video decode failed before EOF; refusing silent frame substitution')
                queue.append(frame if ok else queue[-1]);current+=1
            clip=[queue[int(offset)+30] for offset in OFFSETS]
            ids.append(index)
            global_batch.append([crop(frame,(0,0,width,height)) for frame in clip])
            operator_batch.append([crop(frame,roi) for frame in clip])
            if len(ids)>=batch_size: flush()
        flush();elapsed=time.perf_counter()-begin
        if not (np.isfinite(gl).all() and np.isfinite(op).all()):
            raise ValueError('Encoder produced non-finite features')
        for name,x in [('global_features.npy',gl),('operator_features.npy',op),('timestamps_sec.npy',timestamps)]:
            np.save(output/name,x,allow_pickle=False)
        coverage_end=min(end,float(timestamps[-1]+4/fps))
        meta=dict(stage='visual_feature_extraction',encoder=ENCODER_ID,revision=ENCODER_REVISION,
                  feature_dim=1536,encoder_qkv_bias=False,source_fps=fps,feature_dt=4/fps,anchor_stride_frames=4,
                  clip_frame_offsets=OFFSETS.tolist(),operator_roi=list(roi),
                  analysis_start_sec=float(timestamps[0]),analysis_end_sec=coverage_end,
                  source_video_name=Path(video).name,device=str(device),n_features=n,
                  extraction_seconds=elapsed,encoder_forward_seconds=forward_seconds,
                  timing_scope='decode, crop, processor and both frozen encoders; excludes model initialization and output writes')
        write_json(output/'metadata.json',meta)
        return meta
    finally: cap.release()
