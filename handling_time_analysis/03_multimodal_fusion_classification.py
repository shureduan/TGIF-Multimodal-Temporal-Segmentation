"""Paper subsection 3: fusion/classification and predicted handling durations."""
import argparse,json
from pipeline.inference import infer

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video-features',required=True);p.add_argument('--vibration')
    p.add_argument('--bundle',required=True);p.add_argument('--station-id',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',choices=['cpu','cuda','mps','auto'],default='cpu')
    p.add_argument('--chunk-frames',type=int,default=2247)
    p.add_argument('--labels',help='Optional .npy evaluation labels; never used to generate predictions')
    a=p.parse_args()
    print(json.dumps(infer(a.video_features,a.bundle,a.station_id,a.output,vibration=a.vibration,
                          device=a.device,chunk_frames=a.chunk_frames,labels=a.labels),indent=2))
if __name__=='__main__':main()
