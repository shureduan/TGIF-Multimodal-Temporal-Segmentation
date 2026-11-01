"""Paper subsection 2: raw minute CSVs to aligned vibration.npz, without labels."""
import argparse,json
from pipeline.vibration import extract

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--imu-dir',required=True);p.add_argument('--video-features',required=True)
    p.add_argument('--video-start',required=True,help='Local video-clock start, e.g. 2026-05-21T13:55:00')
    p.add_argument('--offset-seconds',required=True,type=float,help='Fixed synchronization offset; never estimated from inference labels')
    p.add_argument('--output',required=True);p.add_argument('--accel-scale',type=float,default=16384.)
    p.add_argument('--allow-missing',action='store_true')
    a=p.parse_args()
    print(json.dumps(extract(a.imu_dir,a.video_features,a.video_start,a.offset_seconds,a.output,
                            accel_scale=a.accel_scale,allow_missing=a.allow_missing),indent=2))
if __name__=='__main__': main()
