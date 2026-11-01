"""Paper subsection 1: python 01_visual_feature_extraction.py --help."""
import argparse,json
from pathlib import Path
from pipeline.visual import extract

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video',required=True);p.add_argument('--output',required=True)
    r=p.add_mutually_exclusive_group(required=True)
    r.add_argument('--operator-roi',type=int,nargs=4,metavar=('X1','Y1','X2','Y2'))
    r.add_argument('--roi-json',help='JSON with operator_roi as a list or {x1,y1,x2,y2}')
    p.add_argument('--start-seconds',type=float,default=0.);p.add_argument('--end-seconds',type=float)
    p.add_argument('--batch-size',type=int,default=4);p.add_argument('--max-anchors',type=int)
    p.add_argument('--device',choices=['cpu','cuda','mps','auto'],default='cpu')
    p.add_argument('--local-files-only',action='store_true')
    a=p.parse_args();roi=a.operator_roi
    if a.roi_json:
        roi=json.loads(Path(a.roi_json).read_text())['operator_roi']
        if isinstance(roi,dict): roi=[roi[k] for k in ('x1','y1','x2','y2')]
    print(json.dumps(extract(a.video,roi,a.output,start=a.start_seconds,end=a.end_seconds,
                            batch_size=a.batch_size,device=a.device,local_files_only=a.local_files_only,
                            max_anchors=a.max_anchors),indent=2))
if __name__=='__main__':main()
