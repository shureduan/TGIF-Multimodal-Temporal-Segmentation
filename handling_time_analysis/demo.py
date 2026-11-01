"""Create synthetic video/IMU input and execute all three stages with trained models.

Demonstrates file contracts and runtime execution, not recognition accuracy.
"""
import argparse,csv
from pathlib import Path
import numpy as np
from pipeline.visual import extract as visual
from pipeline.vibration import extract as vibration
from pipeline.inference import infer

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',required=True)
    p.add_argument('--output',default='outputs/synthetic_demo');p.add_argument('--device',default='cpu')
    p.add_argument('--local-files-only',action='store_true');a=p.parse_args()
    import cv2
    root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    raw=root/'imu';raw.mkdir(exist_ok=True)
    video=root/'synthetic.avi';writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),30,(160,120))
    if not writer.isOpened(): raise RuntimeError('OpenCV MJPG writer is unavailable')
    for i in range(120):
        frame=np.full((120,160,3),35,np.uint8)
        cv2.rectangle(frame,(20+i%50,30),(70+i%50,100),(70,150,210),-1);writer.write(frame)
    writer.release()
    with open(raw/'imu3_2026-01-01_12-00.csv','w',newline='') as f:
        w=csv.writer(f);w.writerow(['timestamp','ax','ay','az'])
        for i in range(2000):
            t=i/500;w.writerow([i*2,0,0,16384*(1+.04*np.sin(2*np.pi*30*t))])
    visual(video,[20,10,140,120],root/'visual',batch_size=2,device=a.device,local_files_only=a.local_files_only)
    vibration(raw,root/'visual','2026-01-01T12:00:00',0.,root/'vibration.npz')
    result=infer(root/'visual',a.bundle,'station3',root/'predictions',vibration=root/'vibration.npz',device=a.device)
    print('Synthetic interface demonstration completed:',result['analyzed_seconds'],'analyzed seconds')
    print('These predictions are not a TGIF accuracy measurement.')
if __name__=='__main__': main()
