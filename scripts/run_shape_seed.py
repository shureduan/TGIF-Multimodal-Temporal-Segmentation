#!/usr/bin/env python3
"""Frozen, resumable 18-fold/one-seed envelope-shape pilot with paired references."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import uuid
from run_wear_benchmark import dataset_hashes,source_hashes
from tgif_dwa.shape_dwa import ShapeWindowConfig
from tgif_dwa.shape_wear import METHOD,PROTOCOL

REPO=Path(__file__).resolve().parents[1]
REFERENCES=['VIDEO_ONLY','EARLY_CONCAT','FIXED_WINDOW_ATTENTION','FINAL_MODEL','SIGNAL_ADAPTIVE_DWA']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(path)


def reference_manifest(v2,v3,v3_predictions,seed,inputs):
    old=json.loads((v2/'benchmark_plan.json').read_text())
    new=json.loads((v3/'benchmark_plan.json').read_text())
    if old['data_sha256']!=inputs or new['data_sha256']!=inputs or old['fold_devices']!=new['fold_devices']:
        raise ValueError('Reference datasets or fold device assignments differ')
    manifest={};predictions={}
    for method in REFERENCES:
        for fold in range(1,19):
            root=v3 if method=='SIGNAL_ADAPTIVE_DWA' else v2
            folder=root/'checkpoints'/method/f'seed_{seed}/split_{fold:02d}'
            metadata=json.loads((folder/'training.json').read_text())
            protocol='signal_adaptive_loso_v3_candidate' if method=='SIGNAL_ADAPTIVE_DWA' else 'fixed_epoch_loso_v2'
            expected={'method':method,'seed':seed,'fold':fold,'protocol':protocol,'checkpoint_selection':'fixed_epoch_last','parent_epochs':30}
            if not all(metadata[k]==v for k,v in expected.items()):raise ValueError('Reference checkpoint identity differs')
            if metadata['runtime']['device']!=old['fold_devices'][str(fold)]:raise ValueError('Reference device mismatch')
            if [r['epoch'] for r in metadata['parent_history']]!=list(range(30)):raise ValueError('Incomplete reference parent')
            names=['parent.pt','training.json']
            if method in ('FINAL_MODEL','SIGNAL_ADAPTIVE_DWA'):
                if metadata['probe_epochs']!=15 or [r['epoch'] for r in metadata['probe_history']]!=list(range(15)):
                    raise ValueError('Incomplete reference probe')
                names.append('background_probe.pt')
            for name in names:manifest[str(folder/name)]=sha(folder/name)
            path=(v3_predictions if method=='SIGNAL_ADAPTIVE_DWA' else v2/'predictions')/method/f'seed_{seed}/split_{fold:02d}.npz'
            manifest[str(path)]=sha(path)
            predictions[f'{method}/{fold}']=str(path)
    return manifest,predictions,old['fold_devices']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--v2-root',type=Path,required=True)
    parser.add_argument('--v3-root',type=Path,required=True)
    parser.add_argument('--v3-predictions',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=47)
    parser.add_argument('--cpu-workers',type=int,default=2)
    parser.add_argument('--mps-workers',type=int,default=1)
    parser.add_argument('--plan-only',action='store_true')
    args=parser.parse_args()
    if args.seed not in (41,47,53) or args.cpu_workers<1 or args.mps_workers<1:parser.error('invalid seed or workers')
    output=args.output.resolve();snapshot=output/'source_snapshot'
    inputs=dataset_hashes(args.data_root.resolve())
    artifacts,references,devices=reference_manifest(args.v2_root.resolve(),args.v3_root.resolve(),args.v3_predictions.resolve(),args.seed,inputs)
    plan={'study':'envelope_shape_single_seed_18fold_v1','protocol':PROTOCOL,'method':METHOD,'seed':args.seed,
          'folds':list(range(1,19)),'parent_epochs':30,'probe_epochs':15,'checkpoint_selection':'fixed_epoch_last',
          'shape_config':asdict(ShapeWindowConfig()),'data_root':str(args.data_root.resolve()),'data_sha256':inputs,
          'source_sha256':source_hashes(REPO),'reference_artifact_sha256':artifacts,'reference_predictions':references,
          'fold_devices':devices,'cpu_workers':args.cpu_workers,'mps_workers':args.mps_workers,
          'evaluation':'All 18 candidate pairs must complete before test inference. No score-based model selection or rule changes.',
          'primary_metrics':['macro_f1_19','map_at_0.5'],'reference_methods':REFERENCES,
          'qualitative_examples':[f'sbj_1/seed{args.seed}',f'sbj_9/seed{args.seed}'],'scope':'One-seed exploratory comparison; existing subjects were used during prior method development.'}
    plan_path=output/'benchmark_plan.json'
    if plan_path.exists():
        if json.loads(plan_path.read_text())!=plan or source_hashes(snapshot)!=plan['source_sha256']:
            raise ValueError('Frozen source/settings/data changed; use a new output directory')
    else:
        output.mkdir(parents=True,exist_ok=False)
        for name in ('src','scripts','configs'):
            shutil.copytree(REPO/name,snapshot/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        dump(plan_path,plan)
    if args.plan_only:
        print('Frozen 18 full-budget pairs with 90 verified same-seed references.');return
    handle=(output/'.runner.lock').open('a+')
    try:fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('The pilot coordinator is already running')
    # A coordinator can exit while detached training children are still alive.
    # Never archive/restart their folders until those specific children finish.
    active=output/'active_processes';active.mkdir(exist_ok=True)
    for receipt in active.glob('*.json'):
        record=json.loads(receipt.read_text())
        process=subprocess.run(['ps','-p',str(record['pid']),'-o','command='],capture_output=True,text=True)
        if process.returncode==0 and str(snapshot) in process.stdout:
            raise RuntimeError(f"An orphaned study child is still running: PID {record['pid']}. Let it finish before resuming.")
        receipt.unlink()
    import torch
    if any(d=='mps' for d in devices.values()) and not torch.backends.mps.is_available():
        raise RuntimeError('The matched reference assignment requires Apple MPS access')
    environment=dict(os.environ,PYTHONPATH=str(snapshot/'src'),PYTHONDONTWRITEBYTECODE='1',
                     OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
    dump(output/'coordinator.json',{'pid':os.getpid(),'started_utc':datetime.now(timezone.utc).isoformat()})
    states={str(f):{'state':'queued','device':devices[str(f)]} for f in plan['folds']};lock=threading.Lock()
    def status(phase='training',error=None):
        completed=sum(row['state']=='complete' for row in states.values())
        dump(output/'status.json',{'phase':phase,'completed':completed,'total':18,'folds':states,
               'updated_utc':datetime.now(timezone.utc).isoformat(),'error':error})
        lines=['包络形态 DWA 单种子实验',f'种子：{args.seed}；阶段：{phase}；训练完成：{completed}/18 folds',
               'Parent 30 / probe 15 epochs，固定末轮；所有训练完成后统一评估。',
               'CPU：奇数 folds；MPS：偶数 folds，与当前 DWA/SWA 相同。',
               '自动输出：RESULTS.md、source_data/、figures/ 和完整对比图册。']
        for key,row in states.items():
            if row['state']=='running':
                log=output/'checkpoints'/METHOD/f'seed_{args.seed}/split_{int(key):02d}/parent_log.jsonl'
                epoch='尚在初始化'
                if log.exists():
                    entries=log.read_text().splitlines()
                    if entries:
                        try:epoch=f"parent epoch {json.loads(entries[-1])['epoch']+1}/30"
                        except (ValueError,KeyError):pass
                lines.append(f"运行中：fold {key} ({row['device']})，{epoch}")
        if error:lines.append('错误：'+error)
        (output/'进度.txt').write_text('\n'.join(lines)+'\n')
    def execute(script,arguments,log):
        log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('a') as stream:
            command=[sys.executable,'-u',str(snapshot/'scripts'/script),*map(str,arguments)]
            process=subprocess.Popen(command,cwd=snapshot,env=environment,stdout=stream,stderr=subprocess.STDOUT)
            receipt=output/'active_processes'/f'{log.stem}.json'
            dump(receipt,{'pid':process.pid,'command':command,'started_utc':datetime.now(timezone.utc).isoformat()})
            try:
                code=process.wait()
                if code:raise subprocess.CalledProcessError(code,command)
            finally:
                if process.poll() is not None:receipt.unlink(missing_ok=True)
    def validate(fold):
        from tgif_dwa.shape_wear import load_shape_wear
        folder=output/'checkpoints'/METHOD/f'seed_{args.seed}/split_{fold:02d}'
        saved=json.loads((folder/'training.json').read_text())
        expected={'protocol':PROTOCOL,'method':METHOD,'seed':args.seed,'fold':fold,'shape_config':plan['shape_config'],
                  'parent_epochs':30,'probe_epochs':15,'checkpoint_selection':'fixed_epoch_last'}
        if not all(saved[k]==v for k,v in expected.items()) or saved['runtime']['device']!=devices[str(fold)]:
            raise ValueError('Completed pilot pair identity differs')
        for kind,budget in [('parent',30),('probe',15)]:
            if [r['epoch'] for r in saved[kind+'_history']]!=list(range(budget)):raise ValueError('Incomplete budget')
        actual={name:sha(folder/name) for name in ('parent.pt','background_probe.pt','training.json')}
        receipt=folder/'completion_sha256.json'
        if receipt.exists() and json.loads(receipt.read_text())!=actual:raise ValueError('Completed pair hashes changed')
        load_shape_wear(folder/'parent.pt',folder/'background_probe.pt','cpu')
        dump(receipt,actual)
    def train(fold):
        folder=output/'checkpoints'/METHOD/f'seed_{args.seed}/split_{fold:02d}'
        with lock:states[str(fold)].update(state='running',started_utc=datetime.now(timezone.utc).isoformat());status()
        try:
            if not (folder/'training.json').exists():
                if folder.exists():
                    archived=output/'interrupted'/f'fold_{fold:02d}_{uuid.uuid4().hex[:8]}'
                    archived.parent.mkdir(parents=True,exist_ok=True);folder.rename(archived)
                execute('train_wear_shape.py',['--data-root',args.data_root.resolve(),'--output',output/'checkpoints',
                        '--fold',fold,'--seed',args.seed,'--device',devices[str(fold)],'--parent-epochs',30,'--probe-epochs',15],
                        output/'logs'/f'train_fold_{fold:02d}.log')
            validate(fold)
            with lock:states[str(fold)].update(state='complete',finished_utc=datetime.now(timezone.utc).isoformat());status()
            print(f'Complete fold {fold}/18',flush=True)
        except Exception as error:
            with lock:states[str(fold)].update(state='failed',error=str(error));status(error=str(error))
            raise
    def heartbeat(stop):
        while not stop.wait(30):
            with lock:status()
    stopped=threading.Event();pulse=threading.Thread(target=heartbeat,args=(stopped,),daemon=True);pulse.start()
    try:
        errors=[]
        with ThreadPoolExecutor(max_workers=args.cpu_workers) as cpu,ThreadPoolExecutor(max_workers=args.mps_workers) as mps:
            futures=[(mps if devices[str(f)]=='mps' else cpu).submit(train,f) for f in plan['folds']]
            for future in as_completed(futures):
                try:future.result()
                except Exception as error:errors.append(str(error))
        stopped.set();pulse.join()
        if errors:raise RuntimeError('; '.join(errors))
        if dataset_hashes(args.data_root.resolve())!=inputs or source_hashes(snapshot)!=plan['source_sha256']:
            raise ValueError('Frozen inputs or source changed before inference')
        for path,expected in artifacts.items():
            if sha(path)!=expected:raise ValueError('Reference artifact changed before inference')
        status('inference')
        # Sequential held-out inference bounds memory; it never updates weights.
        for fold in plan['folds']:
            validate(fold);folder=output/'checkpoints'/METHOD/f'seed_{args.seed}/split_{fold:02d}'
            prediction=output/'predictions'/METHOD/f'seed_{args.seed}/split_{fold:02d}.npz'
            receipt=prediction.with_suffix('.sha256')
            if receipt.exists():
                if not prediction.exists() or receipt.read_text().strip()!=sha(prediction):raise ValueError('Cached prediction changed')
                continue
            execute('infer_wear_shape.py',['--data-root',args.data_root.resolve(),'--subject',f'sbj_{fold-1}',
                    '--parent',folder/'parent.pt','--probe',folder/'background_probe.pt','--device',devices[str(fold)],'--output',prediction],
                    output/'logs'/f'infer_fold_{fold:02d}.log')
            receipt.write_text(sha(prediction)+'\n');print('Inferred fold',fold,flush=True)
        status('analysis')
        execute('analyze_shape_seed.py',['--benchmark',output],output/'logs'/'analysis.log')
        status('plotting')
        execute('plot_shape_seed.py',['--benchmark',output],output/'logs'/'plotting.log')
        dump(output/'final_verification.json',{'all_18_pairs_complete':True,'fixed_epoch_last':True,
             'source_sha256':plan['source_sha256'],'input_sha256':inputs,'reference_sha256':artifacts,
             'source_manifest_sha256':sha(output/'source_data/manifest.json')})
        status('complete');print('Single-seed pilot complete:',output,flush=True)
    except Exception as error:
        stopped.set();pulse.join();status('failed',str(error));raise
    finally:fcntl.flock(handle.fileno(),fcntl.LOCK_UN);handle.close()

if __name__=='__main__':main()
