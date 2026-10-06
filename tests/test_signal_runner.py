"""Exercise queue retries, the evaluation barrier and resumption without training 270 real models."""
from contextlib import redirect_stdout
from dataclasses import asdict
import fcntl
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPT_DIR=Path(__file__).resolve().parents[1]/'scripts'
if str(SCRIPT_DIR) not in sys.path:sys.path.insert(0,str(SCRIPT_DIR))
import run_signal_benchmark as runner


class QueueTests(unittest.TestCase):
    def test_retry_preserves_protocol_and_resume_skips_completed_train_and_inference(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);repo=base/'repo';out=base/'study'
            for name in ('src','scripts','configs'):(repo/name).mkdir(parents=True)
            old={'fold_devices':{str(f):'cpu' for f in range(1,19)}}
            argv=['run_signal_benchmark.py','--data-root',str(base/'data'),'--baseline',str(base/'old'),'--output',str(out)]
            completed=set();attempts={};inferences=[];phases=[]
            def execute(command,**kwargs):
                script=Path(command[2]).name
                opts=dict(zip(command[3::2],command[4::2]))
                if script=='train_wear_signal.py':
                    m=opts['--method'];s=int(opts['--seed']);f=int(opts['--fold']);key=m,s,f
                    attempts[key]=attempts.get(key,0)+1
                    folder=Path(opts['--output'])/m/f'seed_{s}/split_{f:02d}'
                    folder.mkdir(parents=True,exist_ok=False)
                    if key==(runner.METHOD,41,1) and attempts[key]==1:
                        (folder/'partial.txt').write_text('runtime interruption')
                        raise subprocess.CalledProcessError(1,command)
                    metadata={'protocol':runner.PROTOCOL,'method':m,'seed':s,'fold':f,
                        'parent_epochs':30,'probe_epochs':15,'checkpoint_selection':'fixed_epoch_last',
                        'ablation_config':asdict(runner.VARIANTS[m]),'runtime':{'device':'cpu'},
                        'parent_history':[{'epoch':e} for e in range(30)],'probe_history':[{'epoch':e} for e in range(15)]}
                    (folder/'training.json').write_text(json.dumps(metadata))
                    for name in ('parent.pt','background_probe.pt'):(folder/name).write_bytes(b'synthetic queue test placeholder')
                    completed.add(key)
                elif script=='infer_wear_signal.py':
                    self.assertEqual(len(completed),270,'evaluation started before all training completed')
                    p=Path(opts['--output']);p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'synthetic inference placeholder');inferences.append(p)
                elif script=='analyze_signal_benchmark.py':
                    self.assertEqual(len(inferences),270)
                    phases.append('analysis')
                    (out/'review_conclusions.json').write_text(json.dumps({'total_metric_records':594,'all_270_training_runs_complete':True,'figures_pending':True}))
                elif script=='plot_signal_benchmark.py':
                    phases.append('plot')
                    review=json.loads((out/'review_conclusions.json').read_text());review.update(figures_pending=False,pdf_pages=9)
                    (out/'review_conclusions.json').write_text(json.dumps(review))
                    p=out/'synthetic.pdf';p.write_bytes(b'synthetic artifact placeholder')
                    (out/'figure_outputs.json').write_text(json.dumps({'pages':9,'figure_names':[str(i) for i in range(9)],'sha256':{'synthetic.pdf':hashlib.sha256(p.read_bytes()).hexdigest()}}))
                else:self.fail(script)
            with patch.object(runner,'REPO',repo), patch.object(runner,'source_hashes',return_value={}),patch.object(runner,'dataset_hashes',return_value={}),patch.object(runner,'baseline_identity',return_value=(old,{})),patch.object(runner.subprocess,'run',side_effect=execute),patch.object(sys,'argv',argv),redirect_stdout(io.StringIO()):
                runner.main()
                self.assertEqual(json.loads((out/'status.json').read_text())['phase'],'complete')
                self.assertEqual(len(list((out/'interrupted').iterdir())),1)
                self.assertEqual(attempts[(runner.METHOD,41,1)],2)
                self.assertEqual(sum(attempts.values()),271)
                runner.main()
                self.assertEqual(sum(attempts.values()),271)
                self.assertEqual(len(inferences),270)
                self.assertEqual(phases,['analysis','plot','analysis','plot'])
                with (out/'.runner.lock').open('a+') as handle:
                    fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                    with self.assertRaisesRegex(RuntimeError,'already has a running coordinator'):
                        runner.main()
