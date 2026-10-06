"""Exercise file-based evaluation and benchmark ordering without a full benchmark."""

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def script_module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WearCommandTests(unittest.TestCase):
    def test_evaluator_preserves_subject_metadata_and_rejects_invalid_pairs(self):
        evaluate = script_module('evaluate_wear')
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            truth = np.repeat(np.arange(19), 2)
            payload = dict(id=np.asarray('sbj_0'), true=truth, pred=truth,
                           probabilities=np.eye(19)[truth], protocol=np.asarray('fixed_epoch_loso_v2'),
                           method=np.asarray('FINAL_MODEL'), seed=np.asarray(47), fold=np.asarray(1),
                           run_id=np.asarray('test-run'))
            path = directory / 'prediction.npz'
            output = directory / 'metrics.json'

            def run(*paths):
                with patch.object(sys, 'argv', ['evaluate_wear', *map(str, paths), '--output', str(output)]), \
                        contextlib.redirect_stdout(io.StringIO()):
                    evaluate.main()

            np.savez(path, **payload)
            run(path)
            row = json.loads(output.read_text())['per_subject'][0]
            self.assertEqual(row['subject'], 'sbj_0')
            self.assertEqual(row['seed'], 47)
            self.assertEqual(row['macro_f1_19'], 1.)
            self.assertEqual(row['map_at_0.5'], 1.)
            with self.assertRaisesRegex(ValueError, 'duplicate subject'):
                run(path, path)
            for key, value, error in (
                ('pred', truth.astype(float), 'integers'),
                ('id', np.asarray('sbj_1'), 'identity mismatch'),
                ('probabilities', np.zeros((len(truth), 19)), 'sum to one'),
                ('pred', np.zeros_like(truth), 'argmax'),
            ):
                with self.subTest(key=key):
                    np.savez(path, **{**payload, key: value})
                    with self.assertRaisesRegex(ValueError, error):
                        run(path)

    def test_benchmark_freezes_sources_and_trains_all_jobs_before_scoring(self):
        runner = script_module('run_wear_benchmark')
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'benchmark'
            for kind in ('video', 'imu'):
                (Path(temporary) / kind).mkdir()
                for i in range(18):
                    (Path(temporary) / kind / f'sbj_{i}.npy').write_bytes(b'fixture')
            (Path(temporary) / 'label').mkdir()
            (Path(temporary) / 'label' / 'wear_split_18.json').write_text('{}')
            arguments = ['run_wear_benchmark', '--data-root', temporary, '--output', str(output),
                         '--seeds', '41', '47', '--parent-epochs', '1', '--probe-epochs', '1',
                         '--workers', '2', '--mps-folds', '4']
            calls = []

            def fake_run(cmd, **_):
                calls.append(cmd)
                if Path(cmd[2]).name != 'train_wear.py':
                    return
                options = dict(zip(cmd[3::2], cmd[4::2]))
                method, seed, fold = options['--method'], int(options['--seed']), int(options['--fold'])
                self.assertEqual(options['--device'], 'mps' if fold == 4 else 'cpu')
                folder = Path(options['--output']) / method / f'seed_{seed}' / f'split_{fold:02d}'
                folder.mkdir(parents=True)
                (folder / 'parent.pt').touch()
                if method == 'FINAL_MODEL':
                    (folder / 'background_probe.pt').touch()
                (folder / 'training.json').write_text(json.dumps({
                    'protocol': 'fixed_epoch_loso_v2', 'method': method, 'seed': seed, 'fold': fold,
                    'parent_epochs': 1, 'probe_epochs': 1 if method == 'FINAL_MODEL' else 0,
                    'checkpoint_selection': 'fixed_epoch_last', 'runtime': {'device': options['--device']},
                }))

            with patch.object(sys, 'argv', arguments), \
                    patch.object(runner.subprocess, 'run', side_effect=fake_run), \
                    contextlib.redirect_stdout(io.StringIO()):
                runner.main()
            scripts = [Path(cmd[2]).name for cmd in calls]
            n_jobs = 18 * 2 * 4
            self.assertEqual(scripts[:n_jobs], ['train_wear.py'] * n_jobs)
            self.assertNotIn('train_wear.py', scripts[n_jobs:])
            self.assertEqual(scripts.count('infer_wear.py'), n_jobs)
            self.assertEqual(scripts[-1], 'analyze_wear_statistics.py')
            plan = json.loads((output / 'benchmark_plan.json').read_text())
            self.assertEqual(plan['source_sha256'], runner.source_hashes(output / 'source_snapshot'))
            self.assertEqual(json.loads((output / 'status.json').read_text())['completed'], n_jobs)
            with patch.object(sys, 'argv', arguments + ['--plan-only']), \
                    contextlib.redirect_stdout(io.StringIO()):
                runner.main()
            completion = output / 'checkpoints/FINAL_MODEL/seed_41/split_01/training.json'
            invalid = json.loads(completion.read_text())
            invalid['runtime']['device'] = 'mps'
            completion.write_text(json.dumps(invalid))
            calls.clear()
            with patch.object(sys, 'argv', arguments), \
                    patch.object(runner.subprocess, 'run', side_effect=fake_run), \
                    contextlib.redirect_stdout(io.StringIO()), \
                    self.assertRaisesRegex(RuntimeError, 'test evaluation withheld'):
                runner.main()
            self.assertEqual(calls, [], 'a failed training audit must block test inference')
            self.assertEqual(json.loads((output / 'status.json').read_text())['phase'], 'training_failed')
            frozen = output / 'source_snapshot' / 'scripts' / 'train_wear.py'
            frozen.write_text(frozen.read_text() + '\n# tampered\n')
            with patch.object(sys, 'argv', arguments + ['--plan-only']), \
                    self.assertRaisesRegex(ValueError, 'snapshot changed'):
                runner.main()


if __name__ == '__main__':
    unittest.main()
