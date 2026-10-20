"""Release contracts: matched cohorts, portable provenance and paired weight bundles."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'scripts'))
from download_shape_weights import read_manifest, validate_archive_record
from prepare_shape_results import prepare, read_csv


class ShapeReleaseTests(unittest.TestCase):
    def test_weight_manifest_requires_all_paired_folds(self):
        archive=read_manifest(REPO/'models/wear_shape_v4/manifest.json')[0]
        self.assertEqual(len(validate_archive_record(archive)),36)
        bad=json.loads(json.dumps(archive));bad['files'].pop()
        with self.assertRaises(ValueError):validate_archive_record(bad)
        bad=json.loads(json.dumps(archive));bad['folds']=list(range(1,18))
        with self.assertRaises(ValueError):validate_archive_record(bad)

    def test_public_tables_are_portable_and_controls_share_the_declared_cohort(self):
        source=REPO/'results/wear_shape_v4'
        metadata=json.loads((source/'manifest.json').read_text())
        with tempfile.TemporaryDirectory() as t:
            out=Path(t)/'release'
            result=prepare(source,source/'control_metrics.csv',out,
                           {'data_sha256':metadata['data_sha256'],'fold_devices':metadata['reference_fold_devices']})
            self.assertEqual(result['seeds'],[47])
            self.assertEqual(result['ablation_folds'],list(range(1,19,2)))
            self.assertNotIn('/Users/',(out/'prediction_sources.json').read_text())
            self.assertNotIn('/private/',(out/'prediction_sources.json').read_text())
            rows=read_csv(out/'ablation_aggregate_metrics.csv')
            self.assertEqual(len(rows),4)
            self.assertTrue(all(r['n_subjects']=='9' and r['seed']=='47' for r in rows))

    def test_mismatched_or_duplicate_controls_cannot_enter_the_figure(self):
        source=REPO/'results/wear_shape_v4';meta=json.loads((source/'manifest.json').read_text())
        with tempfile.TemporaryDirectory() as t:
            controls=Path(t)/'controls.csv';rows=read_csv(source/'control_metrics.csv');rows[-1]=rows[0].copy()
            with controls.open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
            with self.assertRaisesRegex(ValueError,'matched controls'):
                prepare(source,controls,Path(t)/'out',meta)

    def test_changed_metric_table_is_rejected_before_publication(self):
        with tempfile.TemporaryDirectory() as t:
            source=Path(t)/'source';shutil.copytree(REPO/'results/wear_shape_v4',source)
            q=source/'per_subject_metrics.csv';q.write_text(q.read_text()+'\n')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                prepare(source,source/'control_metrics.csv',Path(t)/'out',{})

    def test_published_aggregate_matches_subject_records(self):
        import statistics
        source=REPO/'results/wear_shape_v4'
        rows=read_csv(source/'per_subject_metrics.csv')
        for aggregate in read_csv(source/'aggregate_metrics.csv'):
            values=[float(r['macro_f1_19']) for r in rows if r['method']==aggregate['method']]
            self.assertEqual(len(values),18)
            self.assertAlmostEqual(statistics.mean(values),float(aggregate['macro_f1_19']),places=12)
            self.assertAlmostEqual(statistics.stdev(values),float(aggregate['macro_f1_19_subject_sd']),places=12)


if __name__=='__main__':unittest.main()
