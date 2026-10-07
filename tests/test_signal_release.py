"""Release cohorts, statistics and checkpoint-member integrity."""
import copy
import csv
import json
from pathlib import Path
import sys
import unittest
import numpy as np
from scipy import stats

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from download_signal_weights import read_manifest, validate_archive_record
from tgif_dwa.wear_statistics import holm_adjust


class SignalReleaseTests(unittest.TestCase):
    def test_release_archives_require_exact_matched_cohorts_and_pairs(self):
        archives=read_manifest(ROOT/'models/wear_signal_v3/manifest.json')
        self.assertEqual(sum(len(x['files']) for x in archives),216)
        for original in archives:
            broken=copy.deepcopy(original);broken['files'].pop()
            with self.assertRaises(ValueError):validate_archive_record(broken)
            broken=copy.deepcopy(original);broken['folds']=list(range(1,19)) if original['method']!='SIGNAL_ADAPTIVE_DWA' else [1]
            with self.assertRaises(ValueError):validate_archive_record(broken)

    def test_subject_aggregation_uses_the_documented_matched_cohort(self):
        root=ROOT/'results/wear_signal_v3'
        records=list(csv.DictReader((root/'per_subject_seed_metrics.csv').read_text().splitlines()))
        index={(r['method'],int(r['seed']),int(r['fold'])):r for r in records}
        self.assertEqual(len(index),432)
        for filename,folds in [('aggregate_metrics.csv',range(1,19)),('ablation_aggregate_metrics.csv',range(1,19,2))]:
            for row in csv.DictReader((root/filename).read_text().splitlines()):
                self.assertEqual(int(row['n_subjects']),len(folds))
                for metric in ['macro_f1_19','accuracy','map_at_0.5','avg_map']:
                    subjects=[np.mean([float(index[row['method'],s,f][metric]) for s in [41,47,53]]) for f in folds]
                    self.assertAlmostEqual(float(row[metric]),float(np.mean(subjects)),places=12)
                    self.assertAlmostEqual(float(row[metric+'_subject_sd']),float(np.std(subjects,ddof=1)),places=12)

    def test_paired_statistics_recompute_from_subject_differences(self):
        rows=json.loads((ROOT/'results/wear_signal_v3/paired_statistics.json').read_text())['comparisons']
        self.assertEqual(len(rows),28)
        families={}
        for row in rows:
            diff=list(row['per_subject_difference'].values())
            self.assertEqual(len(diff),row['n_subjects'])
            actual=stats.ttest_1samp(diff,0)
            self.assertAlmostEqual(float(actual.pvalue),row['p_two_sided'],places=12)
            families.setdefault(row['family'],[]).append(row)
        for rows in families.values():
            p=[r['p_two_sided'] for r in rows]+[1.]*(rows[0]['family_test_count']-len(rows))
            for row,expected in zip(rows,holm_adjust(p)):
                self.assertAlmostEqual(row['p_holm_family'],expected,places=12)


if __name__=='__main__':
    unittest.main()
