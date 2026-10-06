import copy
import unittest
import numpy as np
from tgif_dwa.signal_study import ALL_METHODS, BASELINES, DIAGNOSTICS, METRICS, validate_records, paired_comparisons, measure
from tgif_dwa.signal_wear import METHOD, PROTOCOL, VARIANTS


def records(folds=(1,2,3), seeds=(41,47,53)):
    rows=[]
    for m in ALL_METHODS:
        for s in seeds:
            for f in folds:
                value=.5 + (f*.01 if m == METHOD else 0) + (s-47)*.001
                rows.append({'method':m,'seed':s,'fold':f,'subject':f'sbj_{f-1}',
                    'protocol':'fixed_epoch_loso_v2' if m in BASELINES else PROTOCOL,
                    **{k:value for k in METRICS}})
    return rows


class StudyTests(unittest.TestCase):
    def test_all_models_seeds_and_subjects_required_without_protocol_relabeling(self):
        rows=records();validate_records(rows,folds=(1,2,3))
        with self.assertRaisesRegex(ValueError,'incomplete'):
            validate_records(rows[:-1],folds=(1,2,3))
        bad=copy.deepcopy(rows);bad[0]['protocol']=PROTOCOL
        with self.assertRaisesRegex(ValueError,'protocol'):
            validate_records(bad,folds=(1,2,3))
        with self.assertRaisesRegex(ValueError,'duplicate'):
            validate_records([*rows,rows[0]],folds=(1,2,3))

    def test_seed_repeats_do_not_inflate_paired_subject_sample_size(self):
        families={'baseline':BASELINES,'retrained_components':[m for m in VARIANTS if m != METHOD],
                  'frozen_components':DIAGNOSTICS}
        report=paired_comparisons(records(),families,folds=(1,2,3))
        q=[r for r in report['comparisons'] if r['family']=='baseline']
        self.assertEqual(len(q),8)
        for row in q:
            self.assertEqual(row['n_subjects'],3);self.assertEqual(row['n_seeds'],3)
            self.assertAlmostEqual(row['mean_difference'],.02)
            self.assertEqual(row['family_test_count'],8)
            self.assertGreaterEqual(row['p_holm_family'],row['p_two_sided'])
        sensitivity=[r for r in report['comparisons'] if r['family']=='sensitivity_excluding_pilot_subject']
        self.assertTrue(all(r['n_subjects']==2 and 'sbj_0' not in r['per_subject_mean_difference'] for r in sensitivity))

    def test_metric_calculation_keeps_fixed_19_class_space(self):
        truth=np.arange(19);probabilities=np.eye(19)
        values,classes=measure(truth,probabilities,'synthetic')
        self.assertTrue(all(abs(v-1)<1e-12 for v in values.values()))
        self.assertEqual(len(classes),19)
        with self.assertRaisesRegex(ValueError,'probabilities'):
            measure(truth,probabilities*2,'invalid')


if __name__ == '__main__':
    unittest.main()
