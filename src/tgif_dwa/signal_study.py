"""Metrics and predeclared subject-level comparisons for the v3 study."""
import numpy as np
from scipy import stats
from sklearn.metrics import precision_recall_fscore_support

from .signal_wear import METHOD, PROTOCOL, VARIANTS
from .wear_metrics import official_macro_19, tal_map
from .wear_statistics import holm_adjust

BASELINES = ['VIDEO_ONLY', 'EARLY_CONCAT', 'FIXED_WINDOW_ATTENTION', 'FINAL_MODEL']
DIAGNOSTICS = ['PARENT_NO_PROBE', 'ROUND0']
ALL_METHODS = [*BASELINES, *VARIANTS, *DIAGNOSTICS]
METRICS = ['accuracy', 'macro_precision_19', 'macro_recall_19', 'macro_f1_19',
           'background_f1', 'action_macro_f1', *[f'map_at_{x:.1f}' for x in (.3,.4,.5,.6,.7)], 'avg_map']
PRIMARY = ['macro_f1_19', 'map_at_0.5']


def measure(truth, probabilities, identity):
    truth = np.asarray(truth)
    probabilities = np.asarray(probabilities)
    if truth.ndim != 1 or not len(truth) or probabilities.shape != (len(truth), 19):
        raise ValueError('inconsistent prediction shapes')
    if not np.issubdtype(truth.dtype, np.integer) or not np.isin(truth, np.arange(19)).all():
        raise ValueError('invalid 19-class targets')
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or not np.allclose(probabilities.sum(-1), 1, atol=1e-5):
        raise ValueError('invalid probabilities')
    prediction = probabilities.argmax(-1)
    precision, recall, f1, support = precision_recall_fscore_support(truth, prediction, labels=np.arange(19), average=None, zero_division=0)
    values = {'accuracy': float(np.mean(truth == prediction)), **official_macro_19(truth, prediction),
              'background_f1': float(f1[18]), 'action_macro_f1': float(f1[:18].mean()),
              **tal_map([{'id': identity, 'true': truth, 'pred': prediction, 'probabilities': probabilities}])}
    per_class = [{'class_id': c, 'precision': float(precision[c]), 'recall': float(recall[c]),
                  'f1': float(f1[c]), 'support': int(support[c])} for c in range(19)]
    return values, per_class


def validate_records(records, seeds=(41,47,53), folds=tuple(range(1,19))):
    expected = {(m,s,f) for m in ALL_METHODS for s in seeds for f in folds}
    seen = set()
    for row in records:
        key = row['method'], int(row['seed']), int(row['fold'])
        if key not in expected or key in seen or row['subject'] != f'sbj_{key[2]-1}':
            raise ValueError('duplicate, unexpected or mismatched study identity')
        protocol = 'fixed_epoch_loso_v2' if row['method'] in BASELINES else PROTOCOL
        if row['protocol'] != protocol:
            raise ValueError('unexpected source protocol for method')
        if not all(np.isfinite(row[k]) and 0 <= row[k] <= 1 for k in METRICS):
            raise ValueError('invalid metric value')
        seen.add(key)
    if seen != expected:
        raise ValueError(f'incomplete study: {len(expected-seen)} records missing')


def paired_comparisons(records, families, *, seeds=(41,47,53), folds=tuple(range(1,19))):
    validate_records(records, seeds, folds)
    indexed = {(r['method'],int(r['seed']),int(r['fold'])):r for r in records}
    rows = []
    groups = [(name,comparators,folds) for name,comparators in families.items()]
    # The pilot subject is already observed. This sensitivity view is declared
    # before the full benchmark, rather than chosen after seeing its results.
    groups.append(('sensitivity_excluding_pilot_subject', BASELINES, tuple(f for f in folds if f != 1)))
    for name, comparators, selected in groups:
        if len(selected) < 2:
            continue
        family_rows = []
        for comparator in comparators:
            for metric in PRIMARY:
                differences = np.array([[indexed[(METHOD,s,f)][metric]-indexed[(comparator,s,f)][metric] for s in seeds] for f in selected])
                paired = differences.mean(1)
                mean = float(paired.mean()); sd = float(paired.std(ddof=1)); n = len(paired)
                half = float(stats.t.ppf(.975,n-1)*sd/np.sqrt(n))
                if sd == 0:
                    t, p = (0.,1.) if mean == 0 else (None,0.)
                else:
                    result = stats.ttest_1samp(paired,0.)
                    t,p = float(result.statistic),float(result.pvalue)
                family_rows.append({'family':name, 'comparator':comparator,'metric':metric,
                    'n_subjects':n,'n_seeds':len(seeds),'mean_difference':mean,'ci95_low':mean-half,'ci95_high':mean+half,
                    't_statistic':t,'p_two_sided':p,'wins':int((paired>0).sum()),'ties':int((paired==0).sum()),
                    'losses':int((paired<0).sum()),'per_seed_mean_difference':dict(zip(map(str,seeds),differences.mean(0).tolist())),
                    'per_subject_mean_difference':{f'sbj_{f-1}':float(v) for f,v in zip(selected,paired)}})
        adjusted = holm_adjust([r['p_two_sided'] for r in family_rows])
        for row,pvalue in zip(family_rows,adjusted):
            row['p_holm_family'] = pvalue
            row['family_test_count'] = len(family_rows)
        rows.extend(family_rows)
    return {'unit':'subject: mean matched differences across seeds before paired testing',
            'interval':'95% t interval on paired subject differences; unadjusted intervals',
            'limitation':'Exploratory: LOSO training sets overlap, and sbj_0/seed47 was observed in the pilot. Nominal p-values are not independent-dataset confirmation.',
            'comparisons':rows}
