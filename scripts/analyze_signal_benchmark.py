#!/usr/bin/env python3
"""Validate and summarize every completed v3 run plus verified v2 baselines."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from tgif_dwa.signal_wear import METHOD, VARIANTS, PROTOCOL
from tgif_dwa.signal_study import ALL_METHODS, BASELINES, DIAGNOSTICS, METRICS, measure, paired_comparisons, validate_records
from tgif_dwa.wear_metrics import official_wear_concat_metrics
from prepare_wear_figures import shifted_boundaries
from run_wear_benchmark import dataset_hashes, source_hashes


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n'); tmp.replace(path)


def csv_write(path, rows):
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


def robustness_job(root, plan, seed, fold):
    import torch
    from tgif_dwa.signal_wear import load_signal_wear
    from tgif_dwa.wear_data import load_sequence
    from tgif_dwa.wear_model import normalize_wear_inertial
    folder = root / 'checkpoints' / METHOD / f'seed_{seed}/split_{fold:02d}'
    hashes = {name:digest(folder/name) for name in ('parent.pt','background_probe.pt')}
    path = root / 'diagnostics' / f'seed_{seed}_fold_{fold:02d}.json'
    if path.exists():
        saved = json.loads(path.read_text())
        assert saved['checkpoint_sha256'] == hashes
        assert (saved['fold'], saved['seed'], saved['device']) == (fold,seed,plan['fold_devices'][str(fold)])
        return saved['rows']
    device = plan['fold_devices'][str(fold)]
    model, mean, std = load_signal_wear(folder/'parent.pt', folder/'background_probe.pt', device)
    sequence = load_sequence(plan['data_root'], f'sbj_{fold-1}')
    video = torch.as_tensor(sequence['video'], dtype=torch.float32, device=device)
    raw = torch.as_tensor(normalize_wear_inertial(sequence['inertial'],mean,std), device=device)
    valid = torch.ones(len(video), dtype=torch.bool, device=device)
    rows = []
    with torch.inference_mode():
        output = model.parent(video,raw)
        labels = output['round_predictions'][0]
        with np.load(root/'predictions'/METHOD/f'seed_{seed}/split_{fold:02d}.npz') as p:
            assert np.array_equal(labels.cpu().numpy(),p['round0_probabilities'].argmax(-1))
        start,end = model.parent.labels_to_bounds(labels)
        for controller, config in [('Sensor resize',VARIANTS[METHOD]),('Seed only',VARIANTS['NO_SENSOR_RESIZE'])]:
            model.parent.ablation_config = config
            base = output['round_contexts'][1] if controller == 'Sensor resize' else model.parent.segment_seed_expand_attention(video,raw,start,end,valid,False)['context']
            for steps in (-1,1,-2,2):
                changed = torch.as_tensor(shifted_boundaries(labels.cpu().numpy(),steps),device=device)
                lo,hi = model.parent.labels_to_bounds(changed)
                context = model.parent.segment_seed_expand_attention(video,raw,lo,hi,valid,False)['context']
                value = float(torch.linalg.vector_norm(base-context,dim=1).mean().cpu())
                assert np.isfinite(value)
                rows.append({'subject':f'sbj_{fold-1}','fold':fold,'seed':seed,'controller':controller,
                             'signed_steps':steps,'jitter_seconds':abs(steps)*.5,'context_l2':value})
    dump(path,{'checkpoint_sha256':hashes,'fold':fold,'seed':seed,'device':device,'rows':rows})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', type=Path, required=True)
    args = parser.parse_args(); root = args.benchmark.resolve()
    plan = json.loads((root/'benchmark_plan.json').read_text())
    status = json.loads((root/'status.json').read_text())
    assert status['completed'] == status['total'] == 270, 'all retraining must finish before analysis'
    assert plan['methods'] == list(VARIANTS) and plan['seeds'] == [41,47,53] and plan['folds'] == list(range(1,19))
    assert dataset_hashes(Path(plan['data_root'])) == plan['data_sha256'], 'input data changed during the study'
    assert source_hashes(root/'source_snapshot') == plan['source_sha256'], 'frozen study code changed'
    baseline = Path(plan['baseline_root'])
    for name, expected in plan['baseline_artifact_sha256'].items():
        assert digest(baseline/name) == expected, f'baseline changed: {name}'
    for name, entry in json.loads((root/'checkpoint_manifest.json').read_text()).items():
        assert digest(root/name) == entry['sha256'], f'candidate checkpoint changed: {name}'
    source = root/'source_data'; source.mkdir(exist_ok=True)
    rows, class_rows, windows, artifacts = [],[],[],[]
    tracks = {}; truths = {}
    old_metrics = {}
    for method in BASELINES:
        for seed in plan['seeds']:
            payload = json.loads((baseline/'metrics'/f'{method}_seed_{seed}.json').read_text())
            old_metrics.update({(method,seed,r['fold']):r for r in payload['per_subject']})
    for method in [*BASELINES,*VARIANTS]:
        origin = baseline if method in BASELINES else root
        for seed in plan['seeds']:
            for fold in plan['folds']:
                path = origin/'predictions'/method/f'seed_{seed}/split_{fold:02d}.npz'
                with np.load(path, allow_pickle=False) as saved:
                    p = {key:saved[key].copy() for key in saved.files}
                assert p['id'].item() == f'sbj_{fold-1}' and int(p['seed']) == seed and int(p['fold']) == fold
                assert p['method'].item() == method
                expected_protocol = 'fixed_epoch_loso_v2' if method in BASELINES else PROTOCOL
                assert p['protocol'].item() == expected_protocol
                assert np.array_equal(p['pred'], p['probabilities'].argmax(-1))
                key = seed,fold
                if key in truths:
                    assert np.array_equal(truths[key],p['true']), 'methods use different held-out labels'
                else:
                    truths[key] = p['true']
                fields = [(method,'probabilities')]
                if method == METHOD:
                    fields += [('PARENT_NO_PROBE','parent_probabilities'),('ROUND0','round0_probabilities')]
                for display,field in fields:
                    measured,classes = measure(p['true'],p[field],p['id'].item())
                    scope = ('reused_baseline' if method in BASELINES else 'retrained_full' if display == METHOD
                             else 'frozen_full_parent_output' if display in DIAGNOSTICS else 'retrained_ablation')
                    identity = {'subject':p['id'].item(),'protocol':expected_protocol,'method':display,
                                'source_method':method,'scope':scope,'fold':fold,'seed':seed,'run_id':p['run_id'].item()}
                    rows.append({**identity,**measured})
                    class_rows.extend({**identity,**c} for c in classes)
                    tracks[(display,seed,fold)] = p[field].argmax(-1)
                    if method in BASELINES:
                        old = old_metrics[(method,seed,fold)]
                        for metric in ('macro_f1_19','accuracy',*[f'map_at_{x:.1f}' for x in (.3,.4,.5,.6,.7)],'avg_map'):
                            assert abs(measured[metric]-old[metric]) < 1e-10, (method,seed,fold,metric)
                artifacts.append({'method':method,'seed':seed,'fold':fold,'file':str(path),'sha256':digest(path)})
                if method in VARIANTS:
                    for round_id in (0,1):
                        def get(name): return p[f'round{round_id}_{name}']
                        lo,hi = get('selected_window_start'),get('selected_window_end')
                        initial = get('seed_segment_end')-get('seed_segment_start')+1
                        selected = hi-lo+1; positions=np.arange(len(lo))
                        assert (lo <= positions).all() and (hi >= positions).all() and (lo >= 0).all() and (hi < len(lo)).all()
                        assert np.array_equal(selected,initial-get('boundary_clipped_tokens')-get('redundancy_trimmed_tokens')+get('expanded_tokens'))
                        if method == 'NO_SENSOR_RESIZE':
                            assert np.array_equal(lo,get('seed_segment_start')) and np.array_equal(hi,get('seed_segment_end'))
                        if method == 'NO_CONTRACTION':
                            assert (lo <= get('seed_segment_start')).all() and (hi >= get('seed_segment_end')).all()
                        if method == 'NO_EXPANSION':
                            assert (lo >= get('seed_segment_start')).all() and (hi <= get('seed_segment_end')).all()
                        windows.append({'method':method,'seed':seed,'fold':fold,'subject':f'sbj_{fold-1}','round':round_id,
                            'initial_mean_tokens':float(initial.mean()),'selected_mean_tokens':float(selected.mean()),
                            'min_tokens':int(selected.min()),'max_tokens':int(selected.max()),
                            'shrink_fraction':float(np.mean(selected<initial)),'expand_fraction':float(np.mean(selected>initial)),
                            'same_length_fraction':float(np.mean(selected==initial)),
                            'bounds_changed_fraction':float(np.mean((lo != get('seed_segment_start')) | (hi != get('seed_segment_end')))),
                            'stable_fraction':float(get('sensor_statistics_stable').mean())})
    validate_records(rows)
    csv_write(source/'per_subject_seed_metrics.csv',rows)
    csv_write(source/'per_class_metrics.csv',class_rows)
    csv_write(source/'window_behavior.csv',windows)
    dump(source/'prediction_sources.json',artifacts)
    indexed = {(r['method'],r['seed'],r['fold']):r for r in rows}
    subject_means, seed_means, aggregate = [],[],[]
    for method in ALL_METHODS:
        method_subjects = []
        for fold in plan['folds']:
            values = {metric:float(np.mean([indexed[(method,seed,fold)][metric] for seed in plan['seeds']])) for metric in METRICS}
            item = {'method':method,'fold':fold,'subject':f'sbj_{fold-1}',**values}
            subject_means.append(item); method_subjects.append(item)
        for seed in plan['seeds']:
            concatenated = official_wear_concat_metrics([truths[(seed,fold)] for fold in plan['folds']],
                                                       [tracks[(method,seed,fold)] for fold in plan['folds']])
            seed_means.append({'method':method,'seed':seed,'concat_macro_f1':concatenated['macro_f1'],
                **{metric:float(np.mean([indexed[(method,seed,fold)][metric] for fold in plan['folds']])) for metric in METRICS}})
        aggregate.append({'method':method,'concat_macro_f1':float(np.mean([r['concat_macro_f1'] for r in seed_means if r['method']==method])),
            **{metric:float(np.mean([r[metric] for r in method_subjects])) for metric in METRICS},
            **{metric+'_subject_sd':float(np.std([r[metric] for r in method_subjects],ddof=1)) for metric in METRICS}})
    csv_write(source/'subject_means.csv',subject_means); csv_write(source/'seed_means.csv',seed_means)
    csv_write(source/'aggregate_metrics.csv',aggregate)
    statistics = paired_comparisons(rows,plan['test_families'])
    dump(root/'paired_statistics.json',statistics)
    csv_write(source/'paired_comparisons.csv',[{k:v for k,v in r.items() if not isinstance(v,dict)} for r in statistics['comparisons']])
    for fold in (2,10):
        example = [{'time_seconds':(i+1)*.5,'track':'GT','label':int(y)} for i,y in enumerate(truths[(47,fold)])]
        for method in [*BASELINES,METHOD]:
            example += [{'time_seconds':(i+1)*.5,'track':method,'label':int(y)} for i,y in enumerate(tracks[(method,47,fold)])]
        csv_write(source/f'timeline_sbj_{fold-1}_seed47.csv',example)
    print('All 594 method/subject/seed records validated; computing frozen-weight boundary diagnostics.',flush=True)
    import torch
    torch.set_num_threads(1)
    robustness = []
    with ThreadPoolExecutor(max_workers=plan['cpu_workers']) as cpu, ThreadPoolExecutor(max_workers=plan['mps_workers']) as mps:
        futures = [(mps if plan['fold_devices'][str(f)] == 'mps' else cpu).submit(robustness_job,root,plan,s,f)
                   for s in plan['seeds'] for f in plan['folds']]
        for future in as_completed(futures):
            robustness.extend(future.result())
    csv_write(source/'boundary_jitter.csv',sorted(robustness,key=lambda r:(r['seed'],r['fold'],r['controller'],r['signed_steps'])))
    full = next(r for r in aggregate if r['method']==METHOD)
    old = next(r for r in aggregate if r['method']=='FINAL_MODEL')
    numeric_better = all(full[k] > old[k] for k in ('macro_f1_19','map_at_0.5'))
    control = [r for r in statistics['comparisons'] if r['family']=='retrained_components' and r['comparator']=='NO_SENSOR_RESIZE']
    component_supported = all(r['mean_difference'] > 0 and r['p_holm_family'] < .05 for r in control)
    review = {'all_270_training_runs_complete':True,'baseline_records_reproduced':216,'total_metric_records':len(rows),
        'numerically_better_than_published_model_on_both_primary_metrics':numeric_better,
        'sensor_resize_component_positive_holm_below_005_on_both_metrics':component_supported,
        'decision':'Review the full package before publication. No automatic push.',
        'nominal_inference_limitations':statistics['limitation'],
        'full_minus_published':{k:full[k]-old[k] for k in METRICS},
        'figures_pending':True}
    dump(root/'review_conclusions.json',review)
    lines = ['传感器驱动 DWA：18 folds × 3 seeds 全量结果', '',
        '新模型及四个消融版本均重新训练；固定 parent 30 / probe 15 epochs。baseline 复用前已核验输入、配置、权重和预测，指标已逐项重算核对。',
        'TGIF、公开 README、旧图表和权重未修改；尚未上传。', '', '各模型汇总（先平均每个 subject 的三个种子，再平均 18 subjects）：']
    for r in aggregate:
        lines.append(f"{r['method']}: Macro-F1={r['macro_f1_19']:.6f}, mAP@0.5={r['map_at_0.5']:.6f}, Avg.mAP={r['avg_map']:.6f}, Accuracy={r['accuracy']:.6f}, concat F1={r['concat_macro_f1']:.6f}")
    lines += ['',f'新模型相对已发布固定-margin版本，两项主要指标均更高：{numeric_better}',
        f'传感器伸缩相对 NO_SENSOR_RESIZE 的两项差值均为正且组内 Holm p<0.05：{component_supported}',
        '以上统计是探索性证据：LOSO 训练集有重叠；sbj_0/seed47 曾在 pilot 中观察过。已额外报告排除 sbj_0 的敏感性分析。',
        '去掉 probe / Round 0 属于已训练完整模型的输出诊断；四个窗口/attention 消融属于独立重新训练，二者分开画图。',
        '数值更高不自动等于统计显著，也不自动触发上传。完整结果（含负向或不显著比较）均保留。', '',
        '图表：WEAR_sensor_DWA_full_comparison.pdf 与 figures/；全部数值：source_data/；配对统计：paired_statistics.json。',
        '状态未显示 complete 时，绘图或最终核验可能仍在进行。']
    (root/'先看这里.txt').write_text('\n'.join(lines)+'\n')
    dump(source/'manifest.json',{'study':plan['study'],'protocols':{'new':PROTOCOL,'baselines':'fixed_epoch_loso_v2'},
        'n_subjects':18,'seeds':plan['seeds'],'metric_scope':'2 Hz feature grid; fixed 19-class frame scores; frame-derived TAL, not official detector outputs',
        'aggregation':'average seeds within subject, then equally average subjects; pooled F1 reported separately',
        'error_bars':'SD across 18 subject means; paired difference confidence intervals are 95% unadjusted t intervals',
        'sha256':{p.name:digest(p) for p in sorted(source.glob('*.csv'))}})
    print('Metric tables, paired statistics, window diagnostics and local review complete.',flush=True)


if __name__ == '__main__':
    main()
