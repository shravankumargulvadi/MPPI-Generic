"""Analyze calibration and frozen paired cart-pole experiments; NumPy required."""
import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('directory', type=Path)
parser.add_argument('--plot', action='store_true')
args = parser.parse_args()
root = args.directory
rng = np.random.default_rng(20261023)


def read_rows(path):
    rows = list(csv.DictReader(path.open()))
    keys = set()
    for row in rows:
        assert None not in row, (path, row)
        key = tuple(row[k] for k in ('scenario', 'severity', 'mode', 'seed'))
        assert key not in keys, (path, key)
        keys.add(key)
        for key, value in row.items():
            if key not in ('scenario', 'mode'):
                assert np.isfinite(float(value)), (path, key, value)
    return rows


def interval(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return None
    resamples = values[rng.integers(0, len(values), (10000, len(values)))].mean(axis=1)
    return [float(values.mean()), float(np.quantile(resamples, 0.025)), float(np.quantile(resamples, 0.975))]


def mean(rows, col):
    return float(np.mean([float(r.get(col, 0)) for r in rows]))


# All development summaries are descriptive; no development bootstrap claims.
calibration = []
for phase in ('radius-scan', 'refined-radius', 'parameter-screen', 'cart-cost-followup', 'best-radius'):
    for path in sorted((root/phase).glob('*.csv')):
        groups = defaultdict(list)
        for row in read_rows(path):
            groups[(row['scenario'], float(row['severity']), row['mode'])].append(row)
        for (scenario, severity, mode), rows in sorted(groups.items()):
            calibration.append(dict(phase=phase, case=path.stem, scenario=scenario, severity=severity,
                                    mode=mode, trials=len(rows), successes=sum(int(r['success']) for r in rows),
                                    pole_successes=sum(int(r.get('pole_success', 0)) for r in rows),
                                    **{key: mean(rows, key) for key in ('refreshes_mean', 'budget_hit_fraction',
                                       'latency_p95_ms', 'gain_change_mean', 'refreshed_weight_mass_active_mean')}))
with (root/'calibration_summary.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(calibration[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(calibration)

final = []
lines = ['| Configuration | Condition | Primary stock/adaptive | Pole stock/adaptive | p95 stock/adaptive (ms) | Refreshes/cycle | Budget-hit fraction |',
         '| --- | --- | --- | --- | --- | --- | --- |']
for path in sorted((root/'evaluation').glob('*.csv')):
    rows = read_rows(path)
    groups = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        groups[(row['scenario'], float(row['severity']))][row['mode']][int(row['seed'])] = row
    for (scenario, severity), modes in sorted(groups.items()):
        seeds = sorted(modes['stock'])
        assert seeds == list(range(5000, 5030)), (path, scenario, severity, seeds)
        assert set(seeds) == set(modes['adaptive_v0'])
        stock = [modes['stock'][seed] for seed in seeds]
        adaptive = [modes['adaptive_v0'][seed] for seed in seeds]
        item = dict(case=path.stem, scenario=scenario, severity=severity, trials=len(seeds), modes={})
        for mode, rs in (('stock', stock), ('adaptive_v0', adaptive)):
            item['modes'][mode] = dict(successes=sum(int(r['success']) for r in rs),
                                      pole_successes=sum(int(r['pole_success']) for r in rs),
                                      failed_constraints=sum(int(r['failed_constraint']) for r in rs),
                                      deadline_misses=sum(int(r['deadline_misses']) for r in rs),
                                      max_latency_ms=max(float(r['latency_max_ms']) for r in rs),
                                      **{key: mean(rs, key) for key in ('latency_p95_ms', 'cumulative_cost',
                                      'peak_error', 'refreshes_mean', 'budget_hit_fraction', 'saturation_fraction',
                                      'nominal_gap_mean', 'feedback_abs_mean', 'gain_change_mean',
                                      'refreshed_weight_mass_mean', 'refreshed_weight_mass_active_mean')})
        item['paired_difference_ci95'] = {
            col: interval([float(a[col])-float(s[col]) for s, a in zip(stock, adaptive)])
            for col in ('success', 'pole_success', 'latency_p95_ms', 'peak_error', 'cumulative_cost')}
        completed = [(s,a) for s,a in zip(stock,adaptive) if not int(s['failed_constraint']) and not int(a['failed_constraint'])]
        item['complete_pairs'] = len(completed)
        item['completed_cost_diff_ci95'] = interval([float(a['cumulative_cost'])-float(s['cumulative_cost']) for s,a in completed])
        item['primary_discordant_pairs'] = sum(s['success'] != a['success'] for s,a in zip(stock,adaptive))
        item['pole_discordant_pairs'] = sum(s['pole_success'] != a['pole_success'] for s,a in zip(stock,adaptive))
        s,a = item['modes']['stock'],item['modes']['adaptive_v0']
        lines.append(f"| {path.stem} | {scenario} {severity:g} | {s['successes']}/30 / {a['successes']}/30 | "
                     f"{s['pole_successes']}/30 / {a['pole_successes']}/30 | {s['latency_p95_ms']:.3f} / "
                     f"{a['latency_p95_ms']:.3f} | {a['refreshes_mean']:.3f} | {a['budget_hit_fraction']:.4f} |")
        final.append(item)
(root/'evaluation_summary.json').write_text(json.dumps(final, indent=2)+'\n')
(root/'evaluation_table.md').write_text('\n'.join(lines)+'\n')

# Trace comparison is streamed one (scenario,severity,seed) pair at a time.
trace_results = []
for path in sorted((root/'evaluation').glob('*-trace.csv.gz')):
    totals = dict(file=path.name, compared_pairs=0, equal_control_pairs=0, max_control_difference=0.0,
                  max_state_difference=0.0, truncated_pairs=0, active_refresh_cycles=0,
                  zero_tracking_gap_refresh_cycles=0, active_weight_mass_sum=0.0, max_active_weight_mass=0.0)
    current_key = None
    pair = defaultdict(list)

    def compare():
        if not pair:
            return
        s,a = pair['stock'],pair['adaptive_v0']
        totals['compared_pairs'] += 1
        totals['truncated_pairs'] += len(s) != len(a)
        n = min(len(s),len(a))
        ds = np.asarray(s[:n]); da = np.asarray(a[:n])
        u_delta = float(np.max(np.abs(da[:,4]-ds[:,4])))
        state_delta = float(np.max(np.abs(da[:,:4]-ds[:,:4])))
        totals['equal_control_pairs'] += u_delta == 0 and len(s)==len(a)
        totals['max_control_difference'] = max(totals['max_control_difference'],u_delta)
        totals['max_state_difference'] = max(totals['max_state_difference'],state_delta)

    with gzip.open(path, 'rt') as f:
        for row in csv.DictReader(f):
            key = (row['scenario'],row['severity'],row['seed'])
            if key != current_key:
                compare()
                pair = defaultdict(list)
                current_key = key
            pair[row['mode']].append([float(row[c]) for c in ('x','x_dot','theta','theta_dot','applied_control')])
            if row['mode']=='adaptive_v0' and int(row['refreshes'])>0:
                totals['active_refresh_cycles'] += 1
                totals['zero_tracking_gap_refresh_cycles'] += float(row['nominal_gap']) < 1e-8
                weight = float(row['refreshed_weight_mass'])
                totals['active_weight_mass_sum'] += weight
                totals['max_active_weight_mass'] = max(totals['max_active_weight_mass'],weight)
        compare()
    trace_results.append(totals)
(root/'trace_comparison.json').write_text(json.dumps(trace_results, indent=2)+'\n')

print('\n'.join(lines))
print(json.dumps(trace_results, indent=2))

if args.plot:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,2, figsize=(12,8))
    for mode in ('stock','adaptive_v0'):
        for i,case in enumerate(('default-r4','cart200-force20-r4.5')):
            for scenario, linestyle in (('kick','-'),('mismatch','--')):
                group=sorted([r for r in final if r['case']==case and r['scenario']==scenario],key=lambda r:r['severity'])
                axes[0,i].plot([r['severity'] for r in group], [r['modes'][mode]['successes']/30 for r in group],
                               marker='o' if mode=='stock' else 'x', linestyle=linestyle,
                               label=f'{mode}, {scenario}',alpha=0.8)
            axes[0,i].set(title=case,xlabel='kick (rad/s)',ylabel='Primary recovery fraction',ylim=(-.05,1.05))
            axes[0,i].legend(fontsize=8)
    rad=[r for r in calibration if r['phase']=='radius-scan' and r['mode']=='adaptive_v0' and r['scenario']=='nominal']
    rad=sorted([r for r in rad if float(r['case'][7:])<100],key=lambda r:float(r['case'][7:]))
    axes[1,0].plot([float(r['case'][7:]) for r in rad],[r['refreshes_mean'] for r in rad],marker='o')
    axes[1,0].set(xlabel='Weighted trust radius',ylabel='Nominal refreshes / cycle',xscale='log',title='Development radius sweep')
    cases=['default-r4','cart200-force20-r4.5','no-refresh-control']
    x=np.arange(len(cases))
    for j,mode in enumerate(('stock','adaptive_v0')):
        vals=[np.mean([r['modes'][mode]['latency_p95_ms'] for r in final if r['case']==case]) for case in cases]
        axes[1,1].bar(x+(j-.5)*.35,vals,width=.35,label=mode)
    axes[1,1].set(xticks=x,xticklabels=['Original, r=4','Cart 200 / force 20','No-refresh control'],ylabel='Mean trial p95 (ms)',title='Held-out planning latency')
    axes[1,1].legend()
    fig.tight_layout()
    fig.savefig(root/'cartpole_study.png',dpi=160)
