"""Summarize adaptive_rmppi_recovery_benchmark output with paired-seed uncertainty.

Requires NumPy; --plot also requires Matplotlib.
  python3 examples/adaptive_rmppi_recovery_analyze.py summary.csv [--trace trace.csv] [--out DIR] [--plot]
Writes table.md and summary.json to --out (default: directory of summary.csv).
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

MODES = ('stock', 'adaptive_v0')
CYCLE_DEADLINE_MS = 20.0

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('summary', type=Path)
parser.add_argument('--trace', type=Path)
parser.add_argument('--out', type=Path)
parser.add_argument('--plot', action='store_true')
parser.add_argument('--resamples', type=int, default=10000)
args = parser.parse_args()
out_dir = (args.out or args.summary.parent).resolve()
out_dir.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(20261021)

groups = defaultdict(dict)
for row in csv.DictReader(args.summary.open()):
    groups[(row['scenario'], float(row['severity']))].setdefault(row['mode'], {})[int(row['seed'])] = row


def paired_ci(a, b):
    d = b - a
    idx = rng.integers(0, len(d), size=(args.resamples, len(d)))
    boot = d[idx].mean(axis=1)
    return float(d.mean()), [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]


def wilson(k, n, z=1.96):
    if n == 0:
        return [float('nan'), float('nan')]
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(c - h), float(c + h)]


summary = []
lines = ['| Scenario | Severity | Mode | Recovery success [95% CI] | Mean peak error | Mean trial p95 latency (ms) '
         '| Max latency (ms) | Deadline misses | Refreshes/cycle mean (max trial p95) | Budget-hit cycles |',
         '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
diff_lines = ['| Scenario | Severity | Success diff (adaptive - stock) [95% CI] | Peak error diff [95% CI] '
              '| Cost diff [95% CI] | Added trial p95 latency (ms) [95% CI] |',
              '| --- | --- | --- | --- | --- | --- |']
for (scenario, severity), modes in sorted(groups.items()):
    seeds = sorted(set.intersection(*(set(modes.get(m, {})) for m in MODES)))
    if not seeds:
        continue
    get = {m: {k: np.array([float(modes[m][s][k]) for s in seeds]) for k in modes[m][seeds[0]]
               if k not in ('scenario', 'mode')} for m in MODES}
    item = {'scenario': scenario, 'severity': severity, 'paired_seeds': len(seeds), 'modes': {}}
    for m in MODES:
        g = get[m]
        k = int(g['success'].sum())
        item['modes'][m] = {
            'successes': k, 'success_ci95': wilson(k, len(seeds)),
            'failed_constraint': int(g['failed_constraint'].sum()),
            'peak_error_mean': float(g['peak_error'].mean()),
            'cumulative_cost_mean': float(g['cumulative_cost'].mean()),
            'trial_p95_latency_ms_mean': float(g['latency_p95_ms'].mean()),
            'latency_max_ms': float(g['latency_max_ms'].max()),
            'deadline_misses': int(g['deadline_misses'].sum()),
            'refreshes_per_cycle_mean': float(g['refreshes_mean'].mean()),
            'refreshes_per_cycle_p95_max': float(g['refreshes_p95'].max()),
            'budget_hit_fraction_mean': float(g['budget_hit_fraction'].mean()),
        }
        s = item['modes'][m]
        lines.append(f"| {scenario} | {severity:g} | {m} | {k}/{len(seeds)} [{s['success_ci95'][0]:.2f}, "
                     f"{s['success_ci95'][1]:.2f}] | {s['peak_error_mean']:.4f} | {s['trial_p95_latency_ms_mean']:.3f} "
                     f"| {s['latency_max_ms']:.3f} | {s['deadline_misses']} | {s['refreshes_per_cycle_mean']:.3f} "
                     f"({s['refreshes_per_cycle_p95_max']:.0f}) | {s['budget_hit_fraction_mean']:.3f} |")
    item['diff'] = {key: paired_ci(get['stock'][col], get['adaptive_v0'][col]) for key, col in
                    (('success', 'success'), ('peak_error', 'peak_error'), ('cumulative_cost', 'cumulative_cost'),
                     ('trial_p95_latency_ms', 'latency_p95_ms'))}
    d = item['diff']
    fmt = lambda v: f"{v[0]:.4g} [{v[1][0]:.4g}, {v[1][1]:.4g}]"
    diff_lines.append(f"| {scenario} | {severity:g} | {fmt(d['success'])} | {fmt(d['peak_error'])} | "
                      f"{fmt(d['cumulative_cost'])} | {fmt(d['trial_p95_latency_ms'])} |")
    summary.append(item)

table = '\n'.join(lines) + '\n\n' + '\n'.join(diff_lines) + '\n'
(out_dir / 'table.md').write_text(table)
(out_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
print(table)

if args.plot and args.trace:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    traces = defaultdict(lambda: defaultdict(list))
    latencies = defaultdict(list)
    for row in csv.DictReader(args.trace.open()):
        latencies[row['mode']].append(float(row['latency_ms']))
        if row['scenario'] == 'mismatch' and float(row['severity']) == max(
                sev for sc, sev in groups if sc == 'mismatch'):
            traces[row['mode']][int(row['seed'])].append((float(row['time_s']), float(row['error'])))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for m in MODES:
        runs = traces[m]
        if runs:
            n = min(len(v) for v in runs.values())
            t = np.array([p[0] for p in next(iter(runs.values()))[:n]])
            e = np.array([[p[1] for p in v[:n]] for v in runs.values()])
            ax1.plot(t, np.median(e, axis=0), label=f'{m} median')
            ax1.fill_between(t, np.percentile(e, 10, axis=0), np.percentile(e, 90, axis=0), alpha=0.2)
        ax2.hist(latencies[m], bins=60, alpha=0.5, label=m)
    ax1.set(xlabel='time (s)', ylabel='weighted recovery error', title='Mismatch, largest kick (10-90%)')
    ax1.legend()
    ax2.axvline(CYCLE_DEADLINE_MS, color='k', linestyle='--', label='deadline')
    ax2.set(xlabel='control-cycle latency (ms)', ylabel='cycles', title='Latency, all scenarios', yscale='log')
    ax2.legend()
    fig.tight_layout()
    fig.savefig(out_dir / 'recovery.png', dpi=120)
