"""Summarize the submitted benchmark without relabeling its proxy metrics.

Requires NumPy. With --plot, also requires Matplotlib. Run from any directory:
  python3 doc/adaptive_rmppi_validation/analyze.py [--plot]
"""
import csv
import json
import sys
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parent
rows = list(csv.DictReader((root / 'benchmark.csv').open()))
assert len(rows) == 180
rng = np.random.default_rng(20261021)
metrics = ['success', 'peak_tracking_error', 'cumulative_cost', 'latency_p95_ms', 'refreshes', 'overflows']
summary = {}
lines = ['| Kick (m/s) | Mode | Survival | Mean peak position proxy (m) | Mean state cost | Mean trial p95 (ms) | Mean refreshes/cycle |',
         '| --- | --- | --- | --- | --- | --- | --- |']
for severity in ['0.5', '1', '2']:
    modes = {}
    for mode in ['stock', 'adaptive_v0']:
        records = sorted((r for r in rows if r['severity'] == severity and r['mode'] == mode), key=lambda r: int(r['seed']))
        assert [int(r['seed']) for r in records] == list(range(1000, 1030))
        modes[mode] = {key: np.array([float(r[key]) for r in records]) for key in metrics}
        assert all(np.all(np.isfinite(values)) for values in modes[mode].values())
    item = {}
    indices = rng.integers(0, 30, size=(10000, 30))
    for mode, values in modes.items():
        item[mode] = {key: float(value.mean()) for key, value in values.items()}
        item[mode]['survivals'] = int(values['success'].sum())
        item[mode]['n'] = 30
        item[mode]['refreshes_per_cycle'] = float(values['refreshes'].mean() / 300)
        item[mode]['overflows_per_cycle'] = float(values['overflows'].mean() / 300)
        item[mode]['mean_ci95'] = {
            key: np.percentile(value[indices].mean(axis=1), [2.5, 97.5]).tolist()
            for key, value in values.items()
        }
        v = item[mode]
        lines.append(f"| {severity} | {mode} | {v['survivals']}/30 | {v['peak_tracking_error']:.6f} | {v['cumulative_cost']:.3f} | {v['latency_p95_ms']:.3f} | {v['refreshes_per_cycle']:.3f} |")
    item['paired_adaptive_minus_stock'] = {}
    for key in metrics:
        diff = modes['adaptive_v0'][key] - modes['stock'][key]
        item['paired_adaptive_minus_stock'][key] = {
            'mean': float(diff.mean()),
            'bootstrap_ci95': np.percentile(diff[indices].mean(axis=1), [2.5, 97.5]).tolist(),
        }
    summary[severity] = item
(root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
(root / 'table.md').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
print('\nPaired adaptive minus stock mean differences with 95% bootstrap intervals:')
for severity, item in summary.items():
    print(severity, json.dumps(item['paired_adaptive_minus_stock']))
if '--plot' in sys.argv:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    x = np.arange(3)
    for mode, label, offset, color in [('stock', 'Stock', -.16, '#34495e'), ('adaptive_v0', 'Adaptive V0', .16, '#c56a28')]:
        for ax, key, ylabel in zip(axes, ['peak_tracking_error', 'latency_p95_ms'],
                                    ['Mean peak position proxy (m)', 'Mean of trial planning p95 (ms)']):
            values = np.array([summary[s][mode][key] for s in summary])
            ci = np.array([summary[s][mode]['mean_ci95'][key] for s in summary]).T
            ax.bar(x + offset, values, .30, label=label, color=color,
                   yerr=np.maximum(np.vstack((values-ci[0], ci[1]-values)), 0), capsize=3)
            ax.set_xticks(x, list(summary))
            ax.set_xlabel('Velocity kick (m/s)')
            ax.set_ylabel(ylabel)
            ax.spines[['top', 'right']].set_visible(False)
            ax.grid(axis='y', alpha=.2)
            ax.set_axisbelow(True)
    axes[0].legend(frameon=False)
    fig.suptitle('Submitted linear-plant benchmark: 30 paired seeds per kick\nBars show proxy means; whiskers show 95% bootstrap intervals', fontsize=11)
    fig.savefig(root / 'comparison.png', dpi=180)
