"""Run the registered cart-pole development experiments inside the CUDA container."""
import argparse
import csv
import gzip
import shutil
import json
import subprocess
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('phase', choices=['radius', 'screen', 'followup', 'evaluation'])
p.add_argument('--build', type=Path, default=Path('/validation/build'))
p.add_argument('--out', type=Path, required=True)
p.add_argument('--radius', type=float, default=4.0)
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
binaries = {50: a.build/'mppi/examples/adaptive_rmppi_recovery_benchmark',
            100: a.build/'adaptive_cartpole_h100', 150: a.build/'adaptive_cartpole_h150'}
if a.phase == 'radius':
    cases = [(f'radius-{r:g}', 50, r, []) for r in [3.5, 4, 4.5, 5, 1e6]]
    seeds, offset, kicks = 5, 10, '1'
elif a.phase == 'screen':
    cases = [('default', 50, a.radius, []), ('force20', 50, a.radius, ['--force=20']),
             ('cart200', 50, a.radius, ['--cart-weight=200']),
             ('noise2.5', 50, a.radius, ['--noise=2.5']), ('noise7.5', 50, a.radius, ['--noise=7.5']),
             ('lambda1', 50, a.radius, ['--lambda=1']),
             ('ddp3', 50, a.radius, ['--ddp-iterations=3']), ('budget8', 50, a.radius, ['--budget=8']),
             ('horizon100', 100, a.radius, []), ('horizon150', 150, a.radius, [])]
    seeds, offset, kicks = 3, 20, '0.5,1,1.5'
elif a.phase == 'followup':
    cases = [('cart100', 50, a.radius, ['--cart-weight=100']),
             ('cart200', 50, a.radius, ['--cart-weight=200']),
             ('cart500', 50, a.radius, ['--cart-weight=500']),
             ('cart200-force20', 50, a.radius, ['--cart-weight=200', '--force=20']),
             ('cart200-noise2.5', 50, a.radius, ['--cart-weight=200', '--noise=2.5']),
             ('cart200-lambda1', 50, a.radius, ['--cart-weight=200', '--lambda=1']),
             ('cart200-ddp3', 50, a.radius, ['--cart-weight=200', '--ddp-iterations=3'])]
    seeds, offset, kicks = 3, 100, '0.25,0.5,0.75,1'
else:
    cases = [('default-r4', 50, 4.0, []),
             ('cart200-force20-r4.5', 50, 4.5, ['--cart-weight=200', '--force=20']),
             ('no-refresh-control', 50, 1e6, ['--scenarios=nominal'])]
    seeds, offset, kicks = 30, 5000, '0.25,0.5,0.75,1,1.25,1.5'
manifest = []
for name, horizon, radius, extra in cases:
    csv_path = a.out/f'{name}.csv'
    trace_path = a.out/f'{name}-trace.csv' if a.phase == 'evaluation' else None
    cmd = [str(binaries[horizon]), str(seeds), '300', str(csv_path), str(radius), str(trace_path) if trace_path else '-', str(offset), kicks,
           '--diagnostics', *extra]
    manifest.append(dict(name=name, horizon=horizon, radius=radius, seeds=list(range(offset, offset+seeds)), command=cmd))
    (a.out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    with (a.out/f'{name}.log').open('w') as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
    if trace_path:
        with trace_path.open('rb') as source, gzip.open(str(trace_path)+'.gz', 'wb') as dest:
            shutil.copyfileobj(source, dest)
        trace_path.unlink()
    print(f'{name} finished', flush=True)
print('all cases finished', flush=True)
