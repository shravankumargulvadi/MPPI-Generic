"""Run exact submitted Dubins benchmark plus fixed attribution ablations."""
import argparse
import gzip
import json
import shutil
import subprocess
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--build',type=Path,default=Path('/validation/build'))
p.add_argument('--out',type=Path,required=True)
a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
cases=[('submitted',a.build/'submitted_heading_benchmark',[]),
       ('rollout-only',a.build/'mppi/examples/adaptive_rmppi_heading_benchmark',['--plant-feedback=0']),
       ('plant-only',a.build/'mppi/examples/adaptive_rmppi_heading_benchmark',['--rollout-budget=0']),
       ('neither',a.build/'mppi/examples/adaptive_rmppi_heading_benchmark',['--plant-feedback=0','--rollout-budget=0']),
       ('reuse-ddp-baseline',a.build/'mppi/examples/adaptive_rmppi_heading_benchmark',['--stock-ddp-action=1','--rollout-budget=0'])]
manifest=[]
for name,binary,extra in cases:
 trace=a.out/f'{name}-trace.csv'
 cmd=[str(binary),'30','300',str(a.out/f'{name}.csv'),'0.5',str(trace),'1000','1,2,2.5,3',*extra]
 manifest.append(dict(case=name,command=cmd,seeds=list(range(1000,1030))))
 (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 with (a.out/f'{name}.log').open('w') as f:
  subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
 with trace.open('rb') as src,gzip.open(str(trace)+'.gz','wb') as dst:shutil.copyfileobj(src,dst)
 trace.unlink()
 print(name+' finished',flush=True)
