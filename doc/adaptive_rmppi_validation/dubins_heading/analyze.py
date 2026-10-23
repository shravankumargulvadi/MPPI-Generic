"""Paired Dubins heading benchmark and mechanism-ablation analysis (NumPy)."""
import argparse
import csv
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('directory',type=Path)
p.add_argument('--plot',action='store_true')
a=p.parse_args(); root=a.directory
rng=np.random.default_rng(20261023)

def ci(values):
 v=np.asarray(values,dtype=float)
 if not len(v):return None
 b=v[rng.integers(0,len(v),(10000,len(v)))].mean(axis=1)
 return [float(v.mean()),float(np.quantile(b,.025)),float(np.quantile(b,.975))]

def avg(rows,key):return float(np.mean([float(r.get(key,0)) for r in rows]))

case_rows={}
results=[]
lines=['| Case | Kick (rad) | Recovery stock/adaptive | Mean recovery time, both-success pairs (s) | Cost stock/adaptive | p95 stock/adaptive (ms) | Plant solves/trial |',
       '| --- | --- | --- | --- | --- | --- | --- |']
for path in sorted((root/'cases').glob('*.csv')):
 groups=defaultdict(lambda:defaultdict(dict))
 case_rows[path.stem]=list(csv.DictReader(path.open()))
 for row in case_rows[path.stem]:
  assert None not in row
  for key,value in row.items():
   if key not in ('scenario','mode'):assert np.isfinite(float(value)),(path,key,value)
  group=groups[(row['scenario'],float(row['severity']))][row['mode']]
  seed=int(row['seed']); assert seed not in group
  group[seed]=row
 for (scenario,severity),modes in sorted(groups.items()):
  seeds=sorted(modes['stock'])
  assert seeds==list(range(1000,1030)) and seeds==sorted(modes['adaptive_v0'])
  s=[modes['stock'][k] for k in seeds]; t=[modes['adaptive_v0'][k] for k in seeds]
  item=dict(case=path.stem,scenario=scenario,severity=severity,trials=len(seeds),modes={})
  for mode,rs in [('stock',s),('adaptive_v0',t)]:
   item['modes'][mode]=dict(successes=sum(int(r['success']) for r in rs),failed_constraints=sum(int(r['failed_constraint']) for r in rs),
    deadline_misses=sum(int(r['deadline_misses']) for r in rs),max_latency_ms=max(float(r['latency_max_ms']) for r in rs),
    **{key:avg(rs,key) for key in ['recovery_time_s','peak_error','cumulative_cost','latency_p95_ms','refreshes_mean',
           'budget_hit_fraction','plant_refreshes','plant_refresh_failures','reused_ddp_actions']})
  both=[(x,y) for x,y in zip(s,t) if int(x['success']) and int(y['success'])]
  complete=[(x,y) for x,y in zip(s,t) if not int(x['failed_constraint']) and not int(y['failed_constraint'])]
  item['paired_diff_ci95']={key:ci([float(y[key])-float(x[key]) for x,y in zip(s,t)]) for key in ['success','peak_error','latency_p95_ms']}
  item['complete_pairs']=len(complete)
  item['complete_cost_diff_ci95']=ci([float(y['cumulative_cost'])-float(x['cumulative_cost']) for x,y in complete])
  item['both_success_pairs']=len(both)
  item['both_success_recovery_diff_ci95']=ci([float(y['recovery_time_s'])-float(x['recovery_time_s']) for x,y in both])
  item['adaptive_only_successes']=sum(not int(x['success']) and int(y['success']) for x,y in zip(s,t))
  item['stock_only_successes']=sum(int(x['success']) and not int(y['success']) for x,y in zip(s,t))
  discordant=item['adaptive_only_successes']+item['stock_only_successes']
  tail=min(item['adaptive_only_successes'],item['stock_only_successes'])
  item['exact_paired_success_p_two_sided']=min(1.0,2*sum(math.comb(discordant,k) for k in range(tail+1))/(2**discordant)) if discordant else 1.0
  rec_s=avg([x for x,y in both],'recovery_time_s') if both else -1
  rec_t=avg([y for x,y in both],'recovery_time_s') if both else -1
  ss,tt=item['modes']['stock'],item['modes']['adaptive_v0']
  lines.append(f"| {path.stem} | {severity:g} | {ss['successes']}/30 / {tt['successes']}/30 | {rec_s:.3f} / {rec_t:.3f} | "
   f"{ss['cumulative_cost']:.3f} / {tt['cumulative_cost']:.3f} | {ss['latency_p95_ms']:.3f} / {tt['latency_p95_ms']:.3f} | {tt['plant_refreshes']:.2f} |")
  results.append(item)
aggregates=[]
for case,rows in sorted(case_rows.items()):
 disturbed=[r for r in rows if r['scenario']!='nominal']
 by_seed=defaultdict(lambda:defaultdict(list))
 for r in disturbed:by_seed[int(r['seed'])][r['mode']].append(r)
 rate_diff=[]; cost_diff=[]; time_diff=[]
 for seed,modes in sorted(by_seed.items()):
  stock=modes['stock']; adaptive=modes['adaptive_v0']
  rate_diff.append(avg(adaptive,'success')-avg(stock,'success'))
  cost_diff.append(avg(adaptive,'cumulative_cost')-avg(stock,'cumulative_cost'))
  time_diff.append(avg(adaptive,'latency_p95_ms')-avg(stock,'latency_p95_ms'))
 item=dict(case=case,seeds=len(by_seed),disturbed_trials_per_mode=len(disturbed)//2,
           stock_successes=sum(int(r['success']) for r in disturbed if r['mode']=='stock'),
           adaptive_successes=sum(int(r['success']) for r in disturbed if r['mode']=='adaptive_v0'),
           success_rate_diff_ci95_seed_blocked=ci(rate_diff),cost_diff_ci95_seed_blocked=ci(cost_diff),
           planning_p95_diff_ci95_seed_blocked=ci(time_diff),
           failed_constraint_trials=sum(int(r['failed_constraint']) for r in disturbed))
 aggregates.append(item)
(root/'aggregate_summary.json').write_text(json.dumps(aggregates,indent=2)+'\n')
(root/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
(root/'table.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))

# Summarize clipped actuation and replay all trace errors against the declared recovery norm.
trace_summary=[]
plot_traces=defaultdict(list)
for path in sorted((root/'cases').glob('*-trace.csv.gz')):
 current=None; modes=defaultdict(list)
 summary=dict(case=path.name.replace('-trace.csv.gz',''),pairs=0,max_applied_control_difference=0.0,
              matched_command_pairs_at_1e_5=0,max_trace_error_reconstruction_difference=0.0,plant_failure_events=0)
 def flush():
  if not modes:return
  ss,tt=modes['stock'],modes['adaptive_v0']
  n=min(len(ss),len(tt)); summary['pairs']+=1
  delta=float(np.max(np.abs(np.asarray(ss[:n])[:,1:3]-np.asarray(tt[:n])[:,1:3])))
  summary['max_applied_control_difference']=max(summary['max_applied_control_difference'],delta)
  summary['matched_command_pairs_at_1e_5']+=delta<=1e-5 and len(ss)==len(tt)
 with gzip.open(path,'rt') as f:
  for row in csv.DictReader(f):
   assert None not in row
   key=(row['scenario'],row['severity'],row['seed'])
   if current!=key:
    flush(); modes=defaultdict(list);current=key
   yaw=float(row['yaw']); lateral=float(row['y'])
   error=np.hypot(lateral,np.arctan2(np.sin(yaw),np.cos(yaw)))
   summary['max_trace_error_reconstruction_difference']=max(summary['max_trace_error_reconstruction_difference'],abs(error-float(row['error'])))
   u0=float(row.get('applied_vel',np.clip(float(row['vel']),.5,2)))
   u1=float(row.get('applied_yaw_rate',np.clip(float(row['yaw_rate']),-2,2)))
   assert .49999<=u0<=2.00001 and abs(u1)<=2.00001
   modes[row['mode']].append([float(row['error']),u0,u1])
   summary['plant_failure_events']+=int(row.get('plant_refresh_failures',0))
   if row['scenario']=='heading_kick' and float(row['severity'])==3:
    plot_traces[(summary['case'],row['mode'],int(row['seed']))].append((float(row['time_s']),float(row['error'])))
  flush()
 trace_summary.append(summary)
(root/'trace_summary.json').write_text(json.dumps(trace_summary,indent=2)+'\n')
print(json.dumps(trace_summary,indent=2))

if a.plot:
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(2,2,figsize=(12,8))
 submitted=sorted([r for r in results if r['case']=='submitted' and r['scenario']!='nominal'],key=lambda r:r['severity'])
 for mode,marker in [('stock','o'),('adaptive_v0','x')]:
  axes[0,0].plot([r['severity'] for r in submitted],[r['modes'][mode]['successes']/30 for r in submitted],marker=marker,label=mode)
 axes[0,0].set(xlabel='Heading kick (rad)',ylabel='Recovery fraction',title='Submitted controller comparison',ylim=(-.05,1.05));axes[0,0].legend()
 cases=['submitted','rollout-only','plant-only','neither','reuse-ddp-baseline']
 x=np.arange(len(cases))
 for j,mode in enumerate(['stock','adaptive_v0']):
  vals=[next(r for r in results if r['case']==c and r['severity']==3)['modes'][mode]['cumulative_cost'] for c in cases]
  axes[0,1].bar(x+(j-.5)*.35,vals,.35,label=mode)
 axes[0,1].set(xticks=x,xticklabels=['Submitted','Rollout only','Plant only','Neither','DDP reuse'],ylabel='Post-kick state cost',title='Largest kick: attribution controls');axes[0,1].legend()
 for case,mode,label in [('submitted','stock','stock'),('submitted','adaptive_v0','adaptive: both'),('plant-only','adaptive_v0','adaptive: plant only')]:
  rs=[v for (c,m,s),v in plot_traces.items() if c==case and m==mode]
  n=min(map(len,rs));time=np.array([v[0] for v in rs[0][:n]]);error=np.array([[v[1] for v in r[:n]] for r in rs])
  axes[1,0].plot(time,np.median(error,axis=0),label=label)
  axes[1,0].fill_between(time,np.quantile(error,.1,axis=0),np.quantile(error,.9,axis=0),alpha=.12)
 axes[1,0].axhline(.2,color='k',linestyle=':',label='recovery tolerance');axes[1,0].set(xlabel='Time (s)',ylabel='Recovery error',title='3 rad kick: median and 10-90%');axes[1,0].legend(fontsize=8)
 for j,mode in enumerate(['stock','adaptive_v0']):
  vals=[np.mean([r['modes'][mode]['latency_p95_ms'] for r in results if r['case']==c]) for c in cases]
  axes[1,1].bar(x+(j-.5)*.35,vals,.35,label=mode)
 axes[1,1].set(xticks=x,xticklabels=['Submitted','Rollout only','Plant only','Neither','DDP reuse'],ylabel='Mean trial p95 (ms)',title='Planning latency');axes[1,1].legend()
 fig.tight_layout();fig.savefig(root/'heading_study.png',dpi=160)
