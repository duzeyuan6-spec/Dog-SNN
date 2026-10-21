"""Paired episode-cluster bootstrap of existing quality records; no model selection."""
import json,hashlib
from pathlib import Path
import numpy as np
from common import config,write_json
r=Path(config()['root']);metadata={d['path']:d for d in map(json.loads,(r/'data/frames.jsonl').read_text().splitlines())}
results={}
for folder in ['frozen_t1','decoder_layout_validation']:
 o=r/'runs'/folder;ds={k:json.loads((o/(k+'_quality.json')).read_text()) for k in ['baseline','42','43','44']};base=ds['baseline']['frames'];paths=[x['path'] for x in base];episodes=sorted({metadata[p]['episode'] for p in paths});ei={v:i for i,v in enumerate(episodes)};counts=np.zeros(len(episodes));sums={k:np.zeros((len(episodes),2)) for k in ['42','43','44']}
 for k in sums:assert [x['path'] for x in ds[k]['frames']]==paths
 for j,p in enumerate(paths):
  i=ei[metadata[p]['episode']];counts[i]+=1
  for k in sums:sums[k][i]+=[ds[k]['frames'][j]['psnr']-base[j]['psnr'],ds[k]['frames'][j]['lpips']-base[j]['lpips']]
 rng=np.random.default_rng(20261020);draw=rng.integers(len(episodes),size=(10000,len(episodes)));result={}
 for k,s in sums.items():
  samples=s[draw].sum(1)/counts[draw].sum(1)[:,None];ci=np.percentile(samples,[2.5,97.5],axis=0)
  result[k]={'mean_delta_psnr_lpips':(s.sum(0)/counts.sum()).tolist(),'episode_bootstrap95':ci.tolist(),'quality_noninferior':bool(ci[0,0]>=-.5 and ci[1,1]<=.02)}
 results[folder]={'episodes':len(episodes),'frames':len(paths),'results':result}
write_json(r/'runs/quality_uncertainty/summary.json',dict(results=results,bootstrap_seed=20261020,replicates=10000,unit='episode',stage1_complete=False))
print(json.dumps(results))
