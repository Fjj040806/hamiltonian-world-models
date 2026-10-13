"""Add continuous endpoint diagnostics without changing trained policies."""
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import torch
from sac import Actor
from run_phase3 import evaluate,INITIAL,SPLITS

torch.set_num_threads(1);ROOT=Path(__file__).resolve().parent;OUT=ROOT/'phase3/calibrated'
rows=json.loads((OUT/'results.json').read_text());combined=torch.cat([INITIAL[s] for s in SPLITS])
jobs=sorted(set((r['kind'],r['seed'],r['horizon']) for r in rows));maxdifference=0.
for kind,seed,h in jobs:
    actor=Actor();checkpoint=torch.load(OUT/f'{kind}_seed{seed}_h{h}_u4.pt',weights_only=True)
    actor.load_state_dict(checkpoint['actor']);actor.eval();e=evaluate(SimpleNamespace(actor=actor),combined)
    for j,split in enumerate(SPLITS):
        row=next(r for r in rows if r['kind']==kind and r['seed']==seed and r['horizon']==h and r['split']==split)
        sl=slice(j*32,(j+1)*32);returns=e['per_episode_return'][sl]
        maxdifference=max(maxdifference,float(np.max(np.abs(np.array(returns)-row['per_episode_return']))))
        row['final_angle_rms_mean']=float(np.mean(e['per_episode_angle_rms'][sl]))
        row['final_momentum_rms_mean']=float(np.mean(e['per_episode_momentum_rms'][sl]))
        row['per_episode_angle_rms']=e['per_episode_angle_rms'][sl]
        row['per_episode_momentum_rms']=e['per_episode_momentum_rms'][sl]
assert maxdifference<.005,maxdifference
(OUT/'results.json').write_text(json.dumps(rows,indent=2))
(OUT/'evaluation_reproducibility.json').write_text(json.dumps({'max_per_episode_return_batching_difference':maxdifference,
 'original_returns_retained':True,'new_diagnostics':'final one-time-unit angle and momentum RMS; no checkpoint selection'},indent=2))
print('Continuous endpoint diagnostics added. Maximum return reproduction difference:',maxdifference)
