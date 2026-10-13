from pathlib import Path
import hashlib,json
import numpy as np
from scipy.optimize import brentq
from world_models import WorldModel,KINDS,rollout,gradient_check

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'phase2';gradient_check()
manifest=json.loads((ROOT/'phase1/dataset_manifest.json').read_text())
for name,expected in manifest['files'].items():
    assert hashlib.sha256((ROOT/'phase1'/name).read_bytes()).hexdigest()==expected
r=json.loads((OUT/'results.json').read_text());diags=[];max_difference=0.
q=np.linspace(-np.pi,np.pi,4001);z=np.column_stack([q,np.zeros_like(q)])
for kind in KINDS:
 for seed in range(5):
    m=WorldModel.load(OUT/f'{kind}_{seed}.npz')
    a=np.load(ROOT/'phase1/test_near_above.npz');idx=np.argsort(a['initial_energy'])[16];initial=a['states'][idx:idx+1,0]
    coarse=rollout(m,initial,.02,800);fine=rollout(m,initial,.01,1600)[:,::2]
    difference=float(np.max(np.linalg.norm(coarse-fine,axis=-1)));max_difference=max(max_difference,difference)
    diag={'kind':kind,'seed':seed,'step_halving_max_state_difference':difference}
    if kind in ('force','force_zero_mean','hnn'):
        f=m.field(z)[:,1];diag['force_circulation']=float(np.trapezoid(f,q))
    if kind=='hnn':
        roots=[]
        for j in np.flatnonzero(f[:-1]*f[1:]<0):
            roots.append(brentq(lambda qq:m.field(np.array([[qq,0.]]))[0,1],q[j],q[j+1]))
        assert len(roots)>0
        vals=m.potential(np.array(roots));diag['critical_angles']=roots
        diag['critical_potential_range']=float(vals.max()-vals.min());vmax=float(vals.max())
        for row in r:
            if row['kind']==kind and row['seed']==seed:
                d=np.load(ROOT/'phase1'/f"{row['split']}.npz");zz=d['states'][:,0]
                ehat=.5*zz[:,1]**2+m.potential(zz[:,0])
                row['predicted_rotation_disagreement']=float(np.mean((ehat>vmax)!=(d['initial_energy']>2)))
                row['critical_points']=len(roots);row['critical_potential_range']=diag['critical_potential_range']
    diags.append(diag)
assert max_difference<1e-4,max_difference
assert all(np.isfinite(row['rmse_16']) for row in r)
(OUT/'results.json').write_text(json.dumps(r,indent=2))
import csv
fields=sorted(set().union(*(row.keys() for row in r)))
with (OUT/'results.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(r)
checks={'gradient_checks_passed':True,'dataset_hashes_passed':True,
 'step_halving_max_difference':max_difference,'diagnostics':diags,
 'scope':'Step-halving checks cover one fixed near-above initial state for all 25 models; not every test trajectory.'}
(OUT/'verification.json').write_text(json.dumps(checks,indent=2))
print(json.dumps({'step_halving_max_difference':max_difference,'all_checks_passed':True},indent=2))
for d in diags:
 if d['kind']=='hnn':print('HNN critical angles / barrier:',d['seed'],d['critical_angles'],d['critical_potential_range'])
