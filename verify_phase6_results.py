"""Check selection, paired designs, theoretical floors, grids and saved budgets."""
import json,hashlib,platform
import numpy as np
import scipy,matplotlib,torch
from PIL import Image
from phase6_models import *
from plot_phase6 import prior_floor
from run_phase6_sac import fingerprint as sac_fingerprint
from run_phase6_controllers import fingerprint as controller_fingerprint

selected=json.loads((OUT/'selected_models.json').read_text());candidates=json.loads((OUT/'selection_candidates.json').read_text());pred=json.loads((OUT/'prediction_results.json').read_text())
assert len(selected)==162 and len(candidates)==1620 and len(pred)==486
keys=lambda r:(r['plant'],r['coverage'],r['kind'],r['world_seed'])
assert len({keys(r) for r in selected})==162
for r in selected:
    group=[x for x in candidates if keys(x)==keys(r)];score=min(x['validation_vector_mse'] for x in group)
    assert r['validation_vector_mse']==score
    assert r['normal_equation_residual']<1e-10
    assert r['training_global_count']==round(P['training_samples']*P['coverage'][r['coverage']])
    assert r['validation_global_count']==round(P['validation_samples']*P['coverage'][r['coverage']])
    path=OUT/r['plant']/f'{r["kind"]}_{r["coverage"]}_{r["world_seed"]}.npz'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==r['model_sha256']
for plant in P['plants']:
    initial=np.load(OUT/plant/'initial_states.npz')
    for split,band in P['test_energy_bands'].items():
        energy=plant_energy(initial[split],plant);assert len(energy)==32 and energy.min()>=band[0] and energy.max()<=band[1]
    for w in P['world_seeds']:
        d=np.load(OUT/plant/f'pools_{w}.npz')
        train=[];val=[]
        for scope,n in [('train',4096),('validation',1024)]:
            for dist in ['local','global']:
                z=d[f'{scope}_{dist}_states'];y=d[f'{scope}_{dist}_labels'];assert z.shape==(n,4) and y.shape==(n,2)
                noise=y-plant_force(z[:,:2],plant);assert .027<noise.std()<.033
                if dist=='local':
                    e=plant_energy(z,plant);assert e.min()>=.2 and e.max()<=1.2
                else:assert np.max(np.abs(z[:,2:]))==0
                (train if scope=='train' else val).extend(map(tuple,z))
        assert not (set(train)&set(val))
# Local translated data admit equivariance for full Fourier/gradient classes.
rng=np.random.default_rng(133000);q=rng.uniform(-np.pi,np.pi,(100,2));shift=P['plants']['shifted']['shift'];equiv=[]
for w in P['world_seeds']:
    for k in ['force','curl_only','curl_mean','hamiltonian']:
        a=Model.load(OUT/'misspecified'/f'{k}_local_{w}.npz');b=Model.load(OUT/'shifted'/f'{k}_local_{w}.npz')
        err=float(np.max(np.abs(a.force(q-shift)-b.force(q))));assert err<1e-6;equiv.append(err)
for r in pred:
    assert np.isfinite(r['global_force_vector_rms']) and np.isfinite(r['rmse_t20'])
    assert r['global_force_vector_rms']>=prior_floor(r['plant'],r['kind'].startswith('odd_'))-1e-9
controls=json.loads((OUT/'controllers/results.json').read_text());assert len(controls)==220
assert len({(r['plant'],r['kind'],r['coverage'],r['world_seed'],r['controller']) for r in controls})==220
for r in controls:
    assert r['fingerprint']==controller_fingerprint(r['plant'],r['kind'],r['coverage'],r['world_seed'])
    assert r['mpc_maximum_cost_increase']<=1e-8
    assert np.isfinite(r['return_mean']) and np.isfinite(r['momentum_vector_rms_mean'])
    assert len(r['per_episode_return'])==32
manifest=json.loads((OUT/'sac/run_manifest.json').read_text());rows=json.loads((OUT/'sac/results.json').read_text())
assert len(manifest)==57 and len(rows)==171 and len(list((OUT/'sac').glob('*.pt')))==57
assert len({r['reset_sha256'] for r in manifest})==1
assert all(r['updates']==7492 and r['transitions']==16000 for r in manifest)
assert all(r['nonfinite_transitions']==0 for r in manifest)
assert all(r['fingerprint']==sac_fingerprint(r['kind'],r['coverage'],r['world_seed']) for r in manifest)
assert sum(r['true_training_queries'] for r in manifest)==48000
assert sum(r['model_training_transitions'] for r in manifest)==864000
expected={('true_only','reference',-1,p) for p in P['sac']['policy_seeds']}|{(k,c,w,p) for k in P['sac']['families'] for c in P['sac']['coverage'] for w in P['sac']['world_seeds'] for p in P['sac']['policy_seeds']}
assert {(r['kind'],r['coverage'],r['world_seed'],r['policy_seed']) for r in manifest}==expected
for name in ['coverage_prediction.gif','coverage_control.gif']:assert Image.open(OUT/name).n_frames==101
noise=json.loads((OUT/'noise_audit.json').read_text());assert len(noise)==9 and max(r['relative_monte_carlo_error'] for r in noise)<.1
verification={'selected_models':162,'validation_candidate_fits':1620,'prediction_evaluation_rows':486,'controller_configurations':220,
              'sac_training_runs':57,'sac_evaluation_rows':171,'sac_collected_transitions':912000,'true_sac_training_queries':48000,
              'model_sac_training_transitions':864000,'unique_plant_conditioned_training_force_labels':73728,
              'unique_plant_conditioned_validation_force_labels':18432,'no_nonfinite_training_transitions':True,
              'training_domain_resets':sum(r['domain_exits'] for r in manifest),'runs_with_domain_resets':sum(r['domain_exits']>0 for r in manifest),
              'maximum_local_translation_equivariance_error':max(equiv),'maximum_mpc_surrogate_cost_increase':max(r['mpc_maximum_cost_increase'] for r in controls),
              'maximum_noise_audit_relative_error':max(r['relative_monte_carlo_error'] for r in noise),'animation_frames_each':101,
              'protocol_sha256':hashlib.sha256((ROOT/'phase6_protocol.json').read_bytes()).hexdigest()}
(OUT/'verification.json').write_text(json.dumps(verification,indent=2))
(OUT/'runtime.json').write_text(json.dumps({'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'matplotlib':matplotlib.__version__,'torch':torch.__version__,'platform':platform.platform()},indent=2))
print(json.dumps(verification,indent=2))
