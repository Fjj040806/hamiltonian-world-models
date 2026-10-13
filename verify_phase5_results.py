"""Verify frozen model/control grid, paired data, budgets and saved animations."""
from pathlib import Path
import json,hashlib
import numpy as np
from PIL import Image
from coupled_models import K,M,truth_energy,structural_matrices,FourierForce
from run_phase5_control import fingerprint
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'phase5';P=json.loads((ROOT/'phase5_protocol.json').read_text())
checks={};R,Cu,Me=structural_matrices()
assert len(K)==12 and M==25 and np.linalg.matrix_rank(Cu)==24 and np.linalg.matrix_rank(Cu+Me)==26
assert np.linalg.eigvalsh(R).min()==.5
checks['exact_gradient_subspace_dimension']=24
mr=json.loads((OUT/'model_results.json').read_text());assert len(mr)==120
mv=json.loads((OUT/'model_verification.json').read_text());assert len(mv)==30
assert all(x['normal_equation_residual']<1e-10 for x in mv)
checks['primary_world_models']=30;checks['world_model_evaluation_rows']=120
for plant,eps in P['plants'].items():
    for w in P['world_seeds']:
        d=np.load(OUT/plant/f'training_{w}.npz');assert d['states'].shape==(4096,4)
        e=truth_energy(d['states'],eps);assert e.min()>=.2 and e.max()<=1.2
        noise=d['force_labels']-d['true_force'];assert .028<noise.std()<.032
        for k in P['families']:assert (OUT/plant/f'{k}_{w}.npz').is_file()
    iz=np.load(OUT/plant/'initial_states.npz')
    for split,band in P['test_energy_bands'].items():
        e=truth_energy(iz[split],eps);assert len(e)==32 and e.min()>=band[0] and e.max()<=band[1]
records=json.loads((OUT/'control/run_manifest.json').read_text());rows=json.loads((OUT/'control/results.json').read_text())
assert len(records)==48 and len(rows)==192
expected={('true_only',-1,p) for p in P['policy_seeds']}|{(k,w,p) for k in P['families'] for w in P['world_seeds'] for p in P['policy_seeds']}
actual={(x['kind'],x['world_seed'],x['policy_seed']) for x in records};assert actual==expected
assert len(list((OUT/'control').glob('*.pt')))==48
assert len({x['reset_state_sha256'] for x in records})==1
assert all(x['updates']==7492 and x['collected_transitions']==16000 for x in records)
assert all(x['domain_exits']==0 and x['nonfinite_transitions']==0 for x in records)
assert all(x['fingerprint']==fingerprint(x['kind'],x['world_seed']) for x in records)
assert sum(x['true_training_queries'] for x in records)==48000
assert sum(x['model_training_transitions'] for x in records)==720000
for x in rows:
    assert len(x['per_episode_return'])==len(x['per_episode_momentum_rms'])==len(x['per_episode_success'])==32
    assert np.isfinite(x['return_mean']) and np.isfinite(x['momentum_vector_rms_mean'])
checks.update(policy_runs=48,policy_evaluation_rows=192,collected_policy_transitions=768000,true_policy_training_queries=48000,
              model_policy_training_transitions=720000,unique_force_labeled_snapshots=24576,no_domain_exits=True,paired_reset_pool=True)
sr=json.loads((OUT/'sensitivity_results.json').read_text());assert len(sr)==420
tr=json.loads((OUT/'tracking/results.json').read_text());assert len(tr)==64
checks.update(posthoc_sensitivity_fits=420,fixed_feedback_evaluation_rows=64)
for name in ['coupled_prediction_comparison.gif','coupled_control_comparison.gif']:
    im=Image.open(OUT/name);assert im.n_frames==101
checks['animation_frames_each']=101
le=json.loads((OUT/'learned_energy_verification.json').read_text());assert max(x['learned_energy_max_drift'] for x in le)<1e-8
checks['learned_energy_max_drift']=max(x['learned_energy_max_drift'] for x in le)
checks['protocol_sha256']=hashlib.sha256((ROOT/'phase5_protocol.json').read_bytes()).hexdigest()
(OUT/'verification.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
