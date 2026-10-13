"""Verify acquisition access, aliases, matched budgets and the frozen decision."""
import json,hashlib,platform
import numpy as np
from PIL import Image
from phase7_design import *
from run_phase7_control import fingerprint

records=json.loads((OUT/'acquisition_records.json').read_text());pilots=json.loads((OUT/'pilots.json').read_text())
assert len(records)==840 and len(pilots)==30 and len(list((OUT/'models').glob('*.npz')))==660
paths=list((OUT/'paths').glob('*.json'));assert len(paths)==210
seen=set();calls=0;maximum_residual=0
for path in paths:
    info=json.loads(path.read_text());data=np.load(path.with_suffix('.npz'))
    assert info['pilot_queries']==128 and info['additional_queries']==256 and info['unselected_candidate_labels_computed']==0
    idx=data['candidate_indices'];assert len(idx)==256 and len(set(idx.tolist()))==256 and idx.min()>=0 and idx.max()<1024
    assert data['labels'].shape==(256,2);meta=info['records'][0];source=np.load(OUT/'data'/f'{meta["plant"]}_{meta["world_seed"]}.npz')
    # Validate queried labels without evaluating any unqueried candidate.
    expected=np.array([plant_force(source['candidate_q'][i],meta['plant'])+noise_for_candidate(meta['world_seed'],int(i)) for i in idx])
    assert np.max(np.abs(expected-data['labels']))<1e-12
    calls+=len(idx)
    seen.update((meta['plant'],meta['world_seed'],int(i)) for i in idx)
    assert [r['budget'] for r in info['records']]==[0,16,64,256]
    for r in info['records']:
        assert r['inverse_check_max_error']<1e-7 if r['budget'] else True
        maximum_residual=max(maximum_residual,r.get('normal_equation_residual',0));assert r.get('normal_equation_residual',0)<1e-7
    # Every branch updates the same data-independent prior exactly once per label.
    L=basis_map(meta['kind']);B=blocks(source['pilot_states'][:,:2],L);H=P['prior_precision_force_whitened']*np.eye(L.shape[1])+np.einsum('nfi,nfj->ij',B,B)
    rhs=np.einsum('nfi,nf->i',B,source['pilot_labels']);Bq=blocks(source['candidate_q'][idx[:64]],L)
    H+=np.einsum('nfi,nfj->ij',Bq,Bq);rhs+=np.einsum('nfi,nf->i',Bq,data['labels'][:64])
    direct=posterior_model(meta['kind'],L,np.linalg.inv(H),rhs,meta['plant'])
    saved=Model.load(OUT/'models'/info['records'][2]['model_file']);assert np.max(np.abs(direct.C-saved.C))<1e-8
metrics=json.loads((OUT/'model_metrics.json').read_text());assert len(metrics)==660
assert all(r['global_force_vector_rms']>=r['force_floor']-1e-10 for r in metrics)
rows=json.loads((OUT/'control_results.json').read_text());assert len(rows)==1200
assert len({(r['model_file'],r['task']) for r in rows})==960
for r in rows:
    assert len(r['per_phase_momentum_rms'])==64 and len(r['per_phase_return'])==64
    assert np.isfinite(r['momentum_vector_rms_mean']) and np.isfinite(r['return_mean'])
    assert r['fingerprint']==fingerprint(r['model_file'],r['task'])
# Pilot aliases agree bit-for-bit across all method reports.
for plant in P['plants']:
    for kind in P['families']:
        for w in P['world_seeds']:
            for task in P['tasks']:
                rr=[r for r in rows if r['plant']==plant and r['kind']==kind and r['world_seed']==w and r['task']==task and r['budget']==0]
                assert len(rr)==5 and len({r['momentum_vector_rms_mean'] for r in rr})==1
refs=json.loads((OUT/'references.json').read_text());assert len(refs)==4
phase=np.load(OUT/'data/test_phases.npz')['q'];assert phase.shape==(64,2)
for plant in P['plants']:
    for w in P['world_seeds']:
        d=np.load(OUT/'data'/f'{plant}_{w}.npz');assert d['pilot_states'].shape==(128,4) and d['candidate_q'].shape==(1024,2)
        assert not (set(map(tuple,phase))&set(map(tuple,d['candidate_q'])))
        assert not (set(map(tuple,phase))&set(map(tuple,d['pilot_states'][:,:2])))
D=json.loads((OUT/'decision.json').read_text());assert len(D['cells'])==4
assert D['primary_numerical_gate_passed']==all(r['passed'] for r in D['cells'])
assert not D['primary_numerical_gate_passed'] and not D['confirmation_required']
assert Image.open(OUT/'phase7_acquisition.gif').n_frames==65
assert Image.open(OUT/'phase7_control_comparison.gif').n_frames==101
checks={'pilot_fits':30,'unique_models':660,'acquisition_paths':210,'acquisition_records_with_aliases':840,
        'unique_learned_model_task_evaluations':960,'privileged_reference_evaluations':4,'reported_control_rows_with_pilot_aliases':1200,
        'unique_pilot_force_labels':1280,'additional_query_calls_across_all_paths':calls,
        'unique_plant_seed_candidate_labels_observed':len(seen),'unselected_candidate_labels_computed_during_acquisition':0,
        'maximum_MAP_normal_equation_residual':maximum_residual,'primary_gate_passed':False,'confirmation_required':False,
        'new_SAC_training_runs':0,'acquisition_animation_frames':65,'physical_animation_frames':101,
        'protocol_sha256':hashlib.sha256((ROOT/'phase7_protocol.json').read_bytes()).hexdigest()}
(OUT/'verification.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
