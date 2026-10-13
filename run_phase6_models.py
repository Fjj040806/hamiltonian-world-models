"""Paired fixed-budget coverage intervention and validation-only model selection."""
from pathlib import Path
import json,hashlib
import numpy as np
from phase6_models import *

def pools(plant,w,validation=False):
    n=P['validation_samples'] if validation else P['training_samples']
    rng=np.random.default_rng(81000+w+(10000 if validation else 0))
    local=local_states(n,rng,plant);glob=np.c_[rng.uniform(-np.pi,np.pi,(n,2)),np.zeros((n,2))]
    return {name:{'states':z,'labels':plant_force(z[:,:2],plant)+rng.normal(0,P['noise_std'],(n,2))} for name,z in [('local',local),('global',glob)]}
def mixture(pool,fraction):
    n=len(pool['local']['states']);ng=int(round(n*fraction));nl=n-ng
    return (np.concatenate([pool['local']['states'][:nl],pool['global']['states'][:ng]]),
            np.concatenate([pool['local']['labels'][:nl],pool['global']['labels'][:ng]]),ng)

def main():
    records=[];candidates=[]
    for plant in P['plants']:
        folder=OUT/plant;folder.mkdir(exist_ok=True)
        initial={s:band_states(P['test_states_per_band'],band,np.random.default_rng(101000+i),plant) for i,(s,band) in enumerate(P['test_energy_bands'].items())}
        np.savez_compressed(folder/'initial_states.npz',**initial)
        for w in P['world_seeds']:
            train=pools(plant,w);val=pools(plant,w,True)
            np.savez_compressed(folder/f'pools_{w}.npz',**{f'{scope}_{dist}_{key}':d[key] for scope,pool in [('train',train),('validation',val)] for dist,d in pool.items() for key in ['states','labels']})
            for cov,fraction in P['coverage'].items():
                z,y,ng=mixture(train,fraction);vz,vy,vng=mixture(val,fraction)
                for kind in P['families']:
                    best=None
                    for ridge in P['ridge_grid']:
                        pen=P['structural_penalty_grid'] if kind in ('curl_only','curl_mean') else [1.]
                        for penalty in pen:
                            m,ch=fit_model(kind,z[:,:2],y,ridge,penalty,plant)
                            score=float(np.mean(np.sum((m.force(vz[:,:2])-vy)**2,axis=1)))
                            rec={'plant':plant,'coverage':cov,'world_seed':w,'kind':kind,'ridge':ridge,'penalty':penalty,'validation_vector_mse':score,**ch}
                            assert ch['normal_equation_residual']<1e-10
                            candidates.append(rec)
                            if best is None or score<best[0]:best=(score,m,rec)
                    _,m,rec=best;m.save(folder/f'{kind}_{cov}_{w}.npz')
                    records.append({**rec,**m.statistics(),'training_global_count':ng,'validation_global_count':vng,
                        'model_sha256':hashlib.sha256((folder/f'{kind}_{cov}_{w}.npz').read_bytes()).hexdigest()})
                print(f'Selected {plant}, coverage={cov}, world={w} using validation only',flush=True)
    (OUT/'selected_models.json').write_text(json.dumps(records,indent=2))
    (OUT/'selection_candidates.json').write_text(json.dumps(candidates,indent=2))
    (OUT/'selection_protocol_sha256.txt').write_text(hashlib.sha256((ROOT/'phase6_protocol.json').read_bytes()).hexdigest())
    print('Primary selected models:',len(records),'candidate fits:',len(candidates),flush=True)
if __name__=='__main__':main()
