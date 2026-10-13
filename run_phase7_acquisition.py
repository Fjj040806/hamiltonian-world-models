"""Fixed-budget query experiment. No true labels for unselected candidates."""
import argparse,json,hashlib,time
import numpy as np
from phase7_design import *

DATA=OUT/'data';PATHS=OUT/'paths';MODELS=OUT/'models'
for d in [DATA,PATHS,MODELS]:d.mkdir(exist_ok=True)

def prepare():
    for plant in P['plants']:
        for w in P['world_seeds']:
            z=local_states(P['pilot_force_labels'],np.random.default_rng(151000+w),plant)
            y=plant_force(z[:,:2],plant)+np.random.default_rng(152000+w).normal(0,P['noise_std'],(len(z),2))
            q=np.random.default_rng(153000+w).uniform(-np.pi,np.pi,(P['candidate_pool_size'],2))
            np.savez_compressed(DATA/f'{plant}_{w}.npz',pilot_states=z,pilot_labels=y,candidate_q=q)
    phases=np.random.default_rng(154000).uniform(-np.pi,np.pi,(P['test_phases'],2))
    np.savez_compressed(DATA/'test_phases.npz',q=phases)

def run(plant,kind,w,method,task):
    d=np.load(DATA/f'{plant}_{w}.npz');L=basis_map(kind);Bp=blocks(d['pilot_states'][:,:2],L);pool=d['candidate_q'];B=blocks(pool,L);n=L.shape[1]
    H=P['prior_precision_force_whitened']*np.eye(n)+np.einsum('nfi,nfj->ij',Bp,Bp)
    rhs=np.einsum('nfi,nf->i',Bp,d['pilot_labels']);inv=np.linalg.inv(H)
    oracle=ForceOracle(plant,w,pool);mask=np.ones(len(pool),bool)
    if method=='global_A':W=np.eye(n)
    elif method=='spectral_A':W=normalize_metric(spectral_metric(L,P['tasks'][task]))
    elif method=='occupancy_A':W=normalize_metric(occupancy_metric(L,d['pilot_states'][:,:2],np.array(P['tasks'][task])))
    else:W=None
    order=np.random.default_rng(155000+w).permutation(len(pool));records=[];labels=[];history=[];tick=time.time()
    stem=f'{plant}_{kind}_w{w}_{method}_{task}'
    pilot_name=f'pilot_{plant}_{kind}_w{w}.npz'
    # Identical pilot aliases are not separate fits. Parent process creates them.
    records.append({'plant':plant,'kind':kind,'world_seed':w,'method':method,'design_task':task,'budget':0,'model_file':pilot_name})
    for j in range(max(P['additional_query_budgets'])):
        if method=='random':idx=int(order[j]);score=None
        else:
            values=candidate_scores(inv,B,W);values[~mask]=-np.inf;idx=int(np.argmax(values));score=float(values[idx])
        y=oracle.query(idx);labels.append(y);mask[idx]=False;rhs+=B[idx].T@y;H+=B[idx].T@B[idx];inv=posterior_update(inv,B[idx])
        history.append({'query':j+1,'candidate_index':idx,'acquisition_score':score,
                        'global_posterior_variance':float(P['noise_std']**2*np.trace(inv))})
        if j+1 in P['additional_query_budgets']:
            direct=np.linalg.inv(H);err=float(np.max(np.abs(direct-inv)));assert err<1e-7
            name=f'{stem}_b{j+1}.npz';model=posterior_model(kind,L,inv,rhs,plant);model.save(MODELS/name)
            spec={t:float(P['noise_std']**2*np.trace(spectral_metric(L,p)@inv)) for t,p in P['tasks'].items()}
            records.append({'plant':plant,'kind':kind,'world_seed':w,'method':method,'design_task':task,'budget':j+1,'model_file':name,
                'global_posterior_variance':float(P['noise_std']**2*np.trace(inv)),
                'spectral_posterior_variance':spec,'inverse_check_max_error':err,'normal_equation_residual':float(np.linalg.norm(H@(inv@rhs)-rhs))})
    assert len(oracle.queried)==256 and len(set(oracle.queried))==256
    np.savez_compressed(PATHS/f'{stem}.npz',candidate_indices=np.array(oracle.queried),labels=np.array(labels),metric=np.eye(n) if W is None else W)
    result={'records':records,'history':history,'pilot_queries':len(d['pilot_labels']),'additional_queries':len(oracle.queried),
            'unselected_candidate_labels_computed':0,'elapsed_seconds':time.time()-tick}
    (PATHS/f'{stem}.json').write_text(json.dumps(result,indent=2));print(f'{stem}: 256 selected labels, {time.time()-tick:.1f}s',flush=True)
    return records

def main(workers):
    prepare();pilots=[]
    for plant in P['plants']:
        for kind in P['families']:
            for w in P['world_seeds']:
                d=np.load(DATA/f'{plant}_{w}.npz');L=basis_map(kind);B=blocks(d['pilot_states'][:,:2],L)
                H=P['prior_precision_force_whitened']*np.eye(L.shape[1])+np.einsum('nfi,nfj->ij',B,B)
                rhs=np.einsum('nfi,nf->i',B,d['pilot_labels']);model=posterior_model(kind,L,np.linalg.inv(H),rhs,plant)
                name=f'pilot_{plant}_{kind}_w{w}.npz';model.save(MODELS/name);pilots.append({'plant':plant,'kind':kind,'world_seed':w,'model_file':name})
    jobs=[(plant,k,w,m,t) for plant in P['plants'] for k in P['families'] for w in P['world_seeds'] for m in P['methods'] for t in (list(P['tasks']) if m in ('occupancy_A','spectral_A') else ['shared'])]
    if workers==1:groups=[run(*j) for j in jobs]
    else:
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            futures=[pool.submit(run,*j) for j in jobs];groups=[f.result() for f in futures]
    records=[r for group in groups for r in group];(OUT/'acquisition_records.json').write_text(json.dumps(records,indent=2));(OUT/'pilots.json').write_text(json.dumps(pilots,indent=2))
    (OUT/'protocol_sha256.txt').write_text(hashlib.sha256((ROOT/'phase7_protocol.json').read_bytes()).hexdigest())
    print('Acquisition paths:',len(jobs),'unique saved model fits:',len(list(MODELS.glob('*.npz'))),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=1);a=ap.parse_args();main(a.workers)
