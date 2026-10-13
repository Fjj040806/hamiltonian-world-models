"""Crossed-seed generative-oracle intervention with a fixed SAC update budget."""
from pathlib import Path
import argparse,json,time,hashlib
import numpy as np
import torch
from sac import SAC,Replay,Dynamics,obs
from world_models import WorldModel
from run_phase3 import TRAIN,INITIAL,SPLITS,evaluate

torch.set_num_threads(1);ROOT=Path(__file__).resolve().parent;OUT=ROOT/'phase4';OUT.mkdir(exist_ok=True)
P=json.loads((ROOT/'phase4_protocol.json').read_text())
def run(kind,world_seed,policy_seed,oracle_budget):
    name=f'{kind}_w{world_seed}_p{policy_seed}_b{oracle_budget}'
    torch.manual_seed(4000+policy_seed);rng=torch.Generator().manual_seed(9000+policy_seed)
    select=np.random.default_rng(45000+policy_seed)
    mask=np.zeros(16000,dtype=bool)
    if oracle_budget:mask[select.choice(16000,oracle_budget,replace=False)]=True
    mask=mask.reshape(2000,8);assert mask.sum()==oracle_budget
    agent=SAC();replay=Replay(20000);true=Dynamics()
    model=None if kind=='true_only' else WorldModel.load(ROOT/'phase2'/f'{kind}_{world_seed}.npz')
    imagined=Dynamics(model);z=TRAIN[torch.randint(len(TRAIN),(8,),generator=rng)].clone()
    age=0;queries=0;model_steps=0;updates=0;exits=0;nonfinite=0;history=[];bins=np.zeros(5,dtype=int);tick=time.time()
    for step in range(2000):
        with torch.no_grad():
            u=4*torch.rand((8,1),generator=rng)-2 if step<128 else agent.actor.sample(obs(z))[0]
            truth=torch.tensor(mask[step]);nz=torch.empty_like(z);reward=torch.empty((8,1))
            if truth.any():nz[truth],reward[truth]=true.step(z[truth],u[truth]);queries+=int(truth.sum())
            if (~truth).any():nz[~truth],reward[~truth]=imagined.step(z[~truth],u[~truth]);model_steps+=int((~truth).sum())
            finite=torch.isfinite(nz).all(1)&torch.isfinite(reward).all(1);nonfinite+=int((~finite).sum())
            if finite.any():replay.add(z[finite],u[finite],reward[finite],nz[finite])
            outside=(~finite)|(nz[:,1].abs()>10);exits+=int(outside.sum())
            if finite.any():
                zz=nz[finite];H=(.5*zz[:,1]**2+1-torch.cos(zz[:,0])).numpy()
                bins+=np.histogram(H,[0,1.2,1.8,2.2,3.,np.inf])[0]
            z=nz
            if outside.any():z[outside]=TRAIN[torch.randint(len(TRAIN),(int(outside.sum()),),generator=rng)]
            age+=1
            if age==400:z=TRAIN[torch.randint(len(TRAIN),(8,),generator=rng)].clone();age=0
        if step>=127:
            assert replay.n>=256
            for _ in range(4):
                losses=agent.update(*replay.sample(256));updates+=1
                assert np.isfinite(losses).all(),name
        if (step+1)*8%4000==0:
            e=evaluate(agent,INITIAL['validation'][:8]);history.append({'transitions':(step+1)*8,'return':e['return_mean'],'success':e['success_rate']})
    assert queries==oracle_budget and model_steps+queries==16000 and updates==7492
    agent.save(OUT/f'{name}.pt');rows=[]
    for split in SPLITS:
        e=evaluate(agent,INITIAL[split]);rows.append({'kind':kind,'world_seed':world_seed,'policy_seed':policy_seed,
          'oracle_budget':oracle_budget,'split':split,**e})
    ex=int(np.argsort(np.load(ROOT/'phase1/test_near_above.npz')['initial_energy'])[16])
    trajectory=evaluate(agent,INITIAL['test_near_above'][ex:ex+1],record=True)
    np.savez_compressed(OUT/f'{name}_trajectory.npz',states=trajectory['states'],actions=trajectory['actions'])
    result={'name':name,'evaluations':rows,'history':history,'true_training_queries':queries,
            'model_training_transitions':model_steps,'updates':updates,'domain_exits':exits,'nonfinite_transitions':nonfinite,
            'hybrid_training_state_energy_bins':bins.tolist(),'selection_mask_sha256':hashlib.sha256(mask.tobytes()).hexdigest(),
            'elapsed_seconds':time.time()-tick,'protocol_sha256':hashlib.sha256((ROOT/'phase4_protocol.json').read_bytes()).hexdigest()}
    (OUT/f'{name}.json').write_text(json.dumps(result,indent=2))
    above=next(r for r in rows if r['split']=='test_near_above')
    print(f"{name}: above return {above['return_mean']:.3f}; success {above['success_rate']:.2f}; exits {exits}; {time.time()-tick:.1f}s",flush=True)
    return result

def schedule():
    jobs=[(k,w,s,b) for k in P['model_families'] for b in [0,1000] for w in P['world_model_seeds'] for s in P['policy_seeds']]
    return [('true_only',-1,s,16000) for s in P['policy_seeds']]+jobs

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=1);ap.add_argument('--single',action='store_true');args=ap.parse_args()
    jobs=schedule();pending=[]
    fingerprint=hashlib.sha256((ROOT/'phase4_protocol.json').read_bytes()).hexdigest()
    for job in jobs:
        k,w,s,b=job;f=OUT/f'{k}_w{w}_p{s}_b{b}.json'
        if not f.exists() or json.loads(f.read_text())['protocol_sha256']!=fingerprint:pending.append(job)
    if args.single:run(*pending[0])
    elif args.workers==1:
        for job in pending:run(*job)
    else:
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            futures=[pool.submit(run,*job) for job in pending]
            for f in futures:f.result()
    if not args.single:
        runs=[];rows=[]
        for k,w,s,b in jobs:
            d=json.loads((OUT/f'{k}_w{w}_p{s}_b{b}.json').read_text());runs.append(d);rows.extend(d['evaluations'])
        (OUT/'results.json').write_text(json.dumps(rows,indent=2))
        (OUT/'run_manifest.json').write_text(json.dumps({'unique_runs':len(runs),'total_true_training_queries':sum(d['true_training_queries'] for d in runs),
          'total_model_training_transitions':sum(d['model_training_transitions'] for d in runs),
          'updates_per_run':7492,'domain_exits':sum(d['domain_exits'] for d in runs),'protocol_sha256':fingerprint},indent=2))
