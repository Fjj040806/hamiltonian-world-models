"""Fresh crossed SAC seeds for coverage endpoints and three fixed families."""
from pathlib import Path
import argparse,json,time,hashlib
import numpy as np
import torch
from coupled_sac import SAC,Replay,obs
from phase6_models import P,ROOT,OUT,Model
from phase6_control import Dynamics
S=P['sac'];RESULTS=OUT/'sac';RESULTS.mkdir(exist_ok=True);torch.set_num_threads(1)

def fingerprint(k,c,w):
    h=hashlib.sha256((ROOT/'phase6_protocol.json').read_bytes())
    for name in ['phase6_models.py','phase6_control.py','coupled_sac.py','run_phase6_sac.py']:h.update((ROOT/name).read_bytes())
    if k!='true_only':h.update((OUT/S['plant']/f'{k}_{c}_{w}.npz').read_bytes())
    return h.hexdigest()
@torch.no_grad()
def evaluate(agent,initial,record=False):
    true=Dynamics(plant=S['plant']);z=torch.tensor(initial,dtype=torch.float32);returns=torch.zeros(len(z));errors=[];states=[z.numpy().copy()];actions=[]
    for _ in range(400):
        u=agent.actor.sample(obs(z),True)[0];z,r=true.step(z,u);returns+=r[:,0];errors.append(z[:,2:]-torch.tensor(P['controller_target']))
        if record:states.append(z.numpy().copy());actions.append(u.numpy().copy())
    e=torch.stack(errors[-20:]);norm=e.square().sum(2).sqrt();rms=e.square().sum(2).mean(0).sqrt();success=(norm<.25).all(0)
    d={'return_mean':float(returns.mean()),'momentum_vector_rms_mean':float(rms.mean()),'success_rate':float(success.float().mean()),
       'per_episode_return':returns.tolist(),'per_episode_momentum_rms':rms.tolist(),'per_episode_success':success.tolist()}
    if record:d.update(states=np.array(states),actions=np.array(actions))
    return d

def run(k,c,w,p):
    name=f'{k}_{c}_w{w}_p{p}';torch.manual_seed(111000+p);rng=torch.Generator().manual_seed(112000+p)
    reset=torch.tensor(np.load(OUT/'sac_reset_states.npz')['states'],dtype=torch.float32);initial=np.load(OUT/S['plant']/'initial_states.npz')
    model=None if k=='true_only' else Model.load(OUT/S['plant']/f'{k}_{c}_{w}.npz');dynamics=Dynamics(model,S['plant'])
    agent=SAC();replay=Replay(20000);z=reset[torch.randint(len(reset),(8,),generator=rng)].clone();updates=exits=nonfinite=0;tick=time.time()
    for step in range(S['transitions']//8):
        with torch.no_grad():
            u=4*torch.rand((8,2))-2 if step<128 else agent.actor.sample(obs(z))[0]
            nz,r=dynamics.step(z,u);finite=torch.isfinite(nz).all(1)&torch.isfinite(r).all(1);nonfinite+=int((~finite).sum())
            if finite.any():replay.add(z[finite],u[finite],r[finite],nz[finite])
            outside=(~finite)|(nz[:,2:].abs()>10).any(1);exits+=int(outside.sum());z=nz
            if outside.any():z[outside]=reset[torch.randint(len(reset),(int(outside.sum()),),generator=rng)]
            if (step+1)%400==0:z=reset[torch.randint(len(reset),(8,),generator=rng)].clone()
        if step>=127:
            for _ in range(4):
                losses=agent.update(*replay.sample(S['batch']));updates+=1;assert np.isfinite(losses).all()
    assert updates==S['updates'];agent.save(RESULTS/f'{name}.pt')
    rows=[{'kind':k,'coverage':c,'world_seed':w,'policy_seed':p,'split':split,**evaluate(agent,initial[split])} for split in initial.files]
    tr=evaluate(agent,initial['rotation'][:1],True);np.savez_compressed(RESULTS/f'{name}_trajectory.npz',states=tr['states'],actions=tr['actions'])
    result={'evaluations':rows,'updates':updates,'transitions':S['transitions'],'true_training_queries':S['transitions'] if k=='true_only' else 0,
       'model_training_transitions':0 if k=='true_only' else S['transitions'],'domain_exits':exits,'nonfinite_transitions':nonfinite,
       'fingerprint':fingerprint(k,c,w),'reset_sha256':hashlib.sha256((OUT/'sac_reset_states.npz').read_bytes()).hexdigest(),'elapsed_seconds':time.time()-tick}
    (RESULTS/f'{name}.json').write_text(json.dumps(result,indent=2));e=rows[-1]
    print(f'SAC {name}: RMS {e["momentum_vector_rms_mean"]:.3f}; success {e["success_rate"]:.2f}; exits {exits}; {time.time()-tick:.1f}s',flush=True)

def main(workers):
    rp=OUT/'sac_reset_states.npz'
    if not rp.exists():
        rng=np.random.default_rng(113000);np.savez_compressed(rp,states=np.c_[rng.uniform(-.8,.8,(1024,2)),rng.uniform(-.4,.4,(1024,2))])
    jobs=[('true_only','reference',-1,p) for p in S['policy_seeds']]+[(k,c,w,p) for c in S['coverage'] for k in S['families'] for w in S['world_seeds'] for p in S['policy_seeds']]
    pending=[]
    for k,c,w,p in jobs:
        rec=RESULTS/f'{k}_{c}_w{w}_p{p}.json'
        if not rec.exists() or json.loads(rec.read_text())['fingerprint']!=fingerprint(k,c,w) or not rec.with_suffix('.pt').exists():pending.append((k,c,w,p))
    if workers==1:
        for j in pending:run(*j)
    else:
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            futures=[pool.submit(run,*j) for j in pending]
            for f in futures:f.result()
    rows=[];manifest=[]
    for k,c,w,p in jobs:
        r=json.loads((RESULTS/f'{k}_{c}_w{w}_p{p}.json').read_text());rows+=r['evaluations']
        manifest.append({'kind':k,'coverage':c,'world_seed':w,'policy_seed':p,**{key:r[key] for key in ['updates','transitions','true_training_queries','model_training_transitions','domain_exits','nonfinite_transitions','fingerprint','reset_sha256']}})
    (RESULTS/'results.json').write_text(json.dumps(rows,indent=2));(RESULTS/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=1);a=ap.parse_args();main(a.workers)
