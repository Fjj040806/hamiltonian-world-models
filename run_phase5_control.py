"""Crossed-seed model-only cyclic SAC, with unique true-only references."""
from pathlib import Path
import argparse,json,time,hashlib
import numpy as np
import torch
from coupled_models import FourierForce
from coupled_sac import SAC,Replay,Dynamics,obs
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'phase5/control';OUT.mkdir(exist_ok=True)
P=json.loads((ROOT/'phase5_protocol.json').read_text());EPS=P['plants'][P['control_plant']]
torch.set_num_threads(1)

def fingerprint(kind,w):
    h=hashlib.sha256((ROOT/'phase5_protocol.json').read_bytes())
    if kind!='true_only':h.update((ROOT/'phase5'/P['control_plant']/f'{kind}_{w}.npz').read_bytes())
    return h.hexdigest()
@torch.no_grad()
def evaluate(agent,initial,record=False):
    d=Dynamics(epsilon=EPS);z=torch.tensor(initial,dtype=torch.float32);returns=torch.zeros(len(z));errors=[]
    states=[z.numpy().copy()];actions=[];winding=torch.zeros((len(z),2))
    for _ in range(P['evaluation_steps']):
        u=agent.actor.sample(obs(z),True)[0];nz,r=d.step(z,u);returns+=r[:,0]
        winding+=torch.atan2(torch.sin(nz[:,:2]-z[:,:2]),torch.cos(nz[:,:2]-z[:,:2]))/(2*np.pi)
        z=nz;errors.append(z[:,2:]-torch.tensor(P['control_target_momentum']))
        if record:states.append(z.numpy().copy());actions.append(u.numpy().copy())
    err=torch.stack(errors[-20:]);norm=err.square().sum(2).sqrt();success=(norm<.25).all(0)
    out={'return_mean':float(returns.mean()),'success_rate':float(success.float().mean()),
         'momentum_vector_rms_mean':float(err.square().sum(2).mean(0).sqrt().mean()),
         'winding_mean':winding.mean(0).tolist(),'per_episode_return':returns.tolist(),
         'per_episode_momentum_rms':err.square().sum(2).mean(0).sqrt().tolist(),'per_episode_success':success.tolist()}
    if record:out.update(states=np.array(states),actions=np.array(actions))
    return out

def run(kind,w,p):
    name=f'{kind}_w{w}_p{p}';torch.manual_seed(61000+p);reset_rng=torch.Generator().manual_seed(62000+p)
    reset=torch.tensor(np.load(ROOT/'phase5/policy_reset_states.npz')['states'],dtype=torch.float32)
    initial=np.load(ROOT/'phase5'/P['control_plant']/'initial_states.npz')
    model=None if kind=='true_only' else FourierForce.load(ROOT/'phase5'/P['control_plant']/f'{kind}_{w}.npz')
    dynamics=Dynamics(model,EPS);agent=SAC();replay=Replay(20000)
    z=reset[torch.randint(len(reset),(8,),generator=reset_rng)].clone();updates=exits=nonfinite=0;history=[];tick=time.time()
    for step in range(P['training_transitions_per_policy']//8):
        with torch.no_grad():
            u=4*torch.rand((8,2))-2 if step<128 else agent.actor.sample(obs(z))[0]
            nz,r=dynamics.step(z,u);finite=torch.isfinite(nz).all(1)&torch.isfinite(r).all(1)
            nonfinite+=int((~finite).sum())
            if finite.any():replay.add(z[finite],u[finite],r[finite],nz[finite])
            outside=(~finite)|(nz[:,2:].abs()>10).any(1);exits+=int(outside.sum());z=nz
            if outside.any():z[outside]=reset[torch.randint(len(reset),(int(outside.sum()),),generator=reset_rng)]
            if (step+1)%400==0:z=reset[torch.randint(len(reset),(8,),generator=reset_rng)].clone()
        if step>=127:
            for _ in range(4):
                losses=agent.update(*replay.sample(P['batch_size']));updates+=1;assert np.isfinite(losses).all()
        if (step+1)%500==0:
            history.append({'transitions':(step+1)*8,'critic_loss':losses[0],'actor_loss':losses[1]})
    assert updates==P['updates_per_policy']
    agent.save(OUT/f'{name}.pt')
    rows=[{'kind':kind,'world_seed':w,'policy_seed':p,'split':s,**evaluate(agent,initial[s])} for s in initial.files]
    # Preselected initial state index 0, not chosen by policy outcome.
    tr=evaluate(agent,initial['rotation'][:1],True)
    np.savez_compressed(OUT/f'{name}_trajectory.npz',states=tr['states'],actions=tr['actions'])
    result={'evaluations':rows,'history':history,'updates':updates,'collected_transitions':16000,
            'true_training_queries':16000 if kind=='true_only' else 0,'model_training_transitions':0 if kind=='true_only' else 16000,
            'domain_exits':exits,'nonfinite_transitions':nonfinite,'fingerprint':fingerprint(kind,w),'elapsed_seconds':time.time()-tick,
            'reset_state_sha256':hashlib.sha256((ROOT/'phase5/policy_reset_states.npz').read_bytes()).hexdigest()}
    (OUT/f'{name}.json').write_text(json.dumps(result,indent=2));e=rows[-1]
    print(f'{name}: rotation return {e["return_mean"]:.3f}; success {e["success_rate"]:.2f}; RMS {e["momentum_vector_rms_mean"]:.3f}; exits {exits}; {time.time()-tick:.1f}s',flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=1);args=ap.parse_args()
    reset_path=ROOT/'phase5/policy_reset_states.npz'
    if not reset_path.exists():
        rng=np.random.default_rng(60000);np.savez_compressed(reset_path,states=np.concatenate([rng.uniform(-.8,.8,(1024,2)),rng.uniform(-.4,.4,(1024,2))],axis=1))
    jobs=[('true_only',-1,p) for p in P['policy_seeds']]+[(k,w,p) for k in P['families'] for w in P['world_seeds'] for p in P['policy_seeds']]
    pending=[]
    for k,w,p in jobs:
        rec=OUT/f'{k}_w{w}_p{p}.json'
        if not rec.exists() or json.loads(rec.read_text())['fingerprint']!=fingerprint(k,w) or not rec.with_suffix('.pt').exists():pending.append((k,w,p))
    if args.workers==1:
        for j in pending:run(*j)
    else:
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            futures=[pool.submit(run,*j) for j in pending]
            for f in futures:f.result()
    rows=[];manifest=[]
    for k,w,p in jobs:
        rec=json.loads((OUT/f'{k}_w{w}_p{p}.json').read_text());rows+=rec['evaluations'];manifest.append({'kind':k,'world_seed':w,'policy_seed':p,**{a:rec[a] for a in ['updates','collected_transitions','true_training_queries','model_training_transitions','domain_exits','nonfinite_transitions','fingerprint','reset_state_sha256']}})
    (OUT/'results.json').write_text(json.dumps(rows,indent=2));(OUT/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
