"""Exploratory cyclic task with fixed query, sample and optimization budgets."""
from pathlib import Path
import argparse,json,time,hashlib
import numpy as np
import torch
from sac import SAC,Replay,Dynamics,obs
from world_models import WorldModel
from run_phase3 import TRAIN,INITIAL,SPLITS

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'phase4/rotation';OUT.mkdir(exist_ok=True)
P=json.loads((ROOT/'phase4_rotation_protocol.json').read_text());torch.set_num_threads(1)
class RotationDynamics(Dynamics):
    def step(self,z,u):
        reward=torch.zeros((len(z),1));h=.01
        cost=lambda zz:(zz[:,1]-2.5)**2+.01*u[:,0]**2
        for _ in range(5):
            old=z;k1=self.field(z,u);k2=self.field(z+h*k1/2,u)
            k3=self.field(z+h*k2/2,u);k4=self.field(z+h*k3,u)
            z=z+h*(k1+2*k2+2*k3+k4)/6
            reward-=h*(cost(old)+cost(z))[:,None]/2
        z=torch.stack([torch.atan2(torch.sin(z[:,0]),torch.cos(z[:,0])),z[:,1]],1)
        return z,reward
@torch.no_grad()
def evaluate(agent,initial,record=False):
    d=RotationDynamics();z=initial.clone();returns=torch.zeros(len(z));errors=[];states=[z.numpy().copy()];actions=[];winding=torch.zeros(len(z))
    for _ in range(400):
        u=agent.actor.sample(obs(z),deterministic=True)[0];nz,r=d.step(z,u);returns+=r[:,0]
        winding+=torch.atan2(torch.sin(nz[:,0]-z[:,0]),torch.cos(nz[:,0]-z[:,0]))/(2*np.pi)
        z=nz;errors.append(z[:,1]-2.5)
        if record:states.append(z.numpy().copy());actions.append(u.numpy().copy())
    err=torch.stack(errors[-20:]);success=(err.abs()<.15).all(0)
    out={'return_mean':float(returns.mean()),'success_rate':float(success.float().mean()),
         'momentum_rms_mean':float(err.square().mean(0).sqrt().mean()),'winding_mean':float(winding.mean()),
         'per_episode_return':returns.tolist(),'per_episode_success':success.tolist()}
    if record:out.update(states=np.array(states),actions=np.array(actions))
    return out
def run(kind,w,budget):
    name=f'{kind}_w{w}_p0_b{budget}';torch.manual_seed(4000);rng=torch.Generator().manual_seed(9000)
    select=np.random.default_rng(45000);mask=np.zeros(16000,dtype=bool)
    if budget:mask[select.choice(16000,budget,replace=False)]=True
    mask=mask.reshape(2000,8);a=SAC();replay=Replay(20000);true=RotationDynamics()
    model=None if kind=='true_only' else WorldModel.load(ROOT/'phase2'/f'{kind}_{w}.npz')
    learned=RotationDynamics(model);z=TRAIN[torch.randint(len(TRAIN),(8,),generator=rng)].clone();queries=updates=exits=0;history=[];tick=time.time()
    for step in range(2000):
        with torch.no_grad():
            u=4*torch.rand((8,1),generator=rng)-2 if step<128 else a.actor.sample(obs(z))[0]
            real=torch.tensor(mask[step]);nz=torch.empty_like(z);r=torch.empty((8,1))
            if real.any():nz[real],r[real]=true.step(z[real],u[real]);queries+=int(real.sum())
            if (~real).any():nz[~real],r[~real]=learned.step(z[~real],u[~real])
            finite=torch.isfinite(nz).all(1)&torch.isfinite(r).all(1)
            if finite.any():replay.add(z[finite],u[finite],r[finite],nz[finite])
            outside=(~finite)|(nz[:,1].abs()>10);exits+=int(outside.sum());z=nz
            if outside.any():z[outside]=TRAIN[torch.randint(len(TRAIN),(int(outside.sum()),),generator=rng)]
            if (step+1)%400==0:z=TRAIN[torch.randint(len(TRAIN),(8,),generator=rng)].clone()
        if step>=127:
            for _ in range(4):
                losses=a.update(*replay.sample(256));updates+=1;assert np.isfinite(losses).all()
        if (step+1)*8%4000==0:
            e=evaluate(a,INITIAL['validation'][:8]);history.append({'transitions':(step+1)*8,'return':e['return_mean'],'success':e['success_rate']})
    assert queries==budget and updates==7492
    a.save(OUT/f'{name}.pt');rows=[{'kind':kind,'world_seed':w,'policy_seed':0,'oracle_budget':budget,'split':s,**evaluate(a,INITIAL[s])} for s in SPLITS]
    ex=int(np.argsort(np.load(ROOT/'phase1/test_near_above.npz')['initial_energy'])[16]);tr=evaluate(a,INITIAL['test_near_above'][ex:ex+1],record=True)
    np.savez_compressed(OUT/f'{name}_trajectory.npz',states=tr['states'],actions=tr['actions'])
    result={'evaluations':rows,'history':history,'true_training_queries':queries,'updates':updates,'domain_exits':exits,'elapsed_seconds':time.time()-tick,
            'protocol_sha256':hashlib.sha256((ROOT/'phase4_rotation_protocol.json').read_bytes()).hexdigest()}
    (OUT/f'{name}.json').write_text(json.dumps(result,indent=2));above=rows[3]
    print(f"rotation {name}: return {above['return_mean']:.3f}; success {above['success_rate']:.2f}; RMS {above['momentum_rms_mean']:.3f}; {time.time()-tick:.1f}s",flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=1);args=ap.parse_args()
    jobs=[('true_only',-1,16000)]+[(k,w,b) for k in P['models'] for b in P['oracle_budgets'] for w in P['world_seeds']]
    fingerprint=hashlib.sha256((ROOT/'phase4_rotation_protocol.json').read_bytes()).hexdigest()
    pending=[]
    for j in jobs:
        path=OUT/f'{j[0]}_w{j[1]}_p0_b{j[2]}.json'
        if not path.exists() or json.loads(path.read_text())['protocol_sha256']!=fingerprint:pending.append(j)
    if args.workers==1:
        for j in pending:run(*j)
    else:
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            futures=[pool.submit(run,*j) for j in pending]
            for f in futures:f.result()
    rows=[]
    for k,w,b in jobs:rows+=json.loads((OUT/f'{k}_w{w}_p0_b{b}.json').read_text())['evaluations']
    (OUT/'results.json').write_text(json.dumps(rows,indent=2))
