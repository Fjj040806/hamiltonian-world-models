"""Fixed feedback diagnostic: actual true-plant control, separate from SAC."""
from pathlib import Path
import json
import numpy as np
import torch
from coupled_models import FourierForce
from coupled_sac import Dynamics
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'phase5/tracking';OUT.mkdir(exist_ok=True)
P=json.loads((ROOT/'phase5_protocol.json').read_text());T=json.loads((ROOT/'phase5_tracking_protocol.json').read_text());torch.set_num_threads(1)
@torch.no_grad()
def main():
    initial=np.load(ROOT/'phase5/misspecified/initial_states.npz');true=Dynamics(epsilon=.06);target=torch.tensor([2.5,1.5]);rows=[]
    for k,w in [(k,w) for k in T['families'] for w in T['world_seeds']]+[('known_force',-1)]:
        m=None if k=='known_force' else FourierForce.load(ROOT/'phase5/misspecified'/f'{k}_{w}.npz');approx=Dynamics(m,epsilon=.06)
        for split in initial.files:
            z=torch.tensor(initial[split],dtype=torch.float32);errors=[];returns=torch.zeros(len(z));states=[z.numpy().copy()];actions=[]
            for i in range(400):
                u=(-approx.force(z[:,:2])-T['gain']*(z[:,2:]-target)).clamp(-2,2);z,r=true.step(z,u)
                returns+=r[:,0];errors.append(z[:,2:]-target)
                if split=='rotation':states.append(z.numpy().copy());actions.append(u.numpy().copy())
            err=torch.stack(errors[-20:]);norm=err.square().sum(2).sqrt()
            rows.append({'kind':k,'world_seed':w,'split':split,'return_mean':float(returns.mean()),
                         'momentum_vector_rms_mean':float(err.square().sum(2).mean(0).sqrt().mean()),
                         'success_rate':float((norm<.25).all(0).float().mean()),'per_episode_momentum_rms':err.square().sum(2).mean(0).sqrt().tolist()})
            if split=='rotation':np.savez_compressed(OUT/f'{k}_{w}_trajectory.npz',states=np.array(states)[:,0:1],actions=np.array(actions)[:,0:1])
    (OUT/'results.json').write_text(json.dumps(rows,indent=2))
    print(json.dumps({k:{key:float(np.mean([x[key] for x in rows if x['kind']==k and x['split']=='rotation'])) for key in ['momentum_vector_rms_mean','return_mean','success_rate']} for k in T['families']+['known_force']},indent=2))
if __name__=='__main__':main()
