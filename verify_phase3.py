"""Independent interface checks before expensive policy experiments."""
from pathlib import Path
import json
import numpy as np
import torch
from sac import Dynamics,SAC,Replay,obs
from world_models import WorldModel,KINDS
from simulator import PendulumEnv
torch.set_num_threads(1);ROOT=Path(__file__).resolve().parent
rng=np.random.default_rng(811);z=rng.uniform([-3.,-3.],[3.,3.],size=(40,2));u=rng.uniform(-2,2,size=(40,1))
errors={}
for kind in KINDS:
    m=WorldModel.load(ROOT/'phase2'/f'{kind}_0.npz');d=Dynamics(m)
    prediction=d.field(torch.tensor(z,dtype=torch.float32),torch.tensor(u,dtype=torch.float32)).numpy()
    errors[kind]=float(np.max(np.abs(prediction-m.field(z,u[:,0]))))
assert max(errors.values())<3e-6,errors
env=PendulumEnv();env.reset((.4,.7))
true=Dynamics();zz=torch.tensor([[.4,.7]])
for action in np.linspace(-1.5,1.5,25):
    o,r,_,_,_=env.step(float(action));zz,rr=true.step(zz,torch.tensor([[action]],dtype=torch.float32))
    assert abs(r-float(rr))<2e-6
angle_error=np.arctan2(np.sin(float(zz[0,0])-env.z[0]),np.cos(float(zz[0,0])-env.z[0]))
assert abs(angle_error)<2e-5 and abs(float(zz[0,1])-env.z[1])<2e-5
torch.manual_seed(83);a=SAC();replay=Replay(50)
zs=torch.tensor(z[:8],dtype=torch.float32);actions,lp=a.actor.sample(obs(zs))
assert (actions.abs()<=2).all() and torch.isfinite(lp).all()
nz,reward=true.step(zs,actions);replay.add(zs,actions,reward,nz)
before=[p.detach().clone() for p in a.actor.parameters()]
losses=a.update(*replay.sample(8));assert np.isfinite(losses).all()
assert any(not torch.equal(p,b) for p,b in zip(a.actor.parameters(),before))
checks={'torch_numpy_field_max_errors':errors,'true_numpy_torch_state_agreement_passed':True,
        'reward_agreement_passed':True,'action_and_logprob_checks_passed':True,
        'actor_update_and_finite_losses_passed':True}
(ROOT/'phase3/interface_checks.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
