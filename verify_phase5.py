"""Meaningful interface, derivative, penalty and independent integrator checks."""
from pathlib import Path
import json
import numpy as np
import torch
from scipy.integrate import solve_ivp
from coupled_models import *
from coupled_sac import Dynamics,Actor,Replay,obs
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'phase5';OUT.mkdir(exist_ok=True)
torch.set_num_threads(1)
rng=np.random.default_rng(71);q=rng.uniform(-np.pi,np.pi,(200,2));checks={}
h=1e-5
numeric=np.stack([(truth_potential(q+np.eye(2)[j]*h)-truth_potential(q-np.eye(2)[j]*h))/(2*h) for j in range(2)],1)
checks['true_gradient_max_error']=float(np.max(np.abs(numeric+truth_force(q))))
assert checks['true_gradient_max_error']<1e-8
for kind in FAMILIES:
    model=FourierForce.load(OUT/'misspecified'/f'{kind}_0.npz');z=rng.normal(size=(50,4));u=rng.uniform(-2,2,(50,2))
    npz=step_numpy(z,u,model);tz,r=Dynamics(model).step(torch.tensor(z,dtype=torch.float32),torch.tensor(u,dtype=torch.float32),wrap=False)
    err=float(np.max(np.abs(npz-tz.numpy())));checks[kind+'_numpy_torch_step_error']=err;assert err<2e-6
    dq=feature_derivatives(q)
    nd=np.stack([(features(q+np.eye(2)[j]*h)-features(q-np.eye(2)[j]*h))/(2*h) for j in range(2)],1)
    assert np.max(np.abs(dq-nd))<1e-8
    if kind=='hamiltonian':
        grad=np.stack([(model.potential(q+np.eye(2)[j]*h)-model.potential(q-np.eye(2)[j]*h))/(2*h) for j in range(2)],1)
        checks['learned_potential_gradient_max_error']=float(np.max(np.abs(grad+model.force(q))));assert checks['learned_potential_gradient_max_error']<1e-8
# Independent high-precision controlled endpoint and work balance.
z=np.array([.2,-.3,2.1,.7]);u=np.array([.4,-.2]);v=np.r_[z,0.]
def rhs(t,v):return np.r_[v[2:4],truth_force(v[:2])+u,np.dot(v[2:4],u)]
ref=solve_ivp(rhs,(0,1),v,method='DOP853',rtol=1e-12,atol=1e-13).y[:,-1]
x=z.copy()
for _ in range(20):x=step_numpy(x,u)
checks['dop853_endpoint_max_error']=float(np.max(np.abs(x-ref[:4])));assert checks['dop853_endpoint_max_error']<1e-7
checks['dop853_energy_work_residual']=float(abs(truth_energy(ref[:4])-truth_energy(z)-ref[4]));assert checks['dop853_energy_work_residual']<1e-9
# Analytic structural quadratic forms equal discrete full-torus quadrature.
a=np.linspace(-np.pi,np.pi,32,endpoint=False);grid=np.stack(np.meshgrid(a,a,indexing='ij'),-1).reshape(-1,2)
C=rng.normal(size=(M,2));m=FourierForce('force',C);stat=m.statistics()
checks['curl_quadrature_error']=float(abs(np.mean(m.curl(grid)**2)-stat['curl_rms']**2));assert checks['curl_quadrature_error']<1e-10
checks['mean_quadrature_error']=float(np.max(np.abs(m.force(grid).mean(0)-C[0])));assert checks['mean_quadrature_error']<1e-12
actor=Actor();action,lp=actor.sample(obs(torch.randn(30,4)));assert action.shape==(30,2) and lp.shape==(30,1)
assert torch.isfinite(lp).all() and (action.abs()<=2).all()
checks['actor_interface']='passed';(OUT/'interface_verification.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
