"""Independent derivative and MPC-adjoint checks before outcome evaluation."""
import json
import numpy as np
import torch
from scipy.integrate import solve_ivp
from phase6_models import *
from phase6_control import Planner,Dynamics,true_jacobian
rng=np.random.default_rng(8123);q=rng.uniform(-np.pi,np.pi,(30,2));h=1e-5;checks={}
for plant in P['plants']:
    numeric=np.stack([(plant_force(q+np.eye(2)[j]*h,plant)-plant_force(q-np.eye(2)[j]*h,plant))/(2*h) for j in range(2)],axis=-1)
    err=np.max(np.abs(numeric-true_jacobian(q,plant)));assert err<1e-8;checks[plant+'_force_jacobian_error']=float(err)
    z=np.array([.3,-.1,2.2,.6]);u=np.array([.4,-.2])
    rhs=lambda t,s:np.r_[s[2:],plant_force(s[:2],plant)+u]
    ref=solve_ivp(rhs,(0,1),z,method='DOP853',rtol=1e-12,atol=1e-13).y[:,-1]
    a=z.copy()
    for _ in range(20):a=step_numpy(a,u,plant=plant)
    err=float(np.max(np.abs(a-ref)));assert err<1e-7;checks[plant+'_rk4_dop853_error']=err
for kind in P['families']:
    m=Model.load(OUT/'misspecified'/f'{kind}_local_10.npz')
    numeric=np.stack([(m.force(q+np.eye(2)[j]*h)-m.force(q-np.eye(2)[j]*h))/(2*h) for j in range(2)],axis=-1)
    err=float(np.max(np.abs(numeric-m.jacobian(q))));assert err<1e-8;checks[kind+'_jacobian_error']=err
    z=rng.normal(size=(5,4));u=rng.uniform(-2,2,(5,2));tz,_=Dynamics(m).step(torch.tensor(z,dtype=torch.float32),torch.tensor(u,dtype=torch.float32),wrap=False)
    err=float(np.max(np.abs(tz.numpy()-step_numpy(z,u,m))));assert err<3e-6;checks[kind+'_torch_numpy_error']=err
    planner=Planner(m);U=rng.uniform(-1,1,(planner.H,5,2));cost,g=planner.rollout(z,U,True)
    errors=[]
    for t,b,j in [(0,0,0),(5,2,1),(11,4,0),(3,1,1)]:
        plus=U.copy();minus=U.copy();plus[t,b,j]+=h;minus[t,b,j]-=h
        numeric=(planner.rollout(z,plus)[b]-planner.rollout(z,minus)[b])/(2*h)
        errors.append(abs(numeric-g[t,b,j]))
    assert max(errors)<1e-7;checks[kind+'_adjoint_gradient_error']=float(max(errors))
    action,diagnostics=planner.action(z);assert np.isfinite(action).all() and np.max(np.abs(action))<=2
    checks[kind+'_mpc_cost_increase']=diagnostics['maximum_cost_increase']
    if kind.startswith('odd_'):assert np.max(np.abs(m.force(q)+m.force(-q)))<1e-12
    if kind in ('hamiltonian','odd_hamiltonian'):
        grad=np.stack([(m.potential(q+np.eye(2)[j]*h)-m.potential(q-np.eye(2)[j]*h))/(2*h) for j in range(2)],axis=-1)
        assert np.max(np.abs(grad+m.force(q)))<1e-8
(OUT/'interface_verification.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
