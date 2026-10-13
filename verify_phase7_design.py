"""Independent exact-update and spectral-response checks before full runs."""
import json
import numpy as np
from phase7_design import *
rng=np.random.default_rng(170000);checks={}
for kind in P['families']:
    L=basis_map(kind);q=rng.uniform(-np.pi,np.pi,(120,2));B=blocks(q,L);n=L.shape[1]
    H=.05*np.eye(n)+np.einsum('nfi,nfj->ij',B[:40],B[:40]);inv=np.linalg.inv(H);W=spectral_metric(L,P['tasks']['forward'])
    score=candidate_scores(inv,B[40:],W)
    for idx in [0,20,60]:
        V=B[40+idx];direct=np.linalg.inv(H+V.T@V)
        exact=np.trace(W@(inv-direct));assert abs(score[idx]-exact)<1e-8
        updated=posterior_update(inv,V);assert np.max(np.abs(updated-direct))<1e-8
    checks[kind+'_rank_two_gain_max_error']=float(max(abs(score[i]-np.trace(W@(inv-np.linalg.inv(H+B[40+i].T@B[40+i])))) for i in [0,20,60]))
    beta=rng.normal(size=n);v=L@beta;C=v.reshape(2,M).T;target=np.array(P['tasks']['forward'])
    grid=np.linspace(-np.pi,np.pi,32,endpoint=False);phase=np.stack(np.meshgrid(grid,grid,indexing='ij'),-1).reshape(-1,2)
    eta=steady_response(C,phase,target)
    expected=float(beta@W@beta);actual=float(np.mean(np.sum(eta*eta,axis=1)))
    assert abs(actual-expected)<1e-10;checks[kind+'_spectral_parseval_error']=abs(actual-expected)
    h=1e-5;derivative=(steady_response(C,phase+target*h,target)-steady_response(C,phase-target*h,target))/(2*h)
    forcing=features(phase)@C
    err=float(np.max(np.abs(derivative+P['feedback_gain']*eta-forcing)));assert err<1e-8;checks[kind+'_filter_ode_residual']=err
# First-order validation in nonlinear phase evolution; no saturation in this audit.
L=basis_map('hamiltonian');beta=rng.normal(size=L.shape[1]);beta/=np.linalg.norm(beta);C=(L@beta).reshape(2,M).T
phase=np.random.default_rng(171000).uniform(-np.pi,np.pi,(64,2));target=np.array(P['tasks']['forward']);gain=P['feedback_gain']
errors=[]
for eps in [.001,.0005]:
    z=np.c_[phase,target+eps*steady_response(C,phase,target)]
    def rhs(s):return np.c_[s[:,2:],eps*(features(s[:,:2])@C)-gain*(s[:,2:]-target)]
    for _ in range(2000):
        h=.01;a=rhs(z);b=rhs(z+h*a/2);c=rhs(z+h*b/2);d=rhs(z+h*c);z+=h*(a+2*b+2*c+d)/6
    approximation=steady_response(C,phase+20*target,target)
    errors.append(float(np.max(np.abs((z[:,2:]-target)/eps-approximation))))
assert errors[1]<.65*errors[0]
checks['first_order_response_errors_at_eps_001_0005']=errors
checks['first_order_error_halving_ratio']=errors[1]/errors[0]
(OUT/'design_verification.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
