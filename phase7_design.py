"""Gaussian periodic-feature experimental design with exact rank-two gains.

All criteria are classical design adaptations. The spectral metric is a
first-order tracking metric; it is not a learned policy-value guarantee.
"""
from pathlib import Path
import json
import numpy as np
from phase6_models import K,M,features,gradient_map,structural_matrices,Model,plant_force,step_numpy,local_states
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'phase7';OUT.mkdir(exist_ok=True)
P=json.loads((ROOT/'phase7_protocol.json').read_text())

def basis_map(kind):
    if kind=='force':T=np.eye(2*M)
    elif kind=='hamiltonian':T=gradient_map()
    elif kind=='odd_force':
        n=len(K);ids=np.r_[np.arange(1+n,M),M+np.arange(1+n,M)];T=np.eye(2*M)[:,ids]
    else:raise ValueError(kind)
    R,_,_=structural_matrices();Rt=T.T@R@T;e,Q=np.linalg.eigh(Rt)
    L=T@((Q*(1/np.sqrt(e)))@Q.T)
    assert np.max(np.abs(L.T@R@L-np.eye(L.shape[1])))<1e-12
    return L

def blocks(q,L):
    x=features(np.asarray(q));z=np.zeros_like(x)
    return np.stack([np.c_[x,z],np.c_[z,x]],axis=1)@L

def spectral_metric(L,target):
    gain=P['feedback_gain'];w=1/(gain**2+(K@np.asarray(target))**2)
    diagonal=np.r_[1/gain**2,.5*w,.5*w];Q=np.diag(np.r_[diagonal,diagonal])
    W=L.T@Q@L
    return W

def occupancy_metric(L,pilot_q,target):
    time=np.linspace(0,20,401);q=pilot_q[:8,None,:]+time[None,:,None]*target
    B=blocks(q.reshape(-1,2),L);return np.einsum('nfi,nfj->ij',B,B)/len(B)

def normalize_metric(W):return W*(len(W)/np.trace(W))

def candidate_scores(inv,B,W=None):
    # U = B A^-1; S = B A^-1 B^T. Labels are not involved.
    U=B@inv;S=U@B.transpose(0,2,1)
    if W is None:return np.trace(S,axis1=1,axis2=2)
    T=(U@W)@U.transpose(0,2,1)
    a=1+S[:,0,0];d=1+S[:,1,1];b=S[:,0,1];det=a*d-b*b
    return (d*T[:,0,0]+a*T[:,1,1]-b*(T[:,0,1]+T[:,1,0]))/det

def posterior_update(inv,B):
    U=B@inv;V=np.linalg.solve(np.eye(2)+U@B.T,U)
    new=inv-U.T@V;return (new+new.T)/2

def posterior_model(kind,L,inv,rhs,plant):
    beta=inv@rhs;v=L@beta;eps=0 if plant=='canonical' else .06
    return Model(kind,v.reshape(2,M).T,eps)

def steady_response(C,q,target):
    gain=P['feedback_gain'];n=len(K);omega=K@np.asarray(target);den=gain**2+omega**2
    A=C[1:1+n];B=C[1+n:];cs=(gain*A-omega[:,None]*B)/den[:,None];sn=(omega[:,None]*A+gain*B)/den[:,None]
    t=np.asarray(q)@K.T;return C[0]/gain+np.cos(t)@cs+np.sin(t)@sn

def noise_for_candidate(seed,index):return np.random.default_rng(161000+seed*10000+index).normal(0,P['noise_std'],2)
class ForceOracle:
    def __init__(self,plant,seed,pool):self.plant=plant;self.seed=seed;self.pool=pool;self.queried=[]
    def query(self,index):
        assert index not in self.queried;self.queried.append(index)
        return plant_force(self.pool[index],self.plant)+noise_for_candidate(self.seed,index)
