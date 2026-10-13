"""Phase 6 parity/coverage models; keeps previous-stage numerical code intact."""
from pathlib import Path
import json
import numpy as np
from coupled_models import (K,M,FourierForce,features,feature_derivatives,structural_matrices,fit,
                            truth_force,truth_potential,sample_states)
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'phase6';OUT.mkdir(exist_ok=True)
P=json.loads((ROOT/'phase6_protocol.json').read_text())

def plant_force(q,plant):
    p=P['plants'][plant];return truth_force(np.asarray(q)-p['shift'],p['epsilon'])
def plant_potential(q,plant):
    p=P['plants'][plant];return truth_potential(np.asarray(q)-p['shift'],p['epsilon'])
def plant_energy(z,plant):return .5*np.sum(z[...,2:]**2,axis=-1)+plant_potential(z[...,:2],plant)
def local_states(n,rng,plant):
    p=P['plants'][plant];z=sample_states(n,P['local_energy_band'],rng,p['epsilon']);z[:,:2]+=p['shift']
    z[:,:2]=np.arctan2(np.sin(z[:,:2]),np.cos(z[:,:2]));return z
def band_states(n,band,rng,plant):
    p=P['plants'][plant];z=sample_states(n,band,rng,p['epsilon']);z[:,:2]+=p['shift']
    z[:,:2]=np.arctan2(np.sin(z[:,:2]),np.cos(z[:,:2]));return z

def gradient_map():
    T=np.zeros((2*M,M-1));n=len(K)
    for j,k in enumerate(K):
        for d in range(2):T[d*M+1+n+j,j]=k[d];T[d*M+1+j,n+j]=-k[d]
    return T
class Model(FourierForce):
    def statistics(self):
        s=super().statistics();s['effective_parameters']={'force':50,'odd_force':24,'curl_only':50,'curl_mean':50,'hamiltonian':24,'odd_hamiltonian':12}[self.kind]
        s['parity_even_component_rms']=float(np.sqrt(np.sum(self.C[0]**2)+.5*np.sum(self.C[1:1+len(K)]**2)))
        return s
    def potential(self,q):
        if self.kind not in ('hamiltonian','odd_hamiltonian'):raise ValueError('No exact potential for this model')
        n=len(K);norm=(K*K).sum(1);a=(self.C[1+n:]*K).sum(1)/norm;b=-(self.C[1:1+n]*K).sum(1)/norm
        t=np.asarray(q)@K.T;return np.cos(t)@a+np.sin(t)@b
    @classmethod
    def load(cls,path):
        d=np.load(path);return cls(str(d['kind']),d['C'],float(d['epsilon']))
    def jacobian(self,q):return np.einsum('...dm,mf->...fd',feature_derivatives(q),self.C)

def fit_model(kind,q,y,ridge,penalty=1.,plant='misspecified'):
    eps=P['plants'][plant]['epsilon']
    if kind not in ('odd_force','odd_hamiltonian'):
        m,ch=fit(kind,q,y,ridge,penalty,penalty,eps)
        return Model(kind,m.C,eps),ch
    x=features(q);G=x.T@x/len(q);b=(x.T@y/len(q)).T.reshape(-1)
    R,_,_=structural_matrices();A=np.kron(np.eye(2),G)+ridge*R
    n=len(K)
    if kind=='odd_hamiltonian':T=gradient_map()[:,:n]
    else:
        ids=np.r_[np.arange(1+n,M),M+np.arange(1+n,M)];T=np.eye(2*M)[:,ids]
    theta=np.linalg.solve(T.T@A@T,T.T@b);v=T@theta
    m=Model(kind,v.reshape(2,M).T,eps)
    return m,{'normal_equation_residual':float(np.linalg.norm(T.T@(A@v-b))),
              'hard_gradient_equivalence_max_coefficient_error':None,
              'objective':float(np.mean(np.sum((m.force(q)-y)**2,axis=1))+ridge*v@R@v)}

def step_numpy(z,u,model=None,plant='misspecified',h=.01,substeps=5):
    z=np.asarray(z,dtype=float);u=np.asarray(u)
    def field(s):return np.concatenate([s[...,2:],(plant_force(s[...,:2],plant) if model is None else model.force(s[...,:2]))+u],axis=-1)
    for _ in range(substeps):
        a=field(z);b=field(z+h*a/2);c=field(z+h*b/2);d=field(z+h*c)
        z=z+h*(a+2*b+2*c+d)/6
    return z
