"""Finite periodic feature models with exact population structural penalties.

This is a convex feature-head benchmark, not a trainable-feature deep HNN.
All models know qdot=p, unit mass and the unit torque port.
"""
import numpy as np
from pathlib import Path

K=np.array([(a,b) for a in range(3) for b in range(-2,3)
            if (a>0 or (a==0 and b>0))],dtype=float)
M=1+2*len(K)
FAMILIES=('force','mean_only','curl_only','curl_mean','hamiltonian')

def truth_potential(q,epsilon=.06):
    q=np.asarray(q);d=q[...,0]-q[...,1];h=3*q[...,0]-q[...,1]
    return 1-np.cos(q[...,0])+.8*(1-np.cos(q[...,1]))+.3*(1-np.cos(d))+epsilon*(1-np.cos(h))
def truth_force(q,epsilon=.06):
    q=np.asarray(q);d=q[...,0]-q[...,1];h=3*q[...,0]-q[...,1]
    return np.stack([-np.sin(q[...,0])-.3*np.sin(d)-3*epsilon*np.sin(h),
                     -.8*np.sin(q[...,1])+.3*np.sin(d)+epsilon*np.sin(h)],axis=-1)
def truth_energy(z,epsilon=.06):return .5*np.sum(z[...,2:]**2,axis=-1)+truth_potential(z[...,:2],epsilon)

def features(q):
    t=np.asarray(q)@K.T
    return np.concatenate([np.ones(t.shape[:-1]+(1,)),np.cos(t),np.sin(t)],axis=-1)
def feature_derivatives(q):
    t=np.asarray(q)@K.T;z=np.zeros(t.shape[:-1]+(1,))
    return np.stack([np.concatenate([z,-np.sin(t)*K[:,j],np.cos(t)*K[:,j]],axis=-1) for j in range(2)],axis=-2)

def structural_matrices():
    gram=np.diag(np.r_[1.,np.full(M-1,.5)])
    ridge=np.kron(np.eye(2),gram);curl=np.zeros((2*M,2*M));mean=np.zeros_like(curl)
    for j,k in enumerate(K):
        # Curl sine coefficients for cosine force coefficients and vice versa.
        for off in (1,1+len(K)):
            v=np.zeros(2*M);v[off+j]=-k[1];v[M+off+j]=k[0]
            curl+=.5*np.outer(v,v)
    mean[0,0]=mean[M,M]=1
    return ridge,curl,mean

class FourierForce:
    def __init__(self,kind,C,epsilon=.06):self.kind=kind;self.C=np.asarray(C);self.epsilon=epsilon
    def force(self,q):return features(q)@self.C
    def curl(self,q):
        d=feature_derivatives(q)
        return d[...,0,:]@self.C[:,1]-d[...,1,:]@self.C[:,0]
    def statistics(self):
        r,c,m=structural_matrices();v=self.C.T.reshape(-1)
        return {'force_norm':float(np.sqrt(v@r@v)),'curl_rms':float(np.sqrt(max(0,v@c@v))),
                'mean_force':self.C[0].tolist(),'mean_force_norm':float(np.linalg.norm(self.C[0])),
                'effective_parameters':M-1 if self.kind=='hamiltonian' else 2*M,
                'feature_count':M,'coefficient_count':2*M}
    def potential(self,q):
        # Only valid for exact gradient coefficient arrays (Hamiltonian model).
        if self.kind!='hamiltonian':raise ValueError('No exact periodic potential for this soft-constrained model')
        n=len(K);norm=(K*K).sum(1)
        a=(self.C[1+n:] * K).sum(1)/norm
        b=-(self.C[1:1+n]*K).sum(1)/norm
        t=np.asarray(q)@K.T
        return np.cos(t)@a+np.sin(t)@b
    def save(self,path):np.savez_compressed(path,kind=self.kind,C=self.C,epsilon=self.epsilon)
    @classmethod
    def load(cls,path):
        x=np.load(path);return cls(str(x['kind']),x['C'],float(x['epsilon']))

def fit(kind,q,y,ridge=.001,lambda_curl=1.,lambda_mean=1.,epsilon=.06):
    x=features(q);G=x.T@x/len(q);b=(x.T@y/len(q)).T.reshape(-1)
    R,Cu,Me=structural_matrices();A=np.kron(np.eye(2),G)+ridge*R
    if kind in ('curl_only','curl_mean'):A+=lambda_curl*Cu
    if kind in ('mean_only','curl_mean'):A+=lambda_mean*Me
    if kind=='hamiltonian':
        # Force coefficient subspace for V=sum a_k cos(kq)+b_k sin(kq).
        T=np.zeros((2*M,M-1));n=len(K)
        for j,k in enumerate(K):
            for d in range(2):T[d*M+1+n+j,j]=k[d];T[d*M+1+j,n+j]=-k[d]
        theta=np.linalg.solve(T.T@A@T,T.T@b);v=T@theta
        stationarity=np.linalg.norm(T.T@(A@v-b))
        # Independent constrained-force nullspace solve; same hypothesis class.
        from scipy.linalg import null_space
        constraints=np.concatenate([Cu,Me],axis=0)
        N=null_space(constraints,rcond=1e-11)
        vv=N@np.linalg.solve(N.T@A@N,N.T@b)
        equivalence=float(np.max(np.abs(v-vv)))
    else:v=np.linalg.solve(A,b);stationarity=np.linalg.norm(A@v-b);equivalence=None
    model=FourierForce(kind,v.reshape(2,M).T,epsilon)
    return model,{'normal_equation_residual':float(stationarity),'hard_gradient_equivalence_max_coefficient_error':equivalence,
                  'objective':float(np.mean((model.force(q)-y)**2)*2+ridge*v@R@v+
                    (lambda_curl*v@Cu@v if kind in ('curl_only','curl_mean') else 0)+
                    (lambda_mean*v@Me@v if kind in ('mean_only','curl_mean') else 0))}

def step_numpy(z,u,model=None,epsilon=.06,h=.01,substeps=5):
    z=np.asarray(z,dtype=float);u=np.asarray(u)
    def field(s):return np.concatenate([s[...,2:],truth_force(s[...,:2],epsilon) if model is None else model.force(s[...,:2])],axis=-1)+np.concatenate([np.zeros_like(u),u],axis=-1)
    for _ in range(substeps):
        a=field(z);b=field(z+h*a/2);c=field(z+h*b/2);d=field(z+h*c)
        z=z+h*(a+2*b+2*c+d)/6
    return z

def sample_states(n,band,rng,epsilon=.06):
    chunks=[];left=n
    while left:
        z=np.concatenate([rng.uniform(-np.pi,np.pi,(max(4096,4*left),2)),rng.uniform(-3.5,3.5,(max(4096,4*left),2))],axis=1)
        e=truth_energy(z,epsilon);good=z[(e>=band[0])&(e<=band[1])][:left]
        chunks.append(good);left-=len(good)
    return np.concatenate(chunks)
