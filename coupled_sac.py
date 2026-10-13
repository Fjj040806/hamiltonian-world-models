"""Two-action SAC and periodic coupled dynamics for the Phase 5 protocol."""
import copy
import numpy as np
import torch
from torch import nn
from coupled_models import K
from sac import mlp

def obs(z):return torch.cat([torch.cos(z[:,:2]),torch.sin(z[:,:2]),z[:,2:]/3],dim=1)
class Actor(nn.Module):
    def __init__(self):super().__init__();self.net=mlp(6,4)
    def sample(self,o,deterministic=False):
        v=self.net(o);mean=v[:,:2];ls=v[:,2:].clamp(-5,1)
        pre=mean if deterministic else mean+ls.exp()*torch.randn_like(mean)
        t=torch.tanh(pre)
        lp=-.5*((pre-mean)/ls.exp())**2-ls-.5*np.log(2*np.pi)-torch.log(1-t*t+1e-6)-np.log(2)
        return 2*t,lp.sum(1,keepdim=True)
class SAC:
    def __init__(self):
        self.actor=Actor();self.q1=mlp(8,1);self.q2=mlp(8,1)
        self.t1=copy.deepcopy(self.q1);self.t2=copy.deepcopy(self.q2)
        self.oa=torch.optim.Adam(self.actor.parameters(),lr=3e-4)
        self.oq=torch.optim.Adam(list(self.q1.parameters())+list(self.q2.parameters()),lr=3e-4)
    def update(self,z,a,r,nz):
        o=obs(z);no=obs(nz)
        with torch.no_grad():
            na,lp=self.actor.sample(no);x=torch.cat([no,na/2],1)
            target=r+.99*(torch.minimum(self.t1(x),self.t2(x))-.01*lp)
        x=torch.cat([o,a/2],1);loss=((self.q1(x)-target)**2).mean()+((self.q2(x)-target)**2).mean()
        self.oq.zero_grad();loss.backward();self.oq.step()
        for q in [self.q1,self.q2]:
            for p in q.parameters():p.requires_grad_(False)
        action,lp=self.actor.sample(o);x=torch.cat([o,action/2],1)
        aloss=(.01*lp-torch.minimum(self.q1(x),self.q2(x))).mean()
        self.oa.zero_grad();aloss.backward();self.oa.step()
        for q in [self.q1,self.q2]:
            for p in q.parameters():p.requires_grad_(True)
        with torch.no_grad():
            for target_q,q in [(self.t1,self.q1),(self.t2,self.q2)]:
                for t,p in zip(target_q.parameters(),q.parameters()):t.mul_(.995).add_(p,alpha=.005)
        return float(loss.detach()),float(aloss.detach())
    def save(self,path):torch.save({'actor':self.actor.state_dict(),'q1':self.q1.state_dict(),'q2':self.q2.state_dict()},path)
class Replay:
    def __init__(self,capacity):self.data=torch.zeros((capacity,11));self.n=0;self.pos=0;self.capacity=capacity
    def add(self,z,a,r,nz):
        rows=torch.cat([z,a,r,nz],1).detach();k=len(rows);ids=(torch.arange(k)+self.pos)%self.capacity
        self.data[ids]=rows;self.pos=(self.pos+k)%self.capacity;self.n=min(self.capacity,self.n+k)
    def sample(self,n):
        d=self.data[torch.randint(self.n,(n,))]
        return d[:,:4],d[:,4:6],d[:,6:7],d[:,7:]
class Dynamics:
    def __init__(self,model=None,epsilon=.06):
        self.model=model;self.epsilon=epsilon;self.k=torch.tensor(K,dtype=torch.float32)
        self.C=None if model is None else torch.tensor(model.C,dtype=torch.float32)
    def force(self,q):
        if self.C is not None:
            t=q@self.k.T;phi=torch.cat([torch.ones((len(q),1)),torch.cos(t),torch.sin(t)],1)
            return phi@self.C
        d=q[:,0]-q[:,1];h=3*q[:,0]-q[:,1]
        return torch.stack([-torch.sin(q[:,0])-.3*torch.sin(d)-3*self.epsilon*torch.sin(h),
                            -.8*torch.sin(q[:,1])+.3*torch.sin(d)+self.epsilon*torch.sin(h)],1)
    def field(self,z,u):return torch.cat([z[:,2:],self.force(z[:,:2])+u],1)
    def step(self,z,u,wrap=True):
        r=torch.zeros((len(z),1));h=.01;target=torch.tensor([2.5,1.5])
        def cost(s):return (s[:,2:]-target).square().sum(1)+.01*u.square().sum(1)
        for _ in range(5):
            old=z;a=self.field(z,u);b=self.field(z+h*a/2,u);c=self.field(z+h*b/2,u);d=self.field(z+h*c,u)
            z=z+h*(a+2*b+2*c+d)/6;r-=h*(cost(old)+cost(z))[:,None]/2
        if wrap:z=torch.cat([torch.atan2(torch.sin(z[:,:2]),torch.cos(z[:,:2])),z[:,2:]],1)
        return z,r
