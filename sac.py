"""Compact SAC implementation with twin Q networks and bounded Gaussian actor."""
import copy
import torch
from torch import nn
import numpy as np

def obs(z):return torch.stack([torch.cos(z[:,0]),torch.sin(z[:,0]),z[:,1]/3],dim=1)
def mlp(n,k):return nn.Sequential(nn.Linear(n,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,k))
class Actor(nn.Module):
    def __init__(self):super().__init__();self.net=mlp(3,2)
    def sample(self,o,deterministic=False,generator=None):
        v=self.net(o);mean=v[:,:1];ls=v[:,1:].clamp(-5,1)
        pre=mean if deterministic else mean+ls.exp()*torch.randn(mean.shape,generator=generator)
        t=torch.tanh(pre);action=2*t
        logp=-.5*((pre-mean)/ls.exp())**2-ls-.5*np.log(2*np.pi)
        logp=logp-torch.log(1-t*t+1e-6)-np.log(2)
        return action,logp
class SAC:
    def __init__(self):
        self.actor=Actor();self.q1=mlp(4,1);self.q2=mlp(4,1)
        self.t1=copy.deepcopy(self.q1);self.t2=copy.deepcopy(self.q2)
        self.oa=torch.optim.Adam(self.actor.parameters(),lr=3e-4)
        self.oq=torch.optim.Adam(list(self.q1.parameters())+list(self.q2.parameters()),lr=3e-4)
    def update(self,z,a,r,nz):
        o=obs(z);no=obs(nz)
        with torch.no_grad():
            na,lp=self.actor.sample(no);x=torch.cat([no,na/2],1)
            target=r+.99*(torch.minimum(self.t1(x),self.t2(x))-.01*lp)
        x=torch.cat([o,a/2],1)
        loss=((self.q1(x)-target)**2).mean()+((self.q2(x)-target)**2).mean()
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
    def save(self,path):
        torch.save({'actor':self.actor.state_dict(),'q1':self.q1.state_dict(),'q2':self.q2.state_dict()},path)

class Replay:
    def __init__(self,capacity):
        self.capacity=capacity;self.data=torch.zeros((capacity,6));self.n=0;self.pos=0
    def add(self,z,a,r,nz):
        rows=torch.cat([z,a,r,nz],1).detach();k=len(rows)
        ids=(torch.arange(k)+self.pos)%self.capacity;self.data[ids]=rows
        self.pos=(self.pos+k)%self.capacity;self.n=min(self.capacity,self.n+k)
    def sample(self,n,generator=None):
        ids=torch.randint(self.n,(n,),generator=generator);d=self.data[ids]
        return d[:,:2],d[:,2:3],d[:,3:4],d[:,4:]

class Dynamics:
    def __init__(self,model=None):
        self.model=model
        if model is not None:
            self.W=torch.tensor(model.W,dtype=torch.float32);self.b=torch.tensor(model.b,dtype=torch.float32)
            self.V=torch.tensor(model.V,dtype=torch.float32);self.c=torch.tensor(model.c,dtype=torch.float32)
            if model.kind=='force_zero_mean':
                self.mean=torch.tanh(torch.tensor(model.quadrature(),dtype=torch.float32)@self.W+self.b).mean(0)
    def field(self,z,u):
        q,p=z[:,0],z[:,1]
        if self.model is None:return torch.stack([p,-torch.sin(q)+u[:,0]],1)
        kind=self.model.kind;x=torch.stack([torch.sin(q),torch.cos(q)],1)
        if kind in ('node','second_order'):x=torch.cat([x,p[:,None]],1)
        t=torch.tanh(x@self.W+self.b)
        if kind=='hnn':
            xq=torch.stack([torch.cos(q),-torch.sin(q)],1)
            f=-((1-t*t)*(xq@self.W)*self.V[:,0]).sum(1)
        elif kind=='force_zero_mean':f=((t-self.mean)@self.V)[:,0]
        else:
            o=t@self.V+self.c
            if kind=='node':return o+torch.cat([torch.zeros_like(u),u],1)
            f=o[:,0]
        return torch.stack([p,f+u[:,0]],1)
    def step(self,z,u):
        reward=torch.zeros((len(z),1));h=.01
        cost=lambda zz:1+torch.cos(zz[:,0])+.1*zz[:,1]**2+.01*u[:,0]**2
        for _ in range(5):
            old=z;k1=self.field(z,u);k2=self.field(z+h*k1/2,u)
            k3=self.field(z+h*k2/2,u);k4=self.field(z+h*k3,u)
            z=z+h*(k1+2*k2+2*k3+k4)/6
            reward-=h*(cost(old)+cost(z))[:,None]/2
        # The state is on a cylinder; wrapping does not clamp speed or dissipate energy.
        z=torch.stack([torch.atan2(torch.sin(z[:,0]),torch.cos(z[:,0])),z[:,1]],1)
        return z,reward
