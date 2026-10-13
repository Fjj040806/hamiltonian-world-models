"""Fixed feedback and deterministic projected-gradient MPC with checked adjoints."""
import numpy as np
import torch
from coupled_sac import Dynamics as BaseDynamics
from phase6_models import P,plant_force,Model

class Dynamics(BaseDynamics):
    def __init__(self,model=None,plant='misspecified'):
        self.plant=plant;self.shift=torch.tensor(P['plants'][plant]['shift'],dtype=torch.float32)
        super().__init__(model,P['plants'][plant]['epsilon'])
    def force(self,q):return super().force(q if self.C is not None else q-self.shift)

def true_jacobian(q,plant):
    pars=P['plants'][plant];q=np.asarray(q)-pars['shift'];eps=pars['epsilon']
    a=np.cos(q[...,0]);b=np.cos(q[...,1]);d=.3*np.cos(q[...,0]-q[...,1]);h=eps*np.cos(3*q[...,0]-q[...,1])
    return np.stack([np.stack([-a-d-9*h,d+3*h],-1),np.stack([d+3*h,-.8*b-d-h],-1)],-2)
class Planner:
    def __init__(self,model=None,plant='misspecified'):
        self.model=model;self.plant=plant;self.cfg=P['mpc'];self.last=None
        self.target=np.array(P['controller_target']);self.h=self.cfg['planning_dt'];self.H=self.cfg['horizon']
    def force(self,q):return plant_force(q,self.plant) if self.model is None else self.model.force(q)
    def jac(self,q):return true_jacobian(q,self.plant) if self.model is None else self.model.jacobian(q)
    def feedback(self,z):return np.clip(-self.force(z[:,:2])-P['controller_gain']*(z[:,2:]-self.target),-2,2)
    def transition(self,z,u):
        p=z[:,2:]+self.h*(self.force(z[:,:2])+u);q=z[:,:2]+self.h*p
        return np.c_[q,p]
    def rollout(self,z,U,gradient=False):
        states=[z];jacs=[];cost=np.zeros(len(z))
        for t in range(self.H):
            if gradient:jacs.append(self.jac(z[:,:2]))
            z=self.transition(z,U[t]);states.append(z)
            cost+=self.h*(np.sum((z[:,2:]-self.target)**2,axis=1)+self.cfg['action_cost']*np.sum(U[t]**2,axis=1))
        cost+=self.cfg['terminal_weight']*np.sum((z[:,2:]-self.target)**2,axis=1)
        if not gradient:return cost
        adj=np.c_[np.zeros((len(z),2)),2*self.cfg['terminal_weight']*(z[:,2:]-self.target)]
        grad=np.empty_like(U)
        for t in range(self.H-1,-1,-1):
            adj[:,2:]+=2*self.h*(states[t+1][:,2:]-self.target)
            g=self.h**2*adj[:,:2]+self.h*adj[:,2:]
            grad[t]=g+2*self.h*self.cfg['action_cost']*U[t]
            adj=np.c_[adj[:,:2]+np.einsum('bji,bj->bi',jacs[t],g),adj[:,2:]+self.h*adj[:,:2]]
        return cost,grad
    def initial_feedback_plan(self,z):
        U=[]
        for _ in range(self.H):
            u=self.feedback(z);U.append(u);z=self.transition(z,u)
        return np.array(U)
    def action(self,z):
        U=self.initial_feedback_plan(z);cost=self.rollout(z,U)
        if self.last is not None:
            alt=np.concatenate([self.last[1:],self.last[-1:]],axis=0);ac=self.rollout(z,alt)
            choose=ac<cost;U[:,choose]=alt[:,choose];cost=np.minimum(cost,ac)
        initial_cost=cost.copy();accepted=0
        for _ in range(self.cfg['iterations']):
            cost,grad=self.rollout(z,U,True);eta=np.full(len(z),self.cfg['gradient_step']);pending=np.ones(len(z),dtype=bool)
            for _ in range(self.cfg['backtracks']):
                candidate=np.clip(U-grad*eta[None,:,None],-2,2);cc=self.rollout(z,candidate)
                take=pending & np.isfinite(cc) & (cc<=cost+1e-12)
                U[:,take]=candidate[:,take];cost[take]=cc[take];accepted+=int(take.sum());pending[take]=False
                if not pending.any():break
                eta[pending]*=.5
        final_cost=self.rollout(z,U);assert np.all(final_cost<=initial_cost+1e-8)
        self.last=U.copy()
        return U[0],{'initial_cost':float(initial_cost.mean()),'final_cost':float(final_cost.mean()),'accepted_updates':accepted,
                     'maximum_cost_increase':float(np.max(final_cost-initial_cost))}
