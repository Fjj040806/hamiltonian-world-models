"""Normalized canonical pendulum; action is held constant for each step."""
import numpy as np

def energy(z):
    z=np.asarray(z)
    return .5*z[...,1]**2+1-np.cos(z[...,0])

def rhs(z,u=0.):
    q,p=z[:2]
    return np.array([p,-np.sin(q)+u,p*u])

def rk4(z,h,u=0.):
    k1=rhs(z,u);k2=rhs(z+h*k1/2,u)
    k3=rhs(z+h*k2/2,u);k4=rhs(z+h*k3,u)
    return z+h*(k1+2*k2+2*k3+k4)/6

def verlet(z,h,u=0.):
    q,p,w=z
    ph=p+h*(-np.sin(q)+u)/2
    qn=q+h*ph
    pn=ph+h*(-np.sin(qn)+u)/2
    return np.array([qn,pn,w+u*(qn-q)])

def simulate(z0,h,steps,method='rk4',actions=None):
    if method not in ('rk4','verlet'):raise ValueError('Unknown method')
    step=rk4 if method=='rk4' else verlet
    actions=np.zeros(steps) if actions is None else np.asarray(actions)
    if len(actions)!=steps:raise ValueError('Action length mismatch')
    out=np.empty((steps+1,3));out[0]=[*z0[:2],0.]
    for i in range(steps):out[i+1]=step(out[i],h,float(actions[i]))
    return out

class PendulumEnv:
    """Minimal RL-ready API; no speed clipping or hidden dissipation.

    Observation is [cos(q),sin(q),p]. Internal q stays unwrapped.
    Reward is a time-integrated upright cost; termination is time-limit only.
    """
    def __init__(self,dt=.05,substeps=5,max_steps=400,action_limit=2.):
        self.dt=dt;self.substeps=substeps;self.max_steps=max_steps
        self.action_limit=action_limit;self.reset()
    def observation(self):return np.array([np.cos(self.z[0]),np.sin(self.z[0]),self.z[1]])
    def reset(self,z0=(0.,0.)):
        self.z=np.array([*z0,0.],dtype=float);self.n=0;self.initial_energy=float(energy(self.z))
        return self.observation(),self.info()
    def info(self):
        return {'energy':float(energy(self.z)),'work':float(self.z[2]),
                'balance_error':float(energy(self.z)-self.initial_energy-self.z[2])}
    def step(self,action):
        if self.n>=self.max_steps:raise RuntimeError('Reset after time limit')
        u=float(np.asarray(action).item())
        if not np.isfinite(u):raise ValueError('Non-finite action')
        u=np.clip(u,-self.action_limit,self.action_limit);reward=0.
        h=self.dt/self.substeps
        for _ in range(self.substeps):
            old=self.z.copy();self.z=rk4(self.z,h,u)
            cost=lambda z:1+np.cos(z[0])+.1*z[1]**2+.01*u*u
            reward-=h*(cost(old)+cost(self.z))/2
        self.n+=1
        return self.observation(),float(reward),False,self.n>=self.max_steps,self.info()
