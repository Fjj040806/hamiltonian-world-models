"""Canonical coupled pendulums on T² × R² and explicit force interventions."""
import numpy as np

KINDS=('true','conservative','cycle_bias','curl_error')
def potential(q,kappa=.3):
    return 1-np.cos(q[...,0])+.8*(1-np.cos(q[...,1]))+kappa*(1-np.cos(q[...,0]-q[...,1]))
def energy(z):return .5*np.sum(z[...,2:]**2,axis=-1)+potential(z[...,:2])
def force(q,kind='true',epsilon=.04):
    q=np.asarray(q);diff=q[...,0]-q[...,1]
    f=np.stack([-np.sin(q[...,0])-.3*np.sin(diff),-.8*np.sin(q[...,1])+.3*np.sin(diff)],axis=-1)
    if kind=='conservative':f+=np.stack([-epsilon*np.sin(diff),epsilon*np.sin(diff)],axis=-1)
    elif kind=='cycle_bias':f+=np.array([epsilon,0.])
    elif kind=='curl_error':f+=epsilon*np.stack([np.sin(q[...,1]),-np.sin(q[...,0])],axis=-1)
    elif kind!='true':raise ValueError(kind)
    return f
def rhs(z,kind='true',epsilon=.04):return np.concatenate([z[2:],force(z[:2],kind,epsilon)])
def rk4(z,h,kind):
    a=rhs(z,kind);b=rhs(z+h*a/2,kind);c=rhs(z+h*b/2,kind);d=rhs(z+h*c,kind)
    return z+h*(a+2*b+2*c+d)/6
def simulate(initial,h,steps,kind):
    z=np.empty((steps+1,4));z[0]=initial
    for j in range(steps):z[j+1]=rk4(z[j],h,kind)
    return z
