"""Small fully trained NumPy tanh networks, including analytic HNN gradients."""
import numpy as np

KINDS=('node','second_order','force','force_zero_mean','hnn')
class WorldModel:
    def __init__(self,kind,seed=0,width=32):
        self.kind=kind;self.width=width;rng=np.random.default_rng(seed)
        n=2 if kind in ('force','force_zero_mean','hnn') else 3;k=2 if kind=='node' else 1
        self.W=rng.normal(0,.6,(n,width));self.b=rng.normal(0,.1,width)
        self.V=rng.normal(0,.15,(width,k));self.c=np.zeros(k)
    def features(self,z):
        x=np.column_stack([np.sin(z[:,0]),np.cos(z[:,0])])
        return x if self.kind in ('force','force_zero_mean','hnn') else np.column_stack([x,z[:,1]])
    def quadrature(self):
        q=np.linspace(-np.pi,np.pi,128,endpoint=False)
        return np.column_stack([np.sin(q),np.cos(q)])
    def output(self,z):
        x=self.features(z);t=np.tanh(x@self.W+self.b)
        if self.kind=='hnn':
            xq=np.column_stack([np.cos(z[:,0]),-np.sin(z[:,0])])
            return -np.sum((1-t*t)*(xq@self.W)*self.V[:,0],axis=1,keepdims=True)
        if self.kind=='force_zero_mean':
            tg=np.tanh(self.quadrature()@self.W+self.b)
            return (t-tg.mean(0))@self.V
        return t@self.V+self.c
    def field(self,z,u=0.):
        o=self.output(z)
        if self.kind=='node':return o+np.column_stack([np.zeros(len(z)),np.broadcast_to(u,len(z))])
        return np.column_stack([z[:,1],o[:,0]+u])
    def potential(self,q):
        x=np.column_stack([np.sin(q),np.cos(q)])
        return (np.tanh(x@self.W+self.b)@self.V)[:,0]
    def loss_grad(self,z,y):
        x=self.features(z);t=np.tanh(x@self.W+self.b);s=1-t*t
        pred=self.output(z);e=pred-y;d=2*e/e.size
        if self.kind=='hnn':
            xq=np.column_stack([np.cos(z[:,0]),-np.sin(z[:,0])]);aq=xq@self.W
            a=2*t*s*aq*self.V[:,0]*d
            grads=[x.T@a-xq.T@(s*self.V[:,0]*d),a.sum(0),(-s*aq).T@d,np.zeros_like(self.c)]
        elif self.kind=='force_zero_mean':
            xg=self.quadrature();tg=np.tanh(xg@self.W+self.b)
            a=(d@self.V.T)*s
            ag=-(np.ones((len(xg),1))*d.sum(0)/len(xg)@self.V.T)*(1-tg*tg)
            grads=[x.T@a+xg.T@ag,a.sum(0)+ag.sum(0),
                   (t-tg.mean(0)).T@d,np.zeros_like(self.c)]
        else:
            a=(d@self.V.T)*s
            grads=[x.T@a,a.sum(0),t.T@d,d.sum(0)]
        return float(np.mean(e*e)),grads
    def params(self):return [self.W,self.b,self.V,self.c]
    def save(self,path):np.savez(path,kind=self.kind,width=self.width,W=self.W,b=self.b,V=self.V,c=self.c)
    @classmethod
    def load(cls,path):
        a=np.load(path);m=cls(str(a['kind']),width=int(a['width']))
        for n in ('W','b','V','c'):setattr(m,n,a[n])
        return m

def rollout(model,initial,dt,steps):
    out=np.empty((len(initial),steps+1,2));out[:,0]=initial
    for j in range(steps):
        z=out[:,j];a=model.field(z);b=model.field(z+dt*a/2)
        c=model.field(z+dt*b/2);d=model.field(z+dt*c)
        out[:,j+1]=z+dt*(a+2*b+2*c+d)/6
    return out

def gradient_check():
    rng=np.random.default_rng(77);z=rng.normal(size=(7,2))
    for kind in KINDS:
        m=WorldModel(kind);y=rng.normal(size=(7,2 if kind=='node' else 1));_,g=m.loss_grad(z,y)
        for p,grad in zip(m.params(),g):
            for idx in [tuple(0 for _ in p.shape),tuple(i-1 for i in p.shape)]:
                old=p[idx];eps=1e-6;p[idx]=old+eps;hi=m.loss_grad(z,y)[0]
                p[idx]=old-eps;lo=m.loss_grad(z,y)[0];p[idx]=old
                assert abs((hi-lo)/(2*eps)-grad[idx])<2e-6,(kind,idx)
