"""Bounded final diagnostics, not acquisition tuning. Outputs raw rows and figures."""
from pathlib import Path
import json, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from phase6_models import fit_model,plant_force,local_states,features
from coupled_models import truth_force
R=Path(__file__).resolve().parent; O=R/'phase8';O.mkdir(exist_ok=True)
P=json.loads((R/'phase8_protocol.json').read_text())
# A different configuration domain and potential family: coupled quartic Duffing.
E=[(a,b) for a in range(4) for b in range(4-a)]
V=[(a,b) for a in range(5) for b in range(5-a) if a+b>0]
def phi(q):return np.stack([q[:,0]**a*q[:,1]**b for a,b in E],axis=1)
def blocks(q,kind):
 x=phi(q);n=len(q);B=np.zeros((n,2,2*len(E)));B[:,0,:len(E)]=x;B[:,1,len(E):]=x
 if kind=='force':return B
 if kind=='odd_force':return B[:,:,np.r_[[j for j,(a,b) in enumerate(E) if (a+b)%2], [len(E)+j for j,(a,b) in enumerate(E) if (a+b)%2]]]
 D=np.zeros((n,2,len(V)))
 for j,(a,b) in enumerate(V):
  if a:D[:,0,j]=-a*q[:,0]**(a-1)*q[:,1]**b
  if b:D[:,1,j]=-b*q[:,0]**a*q[:,1]**(b-1)
 return D
def F(q):
 x,y=q.T;return np.stack([-x-.2*(x-y)-x**3,-1.3*y+.2*(x-y)-.8*y**3],axis=1)
rows=[];stationarity=[]
test=np.random.default_rng(981).uniform(-1.5,1.5,(4096,2));yt=F(test)
for seed in P['seeds']:
 rng=np.random.default_rng(seed);loc=rng.uniform(-.35,.35,(512,2));glob=rng.uniform(-1.5,1.5,(512,2));noise=rng.normal(0,.03,(512,2))
 for frac in [0,.1,1]:
  q=loc.copy();ng=round(512*frac);q[:ng]=glob[:ng];y=F(q)+noise
  for kind in ['force','odd_force','hamiltonian']:
   B=blocks(q,kind).reshape(-1,blocks(q,kind).shape[-1]);Bt=blocks(test,kind);A=B.T@B/512+1e-6*np.eye(B.shape[1]);b=B.T@y.reshape(-1)/512;coef=np.linalg.solve(A,b)
   rms=float(np.sqrt(np.mean(np.sum((Bt@coef-yt)**2,axis=1))));stationarity.append(float(np.linalg.norm(A@coef-b)))
   rows.append(dict(seed=seed,global_fraction=frac,kind=kind,parameters=len(coef),force_RMS=rms))
# Recenter the same odd class using the explicitly privileged true symmetry center.
recenter=[];shift=np.array([.45,-.35]);testq=np.random.default_rng(982).uniform(-np.pi,np.pi,(4096,2));target=plant_force(testq,'shifted')
for seed in P['seeds']:
 rng=np.random.default_rng(seed);local=local_states(4096,rng,'shifted')[:,:2];glob=rng.uniform(-np.pi,np.pi,(4096,2));noise=rng.normal(0,.03,(4096,2))
 for frac in [0,1]:
  q=local if frac==0 else glob;y=plant_force(q,'shifted')+noise
  for center in ['zero','true_center']:
   s=np.zeros(2) if center=='zero' else shift;m,_=fit_model('odd_force',q-s,y,1e-6,plant='shifted');err=m.force(testq-s)-target
   recenter.append(dict(seed=seed,global_fraction=frac,center=center,force_RMS=float(np.sqrt(np.mean(np.sum(err**2,axis=1))))))
# True-force unsaturated sample-and-hold diagnostic; independent RK4 integration.
q0=np.random.default_rng(983).uniform(-np.pi,np.pi,(32,2));targetp=np.array([2.5,1.5]);held=[]
def rhs(z,u):return np.concatenate([z[:,2:],truth_force(z[:,:2],.06)+u],axis=1)
for h in [.1,.05,.025,.0125]:
 z=np.concatenate([q0,np.tile(targetp,(32,1))],axis=1);errors=[]
 for i in range(round(10/h)):
  u=-truth_force(z[:,:2],.06)-2*(z[:,2:]-targetp)
  sub=round(h/.0025);dt=h/sub
  for _ in range(sub):
   a=rhs(z,u);b=rhs(z+dt*a/2,u);c=rhs(z+dt*b/2,u);d=rhs(z+dt*c,u);z+=dt*(a+2*b+2*c+d)/6
  if (i+1)*h>5:errors.append(np.sum((z[:,2:]-targetp)**2,axis=1))
 held.append(dict(hold_step=h,momentum_RMS=float(np.sqrt(np.mean(errors)))))
result={'protocol_sha256':hashlib.sha256((R/'phase8_protocol.json').read_bytes()).hexdigest(),'duffing':rows,'recenter':recenter,'held_control':held,'max_normal_equation_residual':max(stationarity),'new_SAC_runs':0,'new_acquisition_variants':0}
assert all(np.isfinite(r['force_RMS']) for r in rows+recenter)
assert max(stationarity)<1e-10
(O/'results.json').write_text(json.dumps(result,indent=2))
fig,ax=plt.subplots(1,3,figsize=(13,3.8))
for kind in ['force','odd_force','hamiltonian']:
 vals=np.array([[r['force_RMS'] for r in rows if r['kind']==kind and r['global_fraction']==f] for f in [0,.1,1]])
 ax[0].plot([0,1,2],vals.mean(1),'o-',label=kind);ax[0].fill_between([0,1,2],vals.min(1),vals.max(1),alpha=.12)
ax[0].set(xticks=[0,1,2],xticklabels=['local','10% global','global'],yscale='log',ylabel='Global force RMS',title='Independent coupled Duffing');ax[0].legend(fontsize=8)
for center in ['zero','true_center']:
 vals=[[r['force_RMS'] for r in recenter if r['center']==center and r['global_fraction']==f] for f in [0,1]];ax[1].plot([0,1],np.mean(vals,axis=1),'o-',label=center)
ax[1].set(xticks=[0,1],xticklabels=['local','global'],yscale='log',title='Odd prior: privileged recentering',ylabel='Global force RMS');ax[1].legend(fontsize=8)
x=np.array([r['hold_step'] for r in held]);y=np.array([r['momentum_RMS'] for r in held]);ax[2].loglog(x,y,'o-',label='actual RK4');ax[2].loglog(x,y[-1]*x/x[-1],'--',label='slope 1');ax[2].set(title='Exact-force held controller',xlabel='Action hold interval',ylabel='Momentum RMS');ax[2].legend(fontsize=8)
fig.tight_layout();fig.savefig(O/'final_validation.png',dpi=180);fig.savefig(O/'final_validation.svg');plt.close(fig)
print(json.dumps({'duffing_means':{k:{str(f):float(np.mean([r['force_RMS'] for r in rows if r['kind']==k and r['global_fraction']==f])) for f in [0,.1,1]} for k in ['force','odd_force','hamiltonian']},'recenter_means':{c:{str(f):float(np.mean([r['force_RMS'] for r in recenter if r['center']==c and r['global_fraction']==f])) for f in [0,1]} for c in ['zero','true_center']},'held_control':held,'held_log_slope':float(np.polyfit(np.log(x),np.log(y),1)[0])},indent=2))
