from pathlib import Path
import json,csv,time
import numpy as np
import matplotlib.pyplot as plt
from world_models import WorldModel,KINDS,rollout,gradient_check
from simulator import energy,simulate

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'phase2';OUT.mkdir(exist_ok=True)
SEEDS=range(5);STEPS=3000;DT=.02
COL={'node':'#81929d','second_order':'#bb9349','force':'#476ca0','force_zero_mean':'#9a78a7','hnn':'#218b87'}
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
def labels(split):
    a=np.load(ROOT/'phase1'/f'{split}.npz')['states']
    return a[:,1:-1].reshape(-1,2),(a[:,2:]-a[:,:-2]).reshape(-1,2)/(.04)
def save(fig,name):
    fig.tight_layout()
    for ext in ('png','svg'):fig.savefig(OUT/f'{name}.{ext}',dpi=175)
    plt.close(fig)

gradient_check();x,y=labels('train');vx,vy=labels('validation');rows=[];histories={};start=time.time()
for kind in KINDS:
 for seed in SEEDS:
    m=WorldModel(kind,seed);rng=np.random.default_rng(100+seed)
    yy=y if kind=='node' else y[:,1:];vyy=vy if kind=='node' else vy[:,1:]
    mom=[np.zeros_like(p) for p in m.params()];var=[v.copy() for v in mom]
    best=np.inf;history=[];path=OUT/f'{kind}_{seed}.npz'
    for step in range(1,STEPS+1):
        ids=rng.integers(len(x),size=512);loss,grads=m.loss_grad(x[ids],yy[ids])
        lr=.003*(.2+.8*.5*(1+np.cos(np.pi*step/STEPS)))
        for p,g,a,b in zip(m.params(),grads,mom,var):
            a[:]=.9*a+.1*g;b[:]=.999*b+.001*g*g
            p-=lr*(a/(1-.9**step))/(np.sqrt(b/(1-.999**step))+1e-8)
        if step%100==0:
            val=m.loss_grad(vx,vyy)[0];history.append([step,loss,val])
            if val<best:best=val;m.save(path)
    histories[f'{kind}_{seed}']=history
    rows.append({'kind':kind,'seed':seed,'parameters':sum(p.size for p in m.params()),'validation_mse':best})
    print(f'Trained {kind} seed {seed}: validation {best:.3g}',flush=True)
(OUT/'training.json').write_text(json.dumps({'steps':STEPS,'batch':512,'seeds':list(SEEDS),'histories':histories,'models':rows},indent=2))

splits=['validation','test_moderate','test_near_below','test_near_above','test_rotation'];results=[];examples={}
for split in splits:
    a=np.load(ROOT/'phase1'/f'{split}.npz');initial=a['states'][:,0]
    # Longer reference trajectories: 16 time units; integration h=.01 then subsample.
    true=np.array([simulate(z,.01,1600)[::2,:2] for z in initial])
    ex=int(np.argsort(a['initial_energy'])[len(initial)//2]);examples[split]={'true':true[ex]}
    for kind in KINDS:
      for seed in SEEDS:
        m=WorldModel.load(OUT/f'{kind}_{seed}.npz');pred=rollout(m,initial,DT,800)
        assert np.isfinite(pred).all()
        dq=np.arctan2(np.sin(pred[:,:,0]-true[:,:,0]),np.cos(pred[:,:,0]-true[:,:,0]))
        err=dq*dq+(pred[:,:,1]-true[:,:,1])**2
        sx,sy=labels(split);field=m.field(sx)
        row={'split':split,'kind':kind,'seed':seed,
          'field_mse':float(np.mean((field-sy)**2)),
          'rmse_4':float(np.sqrt(np.mean(err[:,1:201]))),
          'rmse_16':float(np.sqrt(np.mean(err[:,1:]))),
          'true_energy_drift_16':float(np.mean(np.max(np.abs(energy(pred)-energy(initial)[:,None]),axis=1)))}
        if kind=='hnn':
            grid=np.linspace(-np.pi,np.pi,1001);v=m.potential(grid)
            row['potential_range']=float(v.max()-v.min())
            row['fixed_reference_barrier']=float(m.potential(np.array([np.pi]))[0]-m.potential(np.array([0.]))[0])
            vtrajectory=.5*pred[:,:,1]**2+m.potential(pred[:,:,0].reshape(-1)).reshape(pred.shape[:2])
            row['learned_energy_drift_16']=float(np.mean(np.max(np.abs(vtrajectory-vtrajectory[:,0,None]),axis=1)))
        results.append(row)
        if seed==0:examples[split][kind]=pred[ex]
    print('Evaluated '+split,flush=True)
(OUT/'results.json').write_text(json.dumps(results,indent=2))
fields=sorted(set().union(*(r.keys() for r in results)))
with (OUT/'results.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(results)

fig,axs=plt.subplots(1,2,figsize=(12,4.5))
for kind in KINDS:
    for ax,metric in zip(axs,['rmse_4','rmse_16']):
        vals=np.array([[r[metric] for r in results if r['kind']==kind and r['split']==s] for s in splits])
        ax.errorbar(np.arange(5),vals.mean(1),yerr=vals.std(1,ddof=1),color=COL[kind],marker='o',label=kind,capsize=3)
        ax.set_xticks(np.arange(5),['ID','moderate','below','above','rotation']);ax.set_yscale('log');ax.set_ylabel('Circular-angle / momentum RMSE');ax.set_title(metric.replace('_',' '));ax.grid(alpha=.15)
axs[0].legend(fontsize=8);fig.suptitle('Cross-energy prediction | mean ± seed SD, 5 fitted models per architecture')
save(fig,'01_cross_energy_prediction')
fig,axs=plt.subplots(1,2,figsize=(11,4.5));grid=np.linspace(-np.pi,np.pi,501)
axs[0].plot(grid,1-np.cos(grid),'k--',label='True potential')
for seed in SEEDS:
    m=WorldModel.load(OUT/f'hnn_{seed}.npz');v=m.potential(grid)-m.potential(np.array([0]))[0]
    axs[0].plot(grid,v,color=COL['hnn'],alpha=.55,label='HNN seeds' if seed==0 else None)
axs[0].axvspan(-np.arccos(-.2),np.arccos(-.2),alpha=.08,color=COL['hnn'],label='Training-accessible angles')
axs[0].set(xlabel='Angle q',ylabel='V(q) − V(0)',title='Conservation does not identify an unseen potential');axs[0].legend(fontsize=8)
for kind in KINDS:
    vals=[r['true_energy_drift_16'] for r in results if r['kind']==kind and r['split']=='test_near_above']
    axs[1].scatter(np.repeat(kind,5),vals,color=COL[kind],s=35)
axs[1].set(ylabel='Mean maximum true-energy deviation',title='Near-above test | each point is one fitted seed');axs[1].tick_params(axis='x',rotation=20)
save(fig,'02_potential_and_energy')
fig,axs=plt.subplots(1,2,figsize=(11,4.5));ex=examples['test_near_above'];times=np.arange(801)*DT
for kind,z in ex.items():
    col='black' if kind=='true' else COL[kind]
    axs[0].plot(times,z[:,0],color=col,label=kind,lw=1.2)
    axs[1].plot(times,energy(z),color=col,lw=1.2)
axs[0].set(xlabel='Time',ylabel='Unwrapped angle',title='Representative near-above trajectory (seed 0)');axs[0].legend(fontsize=8)
axs[1].set(xlabel='Time',ylabel='True H evaluated on predictions',title='Same initial state; different learned flows')
save(fig,'03_trajectory_comparison')
np.savez_compressed(OUT/'animation_trajectories.npz',**ex,dt=DT)
(OUT/'run_metadata.json').write_text(json.dumps({'elapsed_seconds':time.time()-start,'numpy':np.__version__,
 'training_target':'central finite-difference derivative from Phase 1 uncontrolled trajectories',
 'selection':'minimum in-distribution validation derivative MSE every 100 steps',
 'evaluation':'common RK4 dt=.02; independent true RK4 h=.01; no test tuning'},indent=2))
print('Phase 2 world-model pilot complete.',flush=True)
