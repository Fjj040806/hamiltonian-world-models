"""Run numerical verification and generate deterministic Phase 2 datasets."""
from pathlib import Path
import json,hashlib,platform
import numpy as np
import scipy
from scipy.integrate import solve_ivp
from scipy.special import ellipk
import matplotlib.pyplot as plt
from simulator import energy,rhs,simulate,PendulumEnv

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'phase1';OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
 'figure.facecolor':'#fbfcfc','axes.facecolor':'#fbfcfc','svg.fonttype':'none'})
COL={'rk4':'#218b87','verlet':'#cc705b'}
def save(fig,name):
    fig.tight_layout()
    for ext in ('png','svg'):fig.savefig(OUT/f'{name}.{ext}',dpi=180)
    plt.close(fig)

metrics={};h=.01;T=100.;cases={'libration':.7,'near_separatrix':1.99,'rotation':2.7}
fig,axs=plt.subplots(1,3,figsize=(12,3.8))
for ax,(name,E) in zip(axs,cases.items()):
    for method in COL:
        z=simulate([0,np.sqrt(2*E)],h,round(T/h),method)
        drift=energy(z)-E
        metrics[f'{name}_{method}_energy_max']=float(np.max(np.abs(drift)))
        ax.plot(np.arange(len(z))*h,drift,color=COL[method],label=method.upper(),lw=.8)
    ax.set(title=f'{name.replace("_"," ")} | H₀={E}',xlabel='Time',ylabel='H(t) − H(0)')
    ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0));ax.legend(fontsize=8)
save(fig,'01_energy_conservation')

hs=np.array([.08,.04,.02,.01,.005]);errors={}
fig,ax=plt.subplots(figsize=(6.7,4.5))
for method in COL:
    errs=[]
    for dt in hs:
        worst=0
        for E in cases.values():
            initial=[0,np.sqrt(2*E),0]
            ref=solve_ivp(lambda t,z:rhs(z),[0,10],initial,method='DOP853',rtol=2e-13,atol=2e-14).y[:,-1]
            z=simulate(initial,dt,round(10/dt),method)[-1]
            worst=max(worst,np.linalg.norm(z[:2]-ref[:2]))
        errs.append(worst)
    order=float(np.polyfit(np.log(hs[-3:]),np.log(errs[-3:]),1)[0])
    errors[method]=errs;metrics[f'{method}_observed_order']=order
    ax.loglog(hs,errs,'o-',color=COL[method],label=f'{method.upper()}: slope {order:.2f}')
ax.set(xlabel='Step size h',ylabel='Worst endpoint error (T = 10)',title='Convergence against an independent DOP853 reference');ax.legend();ax.grid(alpha=.2)
save(fig,'02_step_convergence')

dt=.01;steps=2000;rng=np.random.default_rng(731)
# Piecewise constant actions with discontinuities aligned to integration boundaries.
actions=np.repeat(rng.uniform(-1.5,1.5,40),50)
z=simulate([.4,.7],dt,steps,actions=actions);t=np.arange(steps+1)*dt
dH=energy(z)-energy(z[0]);W=z[:,2];res=dH-W
metrics['controlled_rk4_balance_max']=float(np.max(np.abs(res)))
# Independent reference restarted at each action discontinuity.
ref=np.array([.4,.7,0.]);ref_samples=[ref.copy()]
for j,u in enumerate(actions[::50]):
    ts=np.linspace(0,.5,51)
    sol=solve_ivp(lambda t,y:rhs(y,u),(0,.5),ref,t_eval=ts,method='DOP853',rtol=2e-13,atol=2e-14)
    ref=sol.y[:,-1];ref_samples.extend(sol.y[:,1:].T)
ref_samples=np.array(ref_samples)
metrics['controlled_reference_max_state_error']=float(np.max(np.linalg.norm(z[:,:2]-ref_samples[:,:2],axis=1)))
fig,axs=plt.subplots(1,2,figsize=(11,3.8))
axs[0].plot(t,dH,color=COL['rk4'],label='Energy change');axs[0].plot(t,W,'--',color=COL['verlet'],label='Accumulated work');axs[0].legend();axs[0].set(xlabel='Time',ylabel='Energy / signed work',title='Controlled pendulum: ΔH = ∫pu dt')
axs[1].plot(t,res,color='#476ca0');axs[1].set(xlabel='Time',ylabel='ΔH − W',title='Numerical work-balance residual');axs[1].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
save(fig,'03_control_work_balance')

env=PendulumEnv(max_steps=3);env.reset((.2,.3))
for _ in range(3):o,r,term,trunc,info=env.step(3.)
assert trunc and not term and o.shape==(3,) and np.isfinite(r)
assert abs(np.cos(env.z[0])-o[0])<1e-14
metrics['env_balance_after_smoke_test']=info['balance_error']
assert metrics['rk4_observed_order']>3.7
assert 1.8<metrics['verlet_observed_order']<2.2
assert metrics['controlled_rk4_balance_max']<1e-7
assert metrics['controlled_reference_max_state_error']<1e-5
assert max(metrics[f'{k}_rk4_energy_max'] for k in cases)<1e-6

# Independent energies and uniform time-phase on each periodic orbit, not uniform q.
splits={'train':(.2,1.2,128),'validation':(.2,1.2,32),
 'test_moderate':(1.3,1.7,32),'test_near_below':(1.8,1.95,32),
 'test_near_above':(2.05,2.2,32),'test_rotation':(2.4,3.,32)}
rng=np.random.default_rng(20261013);allz=[];ids=[];names=[];summaries={};idx=0
for name,(low,high,count) in splits.items():
    states=[];energies=[];phases=[]
    for _ in range(count):
        E=rng.uniform(low,high);sign=rng.choice([-1,1]);phase=rng.uniform(0,1)
        period=4*ellipk(E/2) if E<2 else 2*np.sqrt(2/E)*ellipk(2/E)
        start=np.array([0,sign*np.sqrt(2*E),0.])
        if phase>0:
            start=solve_ivp(lambda t,y:rhs(y),(0,phase*period),start,method='DOP853',rtol=1e-11,atol=1e-12).y[:,-1]
        traj=simulate(start,.02,200)[:,:2]
        states.append(traj);energies.append(E);phases.append(phase);allz.append(traj);ids.append(idx);names.append(name);idx+=1
    states=np.array(states)
    np.savez_compressed(OUT/f'{name}.npz',states=states,actions=np.zeros((count,200)),
                        initial_energy=np.array(energies),initial_phase=np.array(phases),
                        trajectory_id=np.arange(idx-count,idx),dt=.02)
    summaries[name]={'trajectories':count,'transitions':count*200,
        'initial_energy_min':float(min(energies)),'initial_energy_max':float(max(energies)),
        'max_energy_drift':float(np.max(np.abs(energy(states)-np.array(energies)[:,None])))}
fig,axs=plt.subplots(1,2,figsize=(11,4.3))
colors=['#218b87','#69b6b0','#bb9349','#d7b76d','#cc705b','#476ca0']
for (name,stats),col in zip(summaries.items(),colors):
    data=np.load(OUT/f'{name}.npz');zs=data['states']
    axs[0].scatter(np.arctan2(np.sin(zs[:,0,0]),np.cos(zs[:,0,0])),zs[:,0,1],s=10,alpha=.7,color=col,label=name)
    axs[1].hist(data['initial_energy'],bins=12,alpha=.6,color=col,label=name)
q=np.linspace(-np.pi,np.pi,400);pp=np.sqrt(2*(1+np.cos(q)))
axs[0].plot(q,pp,color='#81929d',lw=1);axs[0].plot(q,-pp,color='#81929d',lw=1)
axs[0].set(xlabel='Wrapped angle q',ylabel='Momentum p',title='Independent initial states; uniform orbit time-phase')
axs[1].set(xlabel='Initial true energy',ylabel='Trajectory count',title='Disjoint test energy bands');axs[1].legend(fontsize=7)
save(fig,'04_dataset_coverage')
assert len(set(ids))==288
assert max(v['max_energy_drift'] for v in summaries.values())<1e-6
metrics['all_verification_checks_passed']=True
np.savetxt(OUT/'convergence.csv',np.column_stack([hs,errors['rk4'],errors['verlet']]),delimiter=',',header='h,rk4_error,verlet_error',comments='')
np.savetxt(OUT/'work_balance.csv',np.column_stack([t,dH,W,res]),delimiter=',',header='time,energy_change,work,residual',comments='')
(OUT/'metrics.json').write_text(json.dumps(metrics,indent=2))
manifest={'seed':20261013,'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
 'dt':.02,'duration':4.,'controlled_training_data':False,'splits':summaries,
 'files':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in OUT.glob('*.npz')}}
(OUT/'dataset_manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(metrics,indent=2));print('Created 288 trajectories / 57,600 transitions. No trained models or RL results.')
