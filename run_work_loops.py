"""Matched-force-error interventions; these are analytic models, not learned ones."""
from pathlib import Path
import json
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from coupled_pendulum import KINDS,force,energy,potential,rhs,simulate

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'phase4';OUT.mkdir(exist_ok=True)
COL={'true':'#243746','conservative':'#218b87','cycle_bias':'#cc705b','curl_error':'#9a78a7'}
q=np.linspace(-np.pi,np.pi,256,endpoint=False);Q1,Q2=np.meshgrid(q,q)
grid=np.stack([Q1,Q2],axis=-1);stats={};a=.8
def line_work(start,end,kind):
    s=np.linspace(0,1,2001);delta=np.array(end)-np.array(start)
    qs=np.array(start)[None,:]+s[:,None]*delta
    return float(np.trapezoid(force(qs,kind)@delta,s))
corners=[[-a,-a],[a,-a],[a,a],[-a,a],[-a,-a]]
initial=np.array([0.,.4,3.,.3]);h=.005;T=60.;trajs={}
for kind in KINDS:
    error=force(grid,kind)-force(grid,'true')
    rms=float(np.sqrt(np.mean(np.sum(error*error,axis=-1))))
    period=line_work([0,0],[2*np.pi,0],kind)
    contractible=sum(line_work(x,y,kind) for x,y in zip(corners[:-1],corners[1:]))
    eps=1e-5
    d1=(force(grid+np.array([eps,0]),kind)[...,1]-force(grid-np.array([eps,0]),kind)[...,1])/(2*eps)
    d2=(force(grid+np.array([0,eps]),kind)[...,0]-force(grid-np.array([0,eps]),kind)[...,0])/(2*eps)
    curlrms=float(np.sqrt(np.mean((d1-d2)**2)))
    z=simulate(initial,h,round(T/h),kind);trajs[kind]=z
    reference=solve_ivp(lambda t,z:rhs(z,kind),(0,10),initial,method='DOP853',rtol=1e-12,atol=1e-13).y[:,-1]
    endpoint_error=float(np.linalg.norm(z[round(10/h)]-reference))
    st={'uniform_torus_force_rms_error':rms,'fundamental_cycle_work':period,'contractible_square_work':contractible,
        'curl_rms':curlrms,'endpoint_reference_error_T10':endpoint_error,
        'max_true_energy_change':float(np.max(np.abs(energy(z)-energy(initial)))),
        'unwrapped_q1_displacement':float(z[-1,0]-z[0,0])}
    if kind=='true':st['conservation_error']=st['max_true_energy_change']
    if kind=='conservative':
        HH=.5*np.sum(z[:,2:]**2,axis=1)+potential(z[:,:2],kappa=.34)
        st['modified_hamiltonian_conservation_error']=float(np.max(np.abs(HH-HH[0])))
        assert st['max_true_energy_change']<.08001
    if kind=='cycle_bias':
        residual=energy(z)-energy(z[0])-.04*(z[:,0]-z[0,0])
        st['work_identity_max_residual']=float(np.max(np.abs(residual)))
        assert st['work_identity_max_residual']<1e-5
        assert abs(period-2*np.pi*.04)<1e-12 and curlrms<1e-8
    if kind=='curl_error':assert abs(contractible+8*.04*a*np.sin(a))<1e-6
    assert endpoint_error<1e-6
    if kind!='true':assert abs(rms-.04)<1e-12
    stats[kind]=st
assert stats['true']['conservation_error']<1e-6
np.savez_compressed(OUT/'coupled_intervention_trajectories.npz',**trajs,dt=h)
(OUT/'work_loop_results.json').write_text(json.dumps(stats,indent=2))
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
def save(fig,name):
    fig.tight_layout()
    for ext in ['png','svg']:fig.savefig(OUT/f'{name}.{ext}',dpi=175)
    plt.close(fig)
fig,axs=plt.subplots(1,3,figsize=(12,4))
small=np.linspace(-np.pi,np.pi,15,endpoint=False);A,B=np.meshgrid(small,small);gg=np.stack([A,B],-1)
for ax,kind in zip(axs,KINDS[1:]):
    e=force(gg,kind)-force(gg,'true');ax.quiver(A,B,e[...,0],e[...,1],color=COL[kind],scale=.6,width=.004)
    ax.set(xlabel='q₁',ylabel='q₂',title=kind.replace('_',' '));ax.set_aspect('equal')
    ax.set_xticks([-np.pi,0,np.pi],['−π','0','π']);ax.set_yticks([-np.pi,0,np.pi],['−π','0','π'])
fig.suptitle('Matched force error on the torus: vector RMS = 0.04 in all three cases')
save(fig,'06_torus_force_interventions')
fig,axs=plt.subplots(1,2,figsize=(11,4.2));t=np.arange(round(T/h)+1)*h
for kind,z in trajs.items():
    axs[0].plot(t,energy(z)-energy(z[0]),color=COL[kind],label=kind,lw=1.2)
axs[0].set(xlabel='Time',ylabel='Change in true base Hamiltonian',title='Equal force RMS does not imply equal energy behavior');axs[0].legend(fontsize=8)
work=np.array([[stats[k]['fundamental_cycle_work'],stats[k]['contractible_square_work']] for k in KINDS[1:]])
for i,kind in enumerate(KINDS[1:]):axs[1].bar(np.arange(2)+(i-1)*.23,work[i],width=.23,color=COL[kind],label=kind)
axs[1].set_xticks([0,1],['Noncontractible q₁ cycle','Contractible square']);axs[1].set(ylabel='Net force work',title='Different loops detect different structural errors');axs[1].legend(fontsize=8)
save(fig,'07_work_loops_and_energy')
print(json.dumps(stats,indent=2))
