"""Actual deterministic learned-policy rollouts in the true pendulum."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
from simulator import energy

OUT=Path(__file__).resolve().parent/'phase3/calibrated'
KINDS=['model_free','true_model','force_zero_mean','hnn']
TITLES=['Model-free SAC','True-model SAC','Zero-mean model + SAC','HNN model + SAC']
COL=['#cc705b','#243746','#9a78a7','#218b87']
data={k:np.load(OUT/f'{k}_seed0_h{0 if k=="model_free" else 5}_u4_trajectory.npz') for k in KINDS}
initial=[d['states'][0,0] for d in data.values()]
assert all(np.allclose(z,initial[0]) for z in initial)
fig,axs=plt.subplots(2,4,figsize=(11,5.5),gridspec_kw={'height_ratios':[1.3,1]})
fig.patch.set_facecolor('#fbfcfc');rods=[];tracks=[];dots=[];text=[]
q=np.linspace(-np.pi,np.pi,300);p=np.sqrt(2*(1+np.cos(q)))
for j,(k,col,title) in enumerate(zip(KINDS,COL,TITLES)):
    ax=axs[0,j];ax.set(xlim=(-1.3,1.3),ylim=(-1.3,1.3),aspect='equal',title=title);ax.axis('off')
    ax.plot(0,0,'o',color='#81929d',ms=4)
    rod,=ax.plot([],[],color=col,lw=2.5,marker='o',markevery=[1],ms=10);rods.append(rod)
    text.append(ax.text(.5,.005,'',transform=ax.transAxes,ha='center',fontsize=9))
    ax=axs[1,j];ax.plot(q,p,color='#d0d8dc',lw=.8);ax.plot(q,-p,color='#d0d8dc',lw=.8)
    ax.set(xlim=(-np.pi,np.pi),ylim=(-4,4),xlabel='Wrapped angle q');ax.set_xticks([-np.pi,0,np.pi],['−π','0','π']);ax.spines[['top','right']].set_visible(False)
    track,=ax.plot([],[],lw=1,color=col,alpha=.6);dot,=ax.plot([],[],'o',ms=5,color=col);tracks.append(track);dots.append(dot)
axs[1,0].set_ylabel('Momentum p');banner=fig.suptitle('')
fig.text(.5,.015,'Actual true-environment control | seed 0 | fixed median-energy near-above initial state | playback 2×',ha='center',fontsize=9,color='#637582')
fig.tight_layout(rect=[0,.045,1,.95])
def update(i):
    banner.set_text(f'Learned control from the same initial state · t = {i*.05:.1f}')
    for j,k in enumerate(KINDS):
        z=data[k]['states'][:,0];qv=z[i,0];rods[j].set_data([0,np.sin(qv)],[0,-np.cos(qv)])
        traj=z[max(0,i-80):i+1].copy();angles=traj[:,0].copy();angles[np.flatnonzero(np.abs(np.diff(angles))>np.pi)+1]=np.nan
        tracks[j].set_data(angles,traj[:,1]);dots[j].set_data([qv],[z[i,1]])
        u=data[k]['actions'][min(i,399),0,0]
        text[j].set_text(f'H={energy(z[i]):.2f} · u={u:+.2f}')
    return rods+tracks+dots+text+[banner]
update(200);fig.savefig(OUT/'control_animation_preview.png',dpi=140)
ani=FuncAnimation(fig,update,frames=range(0,401,4),interval=100,blit=False)
ani.save(OUT/'closed_loop_comparison.gif',writer=PillowWriter(fps=10),dpi=100);plt.close(fig)
print('Actual closed-loop animation created; initial states verified identical.')
