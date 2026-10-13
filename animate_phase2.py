"""Animate actual fitted-model trajectories; no generated illustration."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
from simulator import energy

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'phase2'
a=np.load(OUT/'animation_trajectories.npz')
names=['true','second_order','force_zero_mean','hnn'];colors=['#243746','#bb9349','#9a78a7','#218b87']
titles=['True dynamics','Second-order NN','Zero-mean force NN','Hamiltonian NN']
fig,axs=plt.subplots(2,4,figsize=(10.5,5.4),gridspec_kw={'height_ratios':[1.1,1]})
fig.patch.set_facecolor('#fbfcfc');rods=[];trails=[];dots=[];texts=[]
q=np.linspace(-np.pi,np.pi,300);sep=np.sqrt(2*(1+np.cos(q)))
for j,(name,col,title) in enumerate(zip(names,colors,titles)):
    ax=axs[0,j];ax.set(xlim=(-1.25,1.25),ylim=(-1.3,1.3),aspect='equal',title=title)
    ax.axis('off');ax.plot(0,0,'o',color='#81929d',ms=4)
    rod,=ax.plot([],[],color=col,lw=2.5,marker='o',markevery=[1],ms=10);rods.append(rod)
    texts.append(ax.text(.5,.01,'',transform=ax.transAxes,ha='center',fontsize=9))
    ax=axs[1,j];ax.plot(q,sep,color='#d0d8dc',lw=.8);ax.plot(q,-sep,color='#d0d8dc',lw=.8)
    ax.set(xlim=(-np.pi,np.pi),ylim=(-3,3),xlabel='Wrapped q');ax.set_xticks([-np.pi,0,np.pi],['−π','0','π'])
    ax.spines[['top','right']].set_visible(False)
    trail,=ax.plot([],[],color=col,lw=1.1,alpha=.65);dot,=ax.plot([],[],'o',color=col,ms=5)
    trails.append(trail);dots.append(dot)
axs[1,0].set_ylabel('Momentum p')
banner=fig.suptitle('')
fig.text(.5,.015,'Actual learned-model rollouts | seed 0 | median-energy near-above test trajectory | playback 2×',ha='center',fontsize=9,color='#637582')
fig.tight_layout(rect=[0,.04,1,.95])
def update(i):
    banner.set_text(f'Cross-energy prediction from the same initial state · t = {i*.02:.1f}')
    artists=[]
    for j,name in enumerate(names):
        z=a[name];angle=z[i,0];rods[j].set_data([0,np.sin(angle)],[0,-np.cos(angle)])
        start=max(0,i-250);trail=z[start:i+1].copy();wrapped=np.arctan2(np.sin(trail[:,0]),np.cos(trail[:,0]))
        jumps=np.flatnonzero(np.abs(np.diff(wrapped))>np.pi)+1
        wrapped[jumps]=np.nan
        trails[j].set_data(wrapped,trail[:,1]);dots[j].set_data([np.arctan2(np.sin(angle),np.cos(angle))],[z[i,1]])
        texts[j].set_text(f'True H = {energy(z[i]):.3f}')
        artists.extend([rods[j],trails[j],dots[j],texts[j]])
    return artists+[banner]
update(400);fig.savefig(OUT/'animation_preview.png',dpi=140)
ani=FuncAnimation(fig,update,frames=range(0,801,10),interval=100,blit=False)
ani.save(OUT/'prediction_comparison.gif',writer=PillowWriter(fps=10),dpi=95)
plt.close(fig);print('Actual prediction animation saved.')
