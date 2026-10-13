"""Animate exact coupled-rotor intervention trajectories, first 20 time units."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
from coupled_pendulum import energy,KINDS

OUT=Path(__file__).resolve().parent/'phase4'
data=np.load(OUT/'coupled_intervention_trajectories.npz');dt=float(data['dt'])
colors=['#243746','#218b87','#cc705b','#9a78a7']
titles=['True Hamiltonian','Conservative error','Zero curl, nonzero cycle work','Nonzero curl error']
fig,axs=plt.subplots(2,4,figsize=(12,5.6),gridspec_kw={'height_ratios':[1.2,1]})
fig.patch.set_facecolor('#fbfcfc');rods=[];lines=[];points=[];texts=[]
end=round(20/dt);time=np.arange(end+1)*dt
drifts={k:energy(data[k][:end+1])-energy(data[k][0]) for k in KINDS}
ymin=min(v.min() for v in drifts.values());ymax=max(v.max() for v in drifts.values())
for j,(kind,col,title) in enumerate(zip(KINDS,colors,titles)):
    ax=axs[0,j];ax.set(xlim=(-1.4,1.4),ylim=(-1.05,1.05),aspect='equal',title=title);ax.axis('off')
    ax.plot([-.65,.65],[0,0],'--',color='#b9c5cd',lw=1)
    pair=[]
    for x in [-.65,.65]:
        ax.plot(x,0,'o',color='#81929d',ms=4)
        rod,=ax.plot([],[],color=col,lw=2.4,marker='o',markevery=[1],ms=8);pair.append(rod)
    rods.append(pair);texts.append(ax.text(.5,.015,'',transform=ax.transAxes,ha='center',fontsize=9))
    ax=axs[1,j];ax.set(xlim=(0,20),ylim=(ymin-.1,ymax+.2),xlabel='Time');ax.spines[['top','right']].set_visible(False)
    line,=ax.plot([],[],color=col,lw=1.4);dot,=ax.plot([],[],'o',color=col,ms=4);lines.append(line);points.append(dot)
axs[1,0].set_ylabel('Change in base Hamiltonian');banner=fig.suptitle('')
fig.text(.5,.012,'Analytic force interventions, not learned policies | identical initial state | angular coupling schematic | first 20 time units',ha='center',fontsize=9,color='#637582')
fig.tight_layout(rect=[0,.045,1,.95])
def update(i):
    banner.set_text(f'Matched RMS force error, different global work structure · t={i*dt:.1f}')
    for j,kind in enumerate(KINDS):
        z=data[kind][i]
        for k,x in enumerate([-.65,.65]):rods[j][k].set_data([x,x+.6*np.sin(z[k])],[0,-.6*np.cos(z[k])])
        lines[j].set_data(time[:i+1],drifts[kind][:i+1]);points[j].set_data([time[i]],[drifts[kind][i]])
        texts[j].set_text(f'p₁={z[2]:.2f}, p₂={z[3]:.2f} · ΔH={drifts[kind][i]:+.2f}')
    return [r for pair in rods for r in pair]+lines+points+texts+[banner]
update(round(10/dt));fig.savefig(OUT/'work_loop_animation_preview.png',dpi=130)
ani=FuncAnimation(fig,update,frames=range(0,end+1,20),interval=100,blit=False)
ani.save(OUT/'coupled_work_loop_comparison.gif',writer=PillowWriter(fps=10),dpi=95);plt.close(fig)
print('Coupled work-loop animation saved.')
