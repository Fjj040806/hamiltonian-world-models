"""Actual rotation-policy transfer from a fixed, common initial state."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter

OUT=Path(__file__).resolve().parent/'phase4';R=OUT/'rotation'
names=['true_only_w-1_p0_b16000','node_w0_p0_b0','force_zero_mean_w0_p0_b0','hnn_w0_p0_b0']
titles=['True-only training','Neural ODE world model','Zero-mean force world model','Hamiltonian world model']
colors=['#243746','#81929d','#9a78a7','#218b87'];data=[np.load(R/f'{n}_trajectory.npz') for n in names]
assert all(np.allclose(d['states'][0],data[0]['states'][0]) for d in data)
fig,axs=plt.subplots(2,4,figsize=(12,5.4),gridspec_kw={'height_ratios':[1.15,1]})
fig.patch.set_facecolor('#fbfcfc');rods=[];lines=[];points=[];labels=[];t=np.arange(401)*.05
for j,(title,col) in enumerate(zip(titles,colors)):
    ax=axs[0,j];ax.set(xlim=(-1.25,1.25),ylim=(-1.25,1.25),aspect='equal',title=title);ax.axis('off');ax.plot(0,0,'o',color='#81929d',ms=4)
    rod,=ax.plot([],[],color=col,lw=2.5,marker='o',markevery=[1],ms=10);rods.append(rod)
    labels.append(ax.text(.5,.01,'',transform=ax.transAxes,ha='center',fontsize=9))
    ax=axs[1,j];ax.axhline(2.5,color='#243746',ls='--',lw=1);ax.axhspan(2.35,2.65,color='#218b87',alpha=.08)
    ax.set(xlim=(0,20),ylim=(-.5,3.3),xlabel='Time');ax.spines[['top','right']].set_visible(False)
    line,=ax.plot([],[],color=col,lw=1.3);point,=ax.plot([],[],'o',color=col,ms=4);lines.append(line);points.append(point)
axs[1,0].set_ylabel('True momentum p');banner=fig.suptitle('')
fig.text(.5,.015,'Actual true-environment rollouts | world seed 0 / policy seed 0 | learned-model policies: zero added training oracle transitions | playback 2×',ha='center',fontsize=8,color='#637582')
fig.tight_layout(rect=[0,.045,1,.95])
def update(i):
    banner.set_text(f'Persistent rotation target p*=2.5 · t={i*.05:.1f}')
    for j,d in enumerate(data):
        z=d['states'][:,0];q=z[i,0];rods[j].set_data([0,np.sin(q)],[0,-np.cos(q)])
        lines[j].set_data(t[:i+1],z[:i+1,1]);points[j].set_data([t[i]],[z[i,1]])
        labels[j].set_text(f'p={z[i,1]:.3f} · error={z[i,1]-2.5:+.3f}')
    return rods+lines+points+labels+[banner]
update(200);fig.savefig(OUT/'rotation_animation_preview.png',dpi=130)
ani=FuncAnimation(fig,update,frames=range(0,401,4),interval=100,blit=False)
ani.save(OUT/'rotation_control_comparison.gif',writer=PillowWriter(fps=10),dpi=95);plt.close(fig)
print('Actual rotation control animation created.')
