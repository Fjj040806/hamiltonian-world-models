from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':'#243746','text.color':'#243746','axes.edgecolor':'#81929d','figure.facecolor':'#fbfcfc','axes.facecolor':'#fbfcfc','svg.fonttype':'none'})
q=np.linspace(-np.pi,np.pi,1200)
p=np.linspace(-2.7,2.7,900)
Q,P=np.meshgrid(q,p)
H=P**2/2+1-np.cos(Q)
teal='#218b87'; coral='#cc705b'; blue='#476ca0'; gold='#bb9349'
fig,ax=plt.subplots(figsize=(10.8,6.5))
ax.contourf(Q,P,H,levels=[.2,1.2],colors=[teal],alpha=.18)
ax.contour(Q,P,H,levels=[.3,.7,1.1],colors=teal,linewidths=1.2)
ax.contour(Q,P,H,levels=[1.8,1.95],colors=gold,linewidths=1.2,linestyles='dashed')
ax.contour(Q,P,H,levels=[2],colors=coral,linewidths=2.4)
ax.contour(Q,P,H,levels=[2.2,2.7,3.0],colors=blue,linewidths=1.1)
ax.scatter([0],[0],s=36,color=teal,zorder=5)
ax.scatter([-np.pi,np.pi],[0,0],marker='x',s=55,color=coral,zorder=5)
ax.set(xlim=(-np.pi,np.pi),ylim=(-2.7,2.7),xlabel='Angle q (radians; endpoints identified)',ylabel='Canonical momentum p',title='Energy regimes of the normalized Hamiltonian pendulum')
ax.set_xticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi],['−π','−π/2','0','π/2','π'])
ax.text(0,-.5,'Libration',ha='center',color=teal)
ax.text(0,2.45,'Rotation',ha='center',color=blue)
ax.legend(handles=[Line2D([0],[0],color=teal,lw=6,alpha=.4,label='Proposed training band: 0.2 ≤ H ≤ 1.2'),Line2D([0],[0],color=gold,ls='--',label='Near-boundary test levels'),Line2D([0],[0],color=coral,lw=2,label='Uncontrolled separatrix: H = 2'),Line2D([0],[0],color=blue,label='Rotation test levels')],loc='lower left',fontsize=9,framealpha=.94)
fig.text(.08,.025,'Exact contours: H = p²/2 + 1 − cos(q). Proposed split, not training results. Control can cross energy levels.',fontsize=9,color='#637582')
fig.tight_layout(rect=[0,.055,1,1])
for ext in ['png','svg']:fig.savefig(OUT/f'01_energy_regimes.{ext}',dpi=190)
plt.close(fig)

fig,axs=plt.subplots(1,2,figsize=(12,5.4))
delta=.12
axs[0].plot(q,1-np.cos(q),color=teal,lw=2.4,label='True potential')
axs[0].plot(q,(1+delta)*(1-np.cos(q)),color=coral,lw=2,ls='--',label='Illustrative perturbed potential')
axs[0].axhline(2,color=teal,alpha=.4,lw=1)
axs[0].axhline(2*(1+delta),color=coral,alpha=.4,lw=1)
axs[0].set(xlabel='Angle q',ylabel='Potential energy',title='A conservative model can learn the wrong barrier')
axs[0].legend(fontsize=9,loc='upper center')
energy=2.12
for a,col,label in [(1,teal,'True model: rotation'),(1+delta,coral,'Perturbed model: libration')]:
    rad=2*(energy-a*(1-np.cos(q)))
    pp=np.sqrt(np.maximum(rad,0));pp[rad<0]=np.nan
    axs[1].plot(q,pp,color=col,lw=2,label=label)
    axs[1].plot(q,-pp,color=col,lw=2)
axs[1].set(xlabel='Angle q',ylabel='Canonical momentum p',title='Same initial state: q = 0, p = √(2 × 2.12)')
axs[1].scatter([0],[np.sqrt(2*energy)],s=35,color='#243746',zorder=5)
axs[1].legend(loc='lower center',fontsize=9)
for ax in axs:ax.set_xticks([-np.pi,0,np.pi],['−π','0','π']);ax.grid(alpha=.12)
fig.text(.05,.02,'Analytic illustration only: Ĥ = p²/2 + 1.12(1 − cos q). Both uncontrolled models conserve their own energy.',fontsize=9,color='#637582')
fig.tight_layout(rect=[0,.06,1,1])
for ext in ['png','svg']:fig.savefig(OUT/f'02_barrier_mismatch.{ext}',dpi=190)
plt.close(fig)
assert energy>2 and energy<2*(1+delta)
print('Created two analytic figures (PNG + SVG); barrier classification verified.')
