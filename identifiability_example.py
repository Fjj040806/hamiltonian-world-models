"""Exact smooth counterexample: same low-energy data, different barrier.

This elementary construction motivates the project; no novelty claim is made.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'phase2'
q=np.linspace(-np.pi,np.pi,2001);width=.6;amplitude=.2
def bump(q):
    r=np.arctan2(np.sin(q-np.pi),np.cos(q-np.pi));b=np.zeros_like(r)
    mask=np.abs(r)<width;s=r[mask]/width
    b[mask]=np.exp(1-1/(1-s*s))
    return b
base=1-np.cos(q);modified=base+amplitude*bump(q)
qmax=np.arccos(1-1.2);training=np.abs(q)<=qmax
assert np.max(np.abs(modified[training]-base[training]))==0
fig,axs=plt.subplots(1,2,figsize=(11,4.2))
axs[0].axvspan(-qmax,qmax,color='#218b87',alpha=.1,label='All angles accessible at H ≤ 1.2')
axs[0].plot(q,base,color='#218b87',lw=2,label='V(q) = 1 − cos(q)')
axs[0].plot(q,modified,color='#cc705b',ls='--',lw=2,label='V(q) + 0.2 b(q)')
axs[0].set(xlabel='Angle q',ylabel='Potential',title='Exactly identical on the entire training region');axs[0].legend(fontsize=8)
E=2.1
for v,col,lab in [(base,'#218b87','Original: rotation'),(modified,'#cc705b','Modified: libration')]:
    rad=2*(E-v);p=np.sqrt(np.maximum(rad,0));p[rad<0]=np.nan
    axs[1].plot(q,p,color=col,lw=2,label=lab);axs[1].plot(q,-p,color=col,lw=2)
axs[1].set(xlabel='Angle q',ylabel='Momentum p',title='Same initial state: (0, √4.2)');axs[1].legend(fontsize=8)
for ax in axs:ax.spines[['top','right']].set_visible(False);ax.set_xticks([-np.pi,0,np.pi],['−π','0','π'])
fig.text(.5,.01,'Analytic construction, not fitted-model results. Both Hamiltonians are smooth, periodic and conservative.',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.045,1,1])
for ext in ('png','svg'):fig.savefig(OUT/f'04_identifiability_obstruction.{ext}',dpi=175)
plt.close(fig)
print('Verified exact potential agreement on training domain and different rotation classification.')
