"""Posthoc diagnostic figures with explicit controller and regularizer scope."""
from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from coupled_models import FourierForce,truth_potential
from plot_phase5 import COLORS,LABELS,KINDS,save,OUT
rr=json.loads((OUT/'tracking/results.json').read_text());cr=json.loads((OUT/'control/results.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
for ax,key,title in zip(axes,['momentum_vector_rms_mean','return_mean'],['Final momentum vector RMS','True-plant return']):
    for i,(k,c) in enumerate(zip(KINDS,COLORS)):
        a=[x[key] for x in cr if x['kind']==k and x['split']=='rotation'];b=[x[key] for x in rr if x['kind']==k and x['split']=='rotation']
        ax.bar(i-.17,np.mean(a),.32,color=c,alpha=.9);ax.bar(i+.17,np.mean(b),.32,color=c,alpha=.35,hatch='//')
    a=np.mean([x[key] for x in cr if x['kind']=='true_only' and x['split']=='rotation'])
    b=np.mean([x[key] for x in rr if x['kind']=='known_force' and x['split']=='rotation'])
    ax.bar(5-.17,a,.32,color='#343c48');ax.bar(5+.17,b,.32,color='#343c48',alpha=.35,hatch='//')
    ax.set_xticks(range(6),LABELS+['Privileged truth'],rotation=25);ax.set_title(title)
axes[0].set_ylim(0,.52)
from matplotlib.patches import Patch
axes[0].legend(handles=[Patch(facecolor='gray',label='SAC, 3 × 3 seeds (truth: 3)'),Patch(facecolor='gray',alpha=.35,hatch='//',label='Fixed gain-2 feedback, 3 world seeds (truth: 1)')],fontsize=8)
fig.suptitle('Feedback diagnostic: same learned forces; separate policy optimization from model transfer',fontsize=12)
save(fig,'07_feedback_diagnostic')
fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
x=np.linspace(-np.pi,np.pi,401);q=np.stack([x,np.zeros_like(x)],1)
for ax,plant,eps in zip(axes,['canonical','misspecified'],[0,.06]):
    y=[]
    for w in [0,1,2]:
        m=FourierForce.load(OUT/plant/f'hamiltonian_{w}.npz');y.append(m.potential(q)-m.potential(np.zeros((1,2)))[0])
    y=np.array(y);ax.plot(x,truth_potential(q,eps),c='#343c48',lw=2,label='True potential')
    ax.plot(x,y.mean(0),c=COLORS[-1],lw=2,label='Learned Hamiltonian potential');ax.fill_between(x,y.min(0),y.max(0),color=COLORS[-1],alpha=.2)
    ax.set_xlabel('$q_1$, with $q_2=0$');ax.set_ylabel('$V(q_1,0)-V(0,0)$');ax.set_title(plant.capitalize())
axes[0].legend(fontsize=9);fig.suptitle('Periodic potentials conserve learned energy but misestimate the unseen landscape',fontsize=12)
save(fig,'08_potential_landscape')
