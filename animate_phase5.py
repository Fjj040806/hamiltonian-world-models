"""Animations of recorded trajectories, never generated schematic motion."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
from coupled_models import truth_energy
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'phase5'
plt.rcParams.update({'figure.facecolor':'#fafaf8','axes.facecolor':'#fafaf8','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
KINDS=['force','curl_only','curl_mean','hamiltonian'];LABELS=['Unconstrained force','Curl penalty','Curl + mean penalties','Hamiltonian features'];COLORS=['#8b6c56','#9b78a8','#359886','#d58047']

def prediction():
    fig,axes=plt.subplots(2,2,figsize=(9.8,7.1));fig.subplots_adjust(top=.88,hspace=.3,wspace=.24,bottom=.06)
    fig.suptitle('Coupled-system prediction outside low-energy training coverage',fontsize=14,y=.98)
    fig.text(.5,.935,'Actual RK4 trajectories · world seed 0 · fixed high-energy initial state 0 · dashed: true plant',ha='center',fontsize=9)
    lines=[];data=[];texts=[]
    for ax,k,label,color in zip(axes.flat,KINDS,LABELS,COLORS):
        d=np.load(OUT/'misspecified'/f'{k}_0_rotation_prediction.npz');tr=d['truth'][:,0];pr=d['prediction'][:,0];data.append((tr,pr))
        ax.set_xlim(-.8,2.1);ax.set_ylim(-.9,.85);ax.set_aspect('equal');ax.set_xticks([]);ax.set_yticks([]);ax.set_title(label,color=color)
        ax.plot([0,1.3],[0,0],':',c='#a0a5a8',lw=1);ax.scatter([0,1.3],[0,0],s=18,c='#343c48')
        ll=[]
        for j,pivot in enumerate([0,1.3]):
            tl,=ax.plot([],[],'--o',c='#9ba4ac',lw=1.5,ms=6,alpha=.7)
            pl,=ax.plot([],[],'-o',c=color,lw=2.2,ms=7);ll.append((tl,pl,pivot,j))
        lines.append(ll);texts.append(ax.text(.02,.03,'',transform=ax.transAxes,fontsize=9))
    clock=fig.text(.5,.02,'',ha='center',fontsize=10)
    def update(fr):
        i=fr*4;clock.set_text(f't = {i*.05:4.1f}  · playback 2×  · parallel angular coupling schematic')
        for ll,(tr,pr),txt in zip(lines,data,texts):
            for tl,pl,pivot,j in ll:
                tl.set_data([pivot,pivot+.65*np.sin(tr[i,j])],[0,-.65*np.cos(tr[i,j])]);pl.set_data([pivot,pivot+.65*np.sin(pr[i,j])],[0,-.65*np.cos(pr[i,j])])
            err=np.sqrt(np.sum(np.r_[np.arctan2(np.sin(pr[i,:2]-tr[i,:2]),np.cos(pr[i,:2]-tr[i,:2])),pr[i,2:]-tr[i,2:]]**2))
            txt.set_text(f'State error: {err:.2f}   True H shift: {truth_energy(pr[i])-truth_energy(pr[0]):+.2f}')
    anim=FuncAnimation(fig,update,frames=101,interval=100,blit=False)
    anim.save(OUT/'coupled_prediction_comparison.gif',writer=PillowWriter(fps=10));update(35);fig.savefig(OUT/'prediction_animation_preview.png',dpi=160,bbox_inches='tight');plt.close(fig)

def control():
    fig,axes=plt.subplots(2,3,figsize=(12,7));fig.subplots_adjust(top=.87,hspace=.28,wspace=.26,bottom=.07)
    fig.suptitle('Learned cyclic-control policies on the same true coupled plant',fontsize=14,y=.98)
    fig.text(.5,.925,'Zero additional true training transitions for learned-model policies · world seed 0 / policy seed 0 · fixed initial state 0',ha='center',fontsize=9)
    kinds=KINDS+['true_only'];labels=LABELS+['True-only training reference'];colors=COLORS+['#343c48'];lines=[];data=[];texts=[]
    for ax,k,label,color in zip(axes.flat,kinds,labels,colors):
        w=-1 if k=='true_only' else 0;z=np.load(OUT/'control'/f'{k}_w{w}_p0_trajectory.npz')['states'][:,0];data.append(z)
        ax.set_xlim(-.8,2.1);ax.set_ylim(-.9,.85);ax.set_aspect('equal');ax.set_xticks([]);ax.set_yticks([]);ax.set_title(label,color=color)
        ax.plot([0,1.3],[0,0],':',c='#a0a5a8',lw=1);ax.scatter([0,1.3],[0,0],s=18,c='#343c48');ll=[]
        for j,pivot in enumerate([0,1.3]):
            l,=ax.plot([],[],'-o',c=color,lw=2.2,ms=7);ll.append((l,pivot,j))
        lines.append(ll);texts.append(ax.text(.02,.03,'',transform=ax.transAxes,fontsize=9))
    ax=axes.flat[-1];ax.set_title('Momentum error in the displayed trajectories');ax.set_xlim(0,20);ax.set_ylim(0,max(np.linalg.norm(z[:,2:]-[2.5,1.5],axis=1).max() for z in data)*1.05)
    curves=[]
    for z,color,label in zip(data,colors,labels):
        l,=ax.plot([],[],c=color,label=label,lw=1.5);curves.append(l)
    ax.axhline(.25,c='gray',ls=':',lw=1);ax.set_xlabel('Time');ax.set_ylabel('$||p-p^*||$');ax.legend(fontsize=7,loc='upper right')
    clock=fig.text(.5,.025,'',ha='center',fontsize=10)
    def update(fr):
        i=fr*4;clock.set_text(f't = {i*.05:4.1f}  · playback 2×  · true dynamics in every panel')
        for ll,z,txt,curve in zip(lines,data,texts,curves):
            for l,pivot,j in ll:l.set_data([pivot,pivot+.65*np.sin(z[i,j])],[0,-.65*np.cos(z[i,j])])
            txt.set_text(f'p = ({z[i,2]:+.2f}, {z[i,3]:+.2f})')
            curve.set_data(np.arange(i+1)*.05,np.linalg.norm(z[:i+1,2:]-[2.5,1.5],axis=1))
    anim=FuncAnimation(fig,update,frames=101,interval=100,blit=False)
    anim.save(OUT/'coupled_control_comparison.gif',writer=PillowWriter(fps=10));update(80);fig.savefig(OUT/'control_animation_preview.png',dpi=160,bbox_inches='tight');plt.close(fig)
if __name__=='__main__':
    prediction()
    if (OUT/'control/results.json').exists():control()
