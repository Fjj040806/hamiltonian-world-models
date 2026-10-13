"""Actual selected query paths and recorded physical control trajectories."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
from matplotlib.colors import LogNorm
from phase7_design import *
METHODS=P['methods'];LABELS=['Random','Uncertainty','Global A','Occupancy A','Spectral A'];COLORS=['#73695f','#477caa','#359886','#9b78a8','#d58047']
plt.rcParams.update({'figure.facecolor':'#fafaf8','axes.facecolor':'#fafaf8','font.size':9,'axes.spines.top':False,'axes.spines.right':False})

def acquisition():
    data=np.load(OUT/'data/misspecified_20.npz');q=data['candidate_q'];pilot=data['pilot_states'][:,:2];L=basis_map('hamiltonian');Bp=blocks(pilot,L)
    H=P['prior_precision_force_whitened']*np.eye(L.shape[1])+np.einsum('nfi,nfj->ij',Bp,Bp);inv0=np.linalg.inv(H)
    a=np.linspace(-np.pi,np.pi,45);grid=np.stack(np.meshgrid(a,a,indexing='ij'),-1);Bg=blocks(grid.reshape(-1,2),L)
    caches=[];ids=[];globalvar=[]
    for method in METHODS:
        task='forward' if method in ['occupancy_A','spectral_A'] else 'shared'
        selected=np.load(OUT/'paths'/f'misspecified_hamiltonian_w20_{method}_{task}.npz')['candidate_indices'][:64];ids.append(selected)
        inv=inv0.copy();fields=[];vv=[]
        for n in range(65):
            S=(Bg@inv)@Bg.transpose(0,2,1);fields.append(P['noise_std']**2*np.trace(S,axis1=1,axis2=2).reshape(45,45));vv.append(P['noise_std']**2*np.trace(inv))
            if n<64:inv=posterior_update(inv,blocks(q[selected[n]:selected[n]+1],L)[0])
        caches.append(fields);globalvar.append(vv)
    vmax=max(float(np.max(x)) for f in caches for x in f);vmin=min(float(np.min(x)) for f in caches for x in f)
    fig,axes=plt.subplots(2,3,figsize=(12,7.5));fig.subplots_adjust(top=.86,hspace=.32,wspace=.3,bottom=.09,right=.88)
    fig.suptitle('Actual force-query acquisition at equal measurement budgets',fontsize=14,y=.98)
    fig.text(.5,.935,'Misspecified plant · Hamiltonian features · world seed 20 · forward task',ha='center',fontsize=10)
    fig.text(.5,.905,'Background shows model posterior force variance; approximation bias is excluded',ha='center',fontsize=9)
    images=[];points=[];last=[]
    for ax,label,color,field in zip(axes.flat,LABELS,COLORS,caches):
        im=ax.imshow(field[0].T,origin='lower',extent=(-np.pi,np.pi,-np.pi,np.pi),norm=LogNorm(vmin=max(vmin,1e-7),vmax=vmax),cmap='magma');images.append(im)
        ax.scatter(pilot[:,0],pilot[:,1],s=4,c='#89a8ac',alpha=.6);sc=ax.scatter([],[],s=20,facecolors='none',edgecolors='cyan',linewidths=1);points.append(sc)
        la,=ax.plot([],[],'*',c='white',ms=10);last.append(la);ax.set_title(label,color=color);ax.set_xlabel('$q_1$');ax.set_ylabel('$q_2$')
    ax=axes.flat[-1];ax.set_title('Integrated posterior force variance');ax.set_xlim(0,64);ax.set_yscale('log');ax.set_ylim(min(min(v) for v in globalvar)*.8,max(max(v) for v in globalvar)*1.2)
    curves=[]
    for color,label in zip(COLORS,LABELS):line,=ax.plot([],[],color=color,label=label,lw=1.5);curves.append(line)
    ax.set_xlabel('Added force queries');ax.set_ylabel('Posterior variance');ax.legend(fontsize=7)
    cax=fig.add_axes([.925,.2,.014,.58]);fig.colorbar(images[0],cax=cax,label='Posterior force variance')
    clock=fig.text(.5,.025,'',ha='center',fontsize=10)
    def update(n):
        clock.set_text(f'128 shared pilot labels + {n} acquired candidate labels · query process, not physical time')
        for image,point,marker,idx,field,curve,var in zip(images,points,last,ids,caches,curves,globalvar):
            image.set_data(field[n].T);point.set_offsets(q[idx[:n]] if n else np.empty((0,2)))
            marker.set_data([q[idx[n-1],0]],[q[idx[n-1],1]]) if n else marker.set_data([],[])
            curve.set_data(np.arange(n+1),var[:n+1])
    anim=FuncAnimation(fig,update,frames=65,interval=120);anim.save(OUT/'phase7_acquisition.gif',writer=PillowWriter(fps=8));update(32)
    fig.savefig(OUT/'acquisition_animation_preview.png',dpi=160,bbox_inches='tight');plt.close(fig)

def control():
    methods=['random','global_A','spectral_A'];labels=['Random','Global A design','Spectral A design'];cols=[COLORS[0],COLORS[2],COLORS[4]]
    fig=plt.figure(figsize=(12,8));outer=fig.add_gridspec(2,3,left=.06,right=.97,bottom=.09,top=.86,hspace=.32,wspace=.28)
    fig.suptitle('Fixed-budget sampling strategies on the same true coupled plant',fontsize=14,y=.98)
    fig.text(.5,.934,'Misspecified plant · Hamiltonian features · world seed 20 · forward task · fixed initial phase 0',ha='center',fontsize=9)
    fig.text(.5,.9,'Top row: 16 added queries     Bottom row: 64 added queries     Every run shares 128 pilot labels',ha='center',fontsize=9)
    lines=[];texts=[];curves=[];arrays=[]
    for r,budget in enumerate([16,64]):
        for c,(method,label,color) in enumerate(zip(methods,labels,cols)):
            task='forward' if method=='spectral_A' else 'shared';stem=f'misspecified_hamiltonian_w20_{method}_{task}_b{budget}_forward'
            z=np.load(OUT/'control'/f'{stem}_trajectory.npz')['states'][:,0];arrays.append(z)
            gs=outer[r,c].subgridspec(2,1,height_ratios=[1.7,1],hspace=.04);ax=fig.add_subplot(gs[0]);ea=fig.add_subplot(gs[1])
            ax.set_xlim(-.8,2.1);ax.set_ylim(-.9,.85);ax.set_aspect('equal');ax.set_xticks([]);ax.set_yticks([]);ax.set_title(label,color=color)
            ax.plot([0,1.3],[0,0],':',c='#a0a5a8',lw=1);ax.scatter([0,1.3],[0,0],s=18,c='#343c48');ll=[]
            for j,pivot in enumerate([0,1.3]):line,=ax.plot([],[],'-o',c=color,lw=2,ms=7);ll.append((line,pivot,j))
            lines.append(ll);texts.append(ax.text(.03,.03,'',transform=ax.transAxes,fontsize=8))
            ea.set_xlim(0,20);ea.set_ylim(.0005,.5);ea.set_yscale('log');ea.set_xlabel('Time');ea.set_ylabel('$||p-p^*||$');line,=ea.plot([],[],c=color,lw=1.5);curves.append(line)
    clock=fig.text(.5,.025,'',ha='center',fontsize=10)
    def update(frame):
        t=frame*4;clock.set_text(f't = {t*.05:.1f} · playback 2× · real recorded RK4 trajectories · target momentum (2.5, 1.5)')
        for z,ll,txt,curve in zip(arrays,lines,texts,curves):
            for line,pivot,j in ll:line.set_data([pivot,pivot+.65*np.sin(z[t,j])],[0,-.65*np.cos(z[t,j])])
            txt.set_text(f'p = ({z[t,2]:+.3f}, {z[t,3]:+.3f})');curve.set_data(np.arange(t+1)*.05,np.maximum(np.linalg.norm(z[:t+1,2:]-[2.5,1.5],axis=1),1e-7))
    anim=FuncAnimation(fig,update,frames=101,interval=100);anim.save(OUT/'phase7_control_comparison.gif',writer=PillowWriter(fps=10));update(80)
    fig.savefig(OUT/'control_animation_preview.png',dpi=160,bbox_inches='tight');plt.close(fig)
if __name__=='__main__':acquisition();control()
