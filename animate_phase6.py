"""Recorded prediction/control animations with fixed preselected instances."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
from phase6_models import OUT,P,plant_energy
plt.rcParams.update({'figure.facecolor':'#fafaf8','axes.facecolor':'#fafaf8','font.size':9,'axes.spines.top':False,'axes.spines.right':False})

def rods(ax,color,dashed_truth=False):
    ax.set_xlim(-.8,2.1);ax.set_ylim(-.9,.85);ax.set_aspect('equal');ax.set_xticks([]);ax.set_yticks([])
    ax.plot([0,1.3],[0,0],':',c='#a0a5a8',lw=1);ax.scatter([0,1.3],[0,0],s=18,c='#343c48');lines=[]
    for j,pivot in enumerate([0,1.3]):
        if dashed_truth:
            true,=ax.plot([],[],'--o',c='#9ba4ac',lw=1.4,ms=6,alpha=.8)
        else:true=None
        line,=ax.plot([],[],'-o',c=color,lw=2,ms=7);lines.append((line,true,pivot,j))
    return lines

def prediction():
    configs=[('force','local','Force / 0% global data','#73695f'),('force','mixed','Force / 10% global data','#73695f'),
             ('hamiltonian','local','Hamiltonian / 0% global data','#d58047'),('hamiltonian','mixed','Hamiltonian / 10% global data','#d58047')]
    fig,axes=plt.subplots(2,2,figsize=(10,7.5));fig.subplots_adjust(top=.87,hspace=.27,wspace=.2,bottom=.08)
    fig.suptitle('Fixed data budget: a small coverage intervention changes extrapolation',fontsize=14,y=.98)
    fig.text(.5,.93,'Misspecified coupled plant · world seed 10 · fixed high-energy state 0 · dashed: true unforced trajectory',ha='center',fontsize=9)
    lines=[];data=[];texts=[]
    for ax,(k,c,label,color) in zip(axes.flat,configs):
        d=np.load(OUT/'misspecified'/f'{k}_{c}_10_rotation_diagnostics.npz');data.append((d['prediction_trace'],d['true_trace']))
        ax.set_title(label,color=color);lines.append(rods(ax,color,True));texts.append(ax.text(.02,.03,'',transform=ax.transAxes,fontsize=9))
    clock=fig.text(.5,.025,'',ha='center')
    def update(frame):
        t=frame*4;clock.set_text(f't = {t*.05:.1f} · playback 2× · recorded RK4 trajectories')
        for ll,(z,tr),txt in zip(lines,data,texts):
            for line,true,pivot,j in ll:
                line.set_data([pivot,pivot+.65*np.sin(z[t,j])],[0,-.65*np.cos(z[t,j])]);true.set_data([pivot,pivot+.65*np.sin(tr[t,j])],[0,-.65*np.cos(tr[t,j])])
            dq=np.arctan2(np.sin(z[t,:2]-tr[t,:2]),np.cos(z[t,:2]-tr[t,:2]));err=np.linalg.norm(np.r_[dq,z[t,2:]-tr[t,2:]])
            txt.set_text(f'State error: {err:.2f}    True H shift: {plant_energy(z[t],"misspecified")-plant_energy(z[0],"misspecified"):+.2f}')
    anim=FuncAnimation(fig,update,frames=101,interval=100);anim.save(OUT/'coverage_prediction.gif',writer=PillowWriter(fps=10))
    update(60);fig.savefig(OUT/'prediction_animation_preview.png',dpi=160,bbox_inches='tight');plt.close(fig)

def control():
    fig=plt.figure(figsize=(12,8.5));outer=fig.add_gridspec(2,3,left=.055,right=.97,bottom=.08,top=.865,hspace=.3,wspace=.27)
    fig.suptitle('Same true plant: data coverage and controller choice',fontsize=15,y=.98)
    fig.text(.5,.93,'Hamiltonian-feature models · world seed 10 · SAC policy seed 10 · fixed high-energy state 0',ha='center',fontsize=10)
    fig.text(.5,.9,'Top row: local data only (0% global)      Bottom row: global configuration data (100%)',ha='center',fontsize=9)
    configs=[];lines=[];texts=[];curves=[];arrays=[]
    for row,cov in enumerate(['local','global']):
        for col,ctrl in enumerate(['sac','feedback','mpc']):
            gs=outer[row,col].subgridspec(2,1,height_ratios=[1.6,1],hspace=.05);ax=fig.add_subplot(gs[0]);err_ax=fig.add_subplot(gs[1])
            color={'sac':'#9b78a8','feedback':'#477caa','mpc':'#359886'}[ctrl]
            path=OUT/'sac'/f'hamiltonian_{cov}_w10_p10_trajectory.npz' if ctrl=='sac' else OUT/'controllers'/f'misspecified_hamiltonian_{cov}_w10_{ctrl}_trajectory.npz'
            z=np.load(path)['states'][:,0];arrays.append(z);lines.append(rods(ax,color));texts.append(ax.text(.03,.03,'',transform=ax.transAxes,fontsize=8))
            ax.set_title({'sac':'Learned SAC','feedback':'Fixed feedback','mpc':'Projected-gradient MPC'}[ctrl],color=color,fontsize=11)
            err_ax.set_xlim(0,20);err_ax.set_ylim(.001,6);err_ax.set_yscale('log');err_ax.axhline(.25,c='gray',ls=':',lw=1)
            err_ax.set_xlabel('Time');err_ax.set_ylabel('$||p-p^*||$');line,=err_ax.plot([],[],c=color,lw=1.5);curves.append(line)
    clock=fig.text(.5,.025,'',ha='center',fontsize=10)
    def update(frame):
        t=frame*4;clock.set_text(f't = {t*.05:.1f} · playback 2× · target momentum (2.5, 1.5) · true RK4 dynamics throughout')
        for z,ll,txt,curve in zip(arrays,lines,texts,curves):
            for line,_,pivot,j in ll:line.set_data([pivot,pivot+.65*np.sin(z[t,j])],[0,-.65*np.cos(z[t,j])])
            txt.set_text(f'p = ({z[t,2]:+.2f}, {z[t,3]:+.2f})')
            curve.set_data(np.arange(t+1)*.05,np.maximum(np.linalg.norm(z[:t+1,2:]-[2.5,1.5],axis=1),1e-6))
    anim=FuncAnimation(fig,update,frames=101,interval=100);anim.save(OUT/'coverage_control.gif',writer=PillowWriter(fps=10))
    update(85);fig.savefig(OUT/'control_animation_preview.png',dpi=160,bbox_inches='tight');plt.close(fig)
if __name__=='__main__':
    prediction()
    if (OUT/'sac/results.json').exists() and (OUT/'controllers/results.json').exists():control()
