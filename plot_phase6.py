"""Source-backed coverage, prior and controller figures for Phase 6."""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from phase6_models import *
KINDS=P['families'];LABELS=['Force','Odd force','Curl only','Curl + mean','Hamiltonian','Odd Hamiltonian']
COLORS=['#73695f','#477caa','#9b78a8','#359886','#d58047','#b95563'];COVS=list(P['coverage'])
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#fafaf8','axes.facecolor':'#fafaf8','savefig.facecolor':'#fafaf8'})
def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=180,bbox_inches='tight');fig.savefig(OUT/(name+'.svg'),bbox_inches='tight');plt.close(fig)
def csv_write(rows,path):
    keys=[k for k,v in rows[0].items() if not isinstance(v,(list,dict))]
    with path.open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');wr.writeheader();wr.writerows(rows)
def prior_floor(plant,odd=False):
    pars=P['plants'][plant];shift=np.array(pars['shift']);eps=pars['epsilon'];floor2=5*eps**2
    if odd:
        for k,a in [(np.array([1,0]),1.),(np.array([0,1]),.8),(np.array([1,-1]),.3)]:floor2+=.5*a*a*np.dot(k,k)*np.sin(k@shift)**2
    return float(np.sqrt(floor2))
def main():
    rows=json.loads((OUT/'prediction_results.json').read_text());selected=json.loads((OUT/'selected_models.json').read_text());csv_write(rows,OUT/'prediction_results.csv');csv_write(selected,OUT/'selected_models.csv')
    summary={}
    for plant in P['plants']:
        summary[plant]={}
        for cov in COVS:
            summary[plant][cov]={}
            for k in KINDS:
                rr=[r for r in rows if r['plant']==plant and r['coverage']==cov and r['kind']==k and r['split']=='rotation']
                summary[plant][cov][k]={key:float(np.mean([r[key] for r in rr])) for key in ['global_force_vector_rms','rmse_t20','mean_max_true_energy_error','validation_vector_mse']}
    (OUT/'prediction_summary.json').write_text(json.dumps(summary,indent=2))
    fig,axes=plt.subplots(1,3,figsize=(12,4.2),constrained_layout=True)
    for ax,plant in zip(axes,P['plants']):
        for k,label,color in zip(KINDS,LABELS,COLORS):
            v=np.array([[r['global_force_vector_rms'] for r in rows if r['plant']==plant and r['coverage']==cov and r['kind']==k and r['split']=='rotation'] for cov in COVS])
            ax.plot(range(3),v.mean(1),'o-',c=color,label=label,ms=4);ax.fill_between(range(3),v.min(1),v.max(1),color=color,alpha=.12)
        ax.set_yscale('log');ax.set_xticks(range(3),['0%','10%','100%']);ax.set_xlabel('Global configurations in training data');ax.set_ylabel('Global force vector RMS');ax.set_title(plant.capitalize())
        if plant!='canonical':ax.axhline(prior_floor(plant),c='#343c48',ls=':',lw=1)
        if plant=='shifted':ax.axhline(prior_floor(plant,True),c=COLORS[1],ls=':',lw=1)
    axes[0].legend(fontsize=8,ncol=2);fig.suptitle('Fixed sample budget and validation-only model selection: coverage and valid priors',fontsize=13)
    save(fig,'01_coverage_and_force_error')
    fig,axes=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
    for j,plant in enumerate(P['plants']):
        for i,key in enumerate(['rmse_t20','mean_max_true_energy_error']):
            for k,color in zip(KINDS,COLORS):
                vv=np.array([[r[key] for r in rows if r['plant']==plant and r['coverage']==cov and r['kind']==k and r['split']=='rotation'] for cov in COVS])
                axes[i,j].plot(range(3),vv.mean(1),'o-',color=color,ms=4);axes[i,j].fill_between(range(3),vv.min(1),vv.max(1),color=color,alpha=.12)
            axes[i,j].set_xticks(range(3),['0%','10%','100%']);axes[i,j].set_xlabel('Global configuration fraction');axes[i,j].set_ylabel('State vector RMSE at t=20' if i==0 else 'Mean maximum true-energy error');axes[i,j].set_title(plant.capitalize())
    fig.suptitle('Fresh high-energy tests: prediction and true-energy accuracy remain separate',fontsize=13);save(fig,'02_prediction_and_energy')
    fig,axes=plt.subplots(1,3,figsize=(12,4.5),constrained_layout=True)
    for ax,plant in zip(axes,P['plants']):
        mat=np.array([[np.log10(next(r['ridge'] for r in selected if r['plant']==plant and r['coverage']==cov and r['world_seed']==w and r['kind']==k)) for k in KINDS] for cov in COVS for w in P['world_seeds']])
        im=ax.imshow(mat,vmin=-6,vmax=-2,cmap='viridis',aspect='auto');ax.set_xticks(range(6),LABELS,rotation=45,ha='right');ax.set_yticks(range(9),[f'{cov}, w{w}' for cov in COVS for w in P['world_seeds']]);ax.set_title(plant.capitalize())
        for (r,c),v in np.ndenumerate(mat):ax.text(c,r,str(int(v)),ha='center',va='center',color='white',fontsize=8)
    fig.colorbar(im,ax=list(axes),label='log10 selected ridge',shrink=.6);fig.suptitle('Ridge selected from independent noisy validation labels, not test error',fontsize=13);save(fig,'03_validation_selection')
    a=np.linspace(-np.pi,np.pi,101);grid=np.stack(np.meshgrid(a,a,indexing='ij'),-1);tf=plant_force(grid,'shifted');fig,axes=plt.subplots(2,3,figsize=(10,7),constrained_layout=True)
    for i,k in enumerate(['odd_force','hamiltonian']):
        for j,cov in enumerate(COVS):
            m=Model.load(OUT/'shifted'/f'{k}_{cov}_10.npz');error=np.linalg.norm(m.force(grid)-tf,axis=-1)
            im=axes[i,j].imshow(np.maximum(error,1e-4).T,origin='lower',extent=(-np.pi,np.pi,-np.pi,np.pi),norm=LogNorm(vmin=.001,vmax=6),cmap='magma')
            axes[i,j].set_title(f'{k.replace("_"," ")} / {cov}');axes[i,j].set_xlabel('$q_1$');axes[i,j].set_ylabel('$q_2$')
    fig.colorbar(im,ax=list(axes.flat),label='Force vector error (log scale)',shrink=.65);fig.suptitle('Translated potential: odd symmetry about the original origin is an invalid prior',fontsize=12);save(fig,'04_coordinate_origin_probe')
    fig,ax=plt.subplots(figsize=(9,4),constrained_layout=True)
    for i,(k,label,col) in enumerate(zip(KINDS,LABELS,COLORS)):
        v=[r['global_force_vector_rms'] for r in rows if r['plant']=='shifted' and r['coverage']=='global' and r['kind']==k and r['split']=='rotation']
        ax.bar(i,np.mean(v),color=col,alpha=.8);ax.scatter(i+np.array([-.12,0,.12]),v,c='#343c48',s=15)
        floor=prior_floor('shifted',k.startswith('odd_'));ax.plot([i-.35,i+.35],[floor]*2,c='#343c48',ls='--',lw=1.5)
    ax.set_ylim(0,.57);ax.set_xticks(range(6),LABELS,rotation=20);ax.set_ylabel('Global force vector RMS');ax.set_title('Global coverage approaches different approximation floors under valid and invalid priors')
    ax.text(.02,.95,'Dashed segments: analytic feature-class approximation floors',transform=ax.transAxes,va='top',fontsize=9);save(fig,'05_approximation_floors')
    if (OUT/'controllers/results.json').exists():
        rr=json.loads((OUT/'controllers/results.json').read_text());csv_write(rr,OUT/'controllers/results.csv');cs={}
        fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
        for i,plant in enumerate(P['controller_plants']):
            cs[plant]={}
            for j,ctrl in enumerate(P['controllers']):
                cs[plant][ctrl]={}
                for k,color in zip(KINDS,COLORS):
                    vv=np.array([[r['momentum_vector_rms_mean'] for r in rr if r['plant']==plant and r['coverage']==cov and r['kind']==k and r['controller']==ctrl] for cov in COVS])
                    axes[i,j].plot(range(3),vv.mean(1),'o-',color=color,label=k.replace('_',' '),ms=4);axes[i,j].fill_between(range(3),vv.min(1),vv.max(1),color=color,alpha=.12)
                    cs[plant][ctrl][k]={cov:{key:float(np.mean([r[key] for r in rr if r['plant']==plant and r['coverage']==cov and r['kind']==k and r['controller']==ctrl])) for key in ['momentum_vector_rms_mean','return_mean','success_rate']} for cov in COVS}
                ref=next(r for r in rr if r['plant']==plant and r['kind']=='known_force' and r['controller']==ctrl)
                cs[plant][ctrl]['known_force']={key:ref[key] for key in ['momentum_vector_rms_mean','return_mean','success_rate']}
                axes[i,j].axhline(ref['momentum_vector_rms_mean'],c='#343c48',ls=':',label='Privileged reference');axes[i,j].set_yscale('log');axes[i,j].set_xticks(range(3),['0%','10%','100%']);axes[i,j].set_xlabel('Global configuration fraction');axes[i,j].set_ylabel('Final momentum vector RMS');axes[i,j].set_title(f'{plant.capitalize()} / {ctrl.upper()}')
        axes[0,0].legend(fontsize=7,ncol=2);fig.suptitle('Stronger control baselines evaluated in the true plant; no policy training',fontsize=13);save(fig,'06_feedback_and_mpc');(OUT/'controller_summary.json').write_text(json.dumps(cs,indent=2))
    if (OUT/'sac/results.json').exists():
        rr=json.loads((OUT/'sac/results.json').read_text());csv_write(rr,OUT/'sac/results.csv');ss={}
        fig,axes=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
        for j,(key,title) in enumerate([('return_mean','True-plant return'),('momentum_vector_rms_mean','Final momentum vector RMS'),('success_rate','Strict success rate')]):
            for i,k in enumerate(P['sac']['families']):
                col=COLORS[KINDS.index(k)]
                for c,offset,alpha in [('local',-.18,.95),('global',.18,.35)]:
                    v=[r[key] for r in rr if r['kind']==k and r['coverage']==c and r['split']=='rotation']
                    axes[j].bar(i+offset,np.mean(v),width=.3,color=col,alpha=alpha);axes[j].scatter(i+offset+np.tile([-.07,0,.07],3),v,c=col,s=15,alpha=.7)
            ref=[r[key] for r in rr if r['kind']=='true_only' and r['split']=='rotation'];axes[j].axhline(np.mean(ref),c='#343c48',ls=':',lw=1.5)
            axes[j].set_xticks(range(3),['Force','Odd force','Hamiltonian']);axes[j].set_title(title)
        from matplotlib.patches import Patch
        axes[0].legend(handles=[Patch(color='gray',alpha=.95,label='Local data'),Patch(color='gray',alpha=.35,label='Global configurations')],fontsize=8)
        fig.suptitle('SAC with matched 16,000-transition budgets and fresh 3 × 3 crossed seeds',fontsize=13);save(fig,'07_sac_coverage')
        for cov in ['local','global']:
            ss[cov]={}
            for k in P['sac']['families']:
                group=[r for r in rr if r['kind']==k and r['coverage']==cov and r['split']=='rotation'];ss[cov][k]={key:float(np.mean([r[key] for r in group])) for key in ['return_mean','momentum_vector_rms_mean','success_rate']}
        group=[r for r in rr if r['kind']=='true_only' and r['split']=='rotation'];ss['true_only']={key:float(np.mean([r[key] for r in group])) for key in ['return_mean','momentum_vector_rms_mean','success_rate']};(OUT/'sac_summary.json').write_text(json.dumps(ss,indent=2))
if __name__=='__main__':main()
