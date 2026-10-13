"""Scientific figures from saved model and policy results; no invented curves."""
from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from coupled_models import FourierForce,truth_force,truth_potential
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'phase5';P=json.loads((ROOT/'phase5_protocol.json').read_text())
KINDS=P['families'];LABELS=['Force','Mean only','Curl only','Curl + mean','Hamiltonian']
COLORS=['#8b6c56','#608697','#9b78a8','#359886','#d58047']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#fafaf8','axes.facecolor':'#fafaf8','savefig.facecolor':'#fafaf8'})
def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=180,bbox_inches='tight');fig.savefig(OUT/(name+'.svg'),bbox_inches='tight');plt.close(fig)
def csv_write(rows,path):
    keys=[k for k,v in rows[0].items() if not isinstance(v,(list,dict))]
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(rows)
def main():
    rows=json.loads((OUT/'model_results.json').read_text());csv_write(rows,OUT/'model_results.csv')
    summary={}
    fig,axes=plt.subplots(2,2,figsize=(11,7.2),constrained_layout=True)
    for r,plant in enumerate(P['plants']):
        for k,c,label in zip(KINDS,COLORS,LABELS):
            vals=[x['global_force_vector_rms'] for x in rows if x['plant']==plant and x['kind']==k and x['split']=='rotation']
            ix=KINDS.index(k);axes[r,0].scatter(np.full(3,ix)+np.array([-.13,0,.13]),vals,color=c,s=35)
            axes[r,0].plot([ix-.23,ix+.23],[np.mean(vals)]*2,color=c,lw=3)
            cur=[]
            for w in P['world_seeds']:cur.append(np.load(OUT/plant/f'{k}_{w}_rotation_prediction.npz')['rmse'])
            cur=np.array(cur);t=np.arange(401)*.05;axes[r,1].plot(t,cur.mean(0),color=c,label=label)
            axes[r,1].fill_between(t,cur.min(0),cur.max(0),color=c,alpha=.12)
        axes[r,0].set_xticks(range(5),LABELS,rotation=20);axes[r,0].set_ylabel('Global force vector RMS');axes[r,0].set_title(plant.capitalize()+' plant: uniform torus')
        axes[r,1].set_title(plant.capitalize()+' plant: high-energy rollout');axes[r,1].set_xlabel('Time');axes[r,1].set_ylabel('Wrapped state vector RMSE')
    axes[0,1].legend(fontsize=8,ncol=2);fig.suptitle('Shared features and convex fitting: consistency does not identify the unseen force',fontsize=13)
    save(fig,'01_cross_energy_prediction')
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),constrained_layout=True)
    for i,key in enumerate(['curl_rms','mean_force_norm','mean_max_true_energy_error']):
        for j,(k,c) in enumerate(zip(KINDS,COLORS)):
            v=[x[key] for x in rows if x['plant']=='misspecified' and x['kind']==k and x['split']=='rotation']
            axes[i].scatter(np.full(3,j)+np.array([-.12,0,.12]),np.maximum(v,1e-10),color=c)
            axes[i].set_xticks(range(5),LABELS,rotation=30)
        axes[i].set_yscale('log');axes[i].set_ylabel(key.replace('_',' '))
    axes[0].set_title('Local inconsistency');axes[1].set_title('Global mean-force obstruction');axes[2].set_title('True energy error along model rollouts')
    fig.suptitle('Misspecified plant: exact structure can coexist with inaccurate energy landscapes',fontsize=13)
    save(fig,'02_structure_and_energy')
    fig,axes=plt.subplots(2,3,figsize=(11,7.5),constrained_layout=True)
    a=np.linspace(-np.pi,np.pi,101);grid=np.stack(np.meshgrid(a,a,indexing='ij'),-1);tf=truth_force(grid)
    maps=[np.sqrt(np.sum((FourierForce.load(OUT/'misspecified'/f'{k}_0.npz').force(grid)-tf)**2,axis=-1)) for k in KINDS]
    vmax=max(x.max() for x in maps)
    for ax,k,label,data in zip(axes.flat,KINDS,LABELS,maps):
        im=ax.imshow(data.T,origin='lower',extent=(-np.pi,np.pi,-np.pi,np.pi),vmin=0,vmax=vmax,cmap='magma',aspect='equal')
        train=np.load(OUT/'misspecified/training_0.npz')['states'];ax.scatter(train[::12,0],train[::12,1],s=1,c='cyan',alpha=.3)
        ax.set_title(label);ax.set_xlabel('$q_1$');ax.set_ylabel('$q_2$')
    axes.flat[-1].axis('off');fig.colorbar(im,ax=list(axes.flat),label='Force vector error',shrink=.6)
    fig.suptitle('Learned force error on the torus; cyan points mark low-energy training coverage',fontsize=13)
    save(fig,'03_force_error_maps')
    control=OUT/'control/results.json'
    if control.exists():
        cr=json.loads(control.read_text());csv_write(cr,OUT/'control/results.csv')
        for split in P['test_energy_bands']:
            summary[split]={}
            for k in KINDS+['true_only']:
                rr=[x for x in cr if x['kind']==k and x['split']==split]
                summary[split][k]={a:float(np.mean([x[a] for x in rr])) for a in ['return_mean','success_rate','momentum_vector_rms_mean']}
        fig,axes=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
        for ax,key,title in zip(axes,['return_mean','momentum_vector_rms_mean','success_rate'],['True-plant return','Final momentum vector RMS','Strict success rate']):
            for j,(k,c) in enumerate(zip(KINDS+['true_only'],COLORS+['#343c48'])):
                rr=[x for x in cr if x['kind']==k and x['split']=='rotation'];v=np.array([x[key] for x in rr])
                offsets=np.tile([-.15,0,.15],3) if len(v)==9 else np.array([-.15,0,.15])
                ax.scatter(j+offsets,v,c=c,alpha=.65,s=25);ax.plot([j-.25,j+.25],[v.mean()]*2,c=c,lw=3)
                # Policy-seed marginal means expose one crossed factor, not IID-cell CIs.
                pm=[np.mean([x[key] for x in rr if x['policy_seed']==p]) for p in P['policy_seeds']]
                ax.scatter(j+np.array([-.15,0,.15]),pm,facecolors='none',edgecolors=c,s=65,linewidths=1.3)
            ax.set_xticks(range(6),LABELS+['True only'],rotation=30);ax.set_title(title)
        fig.suptitle('Learned-model control: 3 × 3 seeds; true-only reference: 3 seeds / 16,000 true transitions',fontsize=13)
        save(fig,'04_cyclic_control')
        fig,axes=plt.subplots(1,5,figsize=(13,3.2),constrained_layout=True)
        vv=[x['momentum_vector_rms_mean'] for x in cr if x['split']=='rotation'];vmin=min(vv);vmax=max(vv)
        for ax,k,label in zip(axes,KINDS,LABELS):
            mat=np.array([[next(x['momentum_vector_rms_mean'] for x in cr if x['kind']==k and x['world_seed']==w and x['policy_seed']==p and x['split']=='rotation') for p in P['policy_seeds']] for w in P['world_seeds']])
            im=ax.imshow(mat,vmin=vmin,vmax=vmax,cmap='viridis');ax.set_title(label);ax.set_xticks(range(3),P['policy_seeds']);ax.set_yticks(range(3),P['world_seeds']);ax.set_xlabel('Policy seed');ax.set_ylabel('World seed')
            for (r,c),v in np.ndenumerate(mat):ax.text(c,r,f'{v:.2f}',ha='center',va='center',color='white',fontsize=9)
        fig.colorbar(im,ax=list(axes),label='Momentum RMS',shrink=.65);save(fig,'05_crossed_seed_cells')
        (OUT/'control_summary.json').write_text(json.dumps(summary,indent=2))
    # Structure vs force-error summaries are three distinct world seeds.
    model_summary={}
    for plant in P['plants']:
        model_summary[plant]={}
        for k in KINDS:
            rr=[x for x in rows if x['plant']==plant and x['kind']==k and x['split']=='rotation']
            model_summary[plant][k]={a:float(np.mean([x[a] for x in rr])) for a in ['global_force_vector_rms','rmse_t20','curl_rms','mean_force_norm','mean_max_true_energy_error']}
    (OUT/'model_summary.json').write_text(json.dumps(model_summary,indent=2))
if __name__=='__main__':main()
