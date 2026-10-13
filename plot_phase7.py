"""Plots of all fixed candidates and declared gates; no outcome selection."""
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from phase7_design import *
METHODS=P['methods'];LABELS=['Random','Uncertainty','Global A','Occupancy A','Spectral A']
COLORS=['#73695f','#477caa','#359886','#9b78a8','#d58047']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#fafaf8','axes.facecolor':'#fafaf8','savefig.facecolor':'#fafaf8'})
def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=180,bbox_inches='tight');fig.savefig(OUT/(name+'.svg'),bbox_inches='tight');plt.close(fig)
def csv_write(rows,path):
    keys=[k for k,v in rows[0].items() if not isinstance(v,(list,dict))]
    with path.open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');wr.writeheader();wr.writerows(rows)
def main():
    rows=json.loads((OUT/'control_results.json').read_text());metrics=json.loads((OUT/'model_metrics.json').read_text());lookup={r['model_file']:r for r in metrics}
    joined=[{**r,'global_force_vector_rms':lookup[r['model_file']]['global_force_vector_rms'],
         'ideal_tracking_rms':float(np.sqrt(lookup[r['model_file']]['ideal_first_order_tracking_mse'][r['task']]))} for r in rows]
    csv_write(joined,OUT/'results.csv');refs=json.loads((OUT/'references.json').read_text());summary={}
    fig,axes=plt.subplots(2,2,figsize=(10.5,7),constrained_layout=True)
    for i,plant in enumerate(P['plants']):
        summary[plant]={}
        for j,task in enumerate(P['tasks']):
            summary[plant][task]={};ax=axes[i,j]
            for method,col,label in zip(METHODS,COLORS,LABELS):
                rr=[r for r in rows if r['plant']==plant and r['task']==task and r['kind']=='hamiltonian' and r['method']==method]
                vv=np.array([[r['momentum_vector_rms_mean'] for r in rr if r['budget']==b] for b in P['additional_query_budgets']])
                ax.plot(range(4),vv.mean(1),'o-',c=col,label=label,ms=4);ax.fill_between(range(4),vv.min(1),vv.max(1),color=col,alpha=.12)
                summary[plant][task][method]={str(b):{k:float(np.mean([r[k] for r in rr if r['budget']==b])) for k in ['momentum_vector_rms_mean','return_mean','success_rate']} for b in P['additional_query_budgets']}
            reference=next(r for r in refs if r['plant']==plant and r['task']==task);ax.axhline(reference['momentum_vector_rms_mean'],ls=':',c='#343c48',lw=1,label='Privileged reference')
            ax.set_yscale('log');ax.set_xticks(range(4),P['additional_query_budgets']);ax.set_xlabel('Additional force queries beyond 128 pilots');ax.set_ylabel('Final momentum vector RMS');ax.set_title(f'{plant.capitalize()} / {task}')
    axes[0,0].legend(fontsize=8,ncol=2);fig.suptitle('Hamiltonian feature models: task-weighted acquisition does not uniformly improve actual tracking',fontsize=12)
    save(fig,'01_fixed_budget_control');(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    # Focused primary budget, with all paired world-seed outcomes exposed.
    fig,axes=plt.subplots(2,2,figsize=(10.5,7),constrained_layout=True)
    for i,plant in enumerate(P['plants']):
        for j,task in enumerate(P['tasks']):
            ax=axes[i,j]
            for k,(method,col) in enumerate(zip(METHODS,COLORS)):
                vals=[r['momentum_vector_rms_mean'] for r in rows if r['plant']==plant and r['task']==task and r['kind']=='hamiltonian' and r['method']==method and r['budget']==64]
                ax.scatter(k+np.linspace(-.16,.16,5),vals,c=col,s=28);ax.plot([k-.25,k+.25],[np.mean(vals)]*2,c=col,lw=3)
            ax.set_xticks(range(5),LABELS,rotation=20);ax.set_ylabel('Momentum RMS');ax.set_title(f'{plant.capitalize()} / {task}')
    fig.suptitle('Primary 64-query gate: five paired world seeds per method; no IID-phase significance test',fontsize=12);save(fig,'02_primary_gate_cells')
    fig,axes=plt.subplots(2,2,figsize=(10.5,7),constrained_layout=True)
    for i,plant in enumerate(P['plants']):
        for j,key in enumerate(['global_force_vector_rms','ideal_tracking_rms']):
            ax=axes[i,j]
            for method,col in zip(METHODS,COLORS):
                rr=[r for r in joined if r['plant']==plant and r['task']=='forward' and r['kind']=='hamiltonian' and r['method']==method]
                vv=np.array([[r[key] for r in rr if r['budget']==b] for b in P['additional_query_budgets']]);ax.plot(range(4),vv.mean(1),'o-',c=col,label=method.replace('_',' '),ms=4)
            ax.set_yscale('log');ax.set_xticks(range(4),P['additional_query_budgets']);ax.set_xlabel('Additional queries');ax.set_ylabel('Global force RMS' if j==0 else 'Ideal first-order tracking RMS');ax.set_title(plant.capitalize()+' / forward task')
    axes[0,0].legend(fontsize=8,ncol=2);fig.suptitle('Model accuracy and ideal response are diagnostics, not substitutes for actual control',fontsize=12);save(fig,'03_force_and_ideal_response')
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for ax,task in zip(axes,P['tasks']):
        rr=[r for r in rows if r['plant']=='misspecified' and r['task']==task and r['budget']==64]
        for k,label,col in [('force','Force','#73695f'),('odd_force','Odd force','#477caa'),('hamiltonian','Hamiltonian','#d58047')]:
            v=[np.mean([r['momentum_vector_rms_mean'] for r in rr if r['kind']==k and r['method']==m]) for m in METHODS]
            ax.plot(range(5),v,'o-',c=col,label=label);ax.set_xticks(range(5),LABELS,rotation=25);ax.set_ylabel('Momentum vector RMS');ax.set_title('Misspecified / '+task)
    axes[0].legend(fontsize=9);fig.suptitle('Structure and acquisition interact: all three classes and five fixed strategies retained',fontsize=12);save(fig,'04_structure_and_design')
    # Response weights explicitly show which modes are emphasized.
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for ax,(task,target) in zip(axes,P['tasks'].items()):
        weights=1/(P['feedback_gain']**2+(K@target)**2)
        ax.bar(range(len(K)),weights,color='#d58047');ax.set_xticks(range(len(K)),[f'({int(k[0])},{int(k[1])})' for k in K],rotation=55)
        ax.set_xlabel('Fourier frequency');ax.set_ylabel('Steady tracking weight');ax.set_title(f'{task}: target {target}')
    fig.suptitle('Classical stable-filter weights depend on frequency along the requested winding',fontsize=12);save(fig,'05_spectral_task_weights')
    # Posterior certainty compared with force-error diagnostics at the same query budget.
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for ax,plant in zip(axes,P['plants']):
        for method,col in zip(METHODS,COLORS):
            rr=[r for r in joined if r['plant']==plant and r['task']=='forward' and r['kind']=='hamiltonian' and r['method']==method and r['budget']>0]
            x=[np.sqrt(r['spectral_posterior_variance']['forward']) for r in rr];y=[r['momentum_vector_rms_mean'] for r in rr]
            ax.scatter(x,y,color=col,s=20,alpha=.6,label=method.replace('_',' '))
        ax.set_xscale('log');ax.set_yscale('log');ax.set_xlabel('Posterior first-order tracking standard deviation');ax.set_ylabel('Actual momentum vector RMS');ax.set_title(plant.capitalize())
    axes[0].legend(fontsize=8);fig.suptitle('Posterior uncertainty excludes approximation bias and sampled-controller error',fontsize=12);save(fig,'06_uncertainty_and_control')
if __name__=='__main__':main()
