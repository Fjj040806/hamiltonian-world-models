"""Declared posthoc sensitivity diagnostic, never used for main-model selection."""
from pathlib import Path
import json
import numpy as np
from coupled_models import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'phase5';P=json.loads((ROOT/'phase5_protocol.json').read_text());S=json.loads((ROOT/'phase5_sensitivity_protocol.json').read_text())
a=np.linspace(-np.pi,np.pi,64,endpoint=False);grid=np.stack(np.meshgrid(a,a,indexing='ij'),-1).reshape(-1,2)
rows=[]
for plant,eps in P['plants'].items():
    val=sample_states(1024,P['training_energy_band'],np.random.default_rng(73000),eps)
    for w in P['world_seeds']:
        d=np.load(OUT/plant/f'training_{w}.npz')
        for ridge in S['ridge_values']:
            for kind in S['models']:
                penalties=S['structural_penalty_values'] if kind in ('mean_only','curl_only','curl_mean') else [1.]
                for penalty in penalties:
                    m,ch=fit(kind,d['states'][:,:2],d['force_labels'],ridge,penalty,penalty,eps)
                    assert ch['normal_equation_residual']<1e-9
                    rows.append({'plant':plant,'world_seed':w,'kind':kind,'ridge':ridge,'structural_penalty':penalty,
                     'low_energy_force_vector_rms':float(np.sqrt(np.mean(np.sum((m.force(val[:,:2])-truth_force(val[:,:2],eps))**2,axis=1)))),
                     'global_force_vector_rms':float(np.sqrt(np.mean(np.sum((m.force(grid)-truth_force(grid,eps))**2,axis=1)))),**m.statistics()})
(OUT/'sensitivity_results.json').write_text(json.dumps(rows,indent=2))
fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
colors=['#8b6c56','#608697','#9b78a8','#359886','#d58047']
for j,plant in enumerate(P['plants']):
    for kind,color in zip(S['models'],colors):
        rr=[r for r in rows if r['plant']==plant and r['kind']==kind and r['structural_penalty']==1.]
        for i,key in enumerate(['low_energy_force_vector_rms','global_force_vector_rms']):
            means=[np.mean([r[key] for r in rr if r['ridge']==lam]) for lam in S['ridge_values']]
            axes[j,i].loglog(S['ridge_values'],means,'o-',color=color,label=kind.replace('_',' '),ms=4)
            axes[j,i].axvline(.001,color='gray',ls=':',lw=1);axes[j,i].set_xlabel('Force-norm ridge strength');axes[j,i].set_ylabel(key.replace('_',' '));axes[j,i].set_title(plant.capitalize())
axes[0,0].legend(fontsize=8);fig.suptitle('Posthoc sensitivity: dotted line marks the frozen primary ridge, not a tuned optimum')
fig.savefig(OUT/'06_regularization_sensitivity.png',dpi=180,bbox_inches='tight');fig.savefig(OUT/'06_regularization_sensitivity.svg',bbox_inches='tight');plt.close(fig)
print('Sensitivity completed:',len(rows),'fits; primary checkpoints unchanged')
