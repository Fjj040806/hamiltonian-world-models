"""Summarize all calibrated runs; never select policies on test results."""
from pathlib import Path
import json,csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'phase3/calibrated'
rows=json.loads((OUT/'results.json').read_text());controllers=json.loads((OUT/'controller.json').read_text())
pred=json.loads((ROOT/'phase2/results.json').read_text())
KINDS=['node','second_order','force','force_zero_mean','hnn','true_model','model_free']
SPLITS=['validation','test_moderate','test_near_below','test_near_above','test_rotation']
COL={'node':'#81929d','second_order':'#bb9349','force':'#476ca0','force_zero_mean':'#9a78a7','hnn':'#218b87','true_model':'#243746','model_free':'#cc705b'}
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
def group(kind,split,h=None):
    if h is None:h=0 if kind=='model_free' else 5
    return [r for r in rows if r['kind']==kind and r['split']==split and r['horizon']==h]
def save(fig,name):
    fig.tight_layout()
    for ext in ('png','svg'):fig.savefig(OUT/f'{name}.{ext}',dpi=170)
    plt.close(fig)

fig,axs=plt.subplots(1,2,figsize=(12,4.6))
for kind in KINDS:
    runs=[json.loads((OUT/f'{kind}_seed{s}_h{0 if kind=="model_free" else 5}_u4.json').read_text()) for s in range(5)]
    ys=np.array([[v['validation_return'] for v in r['history']] for r in runs]);x=[v['real_steps'] for v in runs[0]['history']]
    axs[0].plot(x,ys.mean(0),marker='o',color=COL[kind],label=kind)
    for line in ys:axs[0].plot(x,line,color=COL[kind],alpha=.1,lw=.8)
    val=np.array([[r['return_mean'] for r in group(kind,s)] for s in SPLITS])
    axs[1].plot(range(5),val.mean(1),marker='o',color=COL[kind],label=kind)
axs[0].set(xlabel='Real training transitions',ylabel='Validation return (higher is better)',title='Training budget and in-distribution learning')
axs[1].set_xticks(range(5),['ID','moderate','below','above','rotation']);axs[1].set(ylabel='Mean final return',title='Paired energy-regime evaluation | 5 seeds')
axs[0].legend(fontsize=7);save(fig,'01_learning_and_generalization')

mat=np.array([[np.mean([r['success_rate'] for r in group(k,s)]) for s in SPLITS] for k in KINDS])
fig,ax=plt.subplots(figsize=(8.5,5.2));im=ax.imshow(mat,vmin=0,vmax=1,cmap='YlGnBu',aspect='auto')
ax.set_xticks(range(5),['ID','moderate','below','above','rotation']);ax.set_yticks(range(7),KINDS)
for i in range(7):
 for j in range(5):ax.text(j,i,f'{100*mat[i,j]:.1f}%',ha='center',va='center',color='white' if mat[i,j]>.55 else '#243746')
ax.set_title('Sustained upright stabilization | final 1 time unit');fig.colorbar(im,ax=ax,label='Success fraction');save(fig,'02_success_rates')

fig,axs=plt.subplots(1,2,figsize=(11,4.6))
for kind in ['force_zero_mean','hnn']:
    for ax,metric in zip(axs,['return_mean','success_rate']):
        vals=np.array([[r[metric] for r in group(kind,'test_near_above',h)] for h in [1,5,15]])
        ax.plot([1,5,15],vals.mean(1),marker='o',color=COL[kind],label=kind)
        for j,h in enumerate([1,5,15]):ax.scatter(np.repeat(h,5),vals[j],color=COL[kind],s=12,alpha=.45)
        ax.set(xlabel='Synthetic rollout depth (control intervals)',ylabel='Mean episode return' if metric=='return_mean' else 'Success fraction',title='Near-above test | equal attempted synthetic budget');ax.set_xticks([1,5,15]);ax.legend(fontsize=8)
save(fig,'03_horizon_ablation')

fig,ax=plt.subplots(figsize=(7,4.6));stats={}
for kind in KINDS[:5]:
    for seed in range(5):
        p=next(r for r in pred if r['kind']==kind and r['seed']==seed and r['split']=='test_near_above')
        c=next(r for r in group(kind,'test_near_above') if r['seed']==seed)
        ax.scatter(p['rmse_16'],c['return_mean'],color=COL[kind],s=30,label=kind if seed==0 else None)
x=[];y=[]
for kind in KINDS[:5]:
 for seed in range(5):
    x.append(next(r['rmse_16'] for r in pred if r['kind']==kind and r['seed']==seed and r['split']=='test_near_above'))
    y.append(next(r['return_mean'] for r in group(kind,'test_near_above') if r['seed']==seed))
rho=float(spearmanr(x,y).statistic)
stats['pooled_descriptive_spearman_prediction_error_vs_return']=rho
ax.set(xlabel='Phase 2 long-horizon prediction RMSE',ylabel='Phase 3 true-environment return',title=f'Prediction versus control | descriptive Spearman ρ={rho:.2f}')
ax.legend(fontsize=8);save(fig,'04_prediction_vs_control')

table=['| Model | ID success | Above success | Rotation success | Above return (mean ± seed SD) |','|---|---:|---:|---:|---:|']
for k in KINDS:
    a=np.array([r['return_mean'] for r in group(k,'test_near_above')]);v=[np.mean([r['success_rate'] for r in group(k,s)]) for s in ['validation','test_near_above','test_rotation']]
    table.append(f'| {k} | {v[0]*100:.1f}% | {v[1]*100:.1f}% | {v[2]*100:.1f}% | {a.mean():.2f} ± {a.std(ddof=1):.2f} |')
stats['controller']=controllers
stats['rows']=len(rows);stats['all_returns_finite']=bool(all(np.isfinite(r['return_mean']) for r in rows))
assert stats['rows']==275 and stats['all_returns_finite']
for kind in KINDS:
 for seed in range(5):
    d=json.loads((OUT/f'{kind}_seed{seed}_h{0 if kind=="model_free" else 5}_u4.json').read_text())
    assert d['real_transitions']==16000 and sum(d['visited_energy_bins'])==16000
stats['real_budget_and_coverage_checks_passed']=True
fig,ax=plt.subplots(figsize=(10,4.6));coverage=[]
for kind in KINDS:
    ds=[json.loads((OUT/f'{kind}_seed{s}_h{0 if kind=="model_free" else 5}_u4.json').read_text()) for s in range(5)]
    coverage.append(np.mean([d['visited_energy_bins'] for d in ds],axis=0)/16000)
coverage=np.array(coverage);bottom=np.zeros(7)
for i,(label,col) in enumerate(zip(['≤1.2','1.2–1.8','1.8–2.2','2.2–3.0','>3.0'],['#218b87','#69b6b0','#bb9349','#cc705b','#476ca0'])):
    ax.bar(KINDS,coverage[:,i],bottom=bottom,label=label,color=col);bottom+=coverage[:,i]
ax.set(ylabel='Fraction of visited real training states',title='Low-energy resets do not imply low-energy RL replay')
ax.tick_params(axis='x',rotation=20);ax.set_title('Low-energy resets do not imply low-energy RL replay',pad=55)
ax.legend(title='True energy',fontsize=8,ncol=5,loc='lower center',bbox_to_anchor=(.5,1.02));save(fig,'05_real_replay_energy_coverage')
stats['fraction_real_states_above_world_model_training_band']={k:float(1-coverage[i,0]) for i,k in enumerate(KINDS)}
(OUT/'summary_table.md').write_text('\n'.join(table)+'\n')
(OUT/'summary.json').write_text(json.dumps(stats,indent=2))
keys=['kind','seed','horizon','split','return_mean','success_rate','effort_mean','signed_work_mean','final_angle_rms_mean','final_momentum_rms_mean']
with (OUT/'results.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(rows)
print('\n'.join(table));print('Descriptive rho:',rho)
