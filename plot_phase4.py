"""Budget intervention summaries; distinguish reused references from actual runs."""
from pathlib import Path
import json,csv
import numpy as np
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'phase4'
rows=json.loads((OUT/'results.json').read_text())
KINDS=['node','force_zero_mean','hnn'];COL={'node':'#81929d','force':'#476ca0','force_zero_mean':'#9a78a7','hnn':'#218b87','true_only':'#243746'}
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
def subset(k,b,split='test_near_above'):
    if b==16000:return [r for r in rows if r['kind']=='true_only' and r['split']==split]
    return [r for r in rows if r['kind']==k and r['oracle_budget']==b and r['split']==split]
def save(fig,name):
    fig.tight_layout()
    for ext in ['png','svg']:fig.savefig(OUT/f'{name}.{ext}',dpi=170)
    plt.close(fig)
fig,axs=plt.subplots(1,3,figsize=(13,4.3))
for k in KINDS:
    for ax,metric in zip(axs,['return_mean','success_rate','final_angle_rms_mean']):
        vals=[np.array([r[metric] for r in subset(k,b)]) for b in [0,1000,16000]]
        ax.plot(range(3),[v.mean() for v in vals],marker='o',color=COL[k],label=k)
        for j,v in enumerate(vals):ax.scatter(j+np.linspace(-.04,.04,len(v)),v,color=COL[k],s=12,alpha=.4)
        ax.set_xticks(range(3),['0','1,000','16,000']);ax.set_xlabel('True training transition queries')
axs[0].set_ylabel('Mean episode return');axs[1].set_ylabel('Strict success fraction');axs[2].set_ylabel('Final angle RMS (radians)')
axs[0].legend(fontsize=8);fig.suptitle('Upright transfer: fixed 16,000 samples / 7,492 updates; crossed model and policy seeds')
save(fig,'01_feedback_budget_intervention')

fig,axs=plt.subplots(1,3,figsize=(12,4.3));data={}
for ax,k in zip(axs,KINDS):
    matrix=np.zeros((3,3));delta=np.zeros((3,3))
    for i,w in enumerate([0,3,4]):
      for j,s in enumerate([0,3,4]):
        base=next(r for r in rows if r['kind']==k and r['world_seed']==w and r['policy_seed']==s and r['oracle_budget']==0 and r['split']=='test_near_above')
        corr=next(r for r in rows if r['kind']==k and r['world_seed']==w and r['policy_seed']==s and r['oracle_budget']==1000 and r['split']=='test_near_above')
        matrix[i,j]=base['return_mean'];delta[i,j]=corr['return_mean']-base['return_mean']
    data[k]={'zero_query_return_matrix':matrix.tolist(),'paired_1000_query_return_change':delta.tolist()}
    im=ax.imshow(matrix,vmin=-1.3,vmax=-.8,cmap='viridis')
    for i in range(3):
     for j in range(3):ax.text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center',color='white' if matrix[i,j]<-1.05 else '#243746')
    ax.set_xticks(range(3),[0,3,4]);ax.set_yticks(range(3),[0,3,4]);ax.set(xlabel='Policy seed',ylabel='World-model seed',title=k)
fig.suptitle('Zero-query true-environment return: crossed seeds are distinct factors');save(fig,'02_crossed_seed_returns')
summary={};table=['| Model | Oracle queries | Above return | Strict success | Final angle RMS |','|---|---:|---:|---:|---:|']
for k in KINDS:
 for b in [0,1000]:
    vals=subset(k,b);summary[f'{k}_{b}']={metric:float(np.mean([r[metric] for r in vals])) for metric in ['return_mean','success_rate','final_angle_rms_mean']}
    v=summary[f'{k}_{b}'];table.append(f"| {k} | {b} | {v['return_mean']:.3f} | {v['success_rate']*100:.1f}% | {v['final_angle_rms_mean']:.4f} |")
ref=subset('true_only',16000);summary['true_reference']={m:float(np.mean([r[m] for r in ref])) for m in ['return_mean','success_rate','final_angle_rms_mean']}
summary['crossed_seeds']=data;summary['reference_unique_policies']=3
(OUT/'summary.json').write_text(json.dumps(summary,indent=2));(OUT/'summary_table.md').write_text('\n'.join(table)+'\n')
fields=['kind','world_seed','policy_seed','oracle_budget','split','return_mean','success_rate','final_angle_rms_mean','final_momentum_rms_mean']
with (OUT/'results.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
print('\n'.join(table))

# Secondary task is explicitly an exploratory one-policy-seed screen.
rotpath=OUT/'rotation/results.json'
if rotpath.exists():
    rotation=json.loads(rotpath.read_text());fig,axs=plt.subplots(1,2,figsize=(11,4.3));rsummary={}
    for k in ['node','force','force_zero_mean','hnn']:
        for b in [0,1000]:
            a=[r for r in rotation if r['kind']==k and r['oracle_budget']==b and r['split']=='test_near_above']
            rsummary[f'{k}_{b}']={metric:float(np.mean([r[metric] for r in a])) for metric in ['return_mean','success_rate','momentum_rms_mean']}
        for ax,metric in zip(axs,['momentum_rms_mean','return_mean']):
            vals=[np.array([r[metric] for r in rotation if r['kind']==k and r['oracle_budget']==b and r['split']=='test_near_above']) for b in [0,1000]]
            ax.plot([0,1],[v.mean() for v in vals],marker='o',label=k,color=COL[k])
            for j,v in enumerate(vals):ax.scatter(j+np.linspace(-.025,.025,3),v,color=COL[k],s=18,alpha=.5)
            ax.set_xticks([0,1],['0','1,000']);ax.set_xlabel('True training transition queries')
    reference=next(r for r in rotation if r['kind']=='true_only' and r['split']=='test_near_above')
    axs[0].axhline(reference['momentum_rms_mean'],color=COL['true_only'],ls='--',label='True-only reference')
    axs[0].set_ylabel('Final momentum tracking RMS');axs[1].set_ylabel('Mean episode return');axs[0].legend(fontsize=8)
    fig.suptitle('Persistent rotation transfer | exploratory: 3 world seeds, one policy seed')
    save(fig,'03_rotation_task_probe');rsummary['reference']=reference
    (OUT/'rotation/summary.json').write_text(json.dumps(rsummary,indent=2))
    print('Rotation summaries:',json.dumps({k:v for k,v in rsummary.items() if k!='reference'},indent=2))
