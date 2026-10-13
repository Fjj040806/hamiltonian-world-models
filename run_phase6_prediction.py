"""Fresh test evaluation after validation-only choices have been written."""
from pathlib import Path
import argparse,json
import numpy as np
from phase6_models import *

def evaluate(rec):
    plant=rec['plant'];kind=rec['kind'];cov=rec['coverage'];w=rec['world_seed'];folder=OUT/plant
    model=Model.load(folder/f'{kind}_{cov}_{w}.npz');initial=np.load(folder/'initial_states.npz');rows=[]
    gridq=np.linspace(-np.pi,np.pi,128,endpoint=False);grid=np.stack(np.meshgrid(gridq,gridq,indexing='ij'),-1).reshape(-1,2)
    global_rms=float(np.sqrt(np.mean(np.sum((model.force(grid)-plant_force(grid,plant))**2,axis=1))))
    for split in P['test_energy_bands']:
        true=np.load(folder/f'true_{split}_prediction.npz')['states'];z=initial[split].copy();pred=[z.copy()]
        for _ in range(P['prediction_steps']):
            z=step_numpy(z,np.zeros((len(z),2)),model,plant);pred.append(z.copy())
        pred=np.array(pred);dq=np.arctan2(np.sin(pred[...,:2]-true[...,:2]),np.cos(pred[...,:2]-true[...,:2]))
        err=np.concatenate([dq,pred[...,2:]-true[...,2:]],axis=-1);rms=np.sqrt(np.mean(np.sum(err**2,axis=-1),axis=1))
        ed=plant_energy(pred,plant)-plant_energy(pred[0],plant)
        row={**rec,'split':split,'global_force_vector_rms':global_rms,'rmse_t5':float(rms[100]),'rmse_t20':float(rms[-1]),
             'mean_max_true_energy_error':float(np.mean(np.max(np.abs(ed),axis=0))),'maximum_predicted_momentum':float(np.max(np.abs(pred[...,2:])))}
        if kind in ('hamiltonian','odd_hamiltonian'):
            hh=.5*np.sum(pred[...,2:]**2,axis=-1)+model.potential(pred[...,:2]);row['learned_energy_max_drift']=float(np.max(np.abs(hh-hh[0])))
        rows.append(row)
        # Full temporal mean/error diagnostics; one preselected trace for animation.
        np.savez_compressed(folder/f'{kind}_{cov}_{w}_{split}_diagnostics.npz',rmse=rms,per_episode_final_error=np.sqrt(np.sum(err[-1]**2,axis=1)),
            per_episode_max_true_energy_error=np.max(np.abs(ed),axis=0),prediction_trace=pred[:,0],true_trace=true[:,0])
    print(f'Prediction {plant}/{cov}/{kind}/w{w}: global force RMS {global_rms:.4f}',flush=True)
    return rows

def main(workers):
    selected=json.loads((OUT/'selected_models.json').read_text())
    for plant in P['plants']:
        folder=OUT/plant;initial=np.load(folder/'initial_states.npz')
        for split in initial.files:
            z=initial[split].copy();states=[z.copy()]
            for _ in range(P['prediction_steps']):z=step_numpy(z,np.zeros((len(z),2)),plant=plant);states.append(z.copy())
            np.savez_compressed(folder/f'true_{split}_prediction.npz',states=np.array(states))
    if workers==1:groups=[evaluate(x) for x in selected]
    else:
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:groups=list(pool.map(evaluate,selected))
    rows=[x for g in groups for x in g];(OUT/'prediction_results.json').write_text(json.dumps(rows,indent=2))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=1);a=ap.parse_args();main(a.workers)
