"""Freeze datasets, solve convex world-model fits and evaluate true rollouts."""
from pathlib import Path
import json,hashlib
import numpy as np
from coupled_models import *
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'phase5';OUT.mkdir(exist_ok=True)
P=json.loads((ROOT/'phase5_protocol.json').read_text())

def main():
    rows=[];checks=[];energy_checks=[]
    grid=np.stack(np.meshgrid(np.linspace(-np.pi,np.pi,128,endpoint=False),np.linspace(-np.pi,np.pi,128,endpoint=False),indexing='ij'),-1).reshape(-1,2)
    for plant,epsilon in P['plants'].items():
        target=OUT/plant;target.mkdir(exist_ok=True)
        initial={s:sample_states(P['test_initial_states_per_band'],b,np.random.default_rng(52000+i),epsilon) for i,(s,b) in enumerate(P['test_energy_bands'].items())}
        np.savez_compressed(target/'initial_states.npz',**initial)
        for w in P['world_seeds']:
            rng=np.random.default_rng(51000+w);z=sample_states(P['training_snapshots_per_world'],P['training_energy_band'],rng,epsilon)
            y=truth_force(z[:,:2],epsilon)+rng.normal(0,P['derivative_noise_std'],(len(z),2))
            np.savez_compressed(target/f'training_{w}.npz',states=z,force_labels=y,true_force=truth_force(z[:,:2],epsilon))
            for kind in P['families']:
                m,check=fit(kind,z[:,:2],y,P['ridge'],P['curl_penalty'],P['mean_penalty'],epsilon)
                m.save(target/f'{kind}_{w}.npz');stat=m.statistics()
                check.update(plant=plant,kind=kind,world_seed=w,**stat)
                assert check['normal_equation_residual']<1e-10
                if kind=='hamiltonian':assert check['hard_gradient_equivalence_max_coefficient_error']<1e-9
                checks.append(check)
                global_rmse=np.sqrt(np.mean(np.sum((m.force(grid)-truth_force(grid,epsilon))**2,axis=1)))
                for split,iz in initial.items():
                    tz=iz.copy();pz=iz.copy();truth=[tz.copy()];pred=[pz.copy()]
                    force_rmse=np.sqrt(np.mean(np.sum((m.force(iz[:,:2])-truth_force(iz[:,:2],epsilon))**2,axis=1)))
                    for j in range(400):
                        tz=step_numpy(tz,np.zeros((len(tz),2)),epsilon=epsilon);pz=step_numpy(pz,np.zeros((len(pz),2)),m,epsilon)
                        truth.append(tz.copy());pred.append(pz.copy())
                    truth=np.array(truth);pred=np.array(pred)
                    dq=np.arctan2(np.sin(pred[...,:2]-truth[...,:2]),np.cos(pred[...,:2]-truth[...,:2]))
                    error=np.concatenate([dq,pred[...,2:]-truth[...,2:]],axis=-1)
                    rmse=np.sqrt(np.mean(np.sum(error**2,axis=-1),axis=1))
                    delta=truth_energy(pred,epsilon)-truth_energy(pred[0],epsilon)
                    if kind=='hamiltonian' and split=='rotation':
                        hh=.5*np.sum(pred[:,:,2:]**2,axis=-1)+m.potential(pred[:,:,:2])
                        energy_checks.append({'plant':plant,'world_seed':w,'learned_energy_max_drift':float(np.max(np.abs(hh-hh[0])))})
                    np.savez_compressed(target/f'{kind}_{w}_{split}_prediction.npz',truth=truth,prediction=pred,rmse=rmse,true_energy_error=delta)
                    rows.append({'plant':plant,'kind':kind,'world_seed':w,'split':split,'initial_force_vector_rms':float(force_rmse),
                        'global_force_vector_rms':float(global_rmse),'rmse_t5':float(rmse[100]),'rmse_t20':float(rmse[-1]),
                        'mean_max_true_energy_error':float(np.mean(np.max(np.abs(delta),axis=0))),**stat})
                print(f'{plant} {kind} w{w}: global force RMS={global_rmse:.5f}, curl={stat["curl_rms"]:.5f}, mean={stat["mean_force_norm"]:.5f}',flush=True)
    (OUT/'model_results.json').write_text(json.dumps(rows,indent=2));(OUT/'model_verification.json').write_text(json.dumps(checks,indent=2))
    (OUT/'learned_energy_verification.json').write_text(json.dumps(energy_checks,indent=2))
    (OUT/'protocol_sha256.txt').write_text(hashlib.sha256((ROOT/'phase5_protocol.json').read_bytes()).hexdigest())
if __name__=='__main__':main()
