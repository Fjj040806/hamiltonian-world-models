"""Evaluate actual true-plant feedback/MPC on all selected models, with resumes."""
from pathlib import Path
import argparse,json,time,hashlib
import numpy as np
from phase6_models import *
from phase6_control import Planner
RESULTS=OUT/'controllers';RESULTS.mkdir(exist_ok=True)

def fingerprint(plant,kind,cov,w):
    h=hashlib.sha256((ROOT/'phase6_protocol.json').read_bytes())
    for name in ['phase6_models.py','phase6_control.py','run_phase6_controllers.py']:h.update((ROOT/name).read_bytes())
    if kind!='known_force':h.update((OUT/plant/f'{kind}_{cov}_{w}.npz').read_bytes())
    return h.hexdigest()
def run(plant,kind,cov,w,controller):
    name=f'{plant}_{kind}_{cov}_w{w}_{controller}';model=None if kind=='known_force' else Model.load(OUT/plant/f'{kind}_{cov}_{w}.npz')
    planner=Planner(model,plant);initial=np.load(OUT/plant/'initial_states.npz')['rotation'];z=initial.copy()
    returns=np.zeros(len(z));errors=[];states=[z.copy()];actions=[];diagnostics=[];tick=time.time();max_speed=0.
    for i in range(P['controller_steps']):
        if controller=='feedback':u=planner.feedback(z)
        else:
            u,d=planner.action(z);diagnostics.append(d)
        old=z;z=step_numpy(z,u,plant=plant)
        def cost(s):return np.sum((s[:,2:]-P['controller_target'])**2,axis=1)+.01*np.sum(u*u,axis=1)
        # Five-substep trapezoid reward, independent of the planner's surrogate.
        r=np.zeros(len(z));rz=old.copy()
        for _ in range(5):
            nz=step_numpy(rz,u,plant=plant,substeps=1)
            r-=.01*(cost(rz)+cost(nz))/2;rz=nz
        assert np.max(np.abs(rz-z))<1e-10
        returns+=r;errors.append(z[:,2:]-P['controller_target']);max_speed=max(max_speed,float(np.max(np.abs(z[:,2:]))))
        assert np.isfinite(z).all() and np.isfinite(r).all()
        states.append(z.copy());actions.append(u.copy())
    e=np.array(errors[-20:]);norm=np.linalg.norm(e,axis=2);rms=np.sqrt(np.mean(np.sum(e*e,axis=2),axis=0));success=(norm<.25).all(0)
    result={'plant':plant,'kind':kind,'coverage':cov,'world_seed':w,'controller':controller,'return_mean':float(returns.mean()),
        'momentum_vector_rms_mean':float(rms.mean()),'success_rate':float(success.mean()),'per_episode_return':returns.tolist(),
        'per_episode_momentum_rms':rms.tolist(),'per_episode_success':success.tolist(),'maximum_true_momentum':max_speed,
        'mpc_maximum_cost_increase':max([d['maximum_cost_increase'] for d in diagnostics],default=0),
        'mean_initial_planning_cost':float(np.mean([d['initial_cost'] for d in diagnostics])) if diagnostics else None,
        'mean_final_planning_cost':float(np.mean([d['final_cost'] for d in diagnostics])) if diagnostics else None,
        'elapsed_seconds':time.time()-tick,'fingerprint':fingerprint(plant,kind,cov,w)}
    np.savez_compressed(RESULTS/f'{name}_trajectory.npz',states=np.array(states)[:,0:1],actions=np.array(actions)[:,0:1])
    (RESULTS/f'{name}.json').write_text(json.dumps(result,indent=2))
    print(f'{name}: RMS {rms.mean():.4f}, success {success.mean():.2f}, {time.time()-tick:.1f}s',flush=True)
    return result

def main(workers):
    jobs=[(plant,k,c,w,ctrl) for plant in P['controller_plants'] for c in P['coverage'] for k in P['families'] for w in P['world_seeds'] for ctrl in P['controllers']]
    jobs+=[(plant,'known_force','reference',-1,ctrl) for plant in P['controller_plants'] for ctrl in P['controllers']]
    pending=[]
    for plant,k,c,w,ctrl in jobs:
        rec=RESULTS/f'{plant}_{k}_{c}_w{w}_{ctrl}.json'
        if not rec.exists() or json.loads(rec.read_text())['fingerprint']!=fingerprint(plant,k,c,w):pending.append((plant,k,c,w,ctrl))
    if workers==1:
        for j in pending:run(*j)
    else:
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            futures=[pool.submit(run,*j) for j in pending]
            for f in futures:f.result()
    rows=[json.loads((RESULTS/f'{plant}_{k}_{c}_w{w}_{ctrl}.json').read_text()) for plant,k,c,w,ctrl in jobs]
    (RESULTS/'results.json').write_text(json.dumps(rows,indent=2))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=1);args=ap.parse_args();main(args.workers)
