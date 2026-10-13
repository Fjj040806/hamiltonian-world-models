"""Fresh, paired true-plant tracking tests with no model-selection feedback."""
import argparse,json,time,hashlib
import numpy as np
from phase7_design import *
CONTROLS=OUT/'control';CONTROLS.mkdir(exist_ok=True)

def fingerprint(model_file,task):
    h=hashlib.sha256((ROOT/'phase7_protocol.json').read_bytes())
    for name in ['phase7_design.py','run_phase7_control.py','phase6_models.py','coupled_models.py']:h.update((ROOT/name).read_bytes())
    if model_file!='reference':h.update((OUT/'models'/model_file).read_bytes())
    h.update(task.encode());return h.hexdigest()

def run(model_file,plant,task):
    model=None if model_file=='reference' else Model.load(OUT/'models'/model_file)
    target=np.array(P['tasks'][task]);q=np.load(OUT/'data/test_phases.npz')['q'];z=np.c_[q,np.broadcast_to(target,q.shape)]
    returns=np.zeros(len(z));errors=[];states=[z.copy()];actions=[];clipped=0;tick=time.time()
    for _ in range(P['control_steps']):
        force=plant_force(z[:,:2],plant) if model is None else model.force(z[:,:2])
        raw=-force-P['feedback_gain']*(z[:,2:]-target);clipped+=int((np.abs(raw)>2).sum());u=np.clip(raw,-2,2)
        for _ in range(5):
            old=z;z=step_numpy(z,u,plant=plant,substeps=1)
            cost=lambda s:np.sum((s[:,2:]-target)**2,axis=1)+.01*np.sum(u*u,axis=1)
            returns-=.01*(cost(old)+cost(z))/2
        assert np.isfinite(z).all();errors.append(z[:,2:]-target);states.append(z.copy());actions.append(u.copy())
    err=np.array(errors[-100:]);rms=np.sqrt(np.mean(np.sum(err*err,axis=2),axis=0));success=(np.linalg.norm(err,axis=2)<.25).all(0)
    result={'plant':plant,'task':task,'model_file':model_file,'momentum_vector_rms_mean':float(rms.mean()),
       'return_mean':float(returns.mean()),'success_rate':float(success.mean()),'per_phase_momentum_rms':rms.tolist(),
       'per_phase_return':returns.tolist(),'per_phase_success':success.tolist(),
       'clipped_action_component_fraction':clipped/(400*len(q)*2),'elapsed_seconds':time.time()-tick,'fingerprint':fingerprint(model_file,task)}
    stem=f'{plant}_reference_{task}' if model_file=='reference' else f'{model_file[:-4]}_{task}'
    np.savez_compressed(CONTROLS/f'{stem}_trajectory.npz',states=np.array(states)[:,0:1],actions=np.array(actions)[:,0:1])
    (CONTROLS/f'{stem}.json').write_text(json.dumps(result,indent=2))
    return result

def main(workers):
    records=json.loads((OUT/'acquisition_records.json').read_text());jobs={}
    for r in records:
        tasks=list(P['tasks']) if r['design_task']=='shared' else [r['design_task']]
        for task in tasks:jobs[(r['model_file'],task)]=(r['model_file'],r['plant'],task)
    for plant in P['plants']:
        for task in P['tasks']:jobs[(f'reference_{plant}',task)]=('reference',plant,task)
    pending=[];lookup={}
    for key,(file,plant,task) in jobs.items():
        stem=f'{plant}_reference_{task}' if file=='reference' else f'{file[:-4]}_{task}';path=CONTROLS/f'{stem}.json'
        if path.exists() and json.loads(path.read_text())['fingerprint']==fingerprint(file,task):lookup[key]=json.loads(path.read_text())
        else:pending.append((key,file,plant,task))
    if workers==1:
        for key,file,plant,task in pending:lookup[key]=run(file,plant,task)
    else:
        from concurrent.futures import ProcessPoolExecutor,as_completed
        import multiprocessing
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            fs={pool.submit(run,file,plant,task):key for key,file,plant,task in pending}
            for i,f in enumerate(as_completed(fs)):
                lookup[fs[f]]=f.result()
                if (i+1)%100==0:print(f'Paired feedback evaluations {i+1}/{len(pending)} complete',flush=True)
    rows=[]
    for r in records:
        tasks=list(P['tasks']) if r['design_task']=='shared' else [r['design_task']]
        for task in tasks:rows.append({**r,**lookup[(r['model_file'],task)]})
    refs=[lookup[(f'reference_{plant}',task)] for plant in P['plants'] for task in P['tasks']]
    (OUT/'control_results.json').write_text(json.dumps(rows,indent=2));(OUT/'references.json').write_text(json.dumps(refs,indent=2))
    (OUT/'control_evaluation_manifest.json').write_text(json.dumps({'unique_learned_model_task_evaluations':len(jobs)-4,'reference_evaluations':4,'reported_rows_including_pilot_aliases':len(rows)},indent=2))
    print('Unique feedback evaluations:',len(jobs),'reported rows:',len(rows),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=1);a=ap.parse_args();main(a.workers)
