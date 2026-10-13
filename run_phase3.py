"""Fixed-world-model SAC pilot; protocol frozen in phase3_protocol.json."""
from pathlib import Path
import argparse,json,csv,time,hashlib
import numpy as np
import torch
from world_models import WorldModel
from sac import SAC,Replay,Dynamics,obs

torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'phase3';OUT.mkdir(exist_ok=True)
P=json.loads((ROOT/'phase3_protocol.json').read_text())
TRAIN=torch.tensor(np.load(ROOT/'phase1/train.npz')['states'][:,0],dtype=torch.float32)
SPLITS=['validation','test_moderate','test_near_below','test_near_above','test_rotation']
INITIAL={s:torch.tensor(np.load(ROOT/'phase1'/f'{s}.npz')['states'][:,0],dtype=torch.float32) for s in SPLITS}

@torch.no_grad()
def evaluate(agent,initial,controller=False,record=False):
    dyn=Dynamics();z=initial.clone();ret=torch.zeros(len(z));eff=ret.clone();signed_work=ret.clone()
    success_steps=[];angle_errors=[];momenta=[];states=[z.numpy().copy()];actions=[]
    for step in range(400):
        if controller:
            error=torch.atan2(torch.sin(z[:,0]-np.pi),torch.cos(z[:,0]-np.pi))
            H=.5*z[:,1]**2+1-torch.cos(z[:,0])
            u=1.5*(2-H)*z[:,1]
            local=(error.abs()<.4)&(z[:,1].abs()<1)
            u=torch.where(local,torch.sin(z[:,0])-4*error-2.5*z[:,1],u).clamp(-2,2)[:,None]
        else:u=agent.actor.sample(obs(z),deterministic=True)[0]
        nz,r=dyn.step(z,u);ret+=r[:,0];eff+=.05*u[:,0]**2
        # Signed work is recovered from exact mechanical energy change, only as evaluation diagnostic.
        signed_work+=(.5*nz[:,1]**2+1-torch.cos(nz[:,0]))-(.5*z[:,1]**2+1-torch.cos(z[:,0]))
        z=nz;error=torch.atan2(torch.sin(z[:,0]-np.pi),torch.cos(z[:,0]-np.pi))
        success_steps.append((error.abs()<.2)&(z[:,1].abs()<.3))
        angle_errors.append(error);momenta.append(z[:,1])
        if record:states.append(z.numpy().copy());actions.append(u.numpy().copy())
    success=torch.stack(success_steps[-20:]).all(0)
    angle_rms=torch.stack(angle_errors[-20:]).square().mean(0).sqrt()
    momentum_rms=torch.stack(momenta[-20:]).square().mean(0).sqrt()
    result={'return_mean':float(ret.mean()),'success_rate':float(success.float().mean()),
            'effort_mean':float(eff.mean()),'signed_work_mean':float(signed_work.mean()),
            'final_angle_rms_mean':float(angle_rms.mean()),'final_momentum_rms_mean':float(momentum_rms.mean()),
            'per_episode_angle_rms':angle_rms.tolist(),'per_episode_momentum_rms':momentum_rms.tolist(),
            'per_episode_return':ret.tolist(),'per_episode_success':success.tolist()}
    if record:result['states']=np.array(states);result['actions']=np.array(actions)
    return result

def run(kind,seed,horizon,budget=16000,updates_per_collection=1,validation_only=False,output_subdir=''):
    out=ROOT/'phase3'/output_subdir;out.mkdir(exist_ok=True,parents=True)
    name=f'{kind}_seed{seed}_h{horizon}'
    if updates_per_collection!=1:name+=f'_u{updates_per_collection}'
    torch.manual_seed(4000+seed)
    rng=torch.Generator().manual_seed(9000+seed);synth_rng=torch.Generator().manual_seed(13000+seed)
    agent=SAC();real=Replay(20000);fake=Replay(20000)
    true=Dynamics();model=None if kind in ('model_free','true_model') else WorldModel.load(ROOT/'phase2'/f'{kind}_{seed}.npz')
    imagined=Dynamics(model)
    z=TRAIN[torch.randint(len(TRAIN),(8,),generator=rng)].clone();age=0;history=[];energy_bins=np.zeros(5,dtype=np.int64)
    rejected=generated=0;tick=time.time()
    for batchstep in range(budget//8):
        with torch.no_grad():
            u=4*torch.rand((8,1),generator=rng)-2 if batchstep*8<1024 else agent.actor.sample(obs(z))[0]
            nz,r=true.step(z,u);real.add(z,u,r,nz)
            H=(.5*nz[:,1]**2+1-torch.cos(nz[:,0])).numpy()
            energy_bins+=np.histogram(H,bins=[0,1.2,1.8,2.2,3.,np.inf])[0]
            z=nz;age+=1
            if age==400:z=TRAIN[torch.randint(len(TRAIN),(8,),generator=rng)].clone();age=0
        if real.n>=1024:
            if kind!='model_free' and batchstep%10==0:
                with torch.no_grad():
                    limit=640 if output_subdir=='calibrated' else 128*horizon
                    branches=int(np.ceil(limit/horizon))
                    start=real.sample(branches,generator=synth_rng)[0];attempted=0
                    for _ in range(horizon):
                        start=start[:limit-attempted]
                        if len(start)==0:break
                        a=agent.actor.sample(obs(start),generator=synth_rng)[0];end,rr=imagined.step(start,a)
                        valid=torch.isfinite(end).all(1)&torch.isfinite(rr).all(1)&(end[:,1].abs()<10)
                        generated+=len(start);attempted+=len(start);rejected+=int((~valid).sum())
                        if valid.any():fake.add(start[valid],a[valid],rr[valid],end[valid])
                        start=end[valid]
                        if len(start)==0:break
            for _ in range(updates_per_collection):
                if kind=='model_free' or fake.n==0:batch=real.sample(256)
                else:
                    b1=real.sample(128);b2=fake.sample(128);batch=tuple(torch.cat([x,y]) for x,y in zip(b1,b2))
                losses=agent.update(*batch)
                assert np.isfinite(losses).all(),name
        if (batchstep+1)*8%4000==0:
            e=evaluate(agent,INITIAL['validation'][:8])
            history.append({'real_steps':(batchstep+1)*8,'validation_return':e['return_mean'],'validation_success':e['success_rate']})
    agent.save(out/f'{name}.pt');evaluations=[]
    evaluation_splits=['validation'] if validation_only else SPLITS
    for split in evaluation_splits:
        e=evaluate(agent,INITIAL[split]);evaluations.append({'kind':kind,'seed':seed,'horizon':horizon,'split':split,**e})
    example_split='validation' if validation_only else 'test_near_above'
    ex=int(np.argsort(np.load(ROOT/'phase1'/f'{example_split}.npz')['initial_energy'])[16])
    e=evaluate(agent,INITIAL[example_split][ex:ex+1],record=True)
    np.savez_compressed(out/f'{name}_trajectory.npz',states=e['states'],actions=e['actions'])
    result={'name':name,'history':history,'evaluation':evaluations,'real_transitions':budget,
      'model_pretraining_transitions':25600 if kind not in ('model_free','true_model') else 0,
      'synthetic_generated':generated,'synthetic_rejected':rejected,'visited_energy_bins':energy_bins.tolist(),
      'elapsed_seconds':time.time()-tick,'torch':torch.__version__,
      'updates_per_collection':updates_per_collection,'validation_only':validation_only}
    (out/f'{name}.json').write_text(json.dumps(result,indent=2))
    print(f"{name}: ID return {evaluations[0]['return_mean']:.2f}, success {evaluations[0]['success_rate']:.2f}, {time.time()-tick:.1f}s",flush=True)
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');ap.add_argument('--kind');ap.add_argument('--seed',type=int,default=0);ap.add_argument('--horizon',type=int,default=5);ap.add_argument('--workers',type=int,default=1);ap.add_argument('--updates',type=int,default=1);ap.add_argument('--validation-only',action='store_true');ap.add_argument('--calibrated',action='store_true');args=ap.parse_args()
    if args.smoke:run('hnn',0,5,1600)
    elif args.kind:run(args.kind,args.seed,args.horizon,updates_per_collection=args.updates,validation_only=args.validation_only)
    else:
        schedule=[(k,s,0 if k=='model_free' else 5) for k in P['models'] for s in P['seeds']]
        schedule +=[(k,s,h) for k in P['horizon_ablation_models'] for h in P['additional_horizons'] for s in P['seeds']]
        subdir='calibrated' if args.calibrated else '';updates=4 if args.calibrated else 1
        comparison_out=ROOT/'phase3'/subdir;comparison_out.mkdir(exist_ok=True)
        suffix='_u4' if args.calibrated else ''
        pending=[]
        for kind,seed,horizon in schedule:
            f=comparison_out/f'{kind}_seed{seed}_h{horizon}{suffix}.json'
            if not f.exists() or json.loads(f.read_text())['real_transitions']!=16000:
                pending.append((kind,seed,horizon))
        if args.workers==1:
            for job in pending:run(*job,updates_per_collection=updates,output_subdir=subdir)
        else:
            from concurrent.futures import ProcessPoolExecutor
            import multiprocessing
            with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
                futures=[pool.submit(run,*job,updates_per_collection=updates,output_subdir=subdir) for job in pending]
                for future in futures:future.result()
        rows=[];histories=[]
        for kind,seed,h in schedule:
            f=comparison_out/f'{kind}_seed{seed}_h{h}{suffix}.json';d=json.loads(f.read_text());rows.extend(d['evaluation']);histories.append(d)
        controller=[{'kind':'energy_controller','split':s,**evaluate(None,INITIAL[s],controller=True)} for s in SPLITS]
        (comparison_out/'results.json').write_text(json.dumps(rows,indent=2));(comparison_out/'controller.json').write_text(json.dumps(controller,indent=2))
        protocol=ROOT/('phase3_calibrated_protocol.json' if args.calibrated else 'phase3_protocol.json')
        (comparison_out/'run_manifest.json').write_text(json.dumps({'protocol_sha256':hashlib.sha256(protocol.read_bytes()).hexdigest(),
          'runs':len(schedule),'total_real_training_transitions':sum(d['real_transitions'] for d in histories),
          'synthetic_generated':sum(d['synthetic_generated'] for d in histories),'synthetic_rejected':sum(d['synthetic_rejected'] for d in histories)},indent=2))
