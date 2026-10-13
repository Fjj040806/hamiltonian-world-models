"""Held-out force and first-order tracking diagnostics, never acquisition inputs."""
import json
import numpy as np
from phase7_design import *

def main():
    records=json.loads((OUT/'acquisition_records.json').read_text());unique={r['model_file']:r for r in records};rows=[]
    a=np.linspace(-np.pi,np.pi,128,endpoint=False);q=np.stack(np.meshgrid(a,a,indexing='ij'),-1).reshape(-1,2)
    trueC=np.zeros((M,2));n=len(K)
    for k,vec in [((1,0),[-1,0]),((0,1),[0,-.8]),((1,-1),[-.3,.3])]:
        j=next(j for j,x in enumerate(K) if tuple(x)==k);trueC[1+n+j]=vec
    R,_,_=structural_matrices()
    for name,r in unique.items():
        model=Model.load(OUT/'models'/name);error=model.force(q)-plant_force(q,r['plant']);v=(model.C-trueC).T.reshape(-1)
        eps=0 if r['plant']=='canonical' else .06
        metric={}
        for task,p in P['tasks'].items():
            w=1/(P['feedback_gain']**2+(K@p)**2);diag=np.r_[1/P['feedback_gain']**2,.5*w,.5*w]
            represented=float(np.sum(v*v*np.r_[diag,diag]));missing=5*eps*eps/(P['feedback_gain']**2+np.dot([3,-1],p)**2)
            metric[task]=represented+missing
        exact_force_error=float(v@R@v+5*eps**2);grid_force_error=float(np.mean(np.sum(error*error,axis=1)));assert abs(exact_force_error-grid_force_error)<1e-10
        rows.append({**r,'global_force_vector_rms':float(np.sqrt(grid_force_error)),
                     'ideal_first_order_tracking_mse':metric,'force_floor':float(np.sqrt(5*eps**2)),**model.statistics()})
    (OUT/'model_metrics.json').write_text(json.dumps(rows,indent=2));print('Held-out metrics:',len(rows),'unique models')
if __name__=='__main__':main()
