"""Posthoc classical symmetry explanation; does not change the practical gate."""
import json,numpy as np
from phase7_design import *
rng=np.random.default_rng(175000);R,_,_=structural_matrices();checks=[]
a=np.linspace(-np.pi,np.pi,32,endpoint=False);grid=np.stack(np.meshgrid(a,a,indexing='ij'),-1).reshape(-1,2)
for kind in ['force','hamiltonian']:
    L=basis_map(kind);B=blocks(grid,L);Mhaar=np.einsum('nfi,nfj->ij',B,B)/len(B);d=L.shape[1]
    assert np.max(np.abs(Mhaar-np.eye(d)))<1e-12
    s=rng.uniform(-np.pi,np.pi,2);Rot=np.eye(M)
    for j,k in enumerate(K):
        theta=k@s;ids=[1+j,1+len(K)+j];Rot[np.ix_(ids,ids)]=[[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]]
    Full=np.kron(np.eye(2),Rot);O=L.T@R@Full@L
    assert np.max(np.abs(Full@L-L@O))<1e-12
    assert np.max(np.abs(O.T@O-np.eye(d)))<1e-12
    assert np.max(np.abs(blocks(grid+s,L)-B@O))<1e-12
    q=rng.uniform(-np.pi,np.pi,(80,2));C=blocks(q,L);Mxi=np.einsum('nfi,nfj->ij',C,C)/len(C)
    for task,p in P['tasks'].items():
        W=spectral_metric(L,p);commute=float(np.max(np.abs(O.T@W@O-W)));assert commute<1e-12
        n=256;rho=P['prior_precision_force_whitened'];uniform=float(np.trace(W)/(rho+n));nonuniform=float(np.trace(W@np.linalg.inv(rho*np.eye(d)+n*Mxi)))
        assert nonuniform>=uniform-1e-12
        checks.append({'kind':kind,'task':task,'haar_information_max_identity_error':float(np.max(np.abs(Mhaar-np.eye(d)))),
          'metric_translation_invariance_error':commute,'uniform_objective':uniform,'one_nonuniform_objective':nonuniform})
(OUT/'posthoc_symmetry_check.json').write_text(json.dumps({'status':'posthoc mathematical illustration only','local_pilot_information_included':False,'checks':checks},indent=2));print(json.dumps(checks,indent=2))
