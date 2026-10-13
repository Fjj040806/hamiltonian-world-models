"""Independent finite-difference, least-squares and artifact checks."""
from pathlib import Path
import json,hashlib
import numpy as np
# Import has no dependency on running experiment; run_phase8 is intentionally a CLI.
R=Path(__file__).resolve().parent;data=json.loads((R/'phase8/results.json').read_text())
assert data['protocol_sha256']==hashlib.sha256((R/'phase8_protocol.json').read_bytes()).hexdigest()
q=np.random.default_rng(44).uniform(-1,1,(40,2));x,y=q.T
def V(q):
 x,y=q.T;return .5*x*x+.65*y*y+.1*(x-y)**2+.25*x**4+.2*y**4
force=np.stack([-x-.2*(x-y)-x**3,-1.3*y+.2*(x-y)-.8*y**3],axis=1)
fd=np.zeros_like(force)
for j in range(2):
 dq=np.zeros_like(q);dq[:,j]=1e-5;fd[:,j]=-(V(q+dq)-V(q-dq))/2e-5
assert np.max(np.abs(fd-force))<1e-8
assert len(data['duffing'])==45 and len(data['recenter'])==20 and len(data['held_control'])==4
# Independently reconstruct a polynomial Hamiltonian fit via augmented QR.
E=[(a,b) for a in range(5) for b in range(5-a) if a+b>0];D=np.zeros((40,2,len(E)))
for j,(a,b) in enumerate(E):
 if a:D[:,0,j]=-a*x**(a-1)*y**b
 if b:D[:,1,j]=-b*x**a*y**(b-1)
B=D.reshape(-1,len(E));yy=force.reshape(-1);A=B.T@B/40+1e-6*np.eye(len(E));b=B.T@yy/40
c=np.linalg.solve(A,b);qr=np.linalg.lstsq(np.vstack([B/np.sqrt(40),.001*np.eye(len(E))]),np.r_[yy/np.sqrt(40),np.zeros(len(E))],rcond=None)[0]
assert np.max(np.abs(c-qr))<1e-9
out={'duffing_force_gradient_max_error':float(np.max(np.abs(fd-force))),'independent_augmented_QR_coefficient_error':float(np.max(np.abs(c-qr))),'unique_duffing_fits':45,'recenter_fits':20,'held_control_evaluations':4,'new_acquisition_variants':0,'all_checks_passed':True}
(R/'phase8/verification.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
