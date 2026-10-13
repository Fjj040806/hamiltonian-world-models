"""Independent quadrature check of the newly expanded finite-time filter formulas."""
from pathlib import Path
import json
import numpy as np
from scipy.integrate import quad
R=Path(__file__).resolve().parent
A=np.array([.7,-.2]);B=np.array([.3,.5]);kappa=2.;maxerr=0;parsevalerr=0
for omega in [0.,2.,6.]:
 C=(kappa*A-omega*B)/(kappa*kappa+omega*omega);D=(omega*A+kappa*B)/(kappa*kappa+omega*omega)
 for t in [.0,.2,1.,5.]:
  for theta in [.2,.7,2.1]:
   response=C*np.cos(theta+omega*t)+D*np.sin(theta+omega*t)-np.exp(-kappa*t)*(C*np.cos(theta)+D*np.sin(theta))
   integ=np.array([quad(lambda s:np.exp(-kappa*(t-s))*(A[j]*np.cos(theta+omega*s)+B[j]*np.sin(theta+omega*s)),0,t,epsabs=1e-12)[0] for j in range(2)])
   maxerr=max(maxerr,float(np.max(np.abs(response-integ))))
  theta=np.arange(2048)*2*np.pi/2048
  response=np.cos(theta[:,None]+omega*t)*C+np.sin(theta[:,None]+omega*t)*D-np.exp(-kappa*t)*(np.cos(theta[:,None])*C+np.sin(theta[:,None])*D)
  empirical=np.mean(np.sum(response**2,axis=1))
  theory=(np.sum(A*A)+np.sum(B*B))/(2*(kappa*kappa+omega*omega))*(1+np.exp(-2*kappa*t)-2*np.exp(-kappa*t)*np.cos(omega*t))
  parsevalerr=max(parsevalerr,float(abs(theory-empirical)))
assert maxerr<1e-11 and parsevalerr<1e-12
result={'filter_quadrature_max_error':maxerr,'finite_time_phase_averaged_formula_error':parsevalerr,'new_plant_evaluations':0,'new_policy_runs':0,'all_checks_passed':True}
(R/'math_appendix_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
