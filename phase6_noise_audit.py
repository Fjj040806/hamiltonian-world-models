"""Exact conditional variance checked by independent Gaussian RHS perturbations."""
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from phase6_models import *
from run_phase6_models import mixture
S=json.loads((ROOT/'phase6_noise_audit_protocol.json').read_text())
pool=np.load(OUT/S['plant']/f'pools_{S["world_seed"]}.npz')
train={dist:{key:pool[f'train_{dist}_{key}'] for key in ['states','labels']} for dist in ['local','global']}
rows=[];R,_,_=structural_matrices();rng=np.random.default_rng(121000)
for cov in S['coverage']:
    z,_,_=mixture(train,P['coverage'][cov]);X=features(z[:,:2]);D=np.kron(np.eye(2),X.T@X/len(z));N=len(z)
    for k in S['families']:
        if k=='force':T=np.eye(2*M)
        elif k=='hamiltonian':T=gradient_map()
        else:
            n=len(K);ids=np.r_[np.arange(1+n,M),M+np.arange(1+n,M)];T=np.eye(2*M)[:,ids]
        Rt=T.T@R@T;Dt=T.T@D@T;A=Dt+S['fixed_ridge']*Rt;Ai=np.linalg.inv(A)
        covariance=S['noise_std']**2/N*(Ai@Dt@Ai)
        analytic=float(np.trace(Rt@covariance))
        # Noise of the empirical RHS has covariance sigma^2 D/N. Sampling that
        # sufficient statistic is exactly equivalent to independent label noise.
        L=np.linalg.cholesky(Dt);rhs=S['noise_std']/np.sqrt(N)*L@rng.normal(size=(len(Dt),S['independent_noise_draws']))
        coeff=np.linalg.solve(A,rhs);values=np.sum(coeff*(Rt@coeff),axis=0);mc=float(values.mean())
        eig_R,Q=np.linalg.eigh(Rt);whitener=(Q*(1/np.sqrt(eig_R)))@Q.T
        normalized_eigen=np.linalg.eigvalsh(whitener@Dt@whitener)
        closed_form=S['noise_std']**2/N*np.sum(normalized_eigen/(normalized_eigen+S['fixed_ridge'])**2)
        assert abs(closed_form-analytic)<max(1e-10,analytic*1e-6)
        assert abs(mc-analytic)/analytic<.10
        rows.append({'kind':k,'coverage':cov,'degrees_of_freedom':len(Dt),'analytic_global_variance':analytic,
             'monte_carlo_global_variance':mc,'relative_monte_carlo_error':abs(mc-analytic)/analytic,
             'global_noise_rms':float(np.sqrt(analytic)),'smallest_normalized_design_eigenvalue':float(normalized_eigen.min()),
             'largest_normalized_design_eigenvalue':float(normalized_eigen.max()),'fixed_ridge':S['fixed_ridge']})
(OUT/'noise_audit.json').write_text(json.dumps(rows,indent=2))
fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True);colors=['#73695f','#477caa','#d58047']
for k,c in zip(S['families'],colors):
    rr=[r for r in rows if r['kind']==k];axes[0].semilogy(range(3),[r['global_noise_rms'] for r in rr],'o-',c=c,label=k.replace('_',' '))
    axes[1].loglog([r['analytic_global_variance'] for r in rr],[r['monte_carlo_global_variance'] for r in rr],'o',c=c)
vals=[r['analytic_global_variance'] for r in rows];axes[1].plot([min(vals),max(vals)],[min(vals),max(vals)],':',c='gray')
axes[0].set_xticks(range(3),['0%','10%','100%']);axes[0].set_xlabel('Global configuration fraction');axes[0].set_ylabel('RMS due to label noise alone');axes[0].legend(fontsize=9)
axes[1].set_xlabel('Exact conditional global variance');axes[1].set_ylabel('Monte Carlo variance, 2,000 draws')
fig.suptitle('Explanatory fixed-design audit: same ridge, independent noise; no primary-model selection',fontsize=12)
fig.savefig(OUT/'08_noise_amplification.png',dpi=180,bbox_inches='tight');fig.savefig(OUT/'08_noise_amplification.svg',bbox_inches='tight');plt.close(fig)
print(json.dumps(rows,indent=2))
