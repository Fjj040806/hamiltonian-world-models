# Coverage, coordinate-dependent priors and force-estimation variance

This note specializes standard Fourier regression, parity projection and discrete adjoint calculations to the implemented experiment. These formulas and their numerical audits are not asserted to be new mathematical theorems or guarantees of RL superiority.

## 1. Model spaces

The shared periodic force representation is

\[
F(q)=c+\sum_{k\in\mathcal K}[A_k\cos(k\cdot q)+B_k\sin(k\cdot q)],
\quad \mathcal K=\{(a,b):a=1,2;\ b=-2,\ldots,2\}\cup\{(0,1),(0,2)\}.
\]

There are 12 frequencies, 25 scalar features and 50 unconstrained force coefficients. Every model knows qdot=p and the unit torque port. Exact constraints define these subspaces:

* Odd force: c=A_k=0, with B_k arbitrary; dimension 24.
* Periodic Hamiltonian force: c=0, A_k and B_k parallel to k; dimension 24.
* Odd Hamiltonian force: c=A_k=0 and B_k parallel to k; dimension 12.

The odd-force and Hamiltonian spaces have equal dimensions but different hypothesis classes. Comparing them removes a simple coefficient-count difference, not every difference in capacity or inductive bias. Soft curl/mean models retain 50 free coefficients with additional quadratic regularization; their statistical effective degrees of freedom need not equal 50.

## 2. Translation invariance and parity

For a translated potential V_s(q)=V_0(q-s), the force is F_s(q)=F_0(q-s). At each frequency, translation rotates the coefficient pair (A_k,B_k) by the phase k·s. The full sine/cosine force space is closed under that rotation. Parallel-to-k coefficients remain parallel to k, so the Hamiltonian subspace is also closed under translation. Population force norm and squared curl are invariant because the phase rotation is orthogonal. Constant mean force is unchanged.

By contrast, a sine-only force develops cosine coefficients when translated. Oddness about the original origin is generally lost. Oddness can be restored by recentering at the correct symmetry origin, if that origin is known or learned. Thus the shifted condition is a coordinate-origin probe of the same physical potential family, not an independent physical-system replication or an intrinsic failure of correctly centered parity methods.

In the local-only paired experiment, translated training and validation configurations have identical forces and noises after recentering. Full force, curl/mean and Hamiltonian objectives therefore transform equivariantly. Their fitted functions should agree under translation, up to numerical error. The final verifier checks this fact independently of the aggregate test scores.

## 3. Approximation floors with a missing mode and an incorrect prior

The true potential is

\[
V(q)=\sum_{j\in\{1,2,d\}}a_j[1-\cos(k_j\cdot(q-s))]
+\varepsilon[1-\cos(k_*\cdot(q-s))],
\]

with k1=(1,0), a1=1; k2=(0,1), a2=0.8; kd=(1,-1), ad=0.3; and k*=(3,-1), epsilon=0.06 in the misspecified cases. The first three frequencies lie in the fitted bandwidth, while k* does not. Normalized torus orthogonality gives

\[
\inf_F\|F-F_*\|_{L^2}^2
=\tfrac12\varepsilon^2\|k_*\|^2=5\varepsilon^2
\]

for the unconstrained full-feature and exact Hamiltonian spaces. Both can represent the included part of the true force exactly. Their common force RMS floor is 0.1341641 when epsilon=0.06.

For an odd force about the original origin, every included cosine component must also be discarded. Its force RMS floor squared is

\[
5\varepsilon^2+
\tfrac12\sum_{j\in\{1,2,d\}}a_j^2\|k_j\|^2\sin^2(k_j\cdot s).
\]

The same floor holds for odd Hamiltonian features here: the retained true sine coefficients are already parallel to their frequencies. At s=(0.45,-0.35), this floor is approximately 0.4435. It includes the entire missing-mode energy, so the omitted mode is not double counted as a separate parity error.

These are approximation-class lower bounds. They do not predict the finite-sample optimum from local observations, the selected ridge, a closed-loop policy return, or the cost of identifying an unknown symmetry origin.

## 4. Conditional variance under fixed design and fixed ridge

Collect force coefficients in v and write v=T theta for a chosen exact subspace. Let

\[
D=I_2\otimes\left(\frac1N\sum_i\phi(q_i)\phi(q_i)^\top\right),
\quad R=I_2\otimes\operatorname{diag}(1,1/2,\ldots,1/2).
\]

With fixed positions q_i, independent componentwise label noise of variance sigma^2 and fixed regularizers, define

\[
D_T=T^\top DT,\quad R_T=T^\top RT,\quad
A_T=T^\top(D+\lambda R+\mu_c Q_c+\mu_m Q_m)T.
\]

The noise of the empirical right-hand side has covariance sigma^2 D_T/N. Linearity of the normal-equation solution gives

\[
\operatorname{Cov}(\widehat\theta\mid q)
=\frac{\sigma^2}{N}A_T^{-1}D_T A_T^{-1}.
\]

Integrating the variance of the fitted force over the normalized torus yields the exact identity

\[
\mathbb E_\epsilon\|F_{\widehat\theta}-\mathbb E_\epsilon F_{\widehat\theta}\|_2^2
=\frac{\sigma^2}{N}\operatorname{tr}
\left(R_T A_T^{-1}D_T A_T^{-1}\right).
\]

This is **noise variance**, not total prediction error. Regularization bias, approximation error and extrapolation bias remain. For exact-subspace models without additional soft penalties, whiten the population force metric. If d_j are the eigenvalues of R_T^(-1/2) D_T R_T^(-1/2), then

\[
\mathcal V=\frac{\sigma^2}{N}\sum_j\frac{d_j}{(d_j+\lambda)^2}.
\]

Under broad uniform coverage, the normalized design approaches identity, and the variance is roughly sigma^2 times the subspace dimension divided by N. Under local coverage, small eigenvalues amplify noise. Restricting to a valid gradient-force subspace changes both dimension and the restricted design spectrum; a same-dimension odd-force subspace need not have the same variance. Additional data coverage can raise the small eigenvalues without increasing the sample count.

The separate explanatory audit fixes ridge at 1e-6, world seed 10 and the misspecified plant. It verifies nine cases with 2,000 independent Gaussian sufficient-statistic perturbations each. Sampling the empirical right-hand-side covariance is distributionally equivalent to perturbing every force label independently. Primary validation-selected models are untouched. The identity does not directly give the unconditional risk of a model whose ridge itself was selected from noisy validation data; that selection can depend on both training and validation noise.

## 5. Discrete predictive control and its adjoint

The MPC surrogate uses kick-drift symplectic Euler with step h=0.05:

\[
p_{t+1}=p_t+h[F(q_t)+u_t],\qquad q_{t+1}=q_t+h p_{t+1}.
\]

Let J_t=DF(q_t). The transition Jacobian and control Jacobian are

\[
A_t=\begin{pmatrix}I+h^2J_t&hI\\hJ_t&I\end{pmatrix},
\qquad B_t=\begin{pmatrix}h^2I\\hI\end{pmatrix}.
\]

With stage cost h(||p_next-p*||^2+0.01||u||^2) and terminal cost 0.5||p_N-p*||^2, the usual discrete adjoint propagates the cost derivative backward. The control gradient is B_t^T times the accumulated next-state derivative plus 0.02h u_t. The implementation checks selected coordinates against central finite differences for every family.

Each projected-gradient candidate clips actions to [-2,2]. Per-trajectory backtracking only accepts a nonincreasing **model-surrogate** cost. Initialization chooses between a shifted previous plan and a fresh fixed-feedback rollout under that same fitted model. This does not prove a global optimization solution, stability of the receding-horizon controller, or monotonic improvement in the true plant. The physical environment uses RK4 rather than the planning integrator, so planning discretization is another explicit approximation.

## 6. What follows for the research question

The model-estimation formulas explain why valid structural restrictions can reduce noise amplification under local observations. They also explain why broad coverage shrinks that advantage and why an incorrect symmetry prior leaves an irreducible error. They do not establish an RL ranking. Strong feedback/MPC and finite-budget SAC use the same fitted forces but introduce different approximation and optimization mechanisms; those outcomes must be measured separately. A new control-relevant method or theorem remains a future contribution, rather than being supplied by these standard identities.
