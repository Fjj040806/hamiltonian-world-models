# Convex force learning on the torus: what the constraints do and do not control

This note specializes classical Fourier/gradient facts to the implemented benchmark. It is not a claim of a new Hodge theorem, new Hamiltonian learner or guaranteed RL improvement.

## 1. Shared model space and exact structural quantities

Let the configuration space be the two-torus with normalized measure, and choose one representative of each nonzero frequency pair:

\[
\mathcal K=\{(a,b): a=1,2,\ b=-2,\ldots,2\}\cup\{(0,1),(0,2)\}.
\]

There are 12 frequencies. Every force model uses the same 25 scalar spatial features

\[
\phi(q)=(1,\{\cos(k\cdot q)\}_{k\in\mathcal K},
\{\sin(k\cdot q)\}_{k\in\mathcal K}),\qquad
F_C(q)=c+\sum_k[A_k\cos(k\cdot q)+B_k\sin(k\cdot q)].
\]

Here c, A_k and B_k are two-vectors. Orthogonality gives exact, quadrature-free formulas:

\[
\|F_C\|_2^2=\|c\|^2+\tfrac12\sum_k(\|A_k\|^2+\|B_k\|^2),
\]
\[
\|\operatorname{curl}F_C\|_2^2
=\tfrac12\sum_k[(k_1A_{k,2}-k_2A_{k,1})^2
+(k_1B_{k,2}-k_2B_{k,1})^2],\qquad \overline F_C=c.
\]

The first nonzero term in the curl itself comes with a sine and a minus sign; squaring removes it. The constant coefficient never enters curl. Mean-force penalization and curl penalization therefore target distinct components. For curl-bearing fields, individual fundamental-loop integrals may depend on the fixed transverse coordinate; a small mean alone does not make every loop integral small.

The force space has 50 free coefficients. The exactly curl-free subspace has dimension 26, including two constant bias components. The curl-free, zero-mean subspace has dimension 24. Thus matched spatial bandwidth does not imply matched effective degrees of freedom. Capacity reduction is part of this structural intervention and must not be disguised as an equal-parameter deep-network comparison.

## 2. Hamiltonian head and constrained-force equivalence

Parameterize a periodic potential, omitting its unidentifiable additive constant:

\[
V_\theta(q)=\sum_k[a_k\cos(k\cdot q)+b_k\sin(k\cdot q)].
\]

Then its force coefficients satisfy

\[
c=0,\qquad A_k=-b_k k,\qquad B_k=a_k k.
\]

Conversely, curl-free zero-mean force coefficients are parallel to each k and admit precisely this form. Let v collect force coefficients, and let T map the 24 potential coefficients into v. Its range equals the joint nullspace of the curl and mean constraints. For any identical data loss and force-norm ridge, minimizing over v in that nullspace and minimizing over theta with v=T theta are the same optimization problem. They induce the same unique force, regardless of the chosen nullspace basis. The code independently solves both formulations and measures their coefficient discrepancy.

This is a classical finite-dimensional equivalence. An exactly constrained force model cannot constitute an independent superior architecture against this Hamiltonian head under identical features, data and objective. Soft penalties differ because they do not impose the exact subspace. This benchmark uses a fixed periodic feature map with a learned linear head, not a general trainable-feature deep HNN.

## 3. Convex objective and reproducible optimum

For noisy force labels y_i at low-energy states q_i, fit

\[
\mathcal L(C)=\frac1N\sum_i\|F_C(q_i)-y_i\|^2
+\lambda\|F_C\|_2^2
+\mu_c\|\operatorname{curl}F_C\|_2^2
+\mu_m\|\overline F_C\|^2.
\]

Force, mean-only, curl-only and curl-plus-mean differ only in which structural penalties are active. The main experiment freezes lambda=0.001 and each active mu=1 before fitting. The Hamiltonian model uses the exact subspace and the same force-norm ridge, so no potential-amplitude ridge mismatch is introduced.

Let G be the empirical feature Gram matrix, R the positive-definite population force Gram matrix, and Q_c,Q_m the positive-semidefinite constraint matrices. The normal system is

\[
A v=b,\quad A=I_2\otimes G+\lambda R+\mu_c Q_c+\mu_m Q_m.
\]

Because R has eigenvalues 1 and 1/2, A is positive definite for lambda>0. The objective has a unique global optimum. In the potential coordinates, T has full column rank and T^T A T is positive definite. Numerical normal-equation residuals and independent constrained solves check the implementation. This eliminates nonconvex **world-model** optimization as an explanation within this benchmark; SAC optimization remains nonconvex.

## 4. Deliberate misspecification and its lower bound

Both true plants are Hamiltonian with known unit kinetic energy. The canonical potential is

\[
V_0=1-\cos q_1+0.8(1-\cos q_2)+0.3(1-\cos(q_1-q_2)).
\]

The control plant adds epsilon[1-cos(3q1-q2)] with epsilon=0.06. Frequency k*=(3,-1) is absent from the shared feature set. Its force contribution is -epsilon k* sin(k*·q), orthogonal to every learned force mode. Therefore every model in this feature space has uniform-torus force RMS at least

\[
\frac{|\varepsilon|\,\|k^*\|}{\sqrt2}=0.06\sqrt5\approx0.13416.
\]

The bound is an approximation floor, not the expected fitted error from local noisy observations. Exact Hamiltonian structure does not restore missing frequencies or identify their amplitude. Even the canonical, correctly specified case suffers noise amplification and regularization bias under limited configuration coverage. The separate posthoc sensitivity diagnostic exposes that dependence without selecting new main models.

## 5. Controlled energy and winding

For a fitted force error decomposed as

\[
F_C-F_*=-\nabla\delta V+c+r,
\]

with periodic delta V and residual r, the true Hamiltonian along fitted controlled dynamics obeys

\[
\Delta H_*-\int_0^t p\cdot u\,ds
=c\cdot\Delta q_{\rm lift}-\Delta\delta V+\int_0^t p\cdot r(q)\,ds.
\]

The potential contribution is bounded by its oscillation. The harmonic contribution accumulates with winding. Residual curl-bearing components contribute trajectory-dependent work. Small L2 curl/mean penalties bound distance from a gradient class as in Phase 4, but do not supply a uniform force-error bound, identify the true potential, or rank policies. Conservation of the **learned** Hamiltonian is distinct from preservation of true energy.

## 6. Fixed feedback isolates another confound

The diagnostic law is

\[
u(q,p)=\operatorname{clip}[-F_C(q)-K(p-p^*),-2,2],\qquad K=2I.
\]

When clipping is inactive, define e=p-p*. In the true plant,

\[
\dot e=-K e+[F_*(q)-F_C(q)].
\]

If the force error is uniformly bounded by eta, then the standard variation-of-constants estimate gives

\[
\|e(t)\|\le e^{-2t}\|e(0)\|+\tfrac\eta2(1-e^{-2t}).
\]

This is a conditional feedback estimate; it is not valid through arbitrary actuator saturation without further analysis. The implementation retains torque saturation and a held action for each 0.05 interval, so the continuous-time cancellation argument is explanatory rather than an exact guarantee for its trajectories. With privileged true force, cancellation is accurate enough to demonstrate practical task feasibility. With fitted force, this fixed controller helps separate model transfer from additional SAC learning difficulties. It is not a substitute for a controlled algorithm comparison or a new stability theorem.
