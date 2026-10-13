# Periodic tracking response and Gaussian acquisition: a classical specialization

This note derives the implemented task metric and acquisition identity. It supplies a project-specific calculation, not a new general optimal-design theorem, Hamiltonian learning algorithm or RL performance guarantee.

## 1. A first-order tracking model

For continuous, unsaturated force-cancelling feedback with gain kappa>0,

\[
u(q,p)=-F_0(q)-\kappa(p-p^*),
\]

let the true force be F0+epsilon deltaF. With e=p-p*,

\[
\dot q=p^*+e,\qquad \dot e=-\kappa e+\varepsilon\delta F(q).
\]

At epsilon=0 the nominal flow is q_nom=q0+p* t. Differentiating at zero perturbation gives

\[
\dot e_1+\kappa e_1=\delta F(q_0+p^*t).
\]

The phase perturbation enters deltaF at second order in epsilon. This expansion is local in perturbation amplitude and time; it is not uniformly valid for arbitrarily long phase drift, actuator saturation or large model error.

Write a periodic force error as

\[
\delta F(q)=c+\sum_{k\in\mathcal K}[A_k\cos(k\cdot q)+B_k\sin(k\cdot q)]
\]

and define omega_k=k·p*. The bounded steady first-order response has constant c/kappa and mode coefficients

\[
C_k=\frac{\kappa A_k-\omega_k B_k}{\kappa^2+\omega_k^2},
\qquad D_k=\frac{\omega_k A_k+\kappa B_k}{\kappa^2+\omega_k^2}.
\]

Substitution verifies the linear forced ODE. Averaging over uniform initial phase on the two-torus eliminates cross-frequency terms and yields

\[
\mathbb E_{q_0}\|e_1(t)\|^2
=\frac{\|c\|^2}{\kappa^2}
+\frac12\sum_k\frac{\|A_k\|^2+\|B_k\|^2}{\kappa^2+(k\cdot p^*)^2}.
\]

Thus a force error that varies slowly along the requested winding velocity can affect tracking more strongly than a rapidly varying error with the same global force norm. Resonant modes k·p*=0 attain the largest gain 1/kappa. This is the standard first-order stable-filter response expressed in periodic coordinates.

The metric addresses steady momentum tracking. It does not optimize full return, action-effort sensitivity, start-up transients, reachability cost or an optimal RL policy. The actual test controller holds clipped torque for 0.05 intervals and uses RK4 plant evolution. These approximations, and finite rather than exactly uniform phase samples, can invalidate quantitative equality with the continuous first-order expression.

## 2. Structure and the force metric

Use the same 12 frequency representatives as Phase 6. In full force coefficients v, the normalized torus metric is

\[
R=I_2\otimes\operatorname{diag}(1,1/2,\ldots,1/2).
\]

A class-specific map T enforces either no restriction (50 dimensions), odd force (24), or an exact periodic potential gradient (24). Set

\[
L=T(T^\top RT)^{-1/2},\qquad v=L\beta.
\]

Then the integrated squared force norm is ||beta||^2. If Phi(q) is the 2-by-50 force feature block, define B(q)=Phi(q)L. All models share spatial bandwidth; odd-force and Hamiltonian classes have the same dimension but distinct subspaces.

Let Q(p*) be diagonal in full coefficients, with DC weight 1/kappa^2 and cosine/sine weights

\[
\frac{1}{2[\kappa^2+(k\cdot p^*)^2]}.
\]

The task tracking metric in whitened coefficients is W=L^T Q L. An omitted true harmonic contributes an additional first-order error not represented by this matrix. For the missing k*=(3,-1) force with potential amplitude epsilon=0.06, that contribution is

\[
\frac{5\varepsilon^2}{\kappa^2+(k^*\cdot p^*)^2}.
\]

Its square root is approximately 0.02121 for target (2.5,1.5) and 0.04743 for target (1.5,2.5), at gain two. These are first-order response contributions, **not** exact lower bounds on nonlinear finite-horizon tracking error. The global force approximation floor remains exact by Fourier orthogonality.

## 3. Posterior and exact greedy gains

Assume componentwise measurement noise variance sigma^2 and a Gaussian prior

\[
\beta\sim\mathcal N(0,\sigma^2\rho^{-1}I).
\]

For observed force labels y_i,

\[
A=\rho I+\sum_iB_i^\top B_i,\qquad
b=\sum_iB_i^\top y_i,\qquad
\widehat\beta=A^{-1}b,\quad \Sigma=\sigma^2 A^{-1}.
\]

The fixed rho=0.05 applies throughout the acquisition sequence. In an averaged-loss convention, its corresponding force-norm ridge is rho/N; this changing averaged coefficient is a consequence of keeping the prior fixed, not a hyperparameter search. Per-coordinate prior precision is equal across classes; total prior variance differs with their dimensions.

The global acquisition objective is tr(Sigma). Task-weighted objectives are tr(W Sigma). For a possible two-component force observation with feature block B, Woodbury gives

\[
\Sigma-\Sigma_{\rm new}
=\sigma^2 A^{-1}B^\top(I+BA^{-1}B^\top)^{-1}BA^{-1}.
\]

The greedy task gain is the trace of this reduction weighted by W. Only a two-by-two matrix must be inverted. Multiplying W by a positive scalar does not change the maximizing candidate. The implementation normalizes the task and occupancy matrices for numerical comparison and verifies scores against direct inversions.

This is weighted A-optimal design in force-whitened coordinates, also interpretable as integrated predictive-variance design. Its general principle is already explicit in prior task-optimal exploration work. Covariance is Bayesian within the specified feature class; it does not include misspecification or all regularization bias and is not automatically calibrated to actual control loss.

## 4. Comparators and information access

Random acquisition ignores covariance. Pointwise uncertainty maximizes tr(B Sigma B^T). Global A-design minimizes integrated force posterior variance. Occupancy A-design uses the second moment of force features along prescribed desired flows from eight pilot phases. Spectral A-design uses the response matrix W above, averaged over uniform initial phase.

The three task-independent acquisition paths are shared across both targets. Task-aware paths are separate. All criteria use unlabelled candidate configurations and known prior/task information; true force is evaluated only after a query is selected. The candidate pool assumes unrestricted generative force queries, not a physically reachable exploration trajectory. The analytic criteria update with geometry, not observed label values. No optimal policy posterior is computed, and BARL or DYNAMICOED are not replicated.

## 5. Verification and decision limits

Independent direct-inverse checks, torus Parseval checks and filter-ODE residuals test the calculations. A small-perturbation nonlinear phase-evolution check halves epsilon and verifies the expected first-order convergence. Actual paired true-plant tracking, not posterior variance alone, determines the predeclared practical gate.

The gate compares the fixed candidate with the best of four baselines in each Hamiltonian plant/task cell at 64 additional queries. At least ten-percent mean improvement and four of five paired seed improvements are required in every cell, together with bounded degradation at other budgets. This is a practical descriptive screen with dependent paired phases, not a significance test or theorem of sample optimality. A failed gate stops this algorithm route without additional tuned variants.

## 6. Posthoc symmetry explanation: uniform approximate design already optimizes the invariant metric

The following is a specialization of classical invariant-design symmetrization, added after the practical gate was evaluated. It changes neither the acquisition rules nor the decision. It applies to the full force and Hamiltonian spaces, not the sine-only odd-force space.

Let xi be an approximate probability-measure design on the whole torus and

\[
M(\xi)=\int B(q)^\top B(q)\,d\xi(q).
\]

For either translation-closed space, shifting q by s rotates the paired sine/cosine parameters by an orthogonal O_s. Hence B(q+s)=B(q)O_s. The spectral W is invariant under that rotation because each mode's sine/cosine pair has the same tracking weight:

\[
O_s^\top W O_s=W.
\]

Assume scalar prior information rho I and no fixed nonuniform pilot information. For n additional expected measurements define

\[
\Psi(M)=\operatorname{tr}[W(\rho I+nM)^{-1}].
\]

The inverse-trace functional is convex on positive-definite information matrices. It is also invariant under M->O_s^T M O_s. Averaging a design over all torus translations gives Haar-uniform configuration measure. Force whitening implies M(Haar)=I. Jensen's inequality therefore gives

\[
\Psi(I)
=\Psi\left(\int O_s^\top M(\xi)O_s\,ds\right)
\le\int\Psi(O_s^\top M(\xi)O_s)\,ds
=\Psi(M(\xi)).
\]

Thus uniform configuration measure is an optimal **approximate design** for every such translation-invariant spectral tracking metric in this idealized setup, even though the task changes the relative weights of frequencies. A frequency preference alone does not necessarily create preferred query locations.

This does not prove that a finite random pool or greedy uniform sampling is exactly optimal, that optimal designs are unique, or that random queries win the primary experiment. The actual local pilot information is nonuniform and fixed, so it breaks a key assumption. Sine-only models also fail translation closure. Misspecification, nonuniform task phases, location-dependent observation noise, restricted reachability and finite budgets are further exceptions. The primary test includes several of these departures and must be assessed empirically.

`phase7_symmetry_check.py` verifies feature equivariance, metric commutation, the Haar information matrix and the inequality against one nonuniform design, explicitly **excluding local pilot information**. These numerical illustrations do not extend the theorem's scope. The argument is classical design theory; see Pukelsheim, *Optimal Design of Experiments*, invariant-design chapters, and Idais and Schwabe (2021), https://link.springer.com/article/10.1007/s42519-021-00221-z . It is not claimed as a new general theorem.
