\clearpage

# Mathematical appendix: detailed derivations {.unnumbered}

The following calculations expand the analysis behind the experiments. All norms on the torus use normalized Haar measure. Statements about exact continuous dynamics must be separated from statements about the integrators and held controls. Established identities are not claimed as new general results.

## A. Canonical structure, work balance and non-identifiability {.unnumbered}

For $z=(q,p)$ and $J=\begin{pmatrix}0&I\\-I&0\end{pmatrix}$, a controlled Hamiltonian vector field is

$$\dot z=J\nabla H(z)+G u,\qquad G=\begin{pmatrix}0\\I\end{pmatrix}.$$

Since $J^\top=-J$,

$$\frac{dH}{dt}=\nabla H^\top J\nabla H+\nabla H^\top Gu=p^\top u,$$

$$H(z(T))-H(z(0))=\int_0^T p(t)^\top u(t)\,dt.$$

The experiment therefore measures a work-balance residual, not raw energy drift during actuation. For a learned force $\widehat F$ used without torque, the true-energy derivative is instead

$$\frac{dH(q,\widehat p)}{dt}=\widehat p^\top[\widehat F(q)-F(q)].$$

This term can be nonzero even if the learned Hamiltonian is conserved exactly. To obtain a conditional quantitative bound, assume $\|\widehat F-F\|\le\delta$ and $\|\widehat p\|\le P$ along the entire learned trajectory. Then

$$|H(\widehat z(T))-H(\widehat z(0))|\le TP\delta.$$

A small training-domain force error does not establish either bound along an unseen trajectory.

For explicit non-identifiability, let $U$ contain all observed configurations and choose a smooth bump $b$ supported in an open region disjoint from $U$. Define

$$\widetilde V(q)=V(q)+a b(q),\qquad \widetilde F(q)=F(q)-a\nabla b(q).$$

On $U$, all force labels and all derivatives of the perturbation vanish. Both vector fields are Hamiltonian, yet outside $U$ their difference can be scaled by arbitrary finite $a$. This construction assumes an unrestricted smooth potential class; a fixed finite analytic feature class may instead be identifiable from sufficiently informative samples. Poor conditioning can nevertheless make finite noisy extrapolation inaccurate. The smooth bump argument must not be misrepresented as exact finite-Fourier non-identifiability.

## B. Fourier curl, periods and gradient-force equivalence {.unnumbered}

Write $F=c+\sum_k(A_k\cos k^\top q+B_k\sin k^\top q)$ and let $k^\perp=(-k_2,k_1)$. Then

$$\partial_{q_1}F_2-\partial_{q_2}F_1
=\sum_k[-(k^\perp\!\cdot A_k)\sin(k^\top q)+(k^\perp\!\cdot B_k)\cos(k^\top q)].$$

Orthogonality implies curl vanishes exactly when $A_k$ and $B_k$ are parallel to $k$ for every mode. This does not remove $c$. For each coordinate-cycle at a fixed transverse coordinate, all nonconstant mode contributions integrate to zero once the curl-free parallel condition holds, and the period is

$$\oint_{\gamma_j}F^\top dq=2\pi c_j.$$

Consequently, a finite periodic curl-free force admits a periodic scalar potential precisely when $c=0$. If $A_k=a_k k$ and $B_k=b_k k$, one potential satisfying $F=-\nabla V$ is

$$V(q)=\sum_k[b_k\cos(k^\top q)-a_k\sin(k^\top q)]+C.$$

Thus a Hamiltonian feature head is exactly a constrained-force regressor under matched objectives. Parameterizing the gradient through a scalar does not introduce a different statistical estimator if the constrained subspace, regularizer and labels are the same. The project verifies this equivalence numerically; differences from unconstrained fitting arise from the hypothesis class and objective.

Parseval gives

$$\|F\|_2^2=\|c\|^2+\frac12\sum_k(\|A_k\|^2+\|B_k\|^2),$$

$$\|\operatorname{curl}F\|_2^2=\frac12\sum_k[(k^\perp\!\cdot A_k)^2+(k^\perp\!\cdot B_k)^2].$$

These exact population quadratic forms generate the force-norm and curl penalties; numerical quadrature is unnecessary for those penalties.

## C. Translation and the parity projection floor {.unnumbered}

For $\theta_k=k^\top s$, translation $q\mapsto q-s$ rotates coefficients:

$$\begin{pmatrix}A'_k\\B'_k\end{pmatrix}=
\begin{pmatrix}\cos\theta_k&-\sin\theta_k\\\sin\theta_k&\cos\theta_k\end{pmatrix}
\begin{pmatrix}A_k\\B_k\end{pmatrix}.$$

The rotation acts separately on each force component and preserves Parseval norm. Parallel coefficients remain parallel, hence Hamiltonian closure. Sine-only force is not closed under this action.

A true included force mode $-a k\sin(k^\top(q-s))$ expands as

$$-a k\cos\theta_k\sin(k^\top q)+a k\sin\theta_k\cos(k^\top q).$$

Orthogonal projection onto the zero-centered odd space drops the second term. Its squared error is $a^2\|k\|^2\sin^2\theta_k/2$. Summing included modes and the orthogonal missing frequency yields the approximation-floor expression in the main text. This is a population-class minimum. Empirical test estimates may fluctuate below it, and ridge fits from local data need not approach it.

## D. Exact fixed-design bias-variance decomposition {.unnumbered}

Stack componentwise force observations as $y=X\theta_*+r+\epsilon$, where $r$ is a deterministic class-misspecification residual at the observed locations. Let $X$ denote the subspace design, $D=X^\top X/N$, regularization $\Lambda$ and $A=D+\Lambda$. The fitted coefficients are

$$\widehat\theta=A^{-1}X^\top y/N,$$

$$\overline\theta:=\mathbb E_\epsilon\widehat\theta
=\theta_*-A^{-1}\Lambda\theta_*+A^{-1}X^\top r/N.$$

The first bias term is shrinkage; the second is observed-design aliasing of omitted dynamics. Orthogonality under a global population measure does not force $X^\top r=0$ under a local or finite design.

Let $R$ be the test-domain coefficient Gram matrix and $r_{\rm test}$ the residual of the chosen reference representation. The total expected test error decomposes as

$$\mathbb E_\epsilon\|\widehat F-F_*\|_{\rm test}^2
=\|F_{\overline\theta}-F_*\|_{\rm test}^2
+\frac{\sigma^2}{N}\operatorname{tr}(RA^{-1}DA^{-1}).$$

The cross term vanishes because the centered estimator has zero mean. This formula accommodates approximation and design aliasing through the first term; the posterior covariance $\sigma^2A^{-1}$ is generally different from frequentist noise covariance $\sigma^2A^{-1}DA^{-1}/N$ in the averaged-loss convention. The two expressions use different assumptions and precision scaling and must not be interchanged.

For a correctly specified, unregularized full-rank design, $A=D$ and variance is $\sigma^2\operatorname{tr}(RD^{-1})/N$. If $D$ is close to $R$, the leading term is $\sigma^2 d/N$. With fixed ridge and population whitening, diagonalizing $D$ gives the sum in the main text. With validation-selected ridge, conditioning on that selected value does not generally restore an unconditional fixed-ridge identity because selection depends on noise.

## E. Full periodic filter calculation and finite-time correction {.unnumbered}

For one vector-valued mode $f(t)=A\cos(\theta+\omega t)+B\sin(\theta+\omega t)$, seek $e_s=C\cos(\theta+\omega t)+D\sin(\theta+\omega t)$. Matching coefficients in $\dot e_s+\kappa e_s=f$ gives

$$\kappa C+\omega D=A,\qquad -\omega C+\kappa D=B,$$

$$C=\frac{\kappa A-\omega B}{\kappa^2+\omega^2},\qquad
D=\frac{\omega A+\kappa B}{\kappa^2+\omega^2}.$$

The cross terms cancel in the response norm:

$$\|C\|^2+\|D\|^2=\frac{\|A\|^2+\|B\|^2}{\kappa^2+\omega^2}.$$

Averaging over uniform torus phase makes different frequency representatives orthogonal and gives the steady formula. If $e(0)=0$, the finite-time solution is

$$e_1(t)=e_s(q_0+p_*t)-e^{-\kappa t}e_s(q_0).$$

This includes the DC contribution $c(1-e^{-\kappa t})/\kappa$. The steady expression is therefore not exactly the finite-time zero-initial-error metric. For each mode, uniform-phase squared error becomes

$$\frac{\|A_k\|^2+\|B_k\|^2}{2(\kappa^2+\omega_k^2)}
[1+e^{-2\kappa t}-2e^{-\kappa t}\cos(\omega_k t)].$$

The DC squared term is $\|c\|^2(1-e^{-\kappa t})^2/\kappa^2$. On the final test window with $\kappa=2$, this linear-model transient is negligible, but nonlinear phase drift, saturation and held actions are not thereby negligible. At resonance $\omega_k=0$, the formula remains well-defined; it concerns an ensemble of initial phases, not temporal decorrelation along one resonant trajectory.

## F. Weighted design, Woodbury and invariant information {.unnumbered}

Let $L=T(T^\top RT)^{-1/2}$ and $B=\Phi L$. Since $\int\Phi^\top\Phi\,dq=R$,

$$\int B(q)^\top B(q)\,dq=L^\top RL=I.$$

The scalar Gaussian prior has precision $\rho I/\sigma^2$. A two-component observation adds $B^\top B/\sigma^2$. Setting $A=\rho I+\sum B_i^\top B_i$, Woodbury gives

$$A_{\rm new}^{-1}=A^{-1}-A^{-1}B^\top(I_2+BA^{-1}B^\top)^{-1}BA^{-1}.$$

For a positive-semidefinite task metric $W$, the exact candidate gain is

$$g_W(q)=\sigma^2\operatorname{tr}\left[WA^{-1}B^\top(I_2+BA^{-1}B^\top)^{-1}BA^{-1}\right]\ge0.$$

Nonnegativity follows because the covariance reduction is positive semidefinite. A positive scalar rescaling of $W$ changes all gains by that scalar and cannot change the selected maximizer. These gains depend on locations and covariance, not observed force values, for this fixed Gaussian linear model.

For completeness, convexity of $\Psi(M)=\operatorname{tr}[W(\rho I+nM)^{-1}]$ can be checked along a symmetric direction $H$. With $S=(\rho I+nM)^{-1}$,

$$\frac{d^2}{dt^2}\Psi(M+tH)\bigg|_{t=0}
=2n^2\operatorname{tr}(WSHSHS)\ge0.$$

Indeed $SHSHS=S^{1/2}(S^{1/2}HS^{1/2})^2S^{1/2}$ is positive semidefinite. Translation invariance then permits Jensen averaging as in the main proof. A fixed pilot matrix $A_0$ that does not commute with the translations replaces $\rho I$ and invalidates the invariance step. Consequently the idealization is not a theorem that the observed random design must win.

## G. Predictive-control adjoint and SAC objectives {.unnumbered}

The MPC model uses kick-drift updates,

$$p_{t+1}=p_t+h[\widehat F(q_t)+u_t],\qquad q_{t+1}=q_t+hp_{t+1}.$$

Writing $J_t=D\widehat F(q_t)$, the state and input Jacobians are

$$A_t=\begin{pmatrix}I+h^2J_t&hI\\hJ_t&I\end{pmatrix},\qquad
B_t=\begin{pmatrix}h^2I\\hI\end{pmatrix}.$$

For an objective $C=\sum_{t=0}^{H-1}\ell(z_{t+1},u_t)+\phi(z_H)$, initialize $\lambda_H=\nabla\phi(z_H)$. A backward step has

$$g_{u_t}=\partial_u\ell+B_t^\top[\lambda_{t+1}+\nabla_z\ell],\qquad
\lambda_t=A_t^\top[\lambda_{t+1}+\nabla_z\ell].$$

Here $\ell=h(\|p_{t+1}-p_*\|^2+0.01\|u_t\|^2)$ and $\phi=0.5\|p_H-p_*\|^2$. Projected-gradient backtracking guarantees only nonincrease of this fitted-model objective. It supplies no true-plant monotonicity or global optimum.

The actual coupled SAC implementation observes

$$o(z)=(\cos q_1,\cos q_2,\sin q_1,\sin q_2,p_1/3,p_2/3),$$

and uses squashed Gaussian torque $u=2\tanh a$. Its change-of-variables density is

$$\log\pi(u\mid o)=\sum_j\left[\log\mathcal N(a_j;\mu_j,\sigma_j^2)-\log2-\log(1-\tanh^2a_j)\right],$$

with a small numerical logarithm safeguard in code. Twin critics use target

$$y=r+\gamma[\min_i\overline Q_i(o',u')-\alpha\log\pi(u'\mid o')],$$

$$\mathcal L_Q=\sum_{i=1}^2\mathbb E(Q_i(o,u)-y)^2,\qquad
\mathcal L_\pi=\mathbb E[\alpha\log\pi(u\mid o)-\min_iQ_i(o,u)].$$

In the coupled implementation, $\gamma=0.99$, fixed $\alpha=0.01$, Adam learning rate $3\times10^{-4}$ and target interpolation rate 0.005. Reward integrates $-\|p-p_*\|^2-0.01\|u\|^2$ by substep trapezoidal quadrature. Replay training uses a continuing-task target without a terminal multiplier; finite evaluation episodes and recorded training resets are therefore distinct from a fully specified episodic Bellman model. These details limit interpretation of finite-budget policy rankings. The formulas document the implementation; they are standard SAC objectives, not a new RL method.

## H. Detailed action-hold consistency and missing-mode tracking {.unnumbered}

At a grid state $(q_n,p_n)$ with $e_n=p_n-p_*$, exact-force cancellation yields $\dot p(t_n)=-\kappa e_n$ while the held torque has zero derivative inside the interval. Hence

$$p_{n+1}=p_n-\kappa h e_n+\frac{h^2}{2}DF(q_n)p_n+O(h^3).$$

A finite-horizon boundedness induction and discrete Gronwall give $\sup_n\|e_n\|\le C_T h$. Integrating $\dot q=p_*+e$ gives $q_n=q_0+p_*t_n+O(h)$. Set $r_n=e_n/h$ and use bounded second derivatives to replace $DF(q_n)p_n$:

$$r_{n+1}=r_n+h\left[-\kappa r_n+\frac12DF(q_0+p_*t_n)p_*\right]+O(h^2).$$

Comparing to forward Euler for the response ODE proves $r_n=\eta(t_n)+O(h)$. For $C^2$ periodic forces on a compact torus and finite $T$, the derivative assumptions hold; an unbounded configuration-space system requires a separate bounded-trajectory neighborhood. Plant integration error must be controlled separately. The diagnostic fixes RK4 substeps at 0.0025 while varying action-hold interval, so control sampling is much coarser than integration.

For the periodic missing potential $\varepsilon[1-\cos(k_*^\top q)]$, its force amplitude is $-\varepsilon k_*\sin(k_*^\top q)$. The ideal first-order squared tracking contribution is

$$\frac{\varepsilon^2\|k_*\|^2}{2[\kappa^2+(k_*^\top p_*)^2]}.$$

With $k_*=(3,-1)$ and the two targets, $\omega_*=6$ or 2; at $\kappa=2$ and $\varepsilon=0.06$, RMS contributions are 0.021213 and 0.047434. These are ideal linear response values, not exact nonlinear tracking lower bounds. The Fourier force approximation floor, by contrast, is an exact class-level statement.
