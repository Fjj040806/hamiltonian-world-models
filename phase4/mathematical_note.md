# Local and global force consistency on the configuration torus

This note records classical mathematical background and the project's explicit intervention identities. It does not claim a new Hodge theorem or a proved improvement in reinforcement learning.

## 1. Mechanical models and topology

Consider canonical coordinates (q,p) on T²×R², with q_i modulo 2π and

\[
H(q,p)=\tfrac12\|p\|^2+V(q),\qquad \dot q=p,\quad \dot p=F(q)+u.
\]

A globally defined periodic potential requires F=−∇V. For a smooth periodic force, let its associated one-form be α=F₁dq₁+F₂dq₂. A necessary local condition is

\[
\partial_{q_1}F_2-\partial_{q_2}F_1=0.
\]

This means the one-form is closed. It is not sufficient on the torus. The periods around the two fundamental noncontractible cycles must also vanish. For a closed force field, these periods are independent of the other coordinate and equal 2π times the corresponding mean force components. Therefore

\[
F=-\nabla V\text{ for periodic }V
\quad\Longleftrightarrow\quad
\operatorname{curl}F=0\text{ and }\overline F=0.
\]

This statement applies to the specified natural mechanical form with qdot=p and a position-only force. A general Hamiltonian vector field with momentum-dependent kinetic energy requires the full canonical compatibility condition; checking only force curl is insufficient.

There is a further distinction from symplecticity. For the unforced vector field X=(p,F), with ω=Σdq_i∧dp_i, contraction gives i_Xω=Σp_i dp_i−ΣF_i dq_i. If the force one-form is closed, Cartan's identity gives L_Xω=d(i_Xω)=0, so the flow is symplectic. A constant nonzero mean force still meets this local condition while its force one-form is not exact on the torus. It therefore need not have a globally single-valued periodic Hamiltonian. Symplectic Jacobian checks alone do not detect this global obstruction. Known external control torque is intentionally an energy port; these constraints concern the assumed unforced conservative dynamics, not arbitrary closed-loop feedback.

## 2. Fourier proof and a quantitative L² estimate

Use normalized measure dq/(2π)² and expand a smooth real periodic force as

\[
F(q)=\sum_{k\in\mathbb Z^2}\widehat F_k e^{ik\cdot q}.
\]

For k≠0, the gradient-force projection has coefficient

\[
\widehat{P_{\mathrm{grad}}F}_k=
\frac{kk^\top}{|k|^2}\widehat F_k,
\qquad
\widehat{P_{\mathrm{grad}}F}_0=0.
\]

For example, a potential producing this projected force has coefficients

\[
\widehat V_k=i\frac{k\cdot\widehat F_k}{|k|^2},\quad k\ne0,
\]

with arbitrary additive constant. Reality follows from conjugate symmetry. Parseval and the orthogonal decomposition of each nonzero Fourier coefficient yield

\[
\|F-P_{\mathrm{grad}}F\|_{L^2}^2
=|\overline F|^2+
\sum_{k\ne0}\frac{|k_1\widehat F_{2,k}-k_2\widehat F_{1,k}|^2}{|k|^2}
\le |\overline F|^2+\|\operatorname{curl}F\|_{L^2}^2.
\]

The final inequality uses |k|≥1 for the 2π-periodic torus. It makes the two error sources explicit: a harmonic mean-force component and a non-gradient component measured by curl. This is a classical Fourier/Hodge estimate. It is an L² projection statement, not a uniform force-error, loop-trace or policy-value guarantee.

## 3. Energy drift under a controlled force decomposition

Let the perturbed simulator use

\[
\dot p=-\nabla V(q)-\nabla\delta V(q)+c+u,
\]

with periodic δV and constant c. Along the simulator trajectory,

\[
\frac{dH}{dt}=p\cdot u-p\cdot\nabla\delta V+p\cdot c.
\]

Integrating with a continuous lift of q gives the exact identity

\[
H(t)-H(0)-\int_0^t p\cdot u\,ds
=c\cdot(q(t)-q(0))-\delta V(q(t))+\delta V(q(0)).
\]

For zero c, the base-energy deviation due solely to the potential perturbation is bounded by osc(δV). For a configuration loop with winding vector n∈Z², the periodic potential term vanishes, leaving 2πc·n. Thus a small constant force error can accumulate with winding even though its curl is exactly zero. A constant bias also admits a nonperiodic linear potential on the unwrapped cover R²; the obstruction is to a globally single-valued periodic Hamiltonian on T².

## 4. Explicit matched-error experiments

The base potential is

\[
V=1-\cos q_1+0.8(1-\cos q_2)+0.3(1-\cos(q_1-q_2)).
\]

With ε=.04, compare three force errors:

\[
e_{\rm grad}=(-\varepsilon\sin(q_1-q_2),\varepsilon\sin(q_1-q_2)),
\]
\[
e_{\rm harmonic}=(\varepsilon,0),\qquad
e_{\rm curl}=(\varepsilon\sin q_2,-\varepsilon\sin q_1).
\]

All have uniform-torus Euclidean vector RMS ε. The first is −∇δV with δV=ε[1−cos(q₁−q₂)], hence osc(δV)=2ε=.08. The harmonic field has zero curl but q₁-cycle work 2πε=.2513274. The third has curl −ε(cos q₁+cos q₂), and work −8εa sin(a) around the counterclockwise square [−a,a]²; at a=.8 this is −.1836432. Its q₁ cycle at q₂=0 has zero work, showing why one cycle alone does not detect every nonconservative field.

The code verifies these identities, compares endpoints with independent DOP853 integration and records energy behavior over 60 time units. These are analytic interventions, not fitted two-dimensional world models or learned coupled-pendulum policies.

## 5. Candidate learning objective

A prospective force-model baseline could use

\[
\mathcal L(\theta)=\mathcal L_{\rm data}
+\lambda_{\rm curl}\mathbb E_q|\partial_1F_{\theta,2}-\partial_2F_{\theta,1}|^2
+\lambda_{\rm mean}|\mathbb E_qF_\theta(q)|^2.
\]

For the continuum loss on the whole torus, the previous bound controls distance from a periodic gradient-force class. Finite collocation, quadrature and optimization errors require separate analysis. A potential-network HNN already enforces exactness by construction; this proposed force baseline separates the constraints from the scalar-energy parameterization. It is not claimed to be a new method until nearest-literature comparison and actual learned-model experiments are completed.

Even perfect curl and period consistency do not identify the correct potential outside data support: the Phase 2 bump construction has zero loop error while changing the barrier. Hence global consistency and energy-landscape identification are distinct research axes.
