# Phase 0 — Hamiltonian world models for reinforcement learning

Date: 2026-10-13. Status: initial literature positioning and research protocol completed; novelty remains provisional. This is a source-backed scoping review, not an exhaustive systematic review. No learned-model or RL results have been produced.

## Decision

Proceed with a mechanism-first project: when does preserving Hamiltonian structure improve model-based RL across energy regimes, after controlling for coordinates, second-order structure, model capacity and data coverage? Start with the normalized pendulum and extend to coupled pendulums if the mechanism survives the pilot.

Working title: **When Does Hamiltonian Structure Help Reinforcement Learning? Energy-Regime Generalization Near a Separatrix.**

## Literature map and implications

| Work | Verified scope | Implication for our project | Evidence depth |
|---|---|---|---|
| Greydanus et al., Hamiltonian Neural Networks, NeurIPS 2019 [1] | Learning a scalar energy function and deriving conservative dynamics | HNN architecture and broad generalization claims are prior art | Primary abstract |
| Zhong et al., Symplectic ODE-Net, ICLR 2020 [2] | Hamiltonian learning with control and physically structured representations | Adding control to an HNN is not an original contribution | Primary abstract |
| Miller et al., Negotiating the separatrix with machine learning, 2021 [3] | Single/double pendulum prediction across libration and rotation, cylindrical coordinates | Learning pendulum separatrices alone is not new; closed-loop decision consequences need investigation | Primary article and PDF introduction |
| Gruver et al., Deconstructing the Inductive Biases of Hamiltonian Neural Networks, ICLR 2022 [4] | Their experiments attribute gains to second-order and coordinate/functional-complexity biases rather than conservation alone | Strong second-order baselines and equal periodic representations are mandatory | Primary PDF sections 4.4–5 and appendix excerpts |
| Troch et al., AC-HGN, L4DC 2025 [5] | Action-conditioned Hamiltonian world models used in MBRL | HNN plus MBRL is already established; full experimental protocol still needs reading | Primary proceedings abstract; PDF retrieval unsuccessful |
| Janner et al., MBPO, NeurIPS 2019 [6] | Short model rollouts branching from real replay data | Rollout horizon is a central controlled variable, not a fixed incidental choice | Primary abstract |
| Chen and Tao, Exactly-Symplectic Maps, ICML 2021 [7] | Learning symplectic transition maps using generating functions | Continuous Hamiltonian structure and discrete symplecticity require separate tests | Primary proceedings abstract |
| Eslami and Babazadeh, DAC-pH, 2025 preprint [8] | Proposed decomposition with RL on dissipative/input flow; formal analysis reported as ongoing | Adjacent control work; do not treat its proposed guarantees as established facts | Primary abstract and author comments |

These sources establish overlap, not absence of all competing work. Citation-chain and full-method checks on [3–5] remain required before any first-of-its-kind claim. A search result about temporal generalization (arXiv:2607.07763) could not be verified at its primary source and is retained only as a follow-up lead.

## Proposed contribution, not a novelty claim

1. A controlled benchmark separating initial-energy shift, actual replay support and model-usage horizon.
2. A mechanism study linking incorrect energy barriers or work balance to control failure, with matched prediction accuracy and targeted perturbations.
3. If justified, a structure-aware rollout stopping rule, compared against equal-compute uncertainty-based and fixed-horizon rules. This remains optional; neither its effectiveness nor novelty is established.

## Mathematical anchor

Use H(q,p)=p²/2+1−cos(q), with q on S¹, p real, qdot=p and pdot=−sin(q)+u. The uncontrolled saddle energy is E_s=2. Below 2, non-equilibrium trajectories librate; above 2 they rotate. At 2 the saddle and separatrix require separate treatment. Under control, dH/dt=pu. The uncontrolled separatrix is a reference geometry, not an invariant boundary of arbitrary feedback dynamics.

For an approximate Hamiltonian with the same reference saddle z_s, suppose |Hhat−H|≤ε at both z and z_s. Then |(Hhat(z)−Hhat(z_s))−(H(z)−H(z_s))|≤2ε by the triangle inequality. Consequently a true energy margin |H(z)−H(z_s)|>2ε preserves its sign. This elementary bound is background scaffolding, not a novel theorem, and does not establish correct orbit topology unless the learned critical-point structure is also controlled. Shifted saddles and spurious equilibria must be checked separately.

Learning derivatives leaves H identifiable only up to an additive constant; use energy differences rather than absolute energy errors. If both kinetic energy and force scaling are unknown, identify additional scale ambiguities before interpretation. Initial version assumes known canonical coordinates and known unit actuator gain; this privileged information must be identical across baselines.

## Preregistered pilot protocol (provisional numerical ranges)

Prediction data: independent uncontrolled trajectories from initial H in [0.2,1.2]. Split by whole trajectory, never neighboring time points. Validate in-distribution before held-out testing. Test bands: [1.3,1.7], [1.8,1.95], [2.05,2.2], [2.4,3.0]. Sample energy and orbit phase explicitly; report the distribution and exclude the exact separatrix from finite-period sampling.

Control data: identical bounded actions and initial-state distributions for every model. Record every visited state's true H. Do not call a replay set low-energy solely because episodes started there. Separate a fixed-model protocol (fixed dynamics dataset; policies may collect equally budgeted real transitions but the model cannot update) from online model updating. A controlled low-energy support restriction, if used, must be reported as a truncated-data condition rather than natural exploration. Prediction and control training datasets are separate tracks.

Model ladder: (A) periodic-input neural ODE; (B) second-order model qdot=p, pdot=a_theta(q,p)+u; (C) force-only model pdot=f_theta(q)+u; (D) separable HNN Hhat=p²/2+V_theta(q); (E) general HNN. All use equivalent sin(q), cos(q) representations and control knowledge where applicable. Derivatives of a periodic scalar H are obtained by the chain rule in canonical q,p, not by assuming sin(q),cos(q) are canonical variables.

Important identifiability caveat: in one position dimension, a zero-mean periodic force is the derivative of a periodic potential. Thus force-only and separable-HNN models may have almost equivalent function classes. They are useful controls, but a coupled two-degree-of-freedom system is needed to test genuinely multidimensional conservative-force constraints (including cross-derivative consistency).

Policy training: common Dyna/MBPO-style SAC loop; same real interaction and tuning budgets, replay rules, reward and action bounds. Synthetic rollout horizons 1, 5, 15 initially. The true-model reference uses the same RL loop and is a reference, not a guaranteed performance upper bound. Include model-free SAC and an energy-based controller.

Numerics: common accurate integrator first; step-size convergence checks; symplectic integration as a separate ablation. General nonseparable HNNs require an appropriate method such as implicit midpoint, not an unjustified Verlet implementation. Feedback closed-loop dynamics need not be symplectic.

## Evidence and evaluation

Report trajectory errors using circular q distance; energy-difference error; true-energy drift on uncontrolled model trajectories; learned-energy drift as a numerical check; saddle/barrier displacement and spurious critical points; work-balance residual grad(H_true)·fhat−pu. True H is permitted for simulator evaluation, not an unreported input to the proposed deployable method.

Control: return, success rate, time to goal, integrated u² effort and signed work ∫pu dt. Effort is not physical work. Five independent seed pipelines minimum for main results, paired evaluation initial conditions, uncertainty across training seeds; evaluation episodes alone are not independent training replicates.

Do not infer causation from a correlation plot. Match or adjust for prediction error, horizon and support distance; inject analytically controlled barrier perturbations; compare held-out explanatory performance of structure diagnostics versus one-step error. Perturbation figures are illustrative, not learned-model results.

## Go/no-go rules

- If HNN beats only a weak MLP but not the second-order baseline, revise the mechanism claim rather than declaring structural superiority.
- If long predictions improve but short-rollout RL does not, study horizon dependence; do not silently select only favorable horizons.
- If all models fail outside support, document identifiability/coverage limits and test fixed-budget targeted sampling against random sampling.
- Advance to method design only after a reproducible failure mechanism. Advance toward submission after multidimensional replication and complete nearest-neighbor literature review.

## Figure plan and software

Created now: exact pendulum phase portrait with explicitly proposed energy bands; analytic potential-barrier perturbation illustration. Python/NumPy/Matplotlib exports PNG and SVG. No learned trajectories or fabricated performance curves.

Next: numerics/energy-work checks (Phase 1); matched-model trajectory/error panels and energy contours (Phase 2); return versus real interactions and energy-conditioned control maps (Phase 3); perturbation and horizon ablations (Phase 4); coupled-system sections/animations (Phase 5). Use Python for scientific charts, Plotly for a later interactive demonstration. ParaView is unnecessary for the initial low-dimensional system.

## Remaining distance to goals

Completed: initial positioning, falsifiable question, mathematical starting point, strong baseline ladder, provisional experiment plan and two reproducible concept figures. Remaining: complete primary-method comparison, validated simulator, datasets, model fitting, RL, mechanistic evidence, multidimensional extension, theory beyond the elementary margin bound, manuscript and runnable repository. No numerical percentage is assigned without results.

## References

[1] https://arxiv.org/abs/1906.01563

[2] https://arxiv.org/abs/1909.12077

[3] https://doi.org/10.1587/nolta.12.134

[4] https://arxiv.org/abs/2202.04836

[5] https://proceedings.mlr.press/v283/troch25a.html

[6] https://arxiv.org/abs/1906.08253

[7] https://proceedings.mlr.press/v139/chen21r.html

[8] https://arxiv.org/abs/2506.07079
