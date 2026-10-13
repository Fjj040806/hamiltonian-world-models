# Phase 4: feedback intervention and global work consistency

## Research question and outcome
When does Hamiltonian structure improve cross-energy prediction and control after accounting for coordinates, mechanical priors, coverage and optimization? This stage does not establish universal HNN superiority. Upright control remains close to its success ceiling even without additional true training transitions. Persistent rotation exposes differences, but their ordering depends on correction budget and seeds. An analytic coupled-pendulum intervention isolates a global work mechanism that a local curl test alone misses.

## Primary intervention: 57 actual training runs
The frozen protocol is `phase4_protocol.json`. NODE, zero-mean force and HNN use world-model seeds 0, 3 and 4 crossed with policy seeds 0, 3 and 4 at oracle budgets 0 and 1,000: 54 runs. Three true-only references use budget 16,000; these references are reused across comparisons, not counted as independent aliases. These selected existing seeds support an exploratory comparison, not population-level inference. Nine crossed cells are not nine independent samples of both seed factors.

Each run collects 16,000 transitions and performs 7,492 SAC updates with eight rollout lanes. At predetermined, paired collection indices, a generative true-dynamics oracle supplies the next state and interval reward at the same simulated state/action; other indices use the frozen learned model. No unused oracle transition is computed. This simulation intervention differs from Phase 3 Dyna and is not a deployable online robot-feedback experiment. Zero oracle budget means zero additional true transitions during policy training: world models still used 25,600 true pretraining transitions and known mechanical/reward priors. Evaluation uses true dynamics separately and supplies no training updates or checkpoint selection.

On the near-above-separatrix band, NODE and zero-mean force achieve 100% upright success at both budgets. HNN moves from 88.9% to 100%, representing one failed cell among nine at budget zero. Mean returns are NODE -0.879/-0.919, zero-mean force -0.929/-0.904 and HNN -0.963/-0.937 for budgets 0/1,000. Continuous final angle error does not improve uniformly. Therefore real correction alone does not explain the earlier prediction-control ranking gap; this task also saturates without it.

Training queries total 75,000 in the primary intervention, with 837,000 model transitions. Exact masks, update counts and absence of domain exits are verified. The three true-only actor checkpoints reproduce the matching Phase 3 model-free actors with zero maximum weight difference.

## Exploratory persistent rotation: 25 additional runs
The separately frozen `phase4_rotation_protocol.json` targets momentum 2.5 with integrated cost `(p-2.5)^2 + 0.01 u^2`. NODE, ordinary force, zero-mean force and HNN each use three world seeds and budgets 0/1,000, but only policy seed 0: 24 runs plus one true-only reference. This is a task probe, not a confirmatory multi-policy-seed experiment. Strict success requires momentum error below 0.15 throughout the final 20 evaluation states; continuous momentum RMS is also reported.

| Model | Momentum RMS, B=0 | Momentum RMS, B=1,000 | Success, B=0 | Success, B=1,000 |
|---|---:|---:|---:|---:|
| NODE | 0.1134 | 0.0816 | 28.1% | 65.6% |
| Ordinary force | 0.1388 | 0.0999 | 22.9% | 42.7% |
| Zero-mean force | 0.1535 | 0.1223 | 7.3% | 35.4% |
| HNN | 0.0947 | 0.1337 | 59.4% | 35.4% |

Values average the three world seeds on 32 paired near-above-separatrix initial states. The single true-only reference has momentum RMS 0.0886 and success 87.5%. HNN improves over the zero-mean baseline at budget zero in this screen, but worsens on average when correction is added. No broad superiority or monotonic feedback claim follows.

`learned_model_cycle_diagnostics.json` shows that zero-mean force has very small global force error and zero cycle work, yet does not lead this finite-budget SAC comparison. Structural consistency, force accuracy and policy optimization are distinct quantities. NODE diagnostics at momentum 2.5 are conditional diagnostics, not a general Hamiltonian test.

Across both experiments there are 82 actual training runs, 1,312,000 collected transitions, 103,000 new training oracle queries and 1,209,000 model transitions. Evaluation and earlier world-model pretraining are separate budgets. All checkpoints are final checkpoints; test performance does not select them.

## Analytic coupled-pendulum mechanism
This component uses specified analytic force interventions, not learned two-dimensional models or policies. The configuration space is the torus and the potential is `1-cos(q1) + 0.8(1-cos(q2)) + 0.3(1-cos(q1-q2))`, with unit kinetic energy. Three errors have equal uniform-torus Euclidean force RMS 0.04:

* A conservative coupling error preserves a modified periodic Hamiltonian. True base-energy change is bounded by 0.08; observed maximum is 0.07684 over 60 time units.
* A constant force bias `(0.04,0)` has zero curl but nonzero work `2*pi*0.04 = 0.25133` around the first fundamental cycle. Observed base-energy change reaches 8.83760, matching `0.04 * unwrapped q1 displacement`.
* A curl-bearing error `0.04*(sin(q2),-sin(q1))` has nonzero contractible-square work -0.18364 for half-width 0.8. A single fundamental loop can miss it.

RK4 step 0.005 is checked against independent DOP853 endpoints over 10 time units, with discrepancies approximately 3e-10. Energy/work identities hold to approximately 4e-10. These demonstrate mechanism under matched analytic errors, not learned-RL performance. The animation uses actual trajectories; its dashed coupling schematic does not represent a Cartesian spring.

## Mathematical scope and literature
`mathematical_note.md` proves the classical periodic-potential criterion: a smooth force on the two-torus is globally a periodic gradient precisely when its curl and harmonic mean vanish. It also derives the Fourier projection, an L2 residual bound and the exact energy-work identity with a conservative error plus constant bias. A constant bias can preserve local symplecticity while obstructing a globally single-valued periodic Hamiltonian; an unwrapped-cover Hamiltonian remains possible. These are classical results, not a new theorem or a control-performance guarantee.

A prospective curl-plus-mean regularized force baseline is described but has not been trained. Finite quadrature and optimization errors remain relevant. Global work consistency cannot identify the correct unseen potential, as the Phase 2 smooth-bump example demonstrates.

Relevant primary-source scope checks:

* Farahmand, Barreto and Nikovsky, **Value-Aware Loss Function for Model-based Reinforcement Learning**, AISTATS 2017: https://proceedings.mlr.press/v54/farahmand17a.html . Decision-aware model learning predates this project; a prediction/control mismatch is not itself novelty.
* Zheng, Zhong and Allen-Blanchette, **Topology-Preserving Neural Operator Learning via Hodge Decomposition**, ICML 2026: https://proceedings.mlr.press/v306/zheng26aj.html . Hodge-based learning already exists. The abstract-level review does not establish whether a specific control objective differs from its full method.
* Gruver et al., **Deconstructing the Inductive Biases of Hamiltonian Neural Networks**, 2022: https://arxiv.org/abs/2202.04836 . Mechanical and representation priors require explicit baselines.

This is a scoping review, not an exhaustive novelty clearance. An original paper still needs a distinct method or control-relevant theoretical contribution, full-method literature comparison and learned multidimensional validation.

## Artifacts and next stage
JSON/CSV results, final checkpoints, rollout arrays, frozen protocols and verification records accompany the source. Figures 01–03 summarize policy experiments; figures 06–07 show analytic force/work interventions. Two GIFs show true-plant rotation policies and analytic coupled trajectories respectively.

Phase 5 should train actual coupled-system world models, compare unconstrained force, local-curl, global-work-consistent force and HNN under matched representation/capacity/data, and evaluate cyclic control with multiple crossed policy seeds. Freeze protocols before outcomes and retain continuous errors and exact query budgets. The current evidence motivates that experiment; it does not predeclare its result.
