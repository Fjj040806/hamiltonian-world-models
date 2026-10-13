# Phase 3: Do cross-energy prediction improvements transfer to reinforcement-learning control?

## Abstract

We connect five frozen pendulum world-model architectures to a common model-based reinforcement-learning pipeline. An initial under-optimized pilot fails to solve the task even with true dynamics. Calibration on an independent, in-distribution-only seed identifies an adequate optimization budget, after which 55 calibrated training runs compare model families and synthetic rollout depths. Large differences in long-horizon prediction accuracy do not yield a clear control ranking in this feedback-rich task. Strict success rates also reflect small steady-state biases near the chosen angle tolerance. These results motivate a mechanism study with limited real correction, crossed model/policy seeds and multidimensional systems; they do not establish Hamiltonian-model superiority or a new algorithm.

## 1. Research question

The project asks when Hamiltonian structure improves cross-energy prediction and control, after accounting for coordinates, second-order information, data support and policy optimization. Phase 2 found that a zero-mean periodic force baseline predicted better than the current separable HNN. Phase 3 tests whether that ranking transfers to actual closed-loop control. It is important to distinguish unseen world-model training energies from unseen policy replay energies.

## 2. Task and algorithm

The true plant is qdot=p, pdot=−sin(q)+u on S¹×R, with |u|≤2. Control actions are held constant for 0.05 time units; both real and model transitions use five RK4 substeps of h=.01. Each episode lasts 400 intervals (T=20). Observation is [cos(q),sin(q),p/3]. Momentum is never clipped in real simulation; q is wrapped only to represent the same cylinder state.

The reward is the negative trapezoidal integral of c(q,p,u)=1+cos(q)+0.1p²+0.01u² across each interval. This task reward is known identically to all model-based variants. It is not a learned reward model. Its geometric terms and known unit actuator gain are shared prior information; the benchmark is not an arbitrary unknown-mechanics setting. True H is used for diagnostic coverage, not as a policy observation.

Success requires |wrap(q−π)|<.2 and |p|<.3 at every evaluated state of the final one-time-unit interval. Episode return, integrated squared-torque effort, energy-change diagnostic, final angle RMS and final momentum RMS are also retained. Signed work is inferred from the verified plant energy difference; this differs from effort.

We implement SAC with a squashed Gaussian actor, twin Q networks and Polyak target networks. Hidden layers are 64×64 ReLU. Discount=.99; entropy temperature=.01 is fixed; learning rate=.0003; target coefficient=.005; batch size=256. Time-limit transitions bootstrap because the underlying continuing regulation task does not terminate at the artificial reset boundary. Checkpoints are chosen at the final fixed budget, not by test performance.

For model-based variants, 128 real and 128 synthetic transitions form each training batch. Short model trajectories branch from real replay every ten collection rounds. Models remain fixed at their Phase 2 checkpoints. Model-free SAC uses 256 real transitions. A true-model variant uses exact dynamics only for synthetic generation and is a diagnostic reference, not a guaranteed performance upper bound. A separately evaluated energy-shaping/local-PD controller has privileged true-mechanics knowledge and is not counted as RL.

## 3. Calibration and protocol changes

The initial frozen pilot used one optimization update for each eight newly collected real transitions. Most runs, including true-model SAC, failed sustained stabilization. These results are retained in phase3/, including the original protocol and per-run checkpoints.

An additional true-model calibration used independent seed 101, unchanged task and 16,000 real transitions, but four optimization updates per collection round. Only in-distribution validation states were assessed. This achieved return −2.36 and success 1.0, demonstrating that the task is learnable at the real-data budget.

The calibrated comparison then froze four updates per collection round for every method. It also equalized attempted synthetic generation to 640 transitions per generation across horizons: branch count=ceil(640/horizon), truncating the last depth to the remaining budget. This replaces the unequal synthetic counts of the original exploratory horizon pilot. Both protocols are saved. These are disclosed exploratory amendments, not retrospectively described as a pristine preregistration.

## 4. Experimental design

Main comparison: seven methods × five seeds at horizon 5, except model-free which has no synthetic horizon. Additional horizon comparisons: HNN and zero-mean force × five seeds × depths {1,15}. Total: 55 calibrated runs. Every run collects exactly 16,000 real transitions in eight parallel environments. Initial training states are sampled from the shared Phase 1 low-energy pool. Actor/critic initializations and random streams are paired by seed across methods; world-model and policy seeds are paired rather than factorially crossed.

The five learned model families are Neural ODE, second-order NN, force NN, zero-mean force NN and separable known-kinetic-energy HNN. Each pretrained model used 25,600 transitions before online policy training. Thus its total dynamics-plus-online data exposure is 41,600 transitions, versus 16,000 online transitions for model-free SAC. Equal online budgets must not be described as equal total data budgets or proof of sample efficiency.

Each policy is evaluated deterministically on the same 32 initial states in each of five energy bands: validation, moderate extrapolation, near-below, near-above and rotation. Evaluation interactions are excluded from training and reported separately. No policy is selected by held-out results. Synthetic endpoints outside |p|≤10 or with nonfinite values are rejected, with counts recorded. No rejections occurred in the calibrated comparison.

The unique calibrated comparison comprises 880,000 real training transitions and 5,984,000 synthetic transitions. Transition counts, configurations and timings are available through per-run JSON records; source code regenerates replay data, which are not all stored as full training replay arrays. Policy/Q checkpoints are provided, but optimizer/RNG state is not provided for exact mid-training resumption.

## 5. Results

The complete mean ± seed-SD table is in calibrated/summary_table.md; all seed/energy results are in calibrated/results.csv and results.json. At the main horizon, near-above success fractions are 1.0 for Neural ODE, second-order NN, HNN and model-free SAC, and .8 for force NN, zero-mean force NN and true-model SAC. Near-above mean returns are approximately −.865, −.921, −.956, −.930, −.895, −.932 and −.860 respectively in the same method order. Differences are small compared with the initial failed pilot.

This does not rank HNN as uniquely superior: substantially worse-predicting Neural ODE models also control successfully, and model-free SAC attains comparable returns. The pooled descriptive Spearman correlation between Phase 2 near-above long-horizon prediction error and Phase 3 return is −.22 across 25 paired architecture/seed cases. These observations are not independent experimental units for a causal test; no significance inference is made.

For zero-mean force, near-above success at horizons 1,5,15 is 1.0,.8,1.0; corresponding returns are approximately −.865,−.930,−.887. HNN success is 1.0 at all three depths, with returns about −.891,−.895,−.888. There is no monotonic rollout-depth advantage established by this small experiment.

The apparent .8-versus-1.0 success difference is concentrated in one seed's steady-state precision. At horizon 5, zero-mean-force seed 3 settles at mean angle RMS .2062 radians, just above the .2 success tolerance. True-model seed 3 has .2195; HNN seed 3 has .1706. Final momentum RMS is near zero. These are small biased equilibria rather than trajectories exploding or repeatedly falling. Continuous errors must accompany binary success rates. Threshold sensitivity is a follow-up question, not a reason to revise the primary criterion after seeing results.

## 6. Coverage and interpretation

About 59–60% of visited real training states lie above the world-model training band H≤1.2; the exact method fractions are in calibrated/summary.json. The world models remain trained on low-energy data, but SAC receives real corrective transitions after policies enter higher-energy regions. Therefore this experiment establishes initial-condition generalization under fixed low-energy world models with online real feedback. It does not establish policy zero-shot extrapolation into completely unseen high-energy states.

The results are consistent with real replay and feedback compensating model error, but do not isolate that mechanism causally. The task also approaches a success ceiling for several methods. Perfect dynamics alone do not guarantee that a finite-budget actor/critic optimizer will produce the most precise policy, as the true-model seed-3 result illustrates.

## 7. Mathematical interpretation

For a fixed feedback policy π, let F and Fhat be the discrete controlled transition maps on a common bounded domain, with sup distance(F,Fhat)≤ε and Lipschitz constant L for one map. Coupling trajectories from the same initial state gives e_(k+1)≤L e_k+ε, hence e_k≤ε sum_(j=0)^(k−1)L^j. If integrated reward maps differ by at most δ_r and are L_r-Lipschitz, then the finite-horizon value discrepancy obeys

|J_F(π)−J_Fhat(π)| ≤ sum_(k=0)^(N−1) γ^k [δ_r + L_r ε sum_(j=0)^(k−1)L^j].

The empty sum at k=0 is zero. The reward discrepancy term is retained because integrating the same running cost along different within-step trajectories can produce different interval rewards. Coordinates can be handled using a suitable cylinder metric or compatible lifted trajectories.

This elementary background bound requires uniform errors on the visited domain and a fixed policy. Low training MSE, energy conservation or the Phase 2 support-limited data do not supply that uniform error guarantee. The inequality also does not compare separately optimized policies or prove SAC convergence. It explains why prediction diagnostics alone cannot certify control quality, rather than constituting a claimed new theorem.

## 8. Verification and deliverables

NumPy-to-PyTorch dynamics ports were compared for all five model families, with maximum discrepancy below 8.1e-7 on the checked states. True-state transitions and rewards agree with the Phase 1 simulator within specified tolerances. Bounded actions, finite log probabilities, nonzero actor updates and finite losses were checked. All 55 trained policies were re-evaluated from saved checkpoints; returns reproduced exactly in this run. Budgets and energy-histogram totals were checked. These are implementation checks, not control-theoretic stability certificates.

Five PNG/SVG scientific figures show learning/generalization, success rates, horizon comparisons, prediction-versus-control and real-replay energy coverage. A GIF shows actual deterministic closed-loop trajectories of four seed-0 policies from the same fixed median-energy near-above initial state; it is not selected for HNN advantage. Animation frame arrays and source code are included. Results from failed and calibrated pilots remain separately identifiable.

## 9. Next research decision

Do not turn these results into a claim of Hamiltonian superiority. Phase 4 should vary real corrective feedback while holding optimization and synthetic budgets controlled, cross model and policy seeds, perform matched-error/targeted barrier interventions, and compare continuous precision as well as task success. A coupled two-degree-of-freedom system will be needed to separate genuinely multidimensional conservative-force constraints from one-dimensional periodic zero-mean equivalence.

An interesting theoretical direction is to distinguish local force consistency from global work around closed loops. For a smooth periodic force on a circle, zero angular circulation is the condition for a periodic potential. In multiple angular coordinates, zero curl and zero fundamental-cycle work must both be considered for a global periodic potential. These are classical mathematical facts; whether their explicit use yields a novel ML/control contribution still needs a primary-literature review and controlled experiments.

The project now has trained world models, actual RL policies, repeatable evaluation and dynamic demonstrations. A submission-quality paper still requires a validated original mechanism or method, broader systems, stronger uncertainty analysis and the outstanding nearest-literature checks.
