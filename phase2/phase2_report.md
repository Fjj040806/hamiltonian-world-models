# Phase 2: Cross-energy world-model pilot and structural diagnostics

Date: 2026-10-13. Status: five model architectures, five initialization seeds each, evaluated on fixed Phase 1 splits. No RL policy training has occurred. Results are a small-system pilot, not a publishable superiority or causality claim.

## Core question and current answer

Does Hamiltonian structure aid cross-energy control after accounting for coordinates, second-order priors and coverage? This phase evaluates prediction prerequisites. The separable HNN improves some predictions over unconstrained models, but a periodic zero-mean force model has lower mean long-horizon error in every extrapolation band. Thus the evidence does not support attributing the pilot gains specifically to the HNN parameterization. Control consequences remain unknown.

## Models and optimization

All networks use one fully trained 32-unit tanh hidden layer, periodic sin(q),cos(q) inputs and NumPy implementations of analytic gradients with Adam. They are not fixed random-feature models. No exact potential formula is supplied to learned models. State labels are canonical q,p, and no analytic sin(q) force target is provided. Labels are central finite differences of recorded trajectories (dt=.02). This introduces O(dt²) target error, to be checked against integrated-transition fitting in follow-up experiments.

| Model | Learned quantity | Supplied structure | Stored parameters |
|---|---|---|---:|
| Neural ODE | both derivatives from sin(q),cos(q),p | unit torque added to pdot | 194 |
| Second order | acceleration from sin(q),cos(q),p | qdot=p; unit torque | 161 |
| Force | force from sin(q),cos(q) | same plus no momentum dependence | 129 |
| Zero-mean force | force minus periodic angular average | same plus zero circulation | 129 |
| Separable HNN | scalar V from sin(q),cos(q) | H=p²/2+V; canonical derivatives; unit torque | 129 |

One unused scalar offset is stored in the last two architectures; it has zero gradient. Counts are similar but not identical across the full ladder. The known unit mass in the final four models is extra prior information compared with unrestricted Neural ODE. This is a decomposition experiment, not a fully capacity-matched architecture tournament. A general H(q,p) model is deferred; current HNN results apply only to the separable known-kinetic-energy case.

Each seed uses the same sampling stream across models, 3,000 updates, batch size 512, learning rate cosine-decayed from about .003 to .0006. Checkpoints are selected every 100 updates by the minimum in-distribution validation derivative MSE. Validation and test trajectories are never used for updates. Width, steps and evaluation rules are fixed; no test-selected hyperparameter search.

The zero-mean comparator was added after the initial force/HNN pilot revealed a possible circulation confound. It subtracts a 128-point uniform periodic quadrature mean, with gradients through that mean. The sampled 4,001-point circulation check gives approximately zero, but no claim of exact symbolic quadrature is made. This adaptive addition is disclosed; subsequent confirmatory experiments should freeze it in advance.

## Evaluation

All models are rolled out with the same RK4 dt=.02 for 16 time units from all 32 initial states in each validation/test band. The truth trajectory is regenerated with RK4 h=.01 and subsampled; Phase 1 independently validated this truth simulator. Prediction error uses circular angular difference and canonical momentum difference:

RMSE_T = sqrt(mean over trajectories and times of [wrap(qhat−q)² + (phat−p)²]).

This is a dimensionless combined-state metric, not an angle-only error. Values are averaged over five independent initialization seeds. Error bars show sample standard deviation across seeds, not confidence intervals. The dataset is fixed across seeds, so dataset uncertainty is not estimated.

See summary_table.md for the complete mean ± SD table and results.csv/results.json for per-seed values. The mean 16-unit RMSE in near-above tests is 1.5543 (second order), 1.5835 (force), 0.1806 (zero-mean force), and 0.6322 (HNN). In rotation tests it is 1.1339 (force), 0.0441 (zero-mean force), and 0.1599 (HNN). Large seed variation precludes significance claims from these averages alone.

## Structural diagnostics

1. The unrestricted force model has angular circulation between about -0.197 and +0.117 across seeds. Nonzero circulation implies nonzero net work per complete revolution for a position-only force. This is a plausible drift mechanism, not proof that it explains every error.
2. HNN learned-energy deviations are around 3e-9 to 4e-9 in the near-above experiment, whereas mean maximum true-energy deviations range from about .011 to .080. Conserving a learned Hamiltonian does not imply matching the real energy surface.
3. Root-finding found two force zeros per fitted HNN on the circle, with shifted critical angles. Actual learned potential ranges are approximately 1.999, 1.920, 1.989, 1.924 and 2.066, compared with the true barrier 2. Grid-based discovery and scalar root refinement are numerical diagnostics, not a formal proof of all roots.
4. Under the fitted HNN's energy/barrier criterion, seeds 1 and 3 misclassify 4/32 near-below initial states as rotating; seed 4 misclassifies 2/32 near-above states as nonrotating. These are uncontrolled regime predictions, not RL success rates.

## An elementary identifiability obstruction

Let E_train=1.2 and q_max=acos(1−E_train). Every uncontrolled training state lies within |q|≤q_max, modulo 2π. Choose a smooth periodic bump b supported within angular distance a=.6 of π, with b(π)=1 and 0≤b≤1. The support is disjoint from the whole training-accessible angle region, not just the sampled points.

Define H_A=p²/2+1−cos(q)+A b(q), with A=.2. Both H and H_A are smooth periodic Hamiltonians. Since b and its derivatives vanish on the training region, their vector fields and all low-energy training trajectories there are identical. But their global potential maxima are 2 and 2.2. From the same initial state (q,p)=(0,sqrt(4.2)), initial energy is 2.1 in both models. The original trajectory rotates, while the modified one cannot cross the higher barrier and librates.

The implemented bump is exp(1−1/[1−(r/a)²]) for |r|<a and zero otherwise, with r the circular signed distance to π. Smooth vanishing at support edges makes the periodic construction well-defined. The shaded-region agreement is checked numerically and follows analytically from disjoint support.

This simple construction demonstrates that conservation alone cannot uniquely identify out-of-support dynamics over the smooth periodic Hamiltonian class. It is background mathematical scaffolding, not a claimed new theorem. It does not apply unchanged to a restrictive globally identifiable parametric family, and it does not establish limits for adaptive control that gathers new data.

## Verification and reproducibility

Analytic training gradients checked by central finite differences for all five architectures; original dataset hashes checked; all predictions finite. Step-halving checks for a fixed median-energy near-above state across all 25 models give maximum state discrepancy 2.77e-6 over T=16. This is much smaller than observed pilot prediction errors. It is a limited numerical check, not uniform verification across all trajectories.

Code, 25 checkpoints, validation histories, per-seed metrics, exact source arrays for animation and verification output are included. A seed-0, median-initial-energy near-above trajectory was fixed for the animation; it is not selected for best HNN performance. Four panels show true dynamics, second-order NN, zero-mean force NN and HNN. Playback is 2x physical simulation time. Static plots include every model.

## What follows

Maintain the core question and carry the zero-mean baseline into Phase 3. Freeze control task, reward, seed/data budgets and model rollout horizons before comparing policies. Separate fixed-model from online-updated experiments. Measure whether better long prediction yields better RL, and whether it matters at short synthetic rollout horizons.

Before a causal or submission claim: additional data seeds, capacity/optimizer sweeps without test tuning, transition-loss training, matched-error comparisons, a general HNN ablation, and coupled-pendulum replication remain necessary. In one position dimension, zero-mean periodic forces are gradients of periodic potentials, so the strong comparator is itself Hamiltonian-compatible. The meaningful question is which constraints and parameterizations help, rather than an artificial Hamiltonian/non-Hamiltonian dichotomy.

Candidate paper contributions remain a controlled mechanism study, a coverage/identifiability analysis, and potentially a justified model-usage rule. These results do not yet establish novelty or RL benefits. Primary-method literature checks from Phase 0 are still outstanding.
