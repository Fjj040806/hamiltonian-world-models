# Phase 1: Simulator verification and energy-regime prediction data

Date: 2026-10-13. All quantities are dimensionless. This milestone establishes numerical infrastructure, not Hamiltonian-model superiority or an original paper result.

## Mathematical specification

The phase space is S¹ × R; internally q remains unwrapped. Let H(q,p)=p²/2+1−cos(q). The equations are qdot=p and pdot=−sin(q)+u. Differentiation gives dH/dt=sin(q)p+p(−sin(q)+u)=pu. The saddle (π,0), modulo 2π, has energy 2. Uncontrolled trajectories below/above 2 librate/rotate except equilibrium and separatrix special cases.

We integrate an auxiliary variable Wdot=pu using the same RK4 stages. For constant u, W=u(q−q_initial) also provides an exact identity. The controlled reference is independently computed using DOP853 with each action discontinuity as a solver segment boundary. Merely calculating instantaneous dH/dt algebraically would not establish numerical work balance, hence the accumulated diagnostic.

RK4 is the main accurate reference simulator. Velocity Verlet is included only for this separable mechanical Hamiltonian. Symplecticity does not imply exact conservation of H. No arbitrary-feedback symplecticity claim is made.

## Verification protocol and results

Energy experiments: h=0.01, duration 100, initial q=0 and p=sqrt(2E), E in {0.7,1.99,2.7}. The largest absolute energy drift across these three cases is 5.68e-10 for RK4 and 5.88e-5 for Verlet. This short-to-moderate duration comparison does not establish a universal long-time advantage for RK4.

Convergence: terminal state error at T=10 against independent DOP853 reference (rtol=2e-13, atol=2e-14), worst over the three cases. h in {0.08,0.04,0.02,0.01,0.005}. Log-log regression on the finest three steps gives order 4.043 (RK4) and 2.009 (Verlet).

Control experiment: 40 independently sampled torques in [-1.5,1.5], each held for 0.5 time units; total T=20, RK4 h=0.01, seed 731, initial (q,p)=(0.4,0.7). Maximum |H(t)−H(0)−W(t)|=9.15e-9. Maximum state discrepancy against the segmented independent reference is 1.03e-8. Control injects substantial energy here; this is a numerical stress test rather than a target task.

The automated gate requires RK4 order >3.7, Verlet order in (1.8,2.2), work residual <1e-7, controlled state discrepancy <1e-5, and uncontrolled RK4 drift <1e-6. All passed. Environment API smoke checks confirm observation shape and time-limit truncation; they do not certify learned-policy performance.

## Sampling

Seed 20261013. Draw E uniformly in the split interval, direction sign uniformly and time phase uniformly on [0,1). Starting from (0,±sqrt(2E)), integrate to that phase using high-accuracy DOP853. Uncontrolled orbit period: 4K(E/2) for E<2 and 2sqrt(2/E)K(2/E) for E>2, where K uses its parameter rather than modulus convention. For rotations, the period is modulo the angular identification.

| Split | Initial E range | Trajectories | Transitions |
|---|---|---:|---:|
| Training | 0.2–1.2 | 128 | 25,600 |
| Validation | 0.2–1.2 | 32 | 6,400 |
| Moderate extrapolation | 1.3–1.7 | 32 | 6,400 |
| Near separatrix, below | 1.8–1.95 | 32 | 6,400 |
| Near separatrix, above | 2.05–2.2 | 32 | 6,400 |
| Rotation | 2.4–3.0 | 32 | 6,400 |

Each trajectory lasts four time units at dt=0.02. Largest observed energy drift among all generated samples is 6.59e-9. SHA-256 hashes and exact runtime versions are recorded in dataset_manifest.json. Four time units need not cover an entire near-separatrix orbit; initial uniform time-phase reduces phase concentration, but longer evaluation trajectories will be required in Phase 2.

## Figures

1. Energy drift over time in three regimes.
2. Step-size convergence against the independent reference.
3. Controlled energy change, signed work and residual.
4. Initial-state coverage and initial-energy distributions.

Created with NumPy/SciPy/Matplotlib; PNG for previews and SVG for publication editing. CSV and JSON retain numerical evidence. No invented learning curves.

## Implications and limitations

The verified h=0.01 simulator is suitable for the planned pilot; dataset h=0.02 errors remain small relative to proposed test energy margins. Nearer-to-separatrix studies must verify tolerances again. Phase 1 has not proven that energy conservation improves RL. Uniform orbit phase is different from uniform spatial coverage, and low-energy prediction data do not identify arbitrary high-energy Hamiltonians. The crucial research question remains whether structural constraints aid control beyond strong matched baselines.

Next: Phase 2 world-model comparison. Controlled training datasets, policy learning, multidimensional validation, theory and a full manuscript remain outstanding. Literature novelty checks from Phase 0 remain open.
