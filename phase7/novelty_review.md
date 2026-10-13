# Phase 7 nearest-method check and originality gate

This is a scoped primary-source review, not an exhaustive absence-of-prior-art certificate.

| Prior work | Reviewed scope | Overlap with the candidate | Consequence |
|---|---|---|---|
| Wagenmaker, Simchowitz and Jamieson, *Task-Optimal Exploration in Linear Dynamical Systems*, ICML 2021 | Primary abstract | Task-dependent experiment-design exploration and task-optimal information requirements | Task-aware exploration is not a new principle |
| Wagenmaker, Shi and Jamieson, *Optimal Exploration for Model-Based RL in Nonlinear Systems*, NeurIPS 2023 | Primary abstract and official full paper available; task-metric method examined separately | Estimates system parameters in a task-dependent metric and reduces exploration to experimental design | A weighted task metric alone does not establish a new nonlinear-control algorithm |
| Mehta et al., *An Experimental Design Perspective on Model-Based Reinforcement Learning*, ICLR 2022 | Primary introduction and Sections 3–5 | Generative transition queries, Bayesian acquisition about policy trajectories, MPC | Our generative force-query setting and relevance-weighted acquisition are not unprecedented |
| Ma et al., *Learning Policy-Aware Models for Model-Based Reinforcement Learning via Transition Occupancy Matching*, L4DC 2023 | Primary abstract | Occupancy and downstream policy relevance in model learning | Occupancy weighting is a required comparator, not a stand-alone novelty claim |
| Smith and Egeland, *Learning Hamiltonian Dynamics with Reproducing Kernel Hilbert Spaces and Random Features*, 2024 | Primary abstract; previous-stage feature-method review | Structured Hamiltonian features and odd symmetry | The Hamiltonian feature head is an established baseline |

Primary sources:

* https://proceedings.mlr.press/v139/wagenmaker21a.html
* https://proceedings.neurips.cc/paper_files/paper/2023/hash/31e018f43ab9c7065c058cc2c5848128-Abstract-Conference.html
* https://proceedings.neurips.cc/paper_files/paper/2023/file/31e018f43ab9c7065c058cc2c5848128-Paper-Conference.pdf
* https://arxiv.org/html/2112.05244v2
* https://proceedings.mlr.press/v211/ma23a.html
* https://arxiv.org/abs/2404.07703

The candidate `spectral_A` specializes weighted optimal design to a first-order periodic momentum-tracking response. Its weights depend on mode frequency along a desired winding velocity. This specialization is useful to derive and test, but the Gaussian covariance update and weighted trace criterion are classical. It is not a full replication of BARL or the Wagenmaker algorithms, and comparisons here cannot support claims of outperforming those published methods.

**Gate 0 decision:** the generic new-algorithm claim is not cleared. Phase 7 proceeds only as a bounded empirical and mathematical specialization. A practical benefit gate is frozen before outcomes; failing it stops this algorithm route without additional tuned variants. Even passing it would support follow-up validation, not automatically establish novelty.

A posthoc mathematical interpretation uses translation-invariant design symmetrization. This too is established background: Pukelsheim, *Optimal Design of Experiments*, invariant-design chapters (https://epubs.siam.org/doi/book/10.1137/1.9780898719109), and Idais and Schwabe (2021), https://link.springer.com/article/10.1007/s42519-021-00221-z . Our torus/Fourier specialization is self-contained in the mathematical note; it is not a replacement novelty claim after the candidate fails.
