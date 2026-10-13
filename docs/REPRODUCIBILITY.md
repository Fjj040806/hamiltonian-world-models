# Complete artifact restoration

This repository includes all original datasets/checkpoints in local stage bundles. Run `python restore_artifacts.py` once after cloning before archive-dependent checks. The script verifies each bundle and all restored file hashes.

# Reproducibility

## Environment

Python 3.12, NumPy 2.3.5, SciPy 1.17.0, Matplotlib 3.10.8. SAC additionally uses Torch 2.14.1+cpu. The pinned environment reproduces this release; other platforms may need compatible package versions. CPU runtime depends strongly on worker count and BLAS thread settings.

```bash
python -m pip install -r requirements.txt
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
python run_phase8.py
python verify_phase8.py
```

The above short path recreates the independent final diagnostics; it does not recreate all prior SAC or acquisition runs. Full reproduction uses the command blocks in README. Do not rerun all stages merely to inspect archived outcomes.

## Inspect archived evidence

Read the stage reports and raw JSON/CSV files. Verification scripts check particular numerical assumptions and fingerprints. Some earlier verifiers require datasets/checkpoints. In this repository all are supplied under `artifacts/`; restore them with the script above. Actual animation and image assets can be displayed without training.

## Paper

Requires Pandoc and a LaTeX installation with pdflatex:

```bash
bash paper/rebuild.sh
```

Run from the repository root. The PDF and LaTeX source are already supplied. Figures use relative paths and must remain beside the stage folders.

## Data and information access

All datasets are synthetic. Mass, momentum kinematics and input ports are known. Free generative force queries are assumed; reachability costs are absent. Test states never select primary regression hyperparameters. Final diagnostics were frozen after earlier outcomes and are labelled accordingly. Policy evaluations, seed replications and aliases are counted separately.

## Failed attempts and decisions

The full archive preserves unsuccessful optimization pilots and the failed acquisition gate. `phase7/decision.json` remains unchanged. No new SAC or acquisition rescue variants are introduced by the final diagnostic. Originality boundaries are recorded in `CONTRIBUTIONS.md`.
