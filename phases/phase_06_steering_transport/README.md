# Phase 6 — Cross-model steering transport (the core experiment)

The project's headline experiment. Combines every piece of machinery from
Phases 1–5 to push a steering signal from a *source* model to a different
*target* model using only structural alignment — no paired data, no
target-side concept labels.

Read `chapter.md`. Then either:

- Run the notebook (~3 minutes on the project 4 GB GPU):
  `jupyter notebook notebook.ipynb`.
- Run the full experimental matrix (2 cells × 4 methods × 3 coefficients
  × 3 seeds with bootstrap CIs):
  `python experiments/run_transport.py`. Writes
  `outputs/<run_id>/run_transport.json`.
- Regenerate the four chapter figures from the latest run:
  `python experiments/make_figures.py`.

The transport pipeline lives in `src/ot_steering/steering/transport.py`;
baselines (random / Procrustes / target oracle) are in
`src/ot_steering/steering/transport_baselines.py`.
