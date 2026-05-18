# Phase 4 — Intra-model OT steering (CHaRS-style)

Replaces the difference-of-means steering direction from Phase 3 with an
OT-induced steering *map*: fit a GMM per class, solve discrete OT between
the cluster centroids, and use barycentric projection to get an
input-conditional displacement vector field. At k=1 the construction
reduces exactly to difference-of-means.

Read `chapter.md`. Then either:

- Run the notebook on Pythia-160M for a fast demo (<60 s):
  `jupyter notebook notebook.ipynb`.
- Run the full comparison sweep on GPT-2-small and Pythia-160M:
  `python experiments/compare_baselines.py`. Writes
  `outputs/<run_id>/compare_baselines.json`.
- Regenerate the four chapter figures from scratch:
  `python experiments/make_figures.py`.
