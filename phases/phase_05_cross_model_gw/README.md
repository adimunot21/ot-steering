# Phase 5 — Cross-model Gromov-Wasserstein alignment

Aligns the contrastive activation distributions of two *different* LLMs
using only their intra-distance matrices (Gromov-Wasserstein). Phase 5
runs four sanity checks — self-pair, adjacent-layer, cross-model, and
random-noise — to validate that the alignment is meaningful before
Phase 6 attempts steering transport on top.

Read `chapter.md`. Then either:

- Run the notebook on Pythia-160M + GPT-2-small (~2 minutes):
  `jupyter notebook notebook.ipynb`.
- Run the full sanity-check sweep (two concepts):
  `python experiments/sanity_checks.py`. Writes
  `outputs/<run_id>/sanity_checks.json`.
- Regenerate the four chapter figures:
  `python experiments/make_figures.py`.
