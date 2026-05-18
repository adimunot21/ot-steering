# Phase 1 — Optimal transport foundations

Discrete OT and Sinkhorn, from scratch and via POT. Production wrappers live in
`src/ot_steering/ot/{emd,sinkhorn}.py`; pedagogical from-scratch versions live
in `scratch_ot.py` and `scratch_sinkhorn.py` and are never imported by
downstream code.

Read `chapter.md`, then run the notebook with
`jupyter notebook notebook.ipynb`. Regenerate figures with
`python experiments/make_figures.py`.
