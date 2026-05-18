# Phase 7 — Diagnostic analysis

Asks whether the cross-model GW alignment cost predicts whether GW
transport will succeed on a given (model_pair, layer, k) cell. Runs
the Phase 6 pipeline across a 36-cell matrix and computes Spearman ρ
with bootstrap 95 % CIs.

Read `chapter.md`. Then either:

- Run the full sweep (~10 minutes on the project GPU):
  `python experiments/sweep.py`. Writes
  `outputs/<run_id>/sweep.json`.
- Analyse the latest sweep and regenerate figures:
  `python experiments/analyse_correlations.py`. Writes
  `outputs/<run_id>/correlations.json` and the four chapter figures.

Bootstrap-correlation helper lives in `src/ot_steering/eval/correlation.py`.

**Headline result:** Spearman ρ between GW cost and shift rate is
**−0.17 [95 % CI −0.45, +0.16]** across 36 cells — not significantly
different from zero. The diagnostic hypothesis is not supported at this
matrix size and resolution; the chapter discusses why and what would
be needed to give a more decisive answer.
