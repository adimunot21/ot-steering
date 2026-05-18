# Phase 2 — Gromov-Wasserstein

Cross-space distribution alignment via entropic GW. Production wrappers live
in `src/ot_steering/ot/gw.py` (`solve_entropic_gw`) and
`src/ot_steering/ot/barycentric.py` (`barycentric_project`). The chapter walks
from "OT can't compare incomparable spaces" through the GW objective to a
working rotation-recovery demo.

Read `chapter.md`, then run `jupyter notebook notebook.ipynb`. Regenerate
figures with `python experiments/make_figures.py`.
