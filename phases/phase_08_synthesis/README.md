# Phase 8 — Synthesis and paper draft

The closing phase. Pulls Chapters 0–7 into a coherent narrative and
ships a one-command figure-regeneration target. No new experiments —
this phase is about consolidation.

Read:
- `paper.md` — workshop-length writeup (~5 pages plus references).
- `chapter.md` — the synthesis chapter, with per-phase recaps and an
  honest accounting of what's published-quality vs preliminary.

To regenerate every figure in the paper, from the repo root:

```bash
python scripts/make_all_figures.py
```

The script calls each phase's `make_figures.py` in chapter order and
reuses the latest `outputs/<run_id>/` artefacts when figures depend
on them. Total wall-clock on the project GPU is ~45 minutes.
