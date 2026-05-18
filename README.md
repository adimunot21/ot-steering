# ot-concept-transport

**Cross-architecture concept-vector transport via Gromov-Wasserstein optimal transport.**

This is a research codebase and an interleaved course. The research question:
can we take a useful interpretability artifact — a "refusal direction," a
sentiment steering vector, a truthfulness direction — discovered inside one
large language model, and *transport* it into a different model with no paired
data and no target-side supervision? The hypothesis we test is that while the
raw activation coordinates of different LLMs are not comparable, the
*relational structure* of their contrastive activation distributions
(harmful-vs-harmless prompts, positive-vs-negative reviews, true-vs-false
statements) is approximately model-universal. Gromov-Wasserstein optimal
transport aligns distributions across incomparable spaces using only
intra-distribution distances — exactly the tool this hypothesis needs.

The deliverable is twofold. First, a working end-to-end pipeline that takes
two LLMs and a concept, learns a GW coupling between their activation
geometries, and uses it to push a steering vector from one model to the other.
Second, a course: each phase folder under `phases/` contains a chapter written
as a blog-essay, readable by a curious engineer with strong Python and ML
fundamentals but **zero prior OT and zero prior LLM-internals knowledge**.
Every concept is defined from scratch the first time it appears; the code in
each chapter is the same code that runs in the project, not a parallel
re-implementation.

The plan is conservative on compute (single GTX 1650, 4 GB VRAM, via 4-bit
quantization) and conservative on scope (text-only, inference-only, models
≤1.1 B params in fp16 / ≤3 B in 4-bit). The target venue is a NeurIPS / ICLR
workshop on mechanistic interpretability or OT&ML. See `PROJECT_PLAN.md` for
the full plan, phases, risks, and related work.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

PyTorch's CUDA variant is selected automatically by pip based on your local
toolchain (cu121 / cu124 / cu13 / cpu). If you need a specific variant,
install torch first from <https://download.pytorch.org/whl/cu121> (or the
corresponding URL) before the `pip install -e .` line above.

## Quickstart

Verify your environment:

```bash
python scripts/verify_env.py
```

Run the test suite:

```bash
pytest -q
```

## Where to look

- [`PROJECT_PLAN.md`](PROJECT_PLAN.md) — phases, deliverables, risks, related
  work, out-of-scope items.
- [`CLAUDE.md`](CLAUDE.md) — how we work in this repo: code conventions,
  testing rules, course-writing voice.
- [`PROGRESS.md`](PROGRESS.md) — current phase pointer and session log.
- [`phases/`](phases/) — the course. Each `phase_NN_topic/` folder has
  `chapter.md` (the essay), code, and figures together.

## Layout

```
src/ot_steering/      # the library (well-tested, reusable)
  ot/                 # OT solvers (POT wrappers + barycentric projection)
  activations/        # model loading, hooks, datasets, caching
  steering/           # baselines + OT steering + cross-model transport
  eval/               # steering & transport metrics
  utils/              # seed, logging, config, io
phases/               # interleaved course + per-phase experiments
tests/                # mirrors src/ exactly
scripts/              # CLI entry points for big experiments
configs/              # YAML configs, validated by pydantic at load time
outputs/              # run artifacts (gitignored)
data/                 # dataset & activation cache (gitignored)
```
