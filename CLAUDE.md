# CLAUDE.md — Project context for Claude Code sessions

This file is loaded automatically at the start of every Claude Code session.
It is the source of truth for **how** we work on this repo. The **what** is in `PROJECT_PLAN.md`.

---

## Project context (read this first)

We are building a research project on **cross-architecture concept-vector transport via Gromov-Wasserstein optimal transport**.

The hypothesis: the *relational structure* of contrastive activation distributions in LLMs (e.g. "harmful prompts" vs. "harmless prompts" at layer N) is approximately model-universal, even though the raw activation coordinates are not. If true, Gromov-Wasserstein — which aligns distributions across incomparable spaces using only intra-distribution distances — should let us transport interpretability artifacts (steering vectors, concept directions, refusal directions) from one LLM to another with no paired data and no target-side supervision.

The deliverable is twofold:

1. A working research codebase that runs the experiment end-to-end and produces publishable results.
2. An interleaved **course**: a chapter sits inside each phase folder alongside the code it explains, written as a blog-essay readable by someone with **no prior OT or LLM background**.

Target venue: NeurIPS / ICLR workshop on mechanistic interpretability or OT&ML.
Hardware budget: single NVIDIA GTX 1650 (4GB VRAM), 32GB RAM, Ubuntu 24.04, CUDA-enabled.

## Where to look when starting a session

1. `PROJECT_PLAN.md` — phases, deliverables per phase, risks, related work.
2. `PROGRESS.md` — current phase pointer, what was done last session, what's blocked. **Update at the end of every session.**
3. The current phase's folder — `phases/phase_NN_topic/`. The chapter, code, notebook, and figures live together.

## Tech stack (and why each choice)

- **Python 3.11.** Best balance of typing features and ecosystem stability in 2026.
- **PyTorch 2.x with CUDA 12.1.** Works on Turing-class GPUs (GTX 1650 is compute capability 7.5).
- **transformers (Hugging Face) + accelerate + bitsandbytes (4-bit).** Required to fit TinyLlama / Qwen-0.5B in 4 GB VRAM.
- **POT (Python Optimal Transport).** The canonical OT library. Use `pot.sinkhorn`, `pot.gromov.entropic_gromov_wasserstein`, etc. **Do not** reimplement these — the from-scratch implementations in Phase 1 exist purely for pedagogy and are never imported by downstream code.
- **scikit-learn.** For `GaussianMixture` fitting and clustering. Don't write EM from scratch.
- **einops.** Standard in mech-interp for residual-stream reshaping.
- **pydantic.** All YAML configs are loaded into pydantic models so typos and type errors fail loudly at startup, not silently 20 minutes into a run.
- **matplotlib.** Plots. Seaborn allowed sparingly.
- **pytest + pytest-cov.** Testing.
- **ruff.** Linting + formatting (replaces black + isort + flake8).
- **mypy.** Static typing, strict mode on `src/`.
- **pre-commit.** Hooks run ruff + mypy on staged files.

## Code conventions

- **Type hints everywhere.** Python 3.11+ syntax: `list[int]`, `dict[str, Any]`, `X | None`. Not `List`, `Dict`, `Optional`.
- **No TODOs in shipped code.** If something is unfinished, raise `NotImplementedError` with a clear message **and** record it in `PROGRESS.md` under "open issues" **and** the file must not be imported by any running path.
- **Fail loudly.** Use `assert` for invariants. Raise specific exceptions for expected error cases. Never `except: pass`. Never bare `except`.
- **No `print()` in library code.** Use `logging` via `ot_steering.utils.logging.get_logger(__name__)`. Scripts and notebooks may print.
- **Docstrings on all public functions.** Google-style: one-line summary, blank line, Args, Returns, Raises.
- **No dead code.** No commented-out blocks "in case we need it." Git history is the safety net.
- **Absolute imports only.** `from ot_steering.ot.sinkhorn import ...`. No `from x import *`.
- **One concept per file** in `src/`. If a file grows past ~300 lines, split it.

## Library use rules

- Use POT for OT after Phase 1.
- Use sklearn's `GaussianMixture` for GMM fitting.
- Use `transformers.AutoModel` + PyTorch forward hooks for activation extraction.
- Use bitsandbytes for 4-bit quantization.
- If you're about to reinvent a wheel: stop, search PyPI, ask. If nothing exists, then implement.

## Testing rules

- Every utility function in `src/ot_steering/` has at least one pytest test in `tests/` mirroring the package layout.
- Tests must be fast (<10s each); use tiny synthetic inputs.
- `pytest -q` must pass before any commit.
- Property tests for OT functions where they apply: `W(p, p) == 0`, symmetry, scale invariance under the right normalization, etc.
- For randomized tests, fix seeds.

## Reproducibility rules

- Every experiment script accepts a config path and a seed.
- Every run writes to `outputs/<run_id>/` where `run_id = <ISO timestamp>_<8-char hash of config>`.
- The exact config used is dumped as `config.yaml` inside the run dir, along with `env.json` (library versions, git commit hash).
- All RNGs (Python, NumPy, PyTorch CPU, PyTorch CUDA, cuDNN deterministic) are seeded via `ot_steering.utils.seed.set_all_seeds`.
- Activations are cached on disk under `data/cache/` with a filename derived from a hash of `(model_id, dataset_id, layer, dtype)`. Never re-extract if cache hits.
- **Figures are regenerated from saved outputs**, never tuned interactively. `scripts/make_figures.py` reads `outputs/<run_id>/` and writes `figures/`.

## Course-writing rules

The course is the heart of the repo. Treat it like an essay collection, not API docs.

- **Voice:** blog-essay, conversational, builds intuition before formalism. Imagine the reader as a curious software engineer with strong Python and ML fundamentals but **zero OT and zero LLM internals knowledge**.
- **Every chapter has:**
  1. A "Why this exists" opener (1–2 paragraphs of motivation, no math).
  2. A worked example that you can hold in your head.
  3. The math, with every symbol defined the first time it appears.
  4. A runnable code section that *imports from `src/`*, not a parallel re-implementation.
  5. A "what we just learned" recap (3–5 bullets).
  6. A "go deeper" reading list (3–5 references).
- **Diagrams beat walls of equations.** Generate figures with matplotlib; commit the generating code.
- **No jargon without a gloss the first time.** "Coupling," "residual stream," "sinkhorn divergence" — all need a one-line definition on first use.
- **No condescension.** The reader is smart, just unfamiliar.
- **Code in the chapter is the same code that runs in the project.** If you find yourself simplifying for the chapter, that's a sign the real code is too complicated — go fix that instead.

## Repo layout

```
ot-concept-transport/
├── README.md
├── CLAUDE.md             # this file
├── PROJECT_PLAN.md       # phases, deliverables, risks
├── PROGRESS.md           # session log, updated every session
├── pyproject.toml
├── .pre-commit-config.yaml
├── .gitignore
├── configs/              # YAML configs, validated by pydantic
│   ├── default.yaml
│   ├── models/
│   ├── datasets/
│   └── experiments/
├── src/ot_steering/      # the library (well-tested, reusable)
│   ├── ot/               # OT solvers (POT wrappers + barycentric)
│   ├── activations/      # model loading, hooks, datasets, caching
│   ├── steering/         # baselines + OT steering + transport
│   ├── eval/             # steering & transport metrics
│   └── utils/            # seed, logging, config, io
├── phases/               # interleaved course + per-phase experiments
│   ├── phase_00_introduction/
│   ├── phase_01_ot_foundations/
│   ├── phase_02_gromov_wasserstein/
│   ├── phase_03_llms_and_steering_baselines/
│   ├── phase_04_intra_model_ot_steering/
│   ├── phase_05_cross_model_gw/
│   ├── phase_06_steering_transport/
│   ├── phase_07_diagnostics/
│   └── phase_08_synthesis/
├── tests/                # mirrors src/ structure exactly
├── scripts/              # CLI entry points for big experiments
├── outputs/              # run artifacts (gitignored)
└── data/                 # dataset & activation cache (gitignored)
```

Each `phases/phase_NN_topic/` folder contains:

```
chapter.md            # the blog-essay
notebook.ipynb        # executable companion (imports from src/)
experiments/          # phase-specific scripts (if any)
figures/              # generated plots
README.md             # 3-sentence summary + how to run
```

## Workflow per session

1. Read `PROGRESS.md` to see where we are.
2. Read the current phase's `chapter.md` and any notes.
3. Make a branch: `phase-NN-shortname`, or `fix/...`, `feat/...`, `docs/...`.
4. Do the work. `pytest -q` must pass.
5. Update `PROGRESS.md`: what got done, what's blocked, what's next.
6. Commit with a Conventional Commits message: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`.
7. **Stop.** Don't merge to `main`; the human reviews and merges.

## What NOT to do

- **Do not train any model from scratch.** We only do inference on pretrained models.
- **Do not load models above ~1.1B params in fp16, or ~3B in 4-bit.** The GPU is 4 GB.
- **Do not pull in a heavy new dependency without justification** in `PROGRESS.md`.
- **Do not expand scope.** If you have an interesting idea that isn't in `PROJECT_PLAN.md`, add it to the "Deferred ideas" section there. Do not act on it this session.
- **Do not apologize for asking clarifying questions.** Ask one specific question and stop.
- **Do not hide failures.** Surface broken tests, write up negative results, document why an approach was abandoned.
- **Do not use MNIST or CIFAR.** Anywhere. For any reason.
- **Do not write a 500-line file when 50 lines and an import would do.**
- **Do not ship code with leftover `print()` statements.**
- **Do not produce code with TODOs.** Either finish it, raise NotImplementedError, or move it back to PROGRESS.md as an open issue.

## When something goes wrong

State, in this order:

1. **What** went wrong (concrete: the error message, the unexpected number, the failed assertion).
2. The **smallest hypothesis** for why.
3. The **minimum experiment** to confirm or refute the hypothesis.
4. The **fix.**

Don't apologize repeatedly. Don't guess randomly. Investigate.
