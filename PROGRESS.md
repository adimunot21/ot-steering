# PROGRESS

## Current phase: 03 — LLMs, activations, and steering baselines

Phase 2 (Gromov-Wasserstein) shipped. Next session should branch
`phase-03-llms-and-steering-baselines` and start Phase 3.

## Session log

### 2026-05-18 — Phase 2: Gromov-Wasserstein

Completed:

- [x] Production GW wrapper `src/ot_steering/ot/gw.py`:
      `solve_entropic_gw(C1, C2, p, q, cfg, *, rng)` over POT's
      `ot.gromov.entropic_gromov_wasserstein`, returning
      `(coupling, gw_cost)`. `GWConfig` (subclass of `BaseConfig`):
      `reg`, `loss_fun in {"square_loss","kl_loss"}`, `num_iter_max`,
      `stop_threshold`, `num_restart`, `warn_on_no_convergence`.
      Multi-restart logic implemented in the wrapper (POT has no built-in
      restart) — randomised initial couplings projected onto the
      transport polytope via 50 Sinkhorn rescalings, lowest-cost run kept.
      POT's GW solver has no `warn` knob, so the wrapper uses a
      `warnings.catch_warnings` block to honour `warn_on_no_convergence`.
- [x] `src/ot_steering/ot/barycentric.py`:
      `barycentric_project(coupling, target_features, source_marginal=None)`
      → `(n, d_target)`. Handles empty rows (returns zero vector, not NaN)
      and accepts an explicit source marginal for numerically clean
      couplings.
- [x] Tests in `tests/ot/`:
      - `test_gw.py` (7 tests): GW on identical clouds yields a near-
        diagonal coupling; GW recovers a 2D rotation (3 seeds, accuracy
        ≥ 85 %); GW cost symmetric under side-swap; pydantic validation;
        shape mismatch errors.
      - `test_barycentric.py` (5 tests): permutation coupling maps source
        → matched targets exactly; uniform coupling maps every source to
        target mean; explicit marginal overrides inferred; empty row →
        zero (not NaN); shape mismatches raise.
      All 43 tests pass; longest 7.5 s (still within the <10 s budget).
- [x] Demo / figures / notebook in `phases/phase_02_gromov_wasserstein/`:
      - `experiments/demo.py` — `run_rotation_demo()` returns a
        `RotationDemo` dataclass with source X (n=80, 2D),
        target Y = X @ R + small_noise, coupling, recovered permutation,
        accuracy, and a displacement-interpolation trajectory.
      - `experiments/make_figures.py` — renders the four chapter figures
        (`01_two_clouds_no_alignment`, `02_gw_coupling_lines`,
        `03_coupling_heatmap`, `04_recovered_correspondence_vs_truth`).
      - `notebook.ipynb` — executes top-to-bottom cleanly via
        `jupyter nbconvert --execute`.
- [x] `phases/phase_02_gromov_wasserstein/chapter.md` — full chapter
      (~2 100 words) and `README.md`.
- [x] Ruff, ruff-format, mypy strict on `src/`, `pytest -q` all pass.

Notes / decisions:

- POT's `entropic_gromov_wasserstein` underflows when `epsilon < ~1e-3`
  (inner Sinkhorn is non-log-domain). Chapter and config defaults
  document the safe range; the rotation tests use `reg=0.01`.
- Multi-restart wrapped *inside* `solve_entropic_gw` so callers don't
  have to write the restart loop themselves. The lowest-cost run wins.
- We had earlier guesses of `num_restart` as a POT-exposed parameter; it
  isn't — we own it in the wrapper.

### 2026-05-18 — Phase 1: OT foundations

Completed:

- [x] Pedagogical implementations in `phases/phase_01_ot_foundations/`:
      `scratch_ot.py` (discrete OT via `scipy.optimize.linprog`) and
      `scratch_sinkhorn.py` (log-domain Sinkhorn). Each is runnable standalone
      with a sanity-check `__main__`.
- [x] Production POT wrappers in `src/ot_steering/ot/`:
      `emd.py` (`solve_emd` + `EMDConfig`) and `sinkhorn.py`
      (`solve_sinkhorn` + `SinkhornConfig`, defaults to `method='sinkhorn_log'`).
      Configs subclass `BaseConfig` (frozen, extra=forbid, validated at
      construction).
- [x] Tests in `tests/ot/`:
      - `test_emd.py`: W(p, p)==0, symmetry, scale invariance of plan
        under cost rescaling, agreement with scratch implementation on
        a 5×5 problem, pydantic validation, shape mismatch error.
      - `test_sinkhorn.py`: marginals preserved, convergence to EMD as
        reg shrinks, log-domain stability at reg=1e-3, agreement with
        scratch implementation, pydantic validation, shape mismatch error.
      All 31 tests pass; longest test 7.5s (within the <10s budget).
- [x] Demo / figures in `phases/phase_01_ot_foundations/`:
      - `experiments/demo.py` — shared `run_demo()` returning a
        `GaussianTransportDemo` dataclass with source/target clouds,
        EMD plan, Sinkhorn plans across a reg sweep, and a displacement-
        interpolation trajectory.
      - `experiments/make_figures.py` — renders the four chapter figures
        to `figures/*.png`.
      - `notebook.ipynb` — executable companion that imports the same
        shared demo (no parallel re-implementation). Executes top-to-bottom
        cleanly via `jupyter nbconvert --execute`.
- [x] `phases/phase_01_ot_foundations/chapter.md` — full chapter
      (~1 800 words): worked example → Kantorovich LP → entropic
      regularisation → Sinkhorn algorithm in log-space → barycentric map and
      displacement interpolation → POT as the production solver.
- [x] `phases/phase_01_ot_foundations/README.md`.
- [x] Ruff, mypy strict on `src/`, and `pytest -q` all pass on the
      Phase 1 branch.

Notes / decisions:

- Tests legitimately import the `scratch_*` files via `importlib.util` for
  agreement testing only — the project's import path still excludes
  `phases/`, so no runtime code can accidentally depend on pedagogical
  implementations.
- Two-implementation agreement on Sinkhorn at reg=0.1 holds to atol≈1e-4
  on the plan; both solvers converge to the same fixed point but apply
  slightly different last-step normalisations. Documented inline.
- The figures script and the notebook share a single `demo.py` module to
  keep them in sync.

### 2026-05-18 — Phase 0: repository scaffolding & environment setup

Completed:

- [x] Verified Python ≥3.11 (system has 3.12.3) and CUDA visible
      (`nvidia-smi` shows GTX 1650, 4 GB, driver 595.58.03, CUDA 13.2).
- [x] Created directory tree per `CLAUDE.md` (`src/ot_steering/{ot,activations,steering,eval,utils}`,
      mirrored `tests/`, `phases/phase_00…phase_08`, `configs/{models,datasets,experiments}`,
      `scripts/`, `outputs/`, `data/cache/`).
- [x] Wrote `pyproject.toml` (hatchling, Python 3.11+, runtime + dev deps as specified in CLAUDE.md).
- [x] Configured ruff (line-length 100, target py311, default + UP + I + B + SIM + N),
      mypy (strict on `src/`), pytest (`tests/` root, `pythonpath=src`).
- [x] Wrote `.pre-commit-config.yaml` (ruff + mypy + standard hygiene hooks).
- [x] Wrote thorough `.gitignore`.
- [x] Source skeleton: empty `__init__.py` in every subpackage of `src/ot_steering/`.
- [x] `src/ot_steering/utils/seed.py` (`set_all_seeds`) — Python random, NumPy, torch CPU/CUDA,
      cuDNN deterministic, `PYTHONHASHSEED`.
- [x] `src/ot_steering/utils/logging.py` (`get_logger`) — stderr handler, ISO-8601 timestamps,
      `OT_STEERING_LOG_LEVEL` env-var override, no propagation.
- [x] `src/ot_steering/utils/config.py` (`BaseConfig`) — pydantic v2, `extra="forbid"`,
      `frozen=True`, `validate_assignment=True`.
- [x] Tests in `tests/utils/` for each of the three utilities.
- [x] `scripts/verify_env.py` — prints versions, CUDA info, runs a GPU matmul,
      returns non-zero on failure.
- [x] `README.md` (pitch, install, quickstart, repo map).
- [x] `phases/phase_00_introduction/chapter.md` (≈2 000 words, blog-essay voice, no OT/LLM
      prerequisites) plus `phases/phase_00_introduction/README.md`.
- [x] Installed deps into `.venv` and ran `pre-commit install`.
- [x] `python scripts/verify_env.py` returned 0.
- [x] `pytest -q` passes.

Notes / decisions:

- Used the system Python 3.12.3 for the venv. `pyproject.toml` requires `>=3.11`; 3.12
  is forward-compatible. Mypy `python_version = "3.11"` so we don't accidentally rely
  on 3.12-only syntax.
- Torch is installed against CUDA 12.1 wheels even though the local driver is CUDA 13.2;
  the 12.1 runtime is forward-compatible with newer drivers (PyTorch policy).
- POT-equivalent OT solvers will land in `src/ot_steering/ot/` in Phase 1 alongside
  the from-scratch pedagogical implementations in `phases/phase_01_ot_foundations/`.

## Next session (Phase 3) — LLMs, activations, and steering baselines

Goal: build the LLM-side infrastructure (model loading, activation extraction
via PyTorch hooks, on-disk cache) and reproduce a known steering result.

Deliverables:

- Model loader supporting Pythia-160M, GPT-2-small, TinyLlama-1.1B (4-bit),
  Qwen2.5-0.5B.
- Activation extractor with PyTorch forward hooks and on-disk cache keyed
  by hash of `(model_id, dataset_id, layer, dtype)`.
- Contrastive dataset loaders for sentiment, truthfulness, and refusal
  (Arditi-style benign harmful set).
- Steering baselines: difference-in-means (ActAdd),
  mean-centring (Jorgensen-style), CAA-style if applicable.
- Steering evaluation harness: success rate, off-target perplexity,
  MMLU sanity check (small sample).
- Chapter 3.

Done when: a known steering result (e.g., sentiment direction on GPT-2-small)
is reproduced within reasonable tolerance of published numbers; eval harness
produces stable numbers across seeds; chapter 3 walks the reader from
"what is a transformer" through "what does it mean to steer one."

## Open issues

*(none)*
