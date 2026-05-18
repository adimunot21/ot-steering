# PROGRESS

## Current phase: 01 — OT foundations

The repository scaffolding is complete; the next session should branch
`phase-01-ot-foundations` and start Phase 1.

## Session log

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

## Next session (Phase 1) — OT foundations

Goal: reader and code both understand discrete OT and Sinkhorn from the ground up.

Deliverables:

- From-scratch discrete OT via `scipy.optimize.linprog` (pedagogical only).
- From-scratch Sinkhorn (pedagogical only).
- POT-equivalents called from `src/ot_steering/ot/`.
- 2D-Gaussian transport demo notebook.
- Property tests (W(p, p) == 0, symmetry, scale invariance) for our POT wrappers.
- `phases/phase_01_ot_foundations/chapter.md`.

Done when: pedagogical implementations match POT within tolerance on a 2D test;
chapter 1 explains OT to someone who has never seen it; all tests pass.

## Open issues

*(none)*
