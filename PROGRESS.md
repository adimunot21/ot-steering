# PROGRESS

## Current phase: 07 — Diagnostic analysis

Phase 6 (cross-model steering transport — the core experiment) shipped.
GW transport beat both random and Procrustes baselines on both cells of
the experimental matrix, recovering 75–87 % of the target-supervised
oracle's lift without target-side concept labels. Next session should
branch `phase-07-diagnostics` from `phase-06-steering-transport` and
start Phase 7.

## Session log

### 2026-05-18 — Phase 6: Cross-model steering transport (core experiment)

Completed:

- [x] `src/ot_steering/steering/transport.py`:
      - `SteeringTransportConfig` (pydantic; `n_components`,
        `distance_metric`, nested `gmm_cfg/gw_cfg`, `assignment` for
        the runtime hook).
      - `TransportedSteeringMap` frozen dataclass holding the source
        CHaRS map, two cross-model alignments (POS↔POS, NEG↔NEG), the
        target NEG GMM, both target centroid sets, and the per-B-NEG-
        cluster transported displacements.
      - `build_transport(src_pos, src_neg, tgt_pos, tgt_neg, cfg, *, rng)`
        — chains Phase 4 CHaRS + two Phase 5 cross-model GW solves +
        the Phase 2 barycentric-projection chain to produce the per-B-NEG-
        cluster displacement vector field. Target POS+NEG activations
        are used only for clustering — never to compute concept-axis
        direction (otherwise it would defeat the no-target-supervision
        promise).
      - `add_transported_steering_hook(block, transported_map, coefficient)`
        — forward-pre-hook context manager mirroring the API of
        `add_ot_steering_hook` from Phase 4.
- [x] `src/ot_steering/steering/transport_baselines.py`:
      - `random_direction(d_target, seed)` — unit-norm random vector.
      - `target_supervised_oracle(target_pos, target_neg)` — upper bound,
        unit-norm difference-of-means computed directly on target.
      - `procrustes_aligned(source_direction, source_centers, target_centers)`
        — orthogonal Procrustes on matched centroid clouds, with zero-
        padding to handle different dimensions. Pairing is by Hungarian
        on a centroid cost matrix.
- [x] Tests in `tests/steering/test_transport.py` (12 new): planted-
      direction transport on synthetic toy, dataclass frozen-ness,
      shape mismatches, hook behaviour (planted clusters, hard
      assignment, identity-Linear), hook removal, pydantic validation,
      target-oracle matches diff-of-means, random-direction unit-norm
      determinism, Procrustes recovers a known 2D rotation, Procrustes
      handles different dims, Procrustes shape rejection. All 12 pass
      in ~6 s. Full suite: 98 pass, 1 skipped.
- [x] Core experiment
      `phases/phase_06_steering_transport/experiments/run_transport.py`:
      2 cells × 4 methods × 3 coefficients × 3 seeds, bootstrap 95 % CIs.
      Writes `outputs/<run_id>/{config.json, run_transport.json}`.
- [x] Demo / figures / notebook in `phases/phase_06_steering_transport/`:
      - `experiments/demo.py` — `run_transport_demo()` (single-seed,
        no-CI variant for the notebook).
      - `experiments/make_figures.py` — pipeline-diagram figure plus
        three result figures (method-comparison bars, lift-vs-coefficient
        with 95 % CI ribbons, off-target-perplexity-vs-coefficient on
        log-scale).
      - `notebook.ipynb` — runs the demo cell in ~3 minutes.
- [x] `phases/phase_06_steering_transport/chapter.md` (~1 900 words) and
      README.md.
- [x] Ruff + ruff-format + mypy strict on `src/` + `pytest -q` all pass.

Headline (best coefficient per method, sentiment, layer-6):

  gpt2 → pythia-160m :  random 12 % | Procrustes 12 % | **GW 15 %** | oracle 20 %
  pythia-160m → gpt2 :  random  8 % | Procrustes  5 % | **GW 13 %** | oracle 15 %

GW transport beats both random and Procrustes on both cells. GW recovers
75 % of the target-oracle's lift on cell 1, 87 % on cell 2 — without
ever seeing target-side concept labels.

Notes / decisions:

- **Per-cluster displacements are unit-normalised** in `run_transport.py`
  before the coefficient sweep — keeps the methods comparable on the
  same coefficient scale. The raw transported displacements have very
  different magnitudes from cell to cell (mean norm 86–190 in the
  experiment log).
- **GMM-centroid pairing for Procrustes uses Hungarian on a centroid
  cost matrix**, not the GW coupling. Hungarian gives a clean 1-1
  pairing; the GW coupling's argmax may not be 1-1.
- **Procrustes underperforms on one cell.** On `pythia → gpt2` Procrustes
  achieves only 5 % shift, worse than random 8 %. This is consistent
  with the broader cross-model-alignment literature: linear alignment
  alone is fragile when the two spaces have unrelated coordinate frames.
  GW's structural matching is genuinely different in kind, not just
  degree.
- **Off-target perplexity comparison.** All four methods stay near
  baseline at coef=1; at higher coefficients the random direction blows
  up ppl most because it has no concept-conditional structure to limit
  its effect on neutral text. GW transport and the oracle stay close
  to baseline — same gentler-scalpel finding as Phase 4.
- **Wide CIs.** Three seeds is fine for an exploratory result but not
  for confident claims. Phase 7 should add more seeds or more layer
  cells.

### 2026-05-18 — Phase 5: Cross-model Gromov-Wasserstein alignment

Completed:

- [x] `src/ot_steering/steering/cross_model_align.py`:
      - `CrossModelGWConfig` (pydantic; `n_components_source`,
        `n_components_target`, `distance_metric in {euclidean, cosine}`,
        `normalize_distances=True` by default, nested
        `gmm_cfg/gw_cfg`). Validator catches partial GMM spec.
      - `CrossModelAlignment` frozen dataclass with source/target
        centres, intra-distance matrices, marginals, coupling, GW cost,
        metric.
      - `cross_model_gw_coupling(source, target, *, cfg, rng)` — either
        uses raw activations (uniform marginals) or fits per-side GMMs
        (using `fit_gmm` from `ot_steering.steering.ot_steering`) and
        passes centroids+weights to `solve_entropic_gw` from
        `ot_steering.ot.gw`. Normalises each intra-distance matrix to
        max=1 by default (critical for POT's entropic GW on real LLM
        activations).
- [x] Tests in `tests/steering/test_cross_model_align.py` (8 new): self-
      pair identity, marginal preservation, cosine vs Euclidean distance,
      partial-GMM-spec rejection, GMM-centroid path shapes, shape
      mismatch errors, GW cost ordering (random > self), dataclass
      frozen-ness. All 8 pass in ~12s. Full suite: 86 pass, 1 skipped.
- [x] Sanity-check experiment
      `phases/phase_05_cross_model_gw/experiments/sanity_checks.py` —
      runs the four cases (self / adjacent / random / cross-model) for
      Pythia-160M → GPT-2 on sentiment AND refusal, writes
      `outputs/<run_id>/{config.json, sanity_checks.json}`.
- [x] Demo / figures / notebook in
      `phases/phase_05_cross_model_gw/`:
      - `experiments/demo.py` — `run_cross_model_demo()` returns a
        `CrossModelGWDemo` dataclass with both models' activations,
        per-side PCAs, the cross-model coupling, the four sanity-check
        GW costs, and a class-confusion matrix (does GW preserve
        positive→positive and negative→negative across models?).
      - `experiments/make_figures.py` — four chapter figures:
        side-by-side PCA of the two residual streams, GW coupling
        heatmap, sanity-check cost bar chart, class-preservation
        confusion matrix.
      - `notebook.ipynb` — runs on Pythia + GPT-2 in ~2 minutes; loads
        each model in turn (releases the previous one before loading
        the next) to stay under 4 GB VRAM.
- [x] `phases/phase_05_cross_model_gw/chapter.md` (~1 950 words) and
      README.md.
- [x] Ruff + ruff-format + mypy strict on `src/` + `pytest -q` all pass.

Notes / decisions:

- **Headline numbers (Pythia-160M → GPT-2, sentiment, layer-6):**
    - self-pair       GW cost = 0.0000   diag-mass ≈ 0.4
    - adjacent layer  GW cost = 0.0000   diag-mass ≈ 0.4
    - **cross-model** GW cost = 0.0205   diag-mass ≈ 0.35
    - **random noise** GW cost = 0.1129  diag-mass ≈ 0.4 (~5× higher)
  Cross-model is 5× higher than self-pair (real difference) but 5×
  lower than random (still much better than chance).
- **Class preservation = 61 %** in the row-normalised confusion matrix
  vs. 50 % chance — GW's coupling tends to send positive-source
  clusters to positive-target clusters. Above-chance is the evidence
  Phase 6 needs.
- **Distance-matrix normalisation.** Real LLM intra-distance matrices
  have entries in the hundreds (especially with outlier "attention
  sink" clusters); POT's entropic GW collapses to gw_cost=0 with
  zero-diagonal coupling at reg=0.05 on such matrices. Added a
  `normalize_distances=True` config flag (default on) that rescales
  each matrix by its max so entries lie in `[0, 1]`. With this and
  reg=0.01 the solver behaves sensibly.
- **Tests for GW coupling identity required `reg=0.01`** instead of
  the project-wide GW default of 0.05; at 0.05 the entropic smear on
  normalised distance matrices spreads the self-pair coupling enough
  to fail a >95 % diagonal-mass assertion.

### 2026-05-18 — Phase 4: Intra-model OT steering (CHaRS-style)

Completed:

- [x] `src/ot_steering/steering/ot_steering.py`:
      - `GMMConfig` (pydantic, `n_components/covariance_type/n_init/`
        `max_iter/reg_covar/seed`).
      - `fit_gmm(activations, cfg)` — wraps sklearn's
        `GaussianMixture`; rejects non-2D and too-few-samples.
      - `OTSteeringMap` frozen dataclass holding both fitted GMMs, the
        centroids on each side, the OT coupling, the barycentric targets,
        and the per-cluster displacement vectors.
      - `build_ot_steering_map(positive_acts, negative_acts, ...)` —
        fits per-class GMMs, solves squared-Euclidean EMD between the
        centroids via `src/ot_steering/ot/emd.py`, barycentrically
        projects the target centroids through the coupling via
        `src/ot_steering/ot/barycentric.py`, returns the
        `OTSteeringMap`. **At k=1 reproduces difference-of-means
        exactly** (tested: agreement to ~1e-7).
      - `add_ot_steering_hook(block, steering_map, coefficient)` —
        forward-pre-hook context manager mirroring the API of
        `add_residual_steering_hook` so the same generation code can
        switch between baselines and OT steering.
      - Soft/hard cluster assignment (config-selectable).
- [x] Tests in `tests/steering/test_ot_steering.py` (9 new): synthetic
      2-component recovery, shape rejections, k=1 → diff-of-means
      fallback, k=4 coupling marginal consistency, per-token displacement
      hook (with a planted 2D-toy verifying the right cluster gets the
      right direction), hook removal on context exit, pydantic validation,
      dataclass frozen-ness. All 9 pass in ~5 s.
- [x] Headline comparison
      `phases/phase_04_intra_model_ot_steering/experiments/compare_baselines.py`
      sweeps {gpt2, pythia-160m} × {sentiment, refusal} × k in {1,2,4,8}
      × coefficients {0.5, 1, 2}, writes `outputs/<run_id>/`
      `{config.json, compare_baselines.json}`. Cells that work:
      pythia/sentiment k=1 coef=1 → 45 % shift, k=2 coef=2 → 20 % shift.
      gpt2/refusal cell is structurally meaningless (base GPT-2 is not
      chat-tuned and never refuses anything) — flagged in the chapter.
- [x] Demo / figures / notebook in
      `phases/phase_04_intra_model_ot_steering/`:
      - `experiments/demo.py` — `run_charsy_demo()` returns a
        `CHaRSDemo` dataclass with GMM fits, OT coupling, per-coef
        lift, and per-coef off-target ppl.
      - `experiments/make_figures.py` — four chapter figures
        (`01_gmm_fits`, `02_cluster_couplings`,
        `03_steering_lift_vs_k`, `04_offtarget_vs_k`).
      - `notebook.ipynb` — executes top-to-bottom on Pythia-160M.
- [x] `phases/phase_04_intra_model_ot_steering/chapter.md` — full
      chapter (~1 850 words). Worked-example sketch of input-conditional
      steering, GMM-OT-barycentric pipeline, the *gentler-scalpel*
      empirical finding (k>1 keeps off-target ppl close to baseline at
      matched coefficient at the cost of smaller raw lift). README too.
- [x] Ruff + ruff-format + mypy strict on `src/` + `pytest -q` all pass.

Notes / decisions:

- **Honest headline result.** k=1 wins on raw lift across every (model,
  concept) cell — the higher-k variants didn't beat baseline in
  *positive-shift rate*. The k>1 win shows up on the *off-target* curve:
  at matched coefficient, k=2/k=4 keep neutral-text perplexity within
  ~5 % of baseline while k=1 blows it up 3–5×. That trade-off is the
  chapter's headline.
- **Diagonal covariance for the GMM** (`covariance_type="diag"`) by
  default — full covariance in 768-dim with only 30 samples per class
  is wildly over-parameterised.
- **Per-token responsibilities** at the hook computed on CPU
  (`predict_proba` → numpy) and copied back to the GPU; the per-batch
  cost is negligible for tiny models. For larger models in Phase 6 we
  may want to push the responsibilities to GPU.
- **k=1 fallback verified by test.** `test_k1_steering_map_falls_back_`
  `to_difference_in_means` asserts the displacements match
  `difference_in_means` to atol=1e-5 — non-trivial because the path
  goes through GMM EM + EMD + barycentric projection.

### 2026-05-18 — Phase 3: LLMs, activations, and steering baselines

Completed:

- [x] `scripts/inspect_models.py` — one-shot inspection of all four target
      model families (Pythia-160M, GPT-2-small, Qwen2.5-0.5B, TinyLlama 4-bit).
      Confirmed three distinct block-attribute paths
      (`gpt_neox.layers`, `transformer.h`, `model.layers`), VRAM use after
      load, tokenizer pad/eos quirks, and the residual-stream tensor shape.
      Written before any extraction code per CLAUDE.md's
      "inspect before integrating" rule.
- [x] `src/ot_steering/activations/model_loader.py`:
      `load_model(cfg)` + `ModelLoaderConfig`. Handles GPT-2's missing
      pad_token (aliased to eos), uses bitsandbytes NF4 + double-quant
      for 4-bit, logs VRAM delta. Inspection-table embedded in the
      module docstring.
- [x] `src/ot_steering/activations/extractor.py`:
      `extract_residual_stream(...)` reads `hidden_states[k]` from
      `output_hidden_states=True` (no manual hooks needed). Supports
      `position='last_token'|'all'|int`, batched, CPU-offloads per batch.
      Companion `resolve_blocks(model)` does family-aware block lookup
      for the steering baselines and for Phase 4+'s OT hooks.
- [x] `src/ot_steering/activations/cache.py`:
      `cache_key(...)` (readable prefix + 8-char blake2b hash) and
      `load_or_extract(cache_dir, key, extractor_fn)` (atomic write via
      `.pt.tmp` → rename).
- [x] `src/ot_steering/activations/datasets.py` + three YAML files in
      `configs/datasets/` (sentiment, truthfulness, refusal — 50 hand-
      curated pairs each). Each dataset has a pydantic schema; typos in
      the YAML fail at load.
- [x] `src/ot_steering/steering/baselines.py`:
      `difference_in_means`, `mean_centered_steering`,
      `add_residual_steering_hook` (context manager), and
      `apply_steering_vector` (generation under steering, family-agnostic
      via `block_resolver`).
- [x] `src/ot_steering/eval/steering_eval.py`:
      `steering_success_rate` (lexicon judge for sentiment, refusal-phrase
      detector for refusal, explicit no-op for truthfulness with a warning)
      and `off_target_perplexity` (with optional steering injection during
      the forward pass for the on/off comparison).
- [x] Tests in `tests/activations/`, `tests/steering/`, `tests/eval/`
      (16 new): family-aware `resolve_blocks` stub-model tests, cache
      hit/miss/atomic-write, dataset schema validation, steering math on
      synthetic activations, hook context-manager add/remove, lexicon
      judge cases. Plus a `@pytest.mark.slow` integration test that
      actually loads Pythia-160M and runs `extract_residual_stream`
      (passes with `pytest --run-slow`, ~9s).
- [x] `tests/conftest.py` registers the `slow` marker and the
      `--run-slow` opt-in (default `pytest -q` skips heavy tests).
- [x] Headline experiment:
      `phases/phase_03_llms_and_steering_baselines/experiments/reproduce_sentiment.py`
      runs GPT-2-small, builds the layer-6 difference-in-means direction
      from 30 sentiment pairs, applies it (unit-normalised, coefficient
      +6) to the 20-pair eval split, and reports a **+15 % net lift** in
      positive-shift rate vs. the unsteered baseline. Writes
      `outputs/<run_id>/{reproduce_sentiment.log, config.json, metrics.json}`.
- [x] Demo / figures / notebook in
      `phases/phase_03_llms_and_steering_baselines/`:
      - `experiments/demo.py` — shared `run_sentiment_demo()` returning a
        `SentimentDemo` dataclass.
      - `experiments/make_figures.py` — four chapter figures:
        activation distribution (PCA), steering vector in PCA, success
        rate vs. coefficient, off-target perplexity vs. coefficient.
      - `notebook.ipynb` — runs Pythia-160M end-to-end via the same
        shared demo (~30 s on the 4 GB GPU); executes cleanly via
        `jupyter nbconvert --execute`.
- [x] `phases/phase_03_llms_and_steering_baselines/chapter.md` —
      transformers crash course, residual-stream framing, contrastive
      pairs, the difference-of-means baseline and its OT interpretation
      (the bridge from Chapter 1), the eval harness, honest discussion
      of where steering works (refusal robust, sentiment middling,
      truthfulness fragile). README.md too.
- [x] Ruff, ruff-format, mypy strict on `src/`, `pytest -q` (default,
      skipping slow tests) all pass.

Notes / decisions:

- **Direction normalisation.** Initial reproduce-sentiment run with the
  raw `mean(pos) - mean(neg)` direction at coefficient 6 produced gibberish
  (off-target perplexity 100×) because the unscaled direction had norm
  ~22 vs. activation norm ~47. Switched both the demo and
  `reproduce_sentiment.py` to a unit-normalised direction so coefficient
  is interpretable as "this many activation-norm units of perturbation".
- **Inspection-first approach paid off.** The block-attribute path
  differs across all three of the families we touch
  (`gpt_neox.layers`, `transformer.h`, `model.layers`); without running
  `inspect_models.py` first I'd have hardcoded one and bricked the
  others. `resolve_blocks` tries each candidate in order.
- **Sentiment success-rate metric.** The original
  `steering_success_rate` compared `class_A_with_steering` vs
  `class_B_with_steering`; this is the wrong question for ActAdd-style
  steering (aggressive coefficients push both classes to the same place).
  `reproduce_sentiment.py` and the demo compare *baseline vs. steered*
  on the same negative-class prompts.
- **Pythia-160M is small.** The notebook runs on Pythia for speed, but
  Pythia-160M's sentiment direction at layer 6 lifts only modestly
  (~15 % positive shifts) and the curves are noisier than the GPT-2-small
  numbers reported in the chapter. Both are correct; bigger models
  give cleaner steering.
- **Warning filters.** Added `ignore::FutureWarning:bitsandbytes.*` and
  `ignore::FutureWarning:transformers.*` to the pytest filterwarnings
  block; the `transformers` API has loud warnings about a `torch_dtype`
  rename that we cannot suppress at the call site.

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

## Next session (Phase 7) — Diagnostic analysis

Goal: even if transport works only sometimes, understand *when*. Three
questions, with Phase 6's pipeline as the probe:

Deliverables:

- Per-(concept, model-pair, layer) breakdown of transport success.
- Correlation analysis: does GW cost predict transfer success?
  (Scatter plot + Spearman ρ across all matrix cells.)
- Failure-mode analysis: layer effect, sample-size effect, cluster-
  count effect, GW initialisation sensitivity.
- `phases/phase_07_diagnostics/chapter.md`.

Done when: we can state in one sentence *when* GW transport works for
steering, with evidence.

## Open issues

*(none)*
