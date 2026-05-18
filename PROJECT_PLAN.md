# Project plan — Cross-Architecture Concept Vector Transport via Gromov-Wasserstein

## 1. Contribution

We test the hypothesis that the **relational structure of contrastive activation distributions in LLMs is approximately model-agnostic**, and exploit this via Gromov-Wasserstein (GW) optimal transport to map steering vectors and steering maps from one LLM to another with **no paired data** and **no target-side supervision**.

If this works, it provides:

- A practical tool: zero-shot transfer of interpretability artifacts (refusal directions, sentiment vectors, truthfulness directions) across model families.
- A diagnostic: GW transport cost as a predictor of when a concept is *universally encoded* versus *model-idiosyncratic*.
- A conceptual extension of the OT-as-steering line (CHaRS, Abdullaev et al. 2026) from *intra-model* to *inter-model*.

## 2. Hypothesis and framing

CHaRS (Abdullaev et al., March 2026) showed that within a single model, the standard difference-in-means steering vector is a degenerate case of an optimal transport map (specifically, the OT map between two unimodal Gaussians with identical covariance — i.e., a pure translation). They generalized this by modeling source/target activation distributions as Gaussian mixtures and using discrete OT between cluster centers, getting a richer input-dependent steering map via barycentric projection.

Our project extends this **across** models. The intuition: two different LLMs produce activation distributions for the same concept contrast (e.g., harmful-vs-harmless prompts) that live in incompatible coordinate systems. Standard alignment methods (Procrustes, CCA) assume a shared latent geometry up to a linear transform. Gromov-Wasserstein makes the weaker, more realistic assumption: that the *intra-distribution geometry* (pairwise relations between activations) is preserved up to a non-linear correspondence. This is precisely what we'd expect if the concept has the same conceptual shape in both models but lives in different embedding coordinates.

## 3. Related work and the gap we exploit

| Cluster | Representative work | Why it doesn't cover us |
|---|---|---|
| OT for intra-model steering | CHaRS (Abdullaev et al. 2026) | Single-model only; our novelty is cross-model |
| Steering vectors | ActAdd (Turner 2023), CAA (Rimsky 2024), mean-centering (Jorgensen 2024), refusal (Arditi 2024) | All within-model; vector addition |
| Non-identifiability of steering | Venkatesh & Kurapath 2026 | Argues steering vectors have large equivalence classes — motivates structural matching like GW |
| Cross-model representation alignment | Procrustes/CCA (classical), Relative Reps (Moschella 2023), Platonic Reps (Huh 2024) | Use linear maps; do not apply to interpretability artifacts |
| GW for representation alignment | Alvarez-Melis & Jaakkola 2018 (bilingual word embeddings), SCOT (single-cell), Kawakita 2024 (human-vs-LLM color) | None do steering-vector transport |

The specific combination — **concept-vector transport across LLM architectures via GW on contrastive activation distributions** — does not appear in any search performed during scoping. Estimated window before parallel work plausibly appears: 6–9 months.

## 4. Phases

Each phase produces both code (added to `src/` and `tests/`) and a chapter of the course (in `phases/phase_NN_topic/chapter.md`). The course is written in blog-essay voice for readers with **no prior OT or LLM background**.

### Phase 0 — Environment & repo scaffolding
**Goal:** A clean, tested, lintable repo that runs.
**Deliverables:** `pyproject.toml`; source/tests/configs/phases skeleton; pre-commit hooks; `.gitignore`; README; `scripts/verify_env.py`; `phases/phase_00_introduction/chapter.md`; PROGRESS.md.
**Done when:** `pytest -q` passes on seed/logging/config tests; `python scripts/verify_env.py` exits 0 with CUDA detected; pre-commit hooks installed; initial commit landed.

### Phase 1 — Optimal transport foundations
**Goal:** Reader and code both understand discrete OT and Sinkhorn from the ground up.
**Deliverables:** From-scratch implementations of discrete OT (via `scipy.optimize.linprog`) and Sinkhorn algorithm, in `phases/phase_01_ot_foundations/` (pedagogical only, **not** used by downstream code). POT-equivalents called from `src/ot_steering/ot/`. 2D-Gaussian transport demo notebook. Tests. Chapter 1.
**Done when:** Pedagogical implementations match POT to within numerical tolerance on a 2D test; chapter 1 explains OT to someone who has never seen it; all tests pass.

### Phase 2 — Gromov-Wasserstein
**Goal:** Reader and code understand structural matching across incomparable spaces.
**Deliverables:** Thin wrappers in `src/ot_steering/ot/gw.py` over POT's `entropic_gromov_wasserstein` with sensible defaults and pydantic-validated config. Barycentric projection utility (`src/ot_steering/ot/barycentric.py`). Toy demo: GW recovers rotation between two 2D point clouds with no shared coordinate frame. Tests. Chapter 2.
**Done when:** GW correctly recovers the known correspondence on a 2D toy across multiple seeds; barycentric projection is implemented and tested; chapter 2 reads naturally.

### Phase 3 — LLMs, activations, and steering baselines
**Goal:** Build the LLM-side infrastructure and reproduce a known steering result.
**Deliverables:**
- Model loader supporting Pythia-160M, GPT-2-small, TinyLlama-1.1B (4-bit), Qwen2.5-0.5B.
- Activation extractor with PyTorch forward hooks and on-disk cache keyed by hash of `(model_id, dataset_id, layer, dtype)`.
- Contrastive dataset loaders for **sentiment**, **truthfulness**, and **refusal** (Arditi-style benign harmful set).
- Steering baselines: difference-in-means (ActAdd), mean-centering (Jorgensen-style), CAA-style if applicable.
- Steering evaluation harness: success rate, off-target perplexity, MMLU sanity check (small sample).
- Chapter 3.
**Done when:** A known steering result (e.g., sentiment direction on GPT-2-small) is reproduced within reasonable tolerance of published numbers; eval harness produces stable numbers across seeds; chapter 3 walks the reader from "what is a transformer" through "what does it mean to steer one."

### Phase 4 — Intra-model OT steering (CHaRS-style)
**Goal:** Reproduce the CHaRS intra-model OT-steering result. This is our intra-model upper bound and a sanity check on our OT machinery.
**Deliverables:** GMM fitting on contrastive activations; discrete OT between cluster centers; barycentric-projection steering map; integrated into the eval harness from Phase 3. Comparison to Phase 3 baselines across all 3 concepts × at least 2 models. Chapter 4.
**Done when:** CHaRS-style steering is competitive with or beats the difference-in-means baseline on at least one (concept, model) cell; chapter 4 takes the reader from "concept directions" to "concept-conditional maps."

### Phase 5 — Cross-model Gromov-Wasserstein alignment
**Goal:** Run GW between contrastive activation distributions of *two different* LLMs and verify it does something sensible.
**Deliverables:**
- End-to-end script that takes (source model, target model, dataset, layer) and produces a GW coupling.
- Coupling visualization (heatmap, cluster-cluster mapping).
- Sanity checks:
  1. GW between a model and itself recovers near-identity.
  2. GW between layer-N and layer-N+1 of the same model recovers near-identity.
  3. GW between two unrelated random distributions of the same size gives high cost.
- Chapter 5.
**Done when:** All three sanity checks pass; chapter 5 explains why cross-model alignment needs a *structural* distance, not a pointwise one.

### Phase 6 — Cross-model steering transport (the core experiment)
**Goal:** Run the actual experiment. Does GW-transported steering work?
**Deliverables:**
- Full pipeline: extract source steering structure → extract target contrastive activations → GW-align → barycentric-project the source steering signal through the coupling → apply on target → measure.
- Baselines for transport: random direction (chance), Procrustes-aligned source vector, CCA-aligned source vector, target-supervised oracle (the steering vector found directly on the target with full supervision — an *upper bound*, not a baseline to beat).
- Experimental matrix: 3 concepts × 2–3 model pairs × 3 seeds, multiple layers per pair.
- Chapter 6.
**Done when:** All matrix cells have numbers with confidence intervals; baselines are properly run; writeup is honest about what worked and what didn't.

### Phase 7 — Diagnostic analysis
**Goal:** Even if transport works only sometimes, understand *when*.
**Deliverables:**
- Per-(concept, model-pair) breakdown of transport success.
- Correlation analysis: does GW cost predict transfer success? (Scatter plot + Spearman ρ across all matrix cells.)
- Failure-mode analysis: layer effect, sample-size effect, cluster-count effect in the GMM step, GW initialization sensitivity.
- Chapter 7.
**Done when:** We can state in one sentence *when* GW transport works for steering, with evidence.

### Phase 8 — Paper draft and polish
**Goal:** A workshop-quality writeup and a reproducible repo.
**Deliverables:**
- Consolidated paper draft (markdown or LaTeX) reusing course chapters.
- Reproducibility README: clone → install → `make all-figures` regenerates every figure in the paper.
- Chapter 8 ("synthesis") tying the project together.
- Final pass on figures, citations, claims.
**Done when:** A researcher who has never seen the repo can clone, install, and regenerate the figures with one command, and the writeup states the contribution clearly and honestly.

## 5. Risk register

| Risk | Likelihood | Severity | Mitigation |
|---|---|---|---|
| GW gets stuck in bad local minima on real activation distributions | Medium | High | Multiple random restarts; sanity checks on known-correspondence pairs; low-rank GW as fallback |
| Steering doesn't transfer because concepts genuinely are model-idiosyncratic | Medium | Medium | Reframe as a *diagnostic* paper — "GW cost predicts transferability" is publishable on its own |
| Parallel paper from another group during the project window | Medium | High | Move fast; differentiate via the diagnostic angle and the cross-architecture-family scope even if the basic transport idea overlaps |
| 4 GB VRAM insufficient for the model pairs we want to study | Medium | Medium | 4-bit quantization via bitsandbytes; if still tight, drop to smaller models (Pythia / GPT-2 family) |
| Off-target degradation kills transferred steering | Low-Medium | Medium | The GW map preserves geometry better than raw vector transport; measure off-target carefully; report it honestly |
| Refusal / safety steering raises content-policy issues | Low | Low | Use Arditi-style benign harmful-prompt sets that are already published; document carefully in chapter 6 |
| Reviewers ask "why GW instead of [X]?" | High | Low | The diagnostic angle + the structural-vs-pointwise framing answers this; bake it into the writeup from Phase 6 onward |

## 6. Out of scope

- Training any model from scratch.
- Multi-GPU runs.
- Models above ~1.1 B parameters in fp16, or ~3 B in 4-bit.
- Theoretical proofs of identifiability, convergence, or sample complexity.
- Novel OT solvers — we use POT.
- Audio, vision, or multi-modal models — text-only LLMs.
- MNIST or CIFAR experiments.

## 7. Deferred ideas (interesting but not for this project)

- GW on per-attention-head activations rather than residual-stream.
- Using the recovered coupling to study the Platonic Representation Hypothesis directly.
- Multilingual extension: cross-language steering transport within multilingual LLMs.
- Fused Gromov-Wasserstein (uses both features and structure) — could be a follow-up if pure GW underperforms.

## 8. Open issues (move things here when blocked)

*(empty — populate during the project)*
