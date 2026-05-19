# Chapter 8 — Synthesis: what we built, what we learned, what's next

## Why this exists

Seven chapters in, the project has a working pipeline, a positive headline result, an honest negative result, an interlocking codebase, and roughly fifteen thousand words of course material. This final chapter steps back, ties the thread that runs through Chapters 0–7, and is explicit about what's published-quality, what's preliminary, and what was deferred. It's the chapter to read if you came in late and want to know whether to read the rest.

## The thread, in one paragraph

The hypothesis the project tested is that the *relational structure* of contrastive activation distributions in LLMs is approximately model-universal, even though the raw activation coordinates aren't. If true, the right tool to align two models is Gromov–Wasserstein optimal transport, because GW operates on intra-distance matrices and doesn't need a shared coordinate frame. The project built the OT machinery from scratch in Chapter 1, generalised to GW in Chapter 2, stood up the LLM-side infrastructure in Chapter 3, reproduced the CHaRS intra-model construction in Chapter 4, ran cross-model GW sanity checks in Chapter 5, ran the actual cross-model steering transport experiment in Chapter 6 (positive result on two cells), and asked the diagnostic question in Chapter 7 (null at this matrix size). The thread connecting everything is the same set of solvers, the same projection, the same hook API — reused unchanged from one phase to the next.

## Recap of each phase, in 1–2 sentences

- **Chapter 0** — set the question (interpretability artefacts don't transfer; we'd like them to) and the plan (eight phases on a 4 GB GPU).
- **Chapter 1** — optimal transport from sand piles to Sinkhorn. From-scratch implementations against POT for pedagogy; production wrappers in `src/ot_steering/ot/{emd,sinkhorn}.py`.
- **Chapter 2** — Gromov–Wasserstein for distributions in incomparable spaces. Production wrappers + a barycentric-projection utility that becomes a load-bearing piece of every subsequent phase. Rotation-recovery demo verifies the solver.
- **Chapter 3** — LLM-side infrastructure: model loader (Pythia, GPT-2, Qwen-0.5B, TinyLlama 4-bit), residual-stream extractor, activation cache, contrastive datasets (50 pairs each for sentiment / truth / refusal), steering baselines, an eval harness. Reproduced ActAdd on GPT-2 with a +15 % net lift on the sentiment task.
- **Chapter 4** — CHaRS-style intra-model OT steering. The OT-conditional steering map is a *gentler scalpel* than the single ActAdd direction: at matched coefficient, k > 1 keeps off-target perplexity within ~5 % of baseline while k = 1 blows it up 3–5×.
- **Chapter 5** — cross-model GW sanity checks. Cost ordering `self ≈ adjacent ≪ cross-model ≪ random` passes on a 2 pairs × 2 concepts matrix; class preservation through the cross-model coupling is above-chance (61 %). The infrastructure to ask "does GW alignment do something sensible" is in place.
- **Chapter 6** — the core experiment. GW transport beats both random and Procrustes on both (gpt2 ↔ pythia) cells in our matrix and recovers 75–87 % of the target-supervised oracle's lift without target-side concept labels. Off-target perplexity stays close to baseline.
- **Chapter 7** — the diagnostic question. Across a 36-cell sweep, Spearman ρ between GW alignment cost and shift rate is −0.17 with a 95 % CI containing zero. Honest null; the chapter discusses the three most plausible reasons (small N, discrete shift-rate axis, base-LMs-only matrix) and the infrastructure to test it on richer matrices.

## What's published-quality

- **The construction itself.** The transport pipeline (Phases 1, 2, 4, 5 → 6) is well-typed, fully unit-tested, runs on a 4 GB GPU in well under a minute per cell, and uses only POT, scikit-learn, and Hugging Face Transformers. The code base passes strict mypy on 24 source files and `pytest -q` on 104 fast tests plus one slow integration test (Pythia-160M end-to-end via `pytest --run-slow`).
- **The Phase 6 transport result.** Two cells, four methods × three coefficients × three seeds with bootstrap 95 % CIs, off-target perplexity tracked alongside lift. The numbers are modest in absolute terms (GW transport: 13–15 % positive-shift rate vs ≤12 % for both baselines) but consistent with the off-target story and *cleanly beat the baselines we compare to*.
- **Course material.** Eight chapters totalling ~15 000 words, written in blog-essay voice for a reader with strong Python/ML fundamentals but no prior OT or LLM-internals background. Every concept is defined the first time it appears; every code snippet imports from `src/`, not a parallel re-implementation.

## What's preliminary

- **Matrix scope.** Two model pairs, one concept (sentiment), one layer-choice heuristic. The most informative next experiment is a refusal evaluation on a chat-tuned model — TinyLlama-1.1B-Chat at 4-bit fits in the project budget but was outside this round.
- **Evaluation judge.** A hand-curated sentiment lexicon over 20 prompts is fine for the headline transport story but it quantises the shift rate into 5 % steps and was the dominant reason the Phase 7 diagnostic correlation came out null. Swapping in a calibrated LM judge or a small finetuned classifier is the obvious upgrade.
- **The diagnostic null.** Honest negative result, not a strong claim. The wider matrix that would settle the question (Qwen, TinyLlama, multiple concepts) is the natural next step but needs either more compute or a tighter eval split.
- **No comparison to CCA or Relative Representations.** Procrustes is the reference linear-alignment baseline; CCA and Relative Representations are the standard non-OT alternatives and should be added in any follow-up.

## What was deferred

These ideas came up during the project but were intentionally not pursued in this round (see PROJECT_PLAN.md §7):

- GW on per-attention-head activations rather than residual-stream tokens. Worth trying if a single-direction residual-stream steering vector turns out to be too coarse for a richer concept.
- Multilingual transport within a single multilingual model. Same project structure, different axis.
- Fused Gromov–Wasserstein (Vayer et al. 2019) — uses both features and intra-distance structure. The natural extension if pure GW underperforms on a harder concept.
- Using the recovered GW coupling as a probe of the Platonic Representation Hypothesis directly (Huh et al. 2024), e.g. by measuring how cost decays as a function of model scale.

## Two paragraphs on the thread

What I think the project says, beyond the headline numbers, is that the *combination* of well-tested OT building blocks plus a careful structural-vs-coordinate framing yields a cross-model interpretability transfer pipeline that *works*, even if modestly. The construction reused every solver from Phase 1 to Phase 6 unchanged; nothing in `src/ot_steering/ot/` had to be rewritten when the project moved from intra-model to cross-model. That kind of code-reuse is itself an empirical claim — it says the right level of abstraction for the OT machinery is small and stable, and the per-phase additions live above it.

The honest Phase 7 null is the chapter I learned the most from. A clean positive answer would have been a second contribution; a clean negative answer would have been a different second contribution ("GW cost is not a useful diagnostic — here's why and what to use instead"). What we got is an *uncertain* answer at this matrix size, which is the most common outcome in small-scale ML experiments and the hardest one to write about well. The decision to ship the null as-is, with its three honest explanations, was easier than I expected — once the infrastructure was in place, the experiment did its job whether or not the headline number was the one I wanted.

## Reproducing everything

The top-level `README.md` walks through clone → install → `python scripts/make_all_figures.py`. The script runs every phase's `make_figures.py` in chapter order; on the project hardware the full regeneration takes about 45 minutes. The headline experiment (`run_transport.py`) and the diagnostic sweep (`sweep.py`) each take ~10–15 minutes; the rest is fast.

## What's next

Nothing in this codebase, unless and until I or someone else wants to extend it. The Phase 6 result is enough to write up; the Phase 7 null is enough to qualify it; the code is enough to let a reviewer reproduce both with one command.

A reasonable follow-up agenda, in priority order:

1. Refusal on TinyLlama-1.1B-Chat at 4-bit. Phase 4's `compare_baselines.py` already supports a refusal concept; Phase 6's pipeline runs unchanged on a different concept.
2. A continuous-valued eval judge. The lexicon hits its ceiling fast; a finetuned classifier over the same prompts would make Phase 7's diagnostic question answerable.
3. Bigger matrix: Qwen-0.5B and TinyLlama added to the model pairs. The infrastructure handles this — only the per-pair compute budget is the constraint.

That's the project. Thanks for reading.
