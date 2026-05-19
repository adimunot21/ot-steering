# Cross-Architecture Concept-Vector Transport via Gromov–Wasserstein Optimal Transport

**Aditya Munot** (`adimunot21@gmail.com`)

---

## Abstract

Mechanistic-interpretability artefacts — steering vectors, refusal directions, sentiment axes — are typically discovered inside one language model and expressed in that model's residual-stream coordinates. They do not transfer to other models, even when the underlying behaviour they encode plausibly should. We test the hypothesis that the *relational structure* of contrastive activation distributions is approximately model-universal, and exploit it via **Gromov–Wasserstein (GW) optimal transport** to push a steering signal from one LLM to another with no paired data and no target-side concept labels. The construction chains CHaRS-style intra-model steering (Gaussian-mixture cluster centroids + EMD + barycentric projection) with two cross-model GW couplings on normalised intra-distance matrices, then a barycentric chain that lifts source-side steering targets into target space. On a Pythia-160M ↔ GPT-2-small matrix with sentiment as the target concept, GW transport beats both random and Procrustes baselines on both transport directions and recovers 75–87 % of the target-supervised oracle's lift while keeping off-target perplexity near baseline. We additionally test whether GW alignment cost *predicts* transfer success across a 36-cell sweep (Spearman ρ with bootstrap CI); the answer in this matrix is no (ρ = −0.17, 95 % CI [−0.45, +0.16]), and we discuss why the diagnostic question needs a wider matrix and a continuous-valued judge to give a decisive answer.

## 1. Introduction

Recent work in mechanistic interpretability has produced clean, surgical interventions — Arditi et al. (2024) showed that a single residual-stream direction mediates refusal in chat-tuned models; Turner et al. (2023) showed that arithmetic on activation vectors can steer generations; Rimsky et al. (2024) generalised the construction with contrastive activation pairs. Every one of these artefacts is expressed in the *coordinates of the model it was discovered on*. A "refusal direction" found in Llama-3 is a 4096-dim vector; a "refusal direction" found in Qwen-2 is a 896-dim vector; there is no reason to think they share an axis even when both models exhibit broadly the same refusal behaviour.

The cost of this coordinate-locality is high. Every new model requires the whole interpretability discovery pipeline to be re-run. The benefit of breaking it would also be high: if the steering vector you found on a chat-tuned 1B model could be transported to a 3B model without re-doing the contrastive-prompt curation, the whole effort would amortise across the family.

Existing cross-model alignment methods — Procrustes, CCA, relative-representations (Moschella et al. 2023) — assume the two spaces are comparable up to a linear (or low-rank) map. That assumption is plausible for, say, two views of the same data, but it is much stronger than the situation between two LLMs warrants. The right tool is one that aligns distributions using only *intra-distribution* geometry: **Gromov–Wasserstein** (Mémoli 2011), which makes no assumption about a shared coordinate frame at all.

Our contribution:

1. **A working pipeline** for cross-model steering transport that combines CHaRS (Abdullaev et al. 2026), entropic GW (Peyré, Cuturi & Solomon 2016), and barycentric projection into a single end-to-end recipe runnable on a 4 GB GPU.
2. **A demonstration** that GW transport beats the random and Procrustes baselines on a Pythia-160M ↔ GPT-2-small matrix and recovers most of a target-supervised oracle's lift, without paired data or target-side concept labels.
3. **A honest negative result** on the diagnostic question of whether GW alignment cost predicts transfer success in this matrix (it does not, at this resolution), with infrastructure ready for richer settings.

## 2. Background

**Discrete optimal transport.** Given two non-negative histograms $a \in \mathbb{R}^n$ and $b \in \mathbb{R}^m$ with equal mass and a cost matrix $M \in \mathbb{R}^{n \times m}$, the Kantorovich problem asks for a coupling $P \in \mathbb{R}^{n \times m}_{\geq 0}$ with row sums $a$ and column sums $b$ minimising $\sum_{ij} P_{ij} M_{ij}$. It is a linear program; for small $n, m$ POT's network-flow solver gives the exact answer. Adding $-\varepsilon H(P)$ where $H$ is Shannon entropy yields the strictly convex entropic OT problem with the closed-form solution $P_{ij} = u_i \exp(-M_{ij}/\varepsilon) v_j$ for scaling vectors $(u, v)$ found via Sinkhorn iteration.

**Gromov–Wasserstein.** When the source and target distributions live in incomparable spaces (different dimensions, no shared frame), no pointwise cost is well-defined. GW replaces it with intra-distance matrices $C^1 \in \mathbb{R}^{n \times n}$ and $C^2 \in \mathbb{R}^{m \times m}$ and minimises $\sum_{ijkl} |C^1_{ik} - C^2_{jl}|^2 P_{ij} P_{kl}$ over couplings. The objective is quadratic in $P$, so the problem is non-convex; entropic regularisation plus projected-gradient descent (Peyré, Cuturi & Solomon 2016) gives a practical solver that POT exposes as `ot.gromov.entropic_gromov_wasserstein`. Random restarts are essential because the objective has local minima.

**Barycentric projection.** Given a coupling $P$ between source and target distributions and target features $Y$, the barycentric image of source point $i$ is $\hat{T}(i) = (1/p_i) \sum_j P_{ij} Y_j$. It is the canonical map induced by a many-to-many coupling.

**Difference-of-means as degenerate OT.** Abdullaev et al. (2026) observed that the standard ActAdd steering vector $\mu_{\text{pos}} - \mu_{\text{neg}}$ is exactly the OT map between two Gaussians with equal covariance. Their CHaRS construction generalises this by fitting a Gaussian mixture per class, solving discrete OT between cluster centroids, and barycentrically projecting to obtain an input-conditional steering map.

## 3. Method

Let $A$ be the source model, $B$ the target, with residual-stream dimensions $d_A$ and $d_B$. We extract last-token activations on $n$ contrastive prompts (positive vs. negative class) at a chosen layer for each model.

**Step 1: Source CHaRS.** Fit per-class GMMs with $k$ components on the source-side activations. Solve discrete OT (squared-Euclidean cost) between the NEG and POS centroids; barycentric-project the POS centroids through the coupling to obtain per-source-NEG-cluster *target images* $\hat{\mu}_i^A \in \mathbb{R}^{d_A}$.

**Step 2: Cross-model GW alignments.** Fit GMMs separately on $B$'s NEG and POS activations. Run two entropic GW solves:
- $P_{\text{neg}}$: between A-NEG centroids and B-NEG centroids,
- $P_{\text{pos}}$: between A-POS centroids and B-POS centroids,

both on intra-distance matrices normalised to $[0, 1]$. We found normalisation essential: real LLM intra-distances can be in the hundreds (especially with outlier "attention-sink" clusters), and POT's entropic GW collapses on such matrices without it.

**Step 3: Barycentric chain.** For each source POS cluster $m$, its B-space image is $\bar{\mu}_m^{A \to B} = (1/p_m) \sum_n P^{\text{pos}}_{mn} \mu_n^B$. For each source NEG cluster $i$, its B-space *target* is the further chained sum $\sum_m W_{im} \bar{\mu}_m^{A \to B}$, where $W$ is the row-normalised intra-A CHaRS coupling. Finally, for each target NEG cluster $j$, the **transported displacement** is

$$d_j^B = \frac{1}{q_j} \sum_i P^{\text{neg}}_{ij} \cdot \left( \sum_m W_{im} \bar{\mu}_m^{A \to B} \right) - \mu_j^B.$$

**Step 4: Runtime hook.** At inference on $B$, attach a forward-pre-hook to the chosen layer; for each token, compute B-NEG GMM responsibilities, blend per-cluster transported displacements, add $\text{coef} \cdot \text{blend}$ to the residual stream.

The whole pipeline uses no target-side concept labels for direction computation — target POS and NEG activations are used only for clustering.

## 4. Experiments

**Phase 5 sanity checks.** Before measuring transfer, we verify that cross-model GW behaves sensibly. On Pythia-160M $\to$ GPT-2-small at layer-6 with sentiment activations and $k=4$ clusters: self-pair GW cost = 0.0000, adjacent-layer = 0.0000, **cross-model = 0.0205**, random-noise baseline = 0.1129. Cross-model is 5× higher than self (a real difference between the two models' geometries) but 5× lower than random (the two are still much more alignable than two unrelated distributions). Class preservation through the cross-model coupling (positive-source $\to$ positive-target via argmax) is 61 %, against a 50 % chance baseline.

**Phase 6 core experiment.** For each (source, target) pair, sweep four methods at three coefficients across three seeds; bootstrap 95 % CIs across seeds. Methods: random (chance floor), Procrustes (orthogonal alignment on Hungarian-matched centroid clouds), GW transport (this work), target oracle (difference-of-means computed directly on target activations — the upper bound).

Best-coefficient positive-shift rate per method:

| Cell | random | Procrustes | **GW transport** | target oracle |
|------|--------|------------|------------------|---------------|
| GPT-2 → Pythia-160M | 12 % | 12 % | **15 %** | 20 % |
| Pythia-160M → GPT-2 | 8 % | 5 % | **13 %** | 15 % |

GW transport beats random on both cells, beats Procrustes on both cells (Procrustes actually does *worse* than random on one cell — consistent with the broader finding that single-rotation alignments fail across unrelated frames), and recovers 75 % / 87 % of the target-oracle's lift. Off-target perplexity stays near the unsteered baseline for GW transport at all coefficients, while a coefficient-matched random direction blows it up — the per-cluster transported displacements localise the steering effect to the part of activation space where it should fire.

**Phase 7 diagnostic.** A natural follow-up question: does the GW alignment cost predict whether transport will work on a given cell? Across a 36-cell matrix (2 pairs × 3 layer depths × 3 cluster counts × 2 seeds), Spearman ρ between `gw_cost_neg` and shift rate is **−0.17 [95 % CI −0.45, +0.16]** — not significantly different from zero. Per-pair correlations even disagree in sign (GPT-2 → Pythia weakly negative, Pythia → GPT-2 weakly positive). We attribute the null to (i) the lexicon judge's 5 %-quantised shift-rate axis with most cells at 5–10 %, (ii) the structural degeneracy at $k=2$ where GW cost collapses to 0, and (iii) the limited dynamic range of two ~125 M-parameter base LMs. The infrastructure (sweep runner, bootstrap-CI helper, four diagnostic figures) is in place for richer matrices.

## 5. Related work

The closest precedent is **Alvarez-Melis & Jaakkola (2018)**, who use GW for bilingual word embedding alignment without parallel corpora; our setting differs in the *target artefact* (steering vector, not embedding map) and the *modality* (LLM residual streams rather than static word vectors). **CHaRS** (Abdullaev et al. 2026) is the intra-model construction we generalise across models; their work makes the connection between difference-of-means steering and OT explicit. **Arditi et al. (2024)** establishes the strongest single-direction steering result (refusal direction); we expect future work to test GW transport on refusal specifically (we did not in this paper because the in-budget chat-tuned model TinyLlama-1.1B-4bit was outside our experimental matrix). **Huh et al. (2024)** articulates the Platonic Representation Hypothesis — that capable models converge on a shared representation up to basis — and our successful transport on the two cells we tried is one small empirical vote for the cross-architecture version of that claim. **Moschella et al. (2023)** propose Relative Representations as an alternative cross-space alignment method using pairwise similarities to anchor sets; a head-to-head comparison is the obvious follow-up.

## 6. Discussion and limitations

The transport result is a starting point, not a closing argument.

**Matrix scope.** Two model pairs, one concept (sentiment), one layer depth chosen by relative-midpoint heuristic. The most interesting next step is a refusal evaluation on TinyLlama-1.1B-Chat at 4-bit (within our 4 GB VRAM budget), where the safety-trained "I cannot help with that" behaviour is what the refusal-direction literature has shown to be most cleanly steerable.

**Evaluation judge.** Our positive-shift judge is a hand-curated sentiment lexicon, deliberately cheap so the eval is reproducible without spinning up another model. The 20-prompt eval split quantises the shift rate to multiples of 5 %, which is fine for the headline transport result but kills statistical power on the diagnostic correlation. A small finetuned classifier or a calibrated LM judge would smooth the axis and likely change Phase 7's null into a meaningful effect, whichever direction it lands.

**Per-cluster vs single-direction transport.** We chose the input-conditional CHaRS construction for the source side because Phase 4 showed it's a gentler scalpel than the single difference-of-means direction. A natural ablation is to use single directions everywhere (no GMM) and report Procrustes-style cross-model steering — but this is essentially what the "Procrustes" baseline does, and it underperformed in our matrix.

**Sample size.** The Phase 6 results use 30 training pairs per class and 20 eval prompts. Both are tight for confidence-interval statistics. The infrastructure handles larger splits trivially; we did not run them because the eval lexicon plateaus quickly.

**No comparison to non-OT alignments.** Procrustes is the classical baseline; CCA and Relative Representations (Moschella et al. 2023) are the natural additional points of comparison. We deferred them to keep the experimental matrix bounded.

## 7. Conclusion

We demonstrated that a cross-model GW alignment, combined with intra-model CHaRS steering and a barycentric projection chain, transports a steering signal from one LLM to another with no paired data and no target-side concept labels, beating both random and Procrustes baselines and recovering most of the target-supervised oracle's lift on a Pythia-160M ↔ GPT-2-small sentiment task. The diagnostic question — whether GW alignment cost predicts transfer success — has a honest null at this matrix size; the infrastructure for asking it on richer matrices is in place. The whole pipeline runs on a 4 GB GPU using only POT, scikit-learn, and Hugging Face Transformers, and is structured so that every solver and projection is reused unchanged across phases of the work.

## References

- Abdullaev, et al. (2026). *Concept-conditional Steering via Optimal Transport (CHaRS).*
- Alvarez-Melis, D., & Jaakkola, T. (2018). Gromov-Wasserstein Alignment of Word Embedding Spaces. *EMNLP.*
- Arditi, A., et al. (2024). Refusal in Language Models Is Mediated by a Single Direction. arXiv:2406.11717.
- Cuturi, M. (2013). Sinkhorn Distances: Lightspeed Computation of Optimal Transport. *NeurIPS.*
- Elhage, N., et al. (2021). A Mathematical Framework for Transformer Circuits. *Anthropic.*
- Huh, M., et al. (2024). The Platonic Representation Hypothesis. arXiv:2405.07987.
- Jorgensen, J., et al. (2024). Improving Activation Steering in Language Models with Mean-Centring.
- Mémoli, F. (2011). Gromov-Wasserstein Distances and the Metric Approach to Object Matching. *Foundations of Computational Mathematics.*
- Moschella, L., et al. (2023). Relative Representations Enable Zero-Shot Latent Space Communication. *ICLR.*
- Peyré, G., Cuturi, M., & Solomon, J. (2016). Gromov-Wasserstein Averaging of Kernel and Distance Matrices. *ICML.*
- Peyré, G., & Cuturi, M. (2019). Computational Optimal Transport. *Foundations and Trends in Machine Learning.*
- Rimsky, N., et al. (2024). Steering Llama 2 via Contrastive Activation Addition. arXiv:2312.06681.
- Santambrogio, F. (2015). *Optimal Transport for Applied Mathematicians.* Birkhäuser.
- Turner, A., et al. (2023). Steering Language Models with Activation Engineering. arXiv:2308.10248.

## Reproducibility

All code lives at <https://github.com/adimunot21/ot-steering>. To regenerate every figure in this paper:

```bash
git clone https://github.com/adimunot21/ot-steering.git
cd ot-steering
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python scripts/make_all_figures.py
```

The script runs every chapter's `make_figures.py` in dependency order. Total wall-clock time: ~45 minutes on the project hardware (single GTX 1650, 4 GB VRAM). Each phase folder under `phases/` contains the chapter, its notebook, its experiment scripts, and the regenerated figures; the project library lives in `src/ot_steering/` with strict-mypy and a complete `pytest` suite (`pytest -q` passes 104 fast tests in ~32 s; `--run-slow` adds one Pythia-160M integration test).
