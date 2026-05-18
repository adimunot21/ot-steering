# Chapter 0 — Why this project exists

## The problem in one sentence

The interpretability tricks we learn for one language model don't transfer to the next one, and that's a real problem.

Let me unpack that.

## A worked example you can hold in your head

Suppose you have a chat model — call it Model A — and you've discovered, by running some clever statistics on its internal activations, that there's a particular direction in its hidden state that means "the prompt is asking for something harmful." Adding a multiple of that vector to the model's activations during generation makes it more refusal-prone; subtracting a multiple makes it more compliant. Researchers have actually done this; the trick is called a *steering vector*, and it works embarrassingly well for a wide range of behaviors — sentiment, factuality, refusal, sycophancy, even chain-of-thought verbosity.

So far so good. Now your colleague trains Model B — same architecture family, different scale, different data mix — and asks you for the refusal vector you found in Model A so they can use it in Model B without redoing the whole analysis.

You hand them a 4096-dimensional vector of floats. They add it into Model B's activations. Model B starts producing gibberish.

Why? Because the *coordinates* of Model A's hidden space and Model B's hidden space have nothing to do with each other. They're both 4096-dimensional, sure, but there's no reason dimension 1729 of Model A should mean the same thing — or any thing — in Model B. The vector you handed over is a sentence in a language only Model A speaks.

This is the problem the project is about. The artifact you discovered — the refusal direction — was real. It encoded something true about what the model is doing. But the way you expressed it was tied to one specific coordinate system, and the moment you stepped outside that coordinate system the artifact stopped making sense.

## Why anyone cares

Two reasons.

First, **interpretability research is expensive**. Finding a clean steering vector for a useful behavior on a real model can mean curating thousands of contrastive prompts, running them through the model, doing careful statistical analysis of intermediate activations, validating across layers, and quantifying off-target damage. Doing that work once is fine. Doing it again from scratch every time a new model drops — every six weeks, lately — is a treadmill that nobody can keep up with. If we could *transport* a refusal vector from Llama-3 to Qwen-3 without redoing the discovery work, we'd massively amortize the cost of every clean interpretability result we get.

Second, **transferable interpretability artifacts would be evidence about what models are**. If a "harmful prompt" direction in Model A can be cleanly translated into a working "harmful prompt" direction in Model B, that suggests both models have learned a representation of harmfulness with the same internal *shape*, even if they describe that shape in different coordinates. That's a non-trivial empirical claim about deep nets — close kin to what's now being called the Platonic Representation Hypothesis: the idea that sufficiently capable models, regardless of architecture, converge to representations of the world that differ mostly in basis. Each successful or failed transport is a vote.

We are not the first to try cross-model representation alignment. The standard moves are *Procrustes* (find the best rotation between two point clouds in a shared space) and *CCA* (find directions in two spaces that are maximally correlated). Both have a fatal-for-our-purposes assumption baked in: that the two spaces are comparable up to a linear map. That's a plausible assumption if your two "spaces" are two views of the same data — left eye and right eye, say — but it's a much stronger claim when applied to two different neural networks whose internals are nonlinear in essentially unbounded ways.

So we need a tool that aligns distributions *without* assuming a shared coordinate system. That's where optimal transport comes in.

## Optimal transport, in one paragraph

You have a pile of sand here. You want it in a pile over there. Each grain has to be moved. Moving a grain a longer distance costs more. The minimum total cost to convert one pile into the other is the *Wasserstein distance* between them, and the assignment of grains to destinations that achieves that minimum is the *optimal transport plan*. That's the whole intuition. The math just makes that precise for continuous distributions, and turns out to give you, for free, a meaningful distance between probability measures and a principled way to *interpolate* between them.

Why does this matter for steering vectors? Because — as a recent paper called *CHaRS* pointed out — the standard "difference of means" steering vector is exactly what optimal transport gives you when both your positive-example and negative-example activations look like Gaussian blobs with the same shape. The familiar steering trick is a *special case* of OT. And once you see it that way, you can ask: what if the blobs aren't Gaussian? What if "harmful" prompts cluster into three sub-modes (request-for-violence, request-for-deception, request-for-self-harm) and "harmless" prompts cluster into five? OT handles that. You stop having a single direction and start having a *map* — a function that, given an activation in the "harmless" distribution, tells you where its "harmful" counterpart should sit.

CHaRS does this within a single model. Our project asks: can you do it *across* models?

## Gromov-Wasserstein, in one paragraph

Standard optimal transport needs the two distributions to live in the same space — otherwise "the cost to move a grain from here to there" isn't even defined. Two LLMs do not live in the same space. So we need a generalization called *Gromov-Wasserstein* (GW), which is what you do when you have two distributions in two completely different spaces and the only thing you can compare is *internal* geometry: pairwise distances within each distribution. Concretely, GW asks: "find a correspondence between points in distribution A and points in distribution B such that *if two points are close together in A, their counterparts are close together in B*." It's structural matching, not coordinate matching. And it's exactly the right tool for the LLM-transport problem, because the one thing we *can* compute on both sides is the pairwise relations between activations.

That sentence — *"the relational structure of contrastive activation distributions is approximately model-universal"* — is the empirical bet of this entire project. If it's true, GW should work. If it's false, GW will fail in informative ways, and the project pivots into a *diagnostic* paper about *when* steering transfers and when it doesn't.

## The rough plan

Eight phases. Each one produces working code and a chapter of this course.

- **Phase 1 — Optimal transport foundations.** We build OT from scratch for a 2D toy. Then we throw the from-scratch code away and use the standard library (POT). The point is for you and me to understand the algorithm, not to ship our own version.
- **Phase 2 — Gromov-Wasserstein.** Same drill, generalized. We watch GW recover a known rotation between two 2D point clouds that have no shared coordinate frame. This is the structural-matching reflex we'll lean on for everything that follows.
- **Phase 3 — LLMs, activations, and steering baselines.** We finally meet the language models. Tiny ones — Pythia, GPT-2-small, Qwen-0.5B, TinyLlama in 4-bit — because the GPU has 4 GB. We extract activations with PyTorch hooks, build a caching layer so we don't have to do it twice, and reproduce a published steering result. This is the unit test for everything that comes after.
- **Phase 4 — Intra-model OT steering (CHaRS-style).** Within one model, we replace the difference-of-means steering vector with an OT-derived steering *map*. This is our intra-model upper bound: it tells us how much steering quality we have to play with.
- **Phase 5 — Cross-model Gromov-Wasserstein.** We finally run GW on two real LLMs' contrastive activations. We don't try to steer yet. We just want to see that GW does something sensible: it should recover near-identity when aligning a model with itself, near-identity when aligning consecutive layers, and high cost when aligning two unrelated distributions.
- **Phase 6 — Cross-model steering transport.** The actual experiment. Source model → extract steering structure → GW-align to target → push the steering signal through the coupling → apply it on the target → measure. We measure against a target-supervised oracle (the steering vector found *directly* on the target with full supervision); that's an upper bound, not a baseline to beat.
- **Phase 7 — Diagnostic analysis.** Even when transport works only sometimes, we want to know *when*. Does GW cost predict transfer success? Does it work better in some layers than others? Some concept families more than others?
- **Phase 8 — Synthesis and writeup.** Workshop-quality paper, reproducible repo, one-command figure regeneration.

That's it. No model training. No mega-experiments. One GPU, one notebook of code, one notebook of writing per phase.

## The promise of this course

Every chapter in this repo will be written like a blog essay, not like API documentation. The reader I have in my head is a sharp software engineer who knows Python and ML fundamentals well but has never seen a transformer hook or an OT solver. So:

- We always start with *why* this concept exists before we open the math.
- Every symbol gets a one-line gloss the first time it appears.
- "Coupling," "residual stream," "Sinkhorn divergence" — none of these get used without a definition the first time.
- The math is in service of the intuition; the intuition is in service of the code; the code is the same code that runs in the project. There is no parallel toy implementation. If something gets simplified for the chapter, that's the cue to go simplify the actual codebase.
- Diagrams beat walls of equations. Whenever there's a real picture to draw, we draw it.

And — importantly — the course is for *me* as much as for you. I am building this project to deeply understand it, not to ship copy-pasted boilerplate. If a chapter has a part that's still vague, that's a sign I haven't understood that part yet, and the right move is to fix it before moving on.

## What we just learned

- Interpretability artifacts (steering vectors, refusal directions, etc.) are real but tied to one model's coordinate system, and so don't transfer.
- Standard alignment methods (Procrustes, CCA) assume a shared coordinate frame and so won't fix this.
- Optimal transport gives a principled way to relate two distributions; Gromov-Wasserstein generalizes it to distributions that live in incomparable spaces.
- We're going to test whether the *relational* structure of contrastive activation distributions is universal enough across LLMs for GW to transport steering vectors with no paired data and no target supervision.
- The plan is eight short phases, each producing code, tests, and a chapter readable from scratch.

## Go deeper

- *Optimal Transport for Applied Mathematicians* — Filippo Santambrogio. The friendliest serious OT textbook. The first 30 pages alone are worth their weight.
- *Computational Optimal Transport* — Peyré & Cuturi. The reference for the algorithmic side; the introduction is the best two-paragraph explanation of why OT exploded in ML in the late 2010s.
- *Steering Language Models with Activation Engineering* (Turner et al., 2023). The original "just add a vector" steering paper.
- *Refusal in Language Models Is Mediated by a Single Direction* (Arditi et al., 2024). The cleanest demonstration of a useful, robust, surgically-applied steering direction.
- *The Platonic Representation Hypothesis* (Huh et al., 2024). The conceptual cousin of this project: the claim that capable models converge on the same internal world model up to a change of basis.

## What's next

Phase 1, the foundations of optimal transport. We'll build the discrete OT problem from scratch on a 2D toy, derive Sinkhorn's algorithm, watch entropic regularization smooth a hard problem into a tractable one, and finish by reproducing all of it through the POT library so you can see the wrapper is exactly what we just built. See `phases/phase_01_ot_foundations/chapter.md`.
