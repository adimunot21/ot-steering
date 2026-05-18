# Phase 3 — LLMs, activations, and steering baselines

Builds the LLM-side infrastructure (model loader, residual-stream extractor,
on-disk activation cache, contrastive datasets) and reproduces a basic
ActAdd-style sentiment steering result on GPT-2-small. The chapter walks
from "what is a transformer" through "what does it mean to steer one,"
ending with two evaluation curves: lift vs. coefficient and off-target
perplexity vs. coefficient.

Read `chapter.md`. Then either:

- Run the notebook on Pythia-160M for a fast demo:
  `jupyter notebook notebook.ipynb`.
- Reproduce the headline result on GPT-2-small:
  `python experiments/reproduce_sentiment.py`. Writes
  `outputs/<run_id>/{reproduce_sentiment.log, config.json, metrics.json}`.
- Regenerate the four chapter figures from scratch:
  `python experiments/make_figures.py`.
