"""Phase 3 demo: sentiment activations + steering on Pythia-160M.

Shared between the notebook and the figures script. Defaults to Pythia-160M
so the whole demo runs in well under a minute on the project GPU (4 GB).
GPT-2-small is also a fine drop-in; ``reproduce_sentiment.py`` uses it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.decomposition import PCA

from ot_steering.activations.datasets import load_sentiment_pairs
from ot_steering.activations.extractor import extract_residual_stream, resolve_blocks
from ot_steering.activations.model_loader import ModelLoaderConfig, load_model
from ot_steering.eval.steering_eval import _sentiment_judge, off_target_perplexity
from ot_steering.steering.baselines import (
    apply_steering_vector,
    difference_in_means,
)


@dataclass(frozen=True)
class SentimentDemo:
    """Container for everything Chapter 3 wants to plot or recap.

    Attributes:
        model_id: Hugging Face model id used.
        hook_layer: Residual-stream layer at which activations were
            extracted and steering is injected.
        positive_acts: ``(n, d_model)`` activations for positive prompts.
        negative_acts: ``(n, d_model)`` activations for negative prompts.
        direction: ``(d_model,)`` difference-in-means steering direction.
        pca: A fitted ``PCA(n_components=2)`` for plotting in 2D.
        coefficients: Coefficient sweep values used for the success-rate
            and off-target perplexity curves.
        success_rate_by_coef: Map ``coef -> sentiment lift rate`` on the
            held-out negative eval prompts.
        off_target_ppl_by_coef: Map ``coef -> off-target perplexity`` on
            a small neutral text corpus.
    """

    model_id: str
    hook_layer: int
    positive_acts: torch.Tensor
    negative_acts: torch.Tensor
    direction: torch.Tensor
    pca: PCA
    coefficients: tuple[float, ...]
    success_rate_by_coef: dict[float, float]
    off_target_ppl_by_coef: dict[float, float]


_NEUTRAL_CORPUS: list[str] = [
    "The chemical element carbon has the symbol C and atomic number six.",
    "Photosynthesis is the process by which green plants convert sunlight into energy.",
    "Geological time is divided into eons, eras, periods, epochs, and ages.",
    "The Roman Empire reached its greatest extent under the emperor Trajan.",
    "A right triangle has one angle that measures exactly ninety degrees.",
    "The Mariana Trench is the deepest known oceanic trench on Earth.",
    "Vertebrate animals are characterised by a backbone or spinal column.",
    "Renewable energy comes from sources that are naturally replenished.",
]


def run_sentiment_demo(
    model_id: str = "EleutherAI/pythia-160m",
    hook_layer: int = 6,
    # Coefficients are applied to the UNIT-NORMALISED steering direction, so
    # `coefficient` is interpretable as "how many residual-stream-norm units
    # of perturbation to inject". For Pythia-160M the activations at layer 6
    # have an L2 norm around 45, and ActAdd-style steering tends to start
    # mattering at ~5-15% of that scale.
    coefficients: tuple[float, ...] = (-8.0, -4.0, -2.0, 0.0, 2.0, 4.0, 8.0),
    n_train: int = 30,
    n_eval: int = 20,
    max_new_tokens: int = 25,
) -> SentimentDemo:
    """Build the steering vector and sweep coefficients, returning everything.

    Args:
        model_id: Model to load (default Pythia-160M; small enough that the
            notebook runs end-to-end in well under a minute on the 4 GB
            project GPU).
        hook_layer: Residual-stream layer to extract from and inject at.
        coefficients: Steering coefficients to sweep.
        n_train: Sentiment pairs used to build the direction.
        n_eval: Held-out pairs used for success-rate and ppl curves.
        max_new_tokens: Tokens generated per prompt during eval.

    Returns:
        A populated :class:`SentimentDemo`.
    """
    model, tokenizer = load_model(ModelLoaderConfig(model_id=model_id))
    pairs = load_sentiment_pairs()
    train_pairs = pairs[:n_train]
    eval_pairs = pairs[n_train : n_train + n_eval]

    pos_prompts = [p for p, _ in train_pairs]
    neg_prompts = [n for _, n in train_pairs]

    acts = extract_residual_stream(
        model,
        tokenizer,
        pos_prompts + neg_prompts,
        layer_indices=[hook_layer],
        batch_size=8,
    )[hook_layer]
    pos_acts = acts[: len(pos_prompts)]
    neg_acts = acts[len(pos_prompts) :]
    raw_direction = difference_in_means(pos_acts, neg_acts)
    # Normalise to unit length so the coefficient sweep is interpretable
    # across different layers, models, and dataset sizes.
    direction = raw_direction / raw_direction.norm().clamp(min=1e-8)

    # 2D PCA on the pooled activations for the chapter figure.
    pooled = torch.cat([pos_acts, neg_acts], dim=0).cpu().numpy().astype(np.float32)
    pca = PCA(n_components=2, random_state=0).fit(pooled)

    eval_negatives = [n for _, n in eval_pairs]
    # Baseline (coef=0) generations once, up front — every coefficient is
    # scored as "did the steered continuation land more-positive than this?".
    baseline_outs = apply_steering_vector(
        model,
        tokenizer,
        eval_negatives,
        block_resolver=resolve_blocks,
        hook_layer=hook_layer,
        direction=direction,
        coefficient=0.0,
        max_new_tokens=max_new_tokens,
    )
    success_rate_by_coef: dict[float, float] = {}
    off_target_by_coef: dict[float, float] = {}

    for coef in coefficients:
        if coef == 0.0:
            success_rate_by_coef[coef] = 0.0
        else:
            steered_outs = apply_steering_vector(
                model,
                tokenizer,
                eval_negatives,
                block_resolver=resolve_blocks,
                hook_layer=hook_layer,
                direction=direction,
                coefficient=coef,
                max_new_tokens=max_new_tokens,
            )
            flips = sum(
                1
                for base, steer in zip(baseline_outs, steered_outs, strict=True)
                if _sentiment_judge(steer) > _sentiment_judge(base)
            )
            success_rate_by_coef[coef] = flips / max(len(eval_negatives), 1)

        ppl = off_target_perplexity(
            model,
            tokenizer,
            _NEUTRAL_CORPUS,
            block_resolver=resolve_blocks,
            hook_layer=hook_layer,
            direction=direction,
            coefficient=coef,
            stride=64,
        )
        off_target_by_coef[coef] = ppl

    return SentimentDemo(
        model_id=model_id,
        hook_layer=hook_layer,
        positive_acts=pos_acts,
        negative_acts=neg_acts,
        direction=direction,
        pca=pca,
        coefficients=coefficients,
        success_rate_by_coef=success_rate_by_coef,
        off_target_ppl_by_coef=off_target_by_coef,
    )
