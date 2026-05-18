"""Phase 4 demo: build CHaRS-style steering maps at various k and compare.

Shared by ``make_figures.py`` and the companion notebook. Default model is
Pythia-160M so the notebook runs in under a minute on the 4 GB GPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.decomposition import PCA

from ot_steering.activations.datasets import load_sentiment_pairs
from ot_steering.activations.extractor import extract_residual_stream, resolve_blocks
from ot_steering.activations.model_loader import ModelLoaderConfig, load_model
from ot_steering.eval.steering_eval import _sentiment_judge, off_target_perplexity
from ot_steering.steering.baselines import apply_steering_vector
from ot_steering.steering.ot_steering import (
    GMMConfig,
    OTSteeringMap,
    add_ot_steering_hook,
    build_ot_steering_map,
)

NDArrayF = NDArray[np.float64]


@dataclass(frozen=True)
class CHaRSDemo:
    """Everything Chapter 4 needs to plot or recap.

    Attributes:
        model_id: Model used.
        hook_layer: Residual-stream layer for extraction and steering.
        positive_acts: ``(n_train, d_model)`` positive-class activations.
        negative_acts: ``(n_train, d_model)`` negative-class activations.
        pca: 2-D ``PCA`` fitted on the pooled activations.
        ks: GMM component counts swept.
        steering_maps_by_k: For each ``k``, the fitted :class:`OTSteeringMap`.
        coefficients: Coefficients swept.
        shift_rate_by_k_coef: Nested dict: ``k -> coef -> positive-shift rate``.
        off_target_ppl_by_k_coef: Same structure: ``k -> coef -> perplexity``.
        baseline_off_target_ppl: Off-target perplexity with no steering.
    """

    model_id: str
    hook_layer: int
    positive_acts: torch.Tensor
    negative_acts: torch.Tensor
    pca: PCA
    ks: tuple[int, ...]
    steering_maps_by_k: dict[int, OTSteeringMap]
    coefficients: tuple[float, ...]
    shift_rate_by_k_coef: dict[int, dict[float, float]]
    off_target_ppl_by_k_coef: dict[int, dict[float, float]]
    baseline_off_target_ppl: float


_NEUTRAL_CORPUS: list[str] = [
    "The chemical element carbon has the symbol C and atomic number six.",
    "Photosynthesis is the process by which green plants convert sunlight into energy.",
    "A right triangle has one angle that measures exactly ninety degrees.",
    "Vertebrate animals are characterised by a backbone or spinal column.",
]


def _generate_with_ot(
    model, tokenizer, prompts, hook_layer, steering_map, coefficient, max_new_tokens
):  # type: ignore[no-untyped-def]
    blocks = list(resolve_blocks(model))
    block = blocks[hook_layer]
    device = next(model.parameters()).device
    outs: list[str] = []
    model.eval()  # type: ignore[no-untyped-call]
    with add_ot_steering_hook(block, steering_map, coefficient), torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt", padding=False).to(device)
            generated = model.generate(  # type: ignore[operator]
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            cont = generated[0, inputs["input_ids"].shape[1] :]
            text = tokenizer.decode(cont, skip_special_tokens=True)
            assert isinstance(text, str)
            outs.append(text)
    return outs


def run_charsy_demo(
    model_id: str = "EleutherAI/pythia-160m",
    hook_layer: int = 6,
    ks: tuple[int, ...] = (1, 2, 4),
    coefficients: tuple[float, ...] = (0.5, 1.0, 2.0),
    n_train: int = 30,
    n_eval: int = 20,
    max_new_tokens: int = 25,
    seed: int = 0,
) -> CHaRSDemo:
    """Fit CHaRS steering at several ``k`` and sweep coefficients on a held-out split.

    Args:
        model_id: Model to load (Pythia-160M default for cheap notebook reruns).
        hook_layer: Residual-stream layer to inject at.
        ks: GMM component counts.
        coefficients: Steering coefficients applied to per-cluster displacements.
        n_train: Training pairs used to fit the GMMs.
        n_eval: Held-out negative-class prompts used for the lift curves.
        max_new_tokens: Tokens generated per prompt during eval.
        seed: RNG seed for the GMM EM.

    Returns:
        Populated :class:`CHaRSDemo`.
    """
    model, tokenizer = load_model(ModelLoaderConfig(model_id=model_id))
    try:
        pairs = load_sentiment_pairs()
        train_pairs = pairs[:n_train]
        eval_pairs = pairs[n_train : n_train + n_eval]
        pos_prompts = [p for p, _ in train_pairs]
        neg_prompts = [n for _, n in train_pairs]
        eval_starts = [n for _, n in eval_pairs]

        acts = extract_residual_stream(
            model,
            tokenizer,
            pos_prompts + neg_prompts,
            layer_indices=[hook_layer],
            batch_size=8,
        )[hook_layer]
        pos_acts = acts[: len(pos_prompts)]
        neg_acts = acts[len(pos_prompts) :]

        pooled = torch.cat([pos_acts, neg_acts], dim=0).cpu().numpy().astype(np.float32)
        pca = PCA(n_components=2, random_state=seed).fit(pooled)

        # Baseline (unsteered) generations and off-target ppl, once.
        baseline_outs = apply_steering_vector(
            model,
            tokenizer,
            eval_starts,
            block_resolver=resolve_blocks,
            hook_layer=hook_layer,
            direction=torch.zeros(pos_acts.shape[1]),
            coefficient=0.0,
            max_new_tokens=max_new_tokens,
        )
        baseline_ppl = off_target_perplexity(model, tokenizer, _NEUTRAL_CORPUS, stride=64)

        steering_maps_by_k: dict[int, OTSteeringMap] = {}
        shift_rate_by_k_coef: dict[int, dict[float, float]] = {}
        off_target_ppl_by_k_coef: dict[int, dict[float, float]] = {}
        for k in ks:
            sm = build_ot_steering_map(
                pos_acts,
                neg_acts,
                gmm_cfg=GMMConfig(n_components=k, covariance_type="diag", seed=seed),
            )
            steering_maps_by_k[k] = sm
            shifts: dict[float, float] = {}
            ppls: dict[float, float] = {}
            for coef in coefficients:
                steered_outs = _generate_with_ot(
                    model, tokenizer, eval_starts, hook_layer, sm, coef, max_new_tokens
                )
                shifts[coef] = sum(
                    1
                    for b, s in zip(baseline_outs, steered_outs, strict=True)
                    if _sentiment_judge(s) > _sentiment_judge(b)
                ) / max(len(eval_starts), 1)
                blocks = list(resolve_blocks(model))
                block = blocks[hook_layer]
                with add_ot_steering_hook(block, sm, coef):
                    ppls[coef] = off_target_perplexity(model, tokenizer, _NEUTRAL_CORPUS, stride=64)
            shift_rate_by_k_coef[k] = shifts
            off_target_ppl_by_k_coef[k] = ppls

        return CHaRSDemo(
            model_id=model_id,
            hook_layer=hook_layer,
            positive_acts=pos_acts,
            negative_acts=neg_acts,
            pca=pca,
            ks=ks,
            steering_maps_by_k=steering_maps_by_k,
            coefficients=coefficients,
            shift_rate_by_k_coef=shift_rate_by_k_coef,
            off_target_ppl_by_k_coef=off_target_ppl_by_k_coef,
            baseline_off_target_ppl=baseline_ppl,
        )
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
