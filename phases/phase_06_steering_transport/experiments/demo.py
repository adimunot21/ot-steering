"""Phase 6 demo: cross-model steering transport on Pythia-160M -> GPT-2-small.

Shared between ``make_figures.py`` and the companion notebook. Runs one
representative cell of the full ``run_transport.py`` matrix (single seed,
no bootstrap) so the notebook fits in ~3 minutes on the project 4 GB GPU.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

from ot_steering.activations.datasets import load_sentiment_pairs
from ot_steering.activations.extractor import extract_residual_stream, resolve_blocks
from ot_steering.activations.model_loader import ModelLoaderConfig, load_model
from ot_steering.eval.steering_eval import _sentiment_judge, off_target_perplexity
from ot_steering.steering.baselines import apply_steering_vector
from ot_steering.steering.ot_steering import GMMConfig, fit_gmm
from ot_steering.steering.transport import (
    SteeringTransportConfig,
    TransportedSteeringMap,
    add_transported_steering_hook,
    build_transport,
)
from ot_steering.steering.transport_baselines import (
    procrustes_aligned,
    random_direction,
    target_supervised_oracle,
)

NDArrayF = NDArray[np.float64]


@dataclass(frozen=True)
class TransportDemo:
    """Output of :func:`run_transport_demo`.

    Attributes:
        source_model_id: Source model id.
        target_model_id: Target model id.
        source_layer: Source-side residual-stream layer index.
        target_layer: Target-side residual-stream layer index.
        coefficients: Steering coefficients swept.
        shift_rates_by_method: ``{method: {coef: positive-shift rate}}``.
        off_target_ppl_by_method: ``{method: {coef: perplexity}}``.
        baseline_off_target_ppl: Perplexity with no steering active.
        transported_map: The fitted :class:`TransportedSteeringMap` for
            the GW-transport method (for plotting the pipeline diagram).
    """

    source_model_id: str
    target_model_id: str
    source_layer: int
    target_layer: int
    coefficients: tuple[float, ...]
    shift_rates_by_method: dict[str, dict[float, float]]
    off_target_ppl_by_method: dict[str, dict[float, float]]
    baseline_off_target_ppl: float
    transported_map: TransportedSteeringMap


_NEUTRAL_CORPUS: list[str] = [
    "The chemical element carbon has the symbol C and atomic number six.",
    "Photosynthesis is the process by which green plants convert sunlight into energy.",
    "A right triangle has one angle that measures exactly ninety degrees.",
    "Vertebrate animals are characterised by a backbone or spinal column.",
]


def _shift_rate(baseline: list[str], steered: list[str]) -> float:
    flips = sum(
        1
        for b, s in zip(baseline, steered, strict=True)
        if _sentiment_judge(s) > _sentiment_judge(b)
    )
    return flips / max(len(baseline), 1)


def run_transport_demo(
    source_model_id: str = "EleutherAI/pythia-160m",
    target_model_id: str = "gpt2",
    n_train: int = 30,
    n_eval: int = 20,
    n_components: int = 4,
    coefficients: tuple[float, ...] = (1.0, 3.0, 6.0),
    max_new_tokens: int = 25,
    seed: int = 0,
) -> TransportDemo:
    """Run one cell of the Phase 6 matrix end-to-end.

    Args:
        source_model_id: Source model id.
        target_model_id: Target model id.
        n_train: Training pairs for GMMs / centroids.
        n_eval: Held-out negative-class prompts for lift evaluation.
        n_components: GMM cluster count per side.
        coefficients: Steering coefficients to sweep.
        max_new_tokens: Tokens generated per prompt during eval.
        seed: RNG seed.

    Returns:
        A populated :class:`TransportDemo`.
    """
    pairs = load_sentiment_pairs()
    train_pairs = pairs[:n_train]
    eval_pairs = pairs[n_train : n_train + n_eval]
    pos_prompts = [p for p, _ in train_pairs]
    neg_prompts = [n for _, n in train_pairs]
    eval_starts = [n for _, n in eval_pairs]

    # --- source side ---
    src_model, src_tok = load_model(ModelLoaderConfig(model_id=source_model_id))
    try:
        src_layer = src_model.config.num_hidden_layers // 2
        src_pos = extract_residual_stream(
            src_model, src_tok, pos_prompts, layer_indices=[src_layer], batch_size=8
        )[src_layer]
        src_neg = extract_residual_stream(
            src_model, src_tok, neg_prompts, layer_indices=[src_layer], batch_size=8
        )[src_layer]
        src_dim = src_pos.shape[1]
    finally:
        del src_model, src_tok
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- target side ---
    tgt_model, tgt_tok = load_model(ModelLoaderConfig(model_id=target_model_id))
    try:
        tgt_layer = tgt_model.config.num_hidden_layers // 2
        tgt_pos = extract_residual_stream(
            tgt_model, tgt_tok, pos_prompts, layer_indices=[tgt_layer], batch_size=8
        )[tgt_layer]
        tgt_neg = extract_residual_stream(
            tgt_model, tgt_tok, neg_prompts, layer_indices=[tgt_layer], batch_size=8
        )[tgt_layer]
        tgt_dim = tgt_pos.shape[1]
        device = next(tgt_model.parameters()).device

        baseline_outs = apply_steering_vector(
            tgt_model,
            tgt_tok,
            eval_starts,
            block_resolver=resolve_blocks,
            hook_layer=tgt_layer,
            direction=torch.zeros(tgt_dim),
            coefficient=0.0,
            max_new_tokens=max_new_tokens,
        )
        baseline_ppl = off_target_perplexity(tgt_model, tgt_tok, _NEUTRAL_CORPUS, stride=64)

        # Pre-compute source direction (diff of means, unit norm).
        src_direction = src_pos.float().mean(dim=0) - src_neg.float().mean(dim=0)
        src_direction = src_direction / src_direction.norm().clamp(min=1e-12)

        # --- four methods ---
        # random
        rd = random_direction(tgt_dim, seed=seed).to(device)
        # procrustes: pair centroids via Hungarian on padded-mean costs
        src_pooled = torch.cat([src_pos, src_neg], dim=0).cpu().numpy().astype(np.float64)
        tgt_pooled = torch.cat([tgt_pos, tgt_neg], dim=0).cpu().numpy().astype(np.float64)
        src_gmm = fit_gmm(src_pooled, GMMConfig(n_components=n_components, seed=seed))
        tgt_gmm = fit_gmm(tgt_pooled, GMMConfig(n_components=n_components, seed=seed))
        common = max(src_dim, tgt_dim)
        s_pad = np.concatenate([src_gmm.means_, np.zeros((n_components, common - src_dim))], axis=1)
        t_pad = np.concatenate([tgt_gmm.means_, np.zeros((n_components, common - tgt_dim))], axis=1)
        from scipy.optimize import linear_sum_assignment

        cost = ((s_pad[:, None] - t_pad[None, :]) ** 2).sum(-1)
        s_idx, t_idx = linear_sum_assignment(cost)
        proc_direction = procrustes_aligned(
            src_direction,
            src_gmm.means_[s_idx],
            tgt_gmm.means_[t_idx],
        ).to(device)

        # gw transport
        tmap = build_transport(
            src_pos,
            src_neg,
            tgt_pos,
            tgt_neg,
            cfg=SteeringTransportConfig(
                n_components=n_components,
                gmm_cfg=GMMConfig(n_components=n_components, seed=seed),
            ),
            rng=np.random.default_rng(seed + 33),
        )
        disp_norms = np.linalg.norm(tmap.transported_displacements, axis=1, keepdims=True)
        tmap_norm = dataclasses.replace(
            tmap,
            transported_displacements=(tmap.transported_displacements / disp_norms.clip(min=1e-12)),
        )

        oracle = target_supervised_oracle(tgt_pos, tgt_neg).to(device)

        shift_rates: dict[str, dict[float, float]] = {
            "random": {},
            "procrustes": {},
            "gw_transport": {},
            "target_oracle": {},
        }
        off_target: dict[str, dict[float, float]] = {
            "random": {},
            "procrustes": {},
            "gw_transport": {},
            "target_oracle": {},
        }
        for coef in coefficients:
            # random
            outs = apply_steering_vector(
                tgt_model,
                tgt_tok,
                eval_starts,
                block_resolver=resolve_blocks,
                hook_layer=tgt_layer,
                direction=rd,
                coefficient=coef,
                max_new_tokens=max_new_tokens,
            )
            shift_rates["random"][coef] = _shift_rate(baseline_outs, outs)
            off_target["random"][coef] = off_target_perplexity(
                tgt_model,
                tgt_tok,
                _NEUTRAL_CORPUS,
                block_resolver=resolve_blocks,
                hook_layer=tgt_layer,
                direction=rd,
                coefficient=coef,
                stride=64,
            )
            # procrustes
            outs = apply_steering_vector(
                tgt_model,
                tgt_tok,
                eval_starts,
                block_resolver=resolve_blocks,
                hook_layer=tgt_layer,
                direction=proc_direction,
                coefficient=coef,
                max_new_tokens=max_new_tokens,
            )
            shift_rates["procrustes"][coef] = _shift_rate(baseline_outs, outs)
            off_target["procrustes"][coef] = off_target_perplexity(
                tgt_model,
                tgt_tok,
                _NEUTRAL_CORPUS,
                block_resolver=resolve_blocks,
                hook_layer=tgt_layer,
                direction=proc_direction,
                coefficient=coef,
                stride=64,
            )
            # gw transport
            block = list(resolve_blocks(tgt_model))[tgt_layer]
            outs = []
            tgt_model.eval()  # type: ignore[no-untyped-call]
            with add_transported_steering_hook(block, tmap_norm, coef), torch.no_grad():
                for prompt in eval_starts:
                    inputs = tgt_tok(prompt, return_tensors="pt", padding=False).to(device)
                    generated = tgt_model.generate(  # type: ignore[operator]
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=tgt_tok.pad_token_id,
                    )
                    cont = generated[0, inputs["input_ids"].shape[1] :]
                    text = tgt_tok.decode(cont, skip_special_tokens=True)
                    assert isinstance(text, str)
                    outs.append(text)
            shift_rates["gw_transport"][coef] = _shift_rate(baseline_outs, outs)
            with add_transported_steering_hook(block, tmap_norm, coef):
                off_target["gw_transport"][coef] = off_target_perplexity(
                    tgt_model, tgt_tok, _NEUTRAL_CORPUS, stride=64
                )
            # oracle
            outs = apply_steering_vector(
                tgt_model,
                tgt_tok,
                eval_starts,
                block_resolver=resolve_blocks,
                hook_layer=tgt_layer,
                direction=oracle,
                coefficient=coef,
                max_new_tokens=max_new_tokens,
            )
            shift_rates["target_oracle"][coef] = _shift_rate(baseline_outs, outs)
            off_target["target_oracle"][coef] = off_target_perplexity(
                tgt_model,
                tgt_tok,
                _NEUTRAL_CORPUS,
                block_resolver=resolve_blocks,
                hook_layer=tgt_layer,
                direction=oracle,
                coefficient=coef,
                stride=64,
            )

        return TransportDemo(
            source_model_id=source_model_id,
            target_model_id=target_model_id,
            source_layer=src_layer,
            target_layer=tgt_layer,
            coefficients=coefficients,
            shift_rates_by_method=shift_rates,
            off_target_ppl_by_method=off_target,
            baseline_off_target_ppl=baseline_ppl,
            transported_map=tmap_norm,
        )
    finally:
        del tgt_model, tgt_tok
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
