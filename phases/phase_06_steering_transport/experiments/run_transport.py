"""Phase 6 core experiment: cross-model steering transport vs baselines.

For each (source_model, target_model, concept) cell in a small matrix,
sweep four steering methods at three coefficients and three seeds:

  - random         : a unit-norm random direction in target space (chance).
  - procrustes     : source direction rotated into target space via
                     orthogonal Procrustes on matched centroid clouds.
  - gw_transport   : the headline method — CHaRS on source, cross-model
                     GW (P_neg, P_pos), barycentric projection chain,
                     per-B-NEG-cluster displacement.
  - target_oracle  : difference-of-means computed *directly* on target
                     activations (upper bound, requires target supervision).

Each method-coefficient-seed combination produces:
  - positive_shift_rate : fraction of held-out negative-class prompts on
                          which the steered continuation reads as
                          more-positive than the unsteered baseline
                          (sentiment lexicon judge from Phase 3).
  - off_target_perplexity : per-token perplexity on a small neutral corpus
                            with the steering hook active.

Bootstrap 95% confidence intervals are taken across the 3 seeds.

Writes outputs/<run_id>/run_transport.json with the full grid and a
config.json. Prints a compact per-cell summary.

Run:
    python phases/phase_06_steering_transport/experiments/run_transport.py
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ot_steering.activations.datasets import load_sentiment_pairs
from ot_steering.activations.extractor import extract_residual_stream, resolve_blocks
from ot_steering.activations.model_loader import ModelLoaderConfig, load_model
from ot_steering.eval.steering_eval import _sentiment_judge, off_target_perplexity
from ot_steering.steering.baselines import (
    apply_steering_vector,
)
from ot_steering.steering.ot_steering import GMMConfig
from ot_steering.steering.transport import (
    SteeringTransportConfig,
    add_transported_steering_hook,
    build_transport,
)
from ot_steering.steering.transport_baselines import (
    procrustes_aligned,
    random_direction,
    target_supervised_oracle,
)
from ot_steering.utils.seed import set_all_seeds

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUTS_DIR = _PROJECT_ROOT / "outputs"

CELLS: list[dict[str, object]] = [
    {
        "source_model": "gpt2",
        "target_model": "EleutherAI/pythia-160m",
        "concept": "sentiment",
        "source_relative_layer": 0.5,
        "target_relative_layer": 0.5,
    },
    {
        "source_model": "EleutherAI/pythia-160m",
        "target_model": "gpt2",
        "concept": "sentiment",
        "source_relative_layer": 0.5,
        "target_relative_layer": 0.5,
    },
]
METHODS = ("random", "procrustes", "gw_transport", "target_oracle")
COEFFICIENTS = (1.0, 3.0, 6.0)
SEEDS = (0, 1, 2)
N_TRAIN = 30
N_EVAL = 20
N_COMPONENTS = 4
MAX_NEW_TOKENS = 25

NEUTRAL_CORPUS: list[str] = [
    "The chemical element carbon has the symbol C and atomic number six.",
    "Photosynthesis is the process by which green plants convert sunlight into energy.",
    "A right triangle has one angle that measures exactly ninety degrees.",
    "The Mariana Trench is the deepest known oceanic trench on Earth.",
]


def _load_pairs(concept: str) -> list[tuple[str, str]]:
    if concept == "sentiment":
        return load_sentiment_pairs()
    raise ValueError(f"concept {concept!r} not supported in this experiment")


def _extract(model, tokenizer, prompts: list[str], layer: int) -> torch.Tensor:
    return extract_residual_stream(model, tokenizer, prompts, layer_indices=[layer], batch_size=8)[
        layer
    ]


def _bootstrap_ci(
    values: list[float], n_boot: int = 1000, seed: int = 0
) -> tuple[float, float, float]:
    """Mean and 2.5 / 97.5 percentile of resampled means."""
    if not values:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    means = rng.choice(arr, size=(n_boot, arr.size), replace=True).mean(axis=1)
    return float(arr.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _generate_unsteered(model, tokenizer, prompts: list[str], hook_layer: int) -> list[str]:
    """Unsteered baseline generations (coefficient=0 dummy hook)."""
    direction = torch.zeros(model.config.hidden_size, dtype=torch.float32)
    return apply_steering_vector(
        model,
        tokenizer,
        prompts,
        block_resolver=resolve_blocks,
        hook_layer=hook_layer,
        direction=direction,
        coefficient=0.0,
        max_new_tokens=MAX_NEW_TOKENS,
    )


def _generate_with_direction(
    model,
    tokenizer,
    prompts: list[str],
    hook_layer: int,
    direction: torch.Tensor,
    coefficient: float,
) -> list[str]:
    return apply_steering_vector(
        model,
        tokenizer,
        prompts,
        block_resolver=resolve_blocks,
        hook_layer=hook_layer,
        direction=direction,
        coefficient=coefficient,
        max_new_tokens=MAX_NEW_TOKENS,
    )


def _generate_with_transport(
    model, tokenizer, prompts: list[str], hook_layer: int, transported_map, coefficient: float
) -> list[str]:
    block = list(resolve_blocks(model))[hook_layer]
    device = next(model.parameters()).device
    outs: list[str] = []
    model.eval()  # type: ignore[no-untyped-call]
    with add_transported_steering_hook(block, transported_map, coefficient), torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt", padding=False).to(device)
            generated = model.generate(  # type: ignore[operator]
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            cont = generated[0, inputs["input_ids"].shape[1] :]
            text = tokenizer.decode(cont, skip_special_tokens=True)
            assert isinstance(text, str)
            outs.append(text)
    return outs


def _shift_rate(baseline: list[str], steered: list[str]) -> float:
    flips = sum(
        1
        for b, s in zip(baseline, steered, strict=True)
        if _sentiment_judge(s) > _sentiment_judge(b)
    )
    return flips / max(len(baseline), 1)


def _off_target_with_direction(
    model, tokenizer, hook_layer: int, direction: torch.Tensor, coefficient: float
) -> float:
    return off_target_perplexity(
        model,
        tokenizer,
        NEUTRAL_CORPUS,
        block_resolver=resolve_blocks,
        hook_layer=hook_layer,
        direction=direction,
        coefficient=coefficient,
        stride=64,
    )


def _off_target_with_transport(
    model, tokenizer, hook_layer: int, transported_map, coefficient: float
) -> float:
    block = list(resolve_blocks(model))[hook_layer]
    with add_transported_steering_hook(block, transported_map, coefficient):
        return off_target_perplexity(model, tokenizer, NEUTRAL_CORPUS, stride=64)


def _run_cell(cell: dict[str, object]) -> dict[str, object]:
    src_id = str(cell["source_model"])
    tgt_id = str(cell["target_model"])
    concept = str(cell["concept"])
    src_rel = float(cell["source_relative_layer"])
    tgt_rel = float(cell["target_relative_layer"])
    print(f"\n=== cell: {src_id} -> {tgt_id}  concept={concept} ===")

    pairs = _load_pairs(concept)
    train_pairs = pairs[:N_TRAIN]
    eval_pairs = pairs[N_TRAIN : N_TRAIN + N_EVAL]
    pos_prompts = [p for p, _ in train_pairs]
    neg_prompts = [n for _, n in train_pairs]
    eval_starts = [n for _, n in eval_pairs]

    # ---- source side: extract POS + NEG activations at chosen layer.
    src_model, src_tok = load_model(ModelLoaderConfig(model_id=src_id))
    src_layer = round(src_rel * src_model.config.num_hidden_layers)
    src_pos_acts = _extract(src_model, src_tok, pos_prompts, src_layer)
    src_neg_acts = _extract(src_model, src_tok, neg_prompts, src_layer)
    src_dim = src_pos_acts.shape[1]
    del src_model, src_tok
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- target side: load, extract POS+NEG, do all the sweeps, then release.
    tgt_model, tgt_tok = load_model(ModelLoaderConfig(model_id=tgt_id))
    tgt_layer = round(tgt_rel * tgt_model.config.num_hidden_layers)
    tgt_pos_acts = _extract(tgt_model, tgt_tok, pos_prompts, tgt_layer)
    tgt_neg_acts = _extract(tgt_model, tgt_tok, neg_prompts, tgt_layer)
    tgt_dim = tgt_pos_acts.shape[1]

    # Unsteered baseline (once).
    baseline_outs = _generate_unsteered(tgt_model, tgt_tok, eval_starts, tgt_layer)
    baseline_ppl = off_target_perplexity(tgt_model, tgt_tok, NEUTRAL_CORPUS, stride=64)
    print(f"  baseline off-target ppl = {baseline_ppl:.2f}")

    # Direction-based methods need a precomputed source direction
    # (difference-of-means on source, unit-normalised).
    src_direction = src_pos_acts.float().mean(dim=0) - src_neg_acts.float().mean(dim=0)
    src_direction = src_direction / src_direction.norm().clamp(min=1e-12)

    # Per-method, per-coefficient, per-seed numbers.
    results: dict[str, dict[float, dict[str, list[float]]]] = {
        m: {c: {"shift": [], "ppl": []} for c in COEFFICIENTS} for m in METHODS
    }
    for seed in SEEDS:
        # ---- random
        rd = random_direction(tgt_dim, seed=seed).to(next(tgt_model.parameters()).device)
        for coef in COEFFICIENTS:
            outs = _generate_with_direction(tgt_model, tgt_tok, eval_starts, tgt_layer, rd, coef)
            results["random"][coef]["shift"].append(_shift_rate(baseline_outs, outs))
            results["random"][coef]["ppl"].append(
                _off_target_with_direction(tgt_model, tgt_tok, tgt_layer, rd, coef)
            )

        # ---- procrustes: fit on the same matched-centroid clouds GW would use.
        # Pair source/target GMM centroids by Hungarian on a centroid cost.
        # Train GMM on pooled source and target activations, k=N_COMPONENTS.
        from ot_steering.steering.ot_steering import fit_gmm  # local import

        src_pooled = torch.cat([src_pos_acts, src_neg_acts], dim=0).cpu().numpy().astype(np.float64)
        tgt_pooled = torch.cat([tgt_pos_acts, tgt_neg_acts], dim=0).cpu().numpy().astype(np.float64)
        src_gmm = fit_gmm(src_pooled, GMMConfig(n_components=N_COMPONENTS, seed=seed))
        tgt_gmm = fit_gmm(tgt_pooled, GMMConfig(n_components=N_COMPONENTS, seed=seed))
        # Pair source clusters to target clusters by best matching of means
        # (assignment problem, but we use a quick greedy pairing on
        # squared-Euclidean cost after zero-padding to the bigger dim).
        common = max(src_dim, tgt_dim)
        s_pad = np.concatenate([src_gmm.means_, np.zeros((N_COMPONENTS, common - src_dim))], axis=1)
        t_pad = np.concatenate([tgt_gmm.means_, np.zeros((N_COMPONENTS, common - tgt_dim))], axis=1)
        cost = ((s_pad[:, None] - t_pad[None, :]) ** 2).sum(-1)
        from scipy.optimize import linear_sum_assignment  # local import

        src_idx, tgt_idx = linear_sum_assignment(cost)
        proc_direction = procrustes_aligned(
            src_direction,
            src_gmm.means_[src_idx],
            tgt_gmm.means_[tgt_idx],
        ).to(next(tgt_model.parameters()).device)
        for coef in COEFFICIENTS:
            outs = _generate_with_direction(
                tgt_model, tgt_tok, eval_starts, tgt_layer, proc_direction, coef
            )
            results["procrustes"][coef]["shift"].append(_shift_rate(baseline_outs, outs))
            results["procrustes"][coef]["ppl"].append(
                _off_target_with_direction(tgt_model, tgt_tok, tgt_layer, proc_direction, coef)
            )

        # ---- gw_transport
        tmap = build_transport(
            src_pos_acts,
            src_neg_acts,
            tgt_pos_acts,
            tgt_neg_acts,
            cfg=SteeringTransportConfig(
                n_components=N_COMPONENTS,
                gmm_cfg=GMMConfig(n_components=N_COMPONENTS, seed=seed),
            ),
            rng=np.random.default_rng(seed + 33),
        )
        # Normalise the per-cluster transported displacements to unit norm
        # so the coefficient sweep is comparable across methods.
        disp_norms = np.linalg.norm(tmap.transported_displacements, axis=1, keepdims=True)
        tmap_normed_disps = tmap.transported_displacements / disp_norms.clip(min=1e-12)
        # Replace the displacements field by building a new dataclass.
        import dataclasses as _dc

        tmap_norm = _dc.replace(tmap, transported_displacements=tmap_normed_disps)
        for coef in COEFFICIENTS:
            outs = _generate_with_transport(
                tgt_model, tgt_tok, eval_starts, tgt_layer, tmap_norm, coef
            )
            results["gw_transport"][coef]["shift"].append(_shift_rate(baseline_outs, outs))
            results["gw_transport"][coef]["ppl"].append(
                _off_target_with_transport(tgt_model, tgt_tok, tgt_layer, tmap_norm, coef)
            )

        # ---- target_oracle
        oracle = target_supervised_oracle(tgt_pos_acts, tgt_neg_acts).to(
            next(tgt_model.parameters()).device
        )
        for coef in COEFFICIENTS:
            outs = _generate_with_direction(
                tgt_model, tgt_tok, eval_starts, tgt_layer, oracle, coef
            )
            results["target_oracle"][coef]["shift"].append(_shift_rate(baseline_outs, outs))
            results["target_oracle"][coef]["ppl"].append(
                _off_target_with_direction(tgt_model, tgt_tok, tgt_layer, oracle, coef)
            )

    # Cleanup target.
    del tgt_model, tgt_tok
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Bootstrap CIs per method-coefficient.
    summary: dict[str, dict[str, dict[str, tuple[float, float, float]]]] = {}
    for method in METHODS:
        summary[method] = {}
        for coef in COEFFICIENTS:
            summary[method][str(coef)] = {
                "shift_mean_lo_hi": _bootstrap_ci(results[method][coef]["shift"]),
                "ppl_mean_lo_hi": _bootstrap_ci(results[method][coef]["ppl"]),
            }
            mean_shift, lo, hi = summary[method][str(coef)]["shift_mean_lo_hi"]
            mean_ppl, ppl_lo, ppl_hi = summary[method][str(coef)]["ppl_mean_lo_hi"]
            print(
                f"  {method:14s} coef={coef:>4.1f}  "
                f"shift={mean_shift:.0%} [{lo:.0%}, {hi:.0%}]   "
                f"ppl={mean_ppl:6.1f} [{ppl_lo:5.1f}, {ppl_hi:5.1f}]"
            )

    return {
        "source_model": src_id,
        "target_model": tgt_id,
        "concept": concept,
        "source_layer": src_layer,
        "target_layer": tgt_layer,
        "source_dim": src_dim,
        "target_dim": tgt_dim,
        "baseline_off_target_ppl": baseline_ppl,
        "results": {
            method: {
                str(coef): {
                    "shift": results[method][coef]["shift"],
                    "ppl": results[method][coef]["ppl"],
                }
                for coef in COEFFICIENTS
            }
            for method in METHODS
        },
        "summary": summary,
    }


def _run_id(config: dict[str, object]) -> str:
    iso = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    digest = hashlib.blake2b(
        json.dumps(config, sort_keys=True).encode("utf-8"), digest_size=4
    ).hexdigest()
    return f"{iso}_{digest}"


def main() -> int:
    set_all_seeds(0)
    config = {
        "cells": CELLS,
        "methods": list(METHODS),
        "coefficients": list(COEFFICIENTS),
        "seeds": list(SEEDS),
        "n_train": N_TRAIN,
        "n_eval": N_EVAL,
        "n_components": N_COMPONENTS,
        "max_new_tokens": MAX_NEW_TOKENS,
    }
    run_dir = OUTPUTS_DIR / _run_id(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    results = [_run_cell(c) for c in CELLS]
    (run_dir / "run_transport.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n=== headline ===")
    for res in results:
        print(f"\n  {res['source_model']} -> {res['target_model']} ({res['concept']}):")
        for method in METHODS:
            best_shift, best_coef = -1.0, 0.0
            for coef_str, payload in res["summary"][method].items():  # type: ignore[index]
                m, _, _ = payload["shift_mean_lo_hi"]
                if m > best_shift:
                    best_shift, best_coef = m, float(coef_str)
            print(f"    {method:14s} best coef={best_coef:>4.1f}  best shift={best_shift:.0%}")
    print(f"\nartifacts: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
