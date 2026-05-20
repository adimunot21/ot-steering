"""Phase 6 follow-up: refusal transport on two chat-tuned models.

Phase 6 ran the transport pipeline on base GPT-2 and Pythia-160M with the
sentiment concept. The honest gap in that result is that base LMs don't
really refuse anything, so the canonical "refusal direction" finding
from Arditi et al. (2024) couldn't be tested at all.

This script closes that gap. Source: Qwen2.5-0.5B-Instruct (fp16, ~950 MB
VRAM, 24 layers, d=896). Target: Qwen2.5-1.5B-Instruct (4-bit nf4 +
double-quant, ~1.1 GB VRAM, 28 layers, d=1536). Both refuse harmful
prompts cleanly in their baseline behaviour (verified during loader
inspection). Same family, different scale — a meaningful cross-
architecture test of the project's hypothesis.

The experiment differs from ``run_transport.py`` in three ways:

1. **Chat-template-formatted activations.** For instruction-tuned
   models the residual stream depends on the full prompt + assistant
   prefix; we apply each tokenizer's chat template before extraction
   and generation.
2. **Refusal concept.** POS = harmful prompts (refusal-eliciting),
   NEG = harmless prompts (compliance-eliciting).
3. **Refusal-induction metric.** Eval is held-out *harmless* prompts.
   Baseline on a safety-tuned chat model: complies. Steered: we hope
   to *induce* refusal. The success metric is
   ``refusal_judge(steered) AND NOT refusal_judge(baseline)``.

Run:
    python phases/phase_06_steering_transport/experiments/run_refusal_transport.py
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ot_steering.activations.datasets import load_refusal_pairs
from ot_steering.activations.extractor import extract_residual_stream, resolve_blocks
from ot_steering.activations.model_loader import ModelLoaderConfig, load_model
from ot_steering.eval.steering_eval import _refusal_judge
from ot_steering.steering.baselines import apply_steering_vector
from ot_steering.steering.ot_steering import GMMConfig, fit_gmm
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

# We try both directions so we have a symmetric matrix like Phase 6's sentiment run.
CELLS: list[dict[str, object]] = [
    {
        "source_model": "Qwen/Qwen2.5-0.5B-Instruct",
        "source_quant": None,
        "target_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "target_quant": "4bit",
    },
    {
        "source_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "source_quant": "4bit",
        "target_model": "Qwen/Qwen2.5-0.5B-Instruct",
        "target_quant": None,
    },
]
METHODS = ("random", "procrustes", "gw_transport", "target_oracle")
COEFFICIENTS = (1.0, 3.0, 6.0)
SEEDS = (0, 1, 2)
N_TRAIN = 30
N_EVAL = 20
N_COMPONENTS = 4
MAX_NEW_TOKENS = 40
RELATIVE_LAYER = 0.5

NEUTRAL_CORPUS: list[str] = [
    "The chemical element carbon has the symbol C and atomic number six.",
    "Photosynthesis is the process by which green plants convert sunlight into energy.",
    "A right triangle has one angle that measures exactly ninety degrees.",
    "The Mariana Trench is the deepest known oceanic trench on Earth.",
]


def _format_chat(tokenizer, prompts: list[str]) -> list[str]:
    """Wrap each prompt as a single-turn user message using the model's chat template."""
    formatted = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        formatted.append(text)
    return formatted


def _extract(model, tokenizer, prompts: list[str], layer: int) -> torch.Tensor:
    """Apply chat template then extract last-token residual-stream activations."""
    formatted = _format_chat(tokenizer, prompts)
    return extract_residual_stream(
        model, tokenizer, formatted, layer_indices=[layer], batch_size=4
    )[layer]


def _generate(model, tokenizer, prompts: list[str], *, max_new_tokens: int) -> list[str]:
    """Generate continuations on chat-formatted prompts. No steering hook here."""
    formatted = _format_chat(tokenizer, prompts)
    device = next(model.parameters()).device
    outs: list[str] = []
    model.eval()  # type: ignore[no-untyped-call]
    with torch.no_grad():
        for text in formatted:
            inputs = tokenizer(text, return_tensors="pt").to(device)
            generated = model.generate(  # type: ignore[operator]
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            cont = generated[0, inputs["input_ids"].shape[1] :]
            text_out = tokenizer.decode(cont, skip_special_tokens=True)
            assert isinstance(text_out, str)
            outs.append(text_out)
    return outs


def _generate_with_direction(
    model, tokenizer, prompts, layer, direction, coefficient, *, max_new_tokens
):  # type: ignore[no-untyped-def]
    """Same as apply_steering_vector but applies the chat template first."""
    formatted = _format_chat(tokenizer, prompts)
    return apply_steering_vector(
        model,
        tokenizer,
        formatted,
        block_resolver=resolve_blocks,
        hook_layer=layer,
        direction=direction,
        coefficient=coefficient,
        max_new_tokens=max_new_tokens,
    )


def _generate_with_transport(
    model, tokenizer, prompts, layer, tmap, coefficient, *, max_new_tokens
):  # type: ignore[no-untyped-def]
    formatted = _format_chat(tokenizer, prompts)
    block = list(resolve_blocks(model))[layer]
    device = next(model.parameters()).device
    outs: list[str] = []
    model.eval()  # type: ignore[no-untyped-call]
    with add_transported_steering_hook(block, tmap, coefficient), torch.no_grad():
        for text in formatted:
            inputs = tokenizer(text, return_tensors="pt").to(device)
            generated = model.generate(  # type: ignore[operator]
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            cont = generated[0, inputs["input_ids"].shape[1] :]
            text_out = tokenizer.decode(cont, skip_special_tokens=True)
            assert isinstance(text_out, str)
            outs.append(text_out)
    return outs


def _induction_rate(baseline: list[str], steered: list[str]) -> float:
    """Fraction of held-out harmless prompts where steering INDUCED refusal.

    Counts cases where the steered continuation looks like a refusal AND
    the unsteered baseline did not — i.e. the steering vector did real
    work, not just preserved an already-present refusal.
    """
    induced = sum(
        1
        for b, s in zip(baseline, steered, strict=True)
        if _refusal_judge(s) and not _refusal_judge(b)
    )
    return induced / max(len(baseline), 1)


def _bootstrap_ci(
    values: list[float], n_boot: int = 1000, seed: int = 0
) -> tuple[float, float, float]:
    if not values:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    means = rng.choice(arr, size=(n_boot, arr.size), replace=True).mean(axis=1)
    return float(arr.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _run_cell(cell: dict[str, object]) -> dict[str, object]:
    src_id = str(cell["source_model"])
    src_quant = cell["source_quant"]
    tgt_id = str(cell["target_model"])
    tgt_quant = cell["target_quant"]
    print(f"\n=== cell: {src_id} -> {tgt_id} ===")

    pairs = load_refusal_pairs()
    train_pairs = pairs[:N_TRAIN]
    eval_pairs = pairs[N_TRAIN : N_TRAIN + N_EVAL]
    harmful_train = [h for h, _ in train_pairs]
    harmless_train = [k for _, k in train_pairs]
    harmless_eval = [k for _, k in eval_pairs]

    # --- source side: load, extract POS/NEG activations, release.
    src_model, src_tok = load_model(
        ModelLoaderConfig(model_id=src_id, quantization=src_quant)  # type: ignore[arg-type]
    )
    src_layer = round(RELATIVE_LAYER * src_model.config.num_hidden_layers)
    src_pos = _extract(src_model, src_tok, harmful_train, src_layer)  # POS = harmful
    src_neg = _extract(src_model, src_tok, harmless_train, src_layer)  # NEG = harmless
    src_dim = int(src_pos.shape[1])
    del src_model, src_tok
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- target side: load, extract POS/NEG, run all sweeps, release.
    tgt_model, tgt_tok = load_model(
        ModelLoaderConfig(model_id=tgt_id, quantization=tgt_quant)  # type: ignore[arg-type]
    )
    tgt_layer = round(RELATIVE_LAYER * tgt_model.config.num_hidden_layers)
    tgt_pos = _extract(tgt_model, tgt_tok, harmful_train, tgt_layer)
    tgt_neg = _extract(tgt_model, tgt_tok, harmless_train, tgt_layer)
    tgt_dim = int(tgt_pos.shape[1])

    # Unsteered baseline on harmless eval prompts: how often does the
    # chat model already refuse? Expected near 0%.
    baseline_outs = _generate(tgt_model, tgt_tok, harmless_eval, max_new_tokens=MAX_NEW_TOKENS)
    baseline_refusal_rate = sum(_refusal_judge(t) for t in baseline_outs) / max(
        len(baseline_outs), 1
    )
    print(f"  baseline refusal rate on harmless eval: {baseline_refusal_rate:.0%}")

    src_direction = src_pos.float().mean(dim=0) - src_neg.float().mean(dim=0)
    src_direction = src_direction / src_direction.norm().clamp(min=1e-12)

    results: dict[str, dict[float, list[float]]] = {
        m: {c: [] for c in COEFFICIENTS} for m in METHODS
    }
    device = next(tgt_model.parameters()).device

    for seed in SEEDS:
        rd = random_direction(tgt_dim, seed=seed).to(device)

        # Procrustes baseline: Hungarian-paired pooled GMM centroids,
        # orthogonal Procrustes, zero-padded.
        src_pooled = torch.cat([src_pos, src_neg], dim=0).cpu().numpy().astype(np.float64)
        tgt_pooled = torch.cat([tgt_pos, tgt_neg], dim=0).cpu().numpy().astype(np.float64)
        src_gmm = fit_gmm(src_pooled, GMMConfig(n_components=N_COMPONENTS, seed=seed))
        tgt_gmm = fit_gmm(tgt_pooled, GMMConfig(n_components=N_COMPONENTS, seed=seed))
        common = max(src_dim, tgt_dim)
        s_pad = np.concatenate([src_gmm.means_, np.zeros((N_COMPONENTS, common - src_dim))], axis=1)
        t_pad = np.concatenate([tgt_gmm.means_, np.zeros((N_COMPONENTS, common - tgt_dim))], axis=1)
        from scipy.optimize import linear_sum_assignment  # local import

        cost = ((s_pad[:, None] - t_pad[None, :]) ** 2).sum(-1)
        s_idx, t_idx = linear_sum_assignment(cost)
        proc_direction = procrustes_aligned(
            src_direction,
            src_gmm.means_[s_idx],
            tgt_gmm.means_[t_idx],
        ).to(device)

        # GW transport.
        tmap = build_transport(
            src_pos,
            src_neg,
            tgt_pos,
            tgt_neg,
            cfg=SteeringTransportConfig(
                n_components=N_COMPONENTS,
                gmm_cfg=GMMConfig(n_components=N_COMPONENTS, seed=seed),
            ),
            rng=np.random.default_rng(seed + 33),
        )
        disp_norms = np.linalg.norm(tmap.transported_displacements, axis=1, keepdims=True)
        tmap_norm = dataclasses.replace(
            tmap,
            transported_displacements=(tmap.transported_displacements / disp_norms.clip(min=1e-12)),
        )

        # Target-supervised oracle.
        oracle = target_supervised_oracle(tgt_pos, tgt_neg).to(device)

        for coef in COEFFICIENTS:
            for method, direction_or_tmap in [
                ("random", rd),
                ("procrustes", proc_direction),
                ("target_oracle", oracle),
            ]:
                outs = _generate_with_direction(
                    tgt_model,
                    tgt_tok,
                    harmless_eval,
                    tgt_layer,
                    direction_or_tmap,
                    coef,
                    max_new_tokens=MAX_NEW_TOKENS,
                )
                results[method][coef].append(_induction_rate(baseline_outs, outs))

            outs_gw = _generate_with_transport(
                tgt_model,
                tgt_tok,
                harmless_eval,
                tgt_layer,
                tmap_norm,
                coef,
                max_new_tokens=MAX_NEW_TOKENS,
            )
            results["gw_transport"][coef].append(_induction_rate(baseline_outs, outs_gw))

    del tgt_model, tgt_tok
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    summary: dict[str, dict[str, dict[str, tuple[float, float, float]]]] = {}
    for method in METHODS:
        summary[method] = {}
        for coef in COEFFICIENTS:
            summary[method][str(coef)] = {
                "induction_mean_lo_hi": _bootstrap_ci(results[method][coef]),
            }
            m, lo, hi = summary[method][str(coef)]["induction_mean_lo_hi"]
            print(
                f"  {method:14s} coef={coef:>4.1f}  induced_refusals={m:.0%} [{lo:.0%}, {hi:.0%}]"
            )

    return {
        "source_model": src_id,
        "target_model": tgt_id,
        "source_layer": src_layer,
        "target_layer": tgt_layer,
        "source_dim": src_dim,
        "target_dim": tgt_dim,
        "baseline_refusal_rate_on_harmless": baseline_refusal_rate,
        "results": {
            method: {str(c): results[method][c] for c in COEFFICIENTS} for method in METHODS
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
        "relative_layer": RELATIVE_LAYER,
    }
    run_dir = OUTPUTS_DIR / _run_id(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    results = [_run_cell(c) for c in CELLS]
    (run_dir / "run_refusal_transport.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    print("\n=== headline (refusal induction rate on harmless prompts) ===")
    for res in results:
        print(f"\n  {res['source_model']} -> {res['target_model']}:")
        print(
            f"    baseline refusal rate (unsteered): {res['baseline_refusal_rate_on_harmless']:.0%}"
        )
        for method in METHODS:
            best_m, best_coef = -1.0, 0.0
            for coef_str, payload in res["summary"][method].items():  # type: ignore[index]
                m, _, _ = payload["induction_mean_lo_hi"]
                if m > best_m:
                    best_m, best_coef = m, float(coef_str)
            print(
                f"    {method:14s} best coef={best_coef:>4.1f}  best induced refusals={best_m:.0%}"
            )
    print(f"\nartifacts: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
