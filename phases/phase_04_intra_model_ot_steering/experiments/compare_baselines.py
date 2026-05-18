"""Compare difference-in-means vs CHaRS-style OT steering across k.

For each (model, concept) cell, fit OT steering maps with k in
``{1, 2, 4, 8}`` (k=1 reduces to difference-in-means; we still run it
through the same pipeline so the comparison is apples-to-apples), sweep
coefficients on a held-out eval split, and record:

  positive_shift_rate(coef)   for sentiment / refusal
  off_target_perplexity(coef) on a small neutral corpus

Saves the full grid as JSON at outputs/<run_id>/compare_baselines.json
and prints a compact per-cell summary.

Run:
    python phases/phase_04_intra_model_ot_steering/experiments/compare_baselines.py
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import torch

from ot_steering.activations.datasets import (
    load_refusal_pairs,
    load_sentiment_pairs,
)
from ot_steering.activations.extractor import extract_residual_stream, resolve_blocks
from ot_steering.activations.model_loader import ModelLoaderConfig, load_model
from ot_steering.eval.steering_eval import (
    _refusal_judge,
    _sentiment_judge,
    off_target_perplexity,
)
from ot_steering.steering.baselines import apply_steering_vector
from ot_steering.steering.ot_steering import (
    GMMConfig,
    add_ot_steering_hook,
    build_ot_steering_map,
)
from ot_steering.utils.seed import set_all_seeds

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUTS_DIR = _PROJECT_ROOT / "outputs"

CELLS: list[dict[str, object]] = [
    {"model_id": "gpt2", "concept": "sentiment", "hook_layer": 6},
    {"model_id": "gpt2", "concept": "refusal", "hook_layer": 6},
    {"model_id": "EleutherAI/pythia-160m", "concept": "sentiment", "hook_layer": 6},
]
KS: tuple[int, ...] = (1, 2, 4, 8)
COEFFICIENTS: tuple[float, ...] = (0.5, 1.0, 2.0)
SEED = 0
N_TRAIN = 30
N_EVAL = 20
MAX_NEW_TOKENS = 25

NEUTRAL_CORPUS: list[str] = [
    "The chemical element carbon has the symbol C and atomic number six.",
    "Photosynthesis is the process by which green plants convert sunlight into energy.",
    "Geological time is divided into eons, eras, periods, epochs, and ages.",
    "A right triangle has one angle that measures exactly ninety degrees.",
    "The Mariana Trench is the deepest known oceanic trench on Earth.",
]


def _load_pairs(concept: str) -> list[tuple[str, str]]:
    if concept == "sentiment":
        return load_sentiment_pairs()
    if concept == "refusal":
        return load_refusal_pairs()
    raise ValueError(f"unknown concept {concept!r}")


def _judge_outputs(concept: str, baseline: list[str], steered: list[str]) -> float:
    """Fraction of prompts where steering shifted output in the intended direction."""
    if concept == "sentiment":
        return sum(
            1
            for b, s in zip(baseline, steered, strict=True)
            if _sentiment_judge(s) > _sentiment_judge(b)
        ) / max(len(baseline), 1)
    # Refusal: success = steered output starts looking like a refusal where
    # the baseline did not.
    return sum(
        1
        for b, s in zip(baseline, steered, strict=True)
        if _refusal_judge(s) and not _refusal_judge(b)
    ) / max(len(baseline), 1)


def _generate_with_ot_steering(
    model, tokenizer, prompts, block_resolver, hook_layer, steering_map, coefficient, max_new_tokens
):
    """Generate from a list of prompts under the OT steering hook."""
    blocks = list(block_resolver(model))
    block = blocks[hook_layer]
    device = next(model.parameters()).device
    outputs: list[str] = []
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
            outputs.append(text)
    return outputs


def _run_cell(cell: dict[str, object]) -> dict[str, object]:
    model_id = str(cell["model_id"])
    concept = str(cell["concept"])
    hook_layer = int(cell["hook_layer"])
    print(f"\n=== cell: model={model_id}  concept={concept}  layer={hook_layer} ===")

    model, tokenizer = load_model(ModelLoaderConfig(model_id=model_id))
    try:
        pairs = _load_pairs(concept)
        train_pairs = pairs[:N_TRAIN]
        eval_pairs = pairs[N_TRAIN : N_TRAIN + N_EVAL]
        # Concept-agnostic "positive" = the class we steer TOWARD.
        # For sentiment that's the explicit positive; for refusal that's
        # the harmful prompt (so the steering pushes the model into refusing).
        if concept == "sentiment":
            pos_prompts = [p for p, _ in train_pairs]
            neg_prompts = [n for _, n in train_pairs]
            eval_starts = [n for _, n in eval_pairs]  # start from negative
        else:  # refusal
            pos_prompts = [h for h, _ in train_pairs]  # harmful
            neg_prompts = [k for _, k in train_pairs]  # harmless
            eval_starts = [h for h, _ in eval_pairs]  # harmful prompts

        acts = extract_residual_stream(
            model,
            tokenizer,
            pos_prompts + neg_prompts,
            layer_indices=[hook_layer],
            batch_size=8,
        )[hook_layer]
        pos_acts = acts[: len(pos_prompts)]
        neg_acts = acts[len(pos_prompts) :]
        baseline_outs = apply_steering_vector(
            model,
            tokenizer,
            eval_starts,
            block_resolver=resolve_blocks,
            hook_layer=hook_layer,
            direction=torch.zeros(pos_acts.shape[1]),
            coefficient=0.0,
            max_new_tokens=MAX_NEW_TOKENS,
        )

        baseline_ppl = off_target_perplexity(
            model,
            tokenizer,
            NEUTRAL_CORPUS,
            stride=64,
        )

        per_k: dict[int, dict[str, dict[float, float]]] = {}
        for k in KS:
            sm = build_ot_steering_map(
                pos_acts,
                neg_acts,
                gmm_cfg=GMMConfig(n_components=k, covariance_type="diag", seed=SEED),
            )
            shifts: dict[float, float] = {}
            ppls: dict[float, float] = {}
            for coef in COEFFICIENTS:
                steered_outs = _generate_with_ot_steering(
                    model,
                    tokenizer,
                    eval_starts,
                    resolve_blocks,
                    hook_layer,
                    sm,
                    coef,
                    MAX_NEW_TOKENS,
                )
                shifts[coef] = _judge_outputs(concept, baseline_outs, steered_outs)
                # Off-target ppl with the OT hook active.
                blocks = list(resolve_blocks(model))
                block = blocks[hook_layer]
                with add_ot_steering_hook(block, sm, coef):
                    ppls[coef] = off_target_perplexity(
                        model,
                        tokenizer,
                        NEUTRAL_CORPUS,
                        stride=64,
                    )
            per_k[k] = {"shift_rate": shifts, "off_target_ppl": ppls}
            best_coef = max(shifts, key=shifts.get)  # type: ignore[arg-type]
            print(
                f"  k={k:>2}  best coef={best_coef:>4.1f}  "
                f"shift={shifts[best_coef]:.0%}  "
                f"ppl={ppls[best_coef]:7.2f}  (baseline ppl={baseline_ppl:.2f})"
            )

        return {
            "model_id": model_id,
            "concept": concept,
            "hook_layer": hook_layer,
            "baseline_off_target_ppl": baseline_ppl,
            "per_k": {
                str(k): {
                    "shift_rate": {str(c): r for c, r in v["shift_rate"].items()},
                    "off_target_ppl": {str(c): p for c, p in v["off_target_ppl"].items()},
                }
                for k, v in per_k.items()
            },
        }
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _run_id(config: dict[str, object]) -> str:
    iso = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    digest = hashlib.blake2b(
        json.dumps(config, sort_keys=True).encode("utf-8"), digest_size=4
    ).hexdigest()
    return f"{iso}_{digest}"


def main() -> int:
    set_all_seeds(SEED)
    config = {
        "cells": CELLS,
        "ks": list(KS),
        "coefficients": list(COEFFICIENTS),
        "seed": SEED,
        "n_train": N_TRAIN,
        "n_eval": N_EVAL,
        "max_new_tokens": MAX_NEW_TOKENS,
    }
    run_dir = OUTPUTS_DIR / _run_id(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    results = [_run_cell(c) for c in CELLS]
    (run_dir / "compare_baselines.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n=== headline ===")
    for cell, res in zip(CELLS, results, strict=True):
        per_k = res["per_k"]  # type: ignore[index]
        best_k, best_shift, best_coef = 1, -1.0, 0.0
        for k_str, payload in per_k.items():
            for c_str, sr in payload["shift_rate"].items():
                if sr > best_shift:
                    best_shift, best_k, best_coef = sr, int(k_str), float(c_str)
        print(
            f"  {cell['model_id']:32s} / {cell['concept']:10s} "
            f"best k={best_k}  coef={best_coef:.1f}  shift={best_shift:.0%}"
        )
    print(f"\nartifacts: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
