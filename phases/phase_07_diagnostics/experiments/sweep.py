"""Phase 7 diagnostic sweep.

For each cell in the matrix
    {(gpt2, pythia-160m), (pythia-160m, gpt2)}
  × layer_pair in {(0.25, 0.25), (0.5, 0.5), (0.75, 0.75)}     (relative depth)
  × k in {2, 4, 8}                                              (GMM components)
  × seed in {0, 1}                                              (RNG)
run the Phase 6 GW-transport pipeline and record:

  - gw_cost_neg : entropic GW cost on the NEG-class alignment
                  (the half of the transport pipeline that drives the
                  per-cluster displacement direction at runtime).
  - shift_rate  : fraction of held-out negative-class prompts whose
                  steered continuation reads as more-positive than the
                  unsteered baseline.

Total cells: 2 × 3 × 3 × 2 = 36.

To keep the runtime tractable on the project 4 GB GPU, we amortise
model loads: per (source, target) pair we load each model once,
extract activations at all three layers once, then iterate over
(k, seed) using only the cached activations.

Writes:
  outputs/<run_id>/config.json
  outputs/<run_id>/sweep.json

Run:
    python phases/phase_07_diagnostics/experiments/sweep.py
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ot_steering.activations.datasets import load_sentiment_pairs
from ot_steering.activations.extractor import extract_residual_stream, resolve_blocks
from ot_steering.activations.model_loader import ModelLoaderConfig, load_model
from ot_steering.eval.steering_eval import _sentiment_judge
from ot_steering.steering.baselines import apply_steering_vector
from ot_steering.steering.ot_steering import GMMConfig
from ot_steering.steering.transport import (
    SteeringTransportConfig,
    add_transported_steering_hook,
    build_transport,
)
from ot_steering.utils.seed import set_all_seeds

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUTS_DIR = _PROJECT_ROOT / "outputs"

PAIRS: list[tuple[str, str]] = [
    ("gpt2", "EleutherAI/pythia-160m"),
    ("EleutherAI/pythia-160m", "gpt2"),
]
RELATIVE_LAYERS: tuple[float, ...] = (0.25, 0.5, 0.75)
KS: tuple[int, ...] = (2, 4, 8)
SEEDS: tuple[int, ...] = (0, 1)
COEFFICIENT = 3.0
N_TRAIN = 30
N_EVAL = 20
MAX_NEW_TOKENS = 25


def _shift_rate(baseline: list[str], steered: list[str]) -> float:
    flips = sum(
        1
        for b, s in zip(baseline, steered, strict=True)
        if _sentiment_judge(s) > _sentiment_judge(b)
    )
    return flips / max(len(baseline), 1)


def _extract_all_layers(model, tokenizer, prompts, rel_layers):  # type: ignore[no-untyped-def]
    """Extract activations at every relative-layer depth in one forward pass."""
    n_layers = model.config.num_hidden_layers
    layer_indices = sorted({int(round(r * n_layers)) for r in rel_layers})
    acts = extract_residual_stream(
        model, tokenizer, prompts, layer_indices=layer_indices, batch_size=8
    )
    return acts, layer_indices


def _generate_unsteered(model, tokenizer, prompts, hook_layer):  # type: ignore[no-untyped-def]
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


def _generate_with_transport(
    model,
    tokenizer,
    prompts,
    hook_layer,  # type: ignore[no-untyped-def]
    tmap,
    coefficient,
):
    block = list(resolve_blocks(model))[hook_layer]
    device = next(model.parameters()).device
    outs: list[str] = []
    model.eval()  # type: ignore[no-untyped-call]
    with add_transported_steering_hook(block, tmap, coefficient), torch.no_grad():
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


def _run_pair(src_id: str, tgt_id: str) -> list[dict]:
    """Run the full (layer × k × seed) sweep for a single (source, target) pair."""
    print(f"\n=== pair: {src_id} -> {tgt_id} ===")

    pairs = load_sentiment_pairs()
    train_pairs = pairs[:N_TRAIN]
    eval_pairs = pairs[N_TRAIN : N_TRAIN + N_EVAL]
    pos_prompts = [p for p, _ in train_pairs]
    neg_prompts = [n for _, n in train_pairs]
    eval_starts = [n for _, n in eval_pairs]

    # Source side: load once, extract activations at every layer we need.
    src_model, src_tok = load_model(ModelLoaderConfig(model_id=src_id))
    src_pos_by_layer, src_layer_indices = _extract_all_layers(
        src_model, src_tok, pos_prompts, RELATIVE_LAYERS
    )
    src_neg_by_layer, _ = _extract_all_layers(src_model, src_tok, neg_prompts, RELATIVE_LAYERS)
    src_n_layers = src_model.config.num_hidden_layers
    del src_model, src_tok
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Target side: load, extract activations, run all generations, release.
    tgt_model, tgt_tok = load_model(ModelLoaderConfig(model_id=tgt_id))
    tgt_pos_by_layer, tgt_layer_indices = _extract_all_layers(
        tgt_model, tgt_tok, pos_prompts, RELATIVE_LAYERS
    )
    tgt_neg_by_layer, _ = _extract_all_layers(tgt_model, tgt_tok, neg_prompts, RELATIVE_LAYERS)
    tgt_n_layers = tgt_model.config.num_hidden_layers

    rows: list[dict] = []
    try:
        for rel in RELATIVE_LAYERS:
            src_layer = int(round(rel * src_n_layers))
            tgt_layer = int(round(rel * tgt_n_layers))
            src_pos = src_pos_by_layer[src_layer]
            src_neg = src_neg_by_layer[src_layer]
            tgt_pos = tgt_pos_by_layer[tgt_layer]
            tgt_neg = tgt_neg_by_layer[tgt_layer]

            baseline_outs = _generate_unsteered(tgt_model, tgt_tok, eval_starts, tgt_layer)

            for k in KS:
                for seed in SEEDS:
                    cfg = SteeringTransportConfig(
                        n_components=k,
                        gmm_cfg=GMMConfig(n_components=k, seed=seed),
                    )
                    tmap = build_transport(
                        src_pos,
                        src_neg,
                        tgt_pos,
                        tgt_neg,
                        cfg=cfg,
                        rng=np.random.default_rng(seed + 33),
                    )
                    # Unit-normalise displacements (same as Phase 6).
                    disp_norms = np.linalg.norm(
                        tmap.transported_displacements, axis=1, keepdims=True
                    )
                    tmap_norm = dataclasses.replace(
                        tmap,
                        transported_displacements=(
                            tmap.transported_displacements / disp_norms.clip(min=1e-12)
                        ),
                    )
                    outs = _generate_with_transport(
                        tgt_model,
                        tgt_tok,
                        eval_starts,
                        tgt_layer,
                        tmap_norm,
                        COEFFICIENT,
                    )
                    shift_rate = _shift_rate(baseline_outs, outs)
                    gw_cost_neg = float(tmap.cross_model_alignment_neg.gw_cost)
                    gw_cost_pos = float(tmap.cross_model_alignment_pos.gw_cost)
                    rows.append(
                        {
                            "source_model": src_id,
                            "target_model": tgt_id,
                            "relative_layer": rel,
                            "source_layer": src_layer,
                            "target_layer": tgt_layer,
                            "k": k,
                            "seed": seed,
                            "gw_cost_neg": gw_cost_neg,
                            "gw_cost_pos": gw_cost_pos,
                            "shift_rate": shift_rate,
                        }
                    )
                    print(
                        f"  rel_layer={rel:.2f}  k={k}  seed={seed}  "
                        f"gw_cost_neg={gw_cost_neg:.4f}  shift={shift_rate:.0%}"
                    )
    finally:
        del tgt_model, tgt_tok
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"  source layer indices extracted: {src_layer_indices}")
    print(f"  target layer indices extracted: {tgt_layer_indices}")
    return rows


def _run_id(config: dict) -> str:
    iso = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    digest = hashlib.blake2b(
        json.dumps(config, sort_keys=True).encode("utf-8"), digest_size=4
    ).hexdigest()
    return f"{iso}_{digest}"


def main() -> int:
    set_all_seeds(0)
    config = {
        "pairs": [list(p) for p in PAIRS],
        "relative_layers": list(RELATIVE_LAYERS),
        "ks": list(KS),
        "seeds": list(SEEDS),
        "coefficient": COEFFICIENT,
        "n_train": N_TRAIN,
        "n_eval": N_EVAL,
        "max_new_tokens": MAX_NEW_TOKENS,
    }
    run_dir = OUTPUTS_DIR / _run_id(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    all_rows: list[dict] = []
    for src, tgt in PAIRS:
        all_rows.extend(_run_pair(src, tgt))

    (run_dir / "sweep.json").write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
    print(f"\n=== sweep done: {len(all_rows)} cells ===")
    print(f"artifacts: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
