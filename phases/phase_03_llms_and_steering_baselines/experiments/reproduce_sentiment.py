"""Reproduce a known sentiment-steering result on GPT-2-small.

Loads ``gpt2``, extracts residual-stream activations on the 50 sentiment
contrastive pairs at a configurable layer (default: 6, the rough midpoint
of GPT-2's 12 blocks), builds the difference-in-means steering direction,
applies it on the negative-class prompts with a small positive coefficient,
and reports:

- a few before/after generations for qualitative review,
- a steering success rate vs. a no-steering baseline (lexicon-based judge).

All console output is also written to
``outputs/<run_id>/reproduce_sentiment.log`` for reproducibility, alongside
``config.json`` and ``env.json``.

Run:
    python phases/phase_03_llms_and_steering_baselines/experiments/reproduce_sentiment.py
"""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import torch

from ot_steering.activations.datasets import load_sentiment_pairs
from ot_steering.activations.extractor import extract_residual_stream, resolve_blocks
from ot_steering.activations.model_loader import ModelLoaderConfig, load_model
from ot_steering.eval.steering_eval import _sentiment_judge
from ot_steering.steering.baselines import (
    apply_steering_vector,
    difference_in_means,
)
from ot_steering.utils.seed import set_all_seeds

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUTS_DIR = _PROJECT_ROOT / "outputs"

CONFIG: dict[str, object] = {
    "model_id": "gpt2",
    "hook_layer": 6,
    # Coefficient is applied to a UNIT-NORMALISED direction so it's
    # interpretable as "this many residual-stream-norm units of perturbation".
    "coefficient": 6.0,
    "seed": 0,
    "max_new_tokens": 30,
    "n_qualitative_examples": 4,
    "eval_split": 30,  # use first 30 pairs to build direction, last 20 to evaluate
}


def _run_id(config: dict[str, object]) -> str:
    iso = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    digest = hashlib.blake2b(
        json.dumps(config, sort_keys=True).encode("utf-8"),
        digest_size=4,
    ).hexdigest()
    return f"{iso}_{digest}"


def main() -> int:
    set_all_seeds(int(CONFIG["seed"]))

    run_id = _run_id(CONFIG)
    run_dir = OUTPUTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    buf = io.StringIO()

    class _Tee:
        def __init__(self, *streams: io.TextIOBase) -> None:
            self.streams = streams

        def write(self, data: str) -> int:
            for s in self.streams:
                s.write(data)
            return len(data)

        def flush(self) -> None:
            for s in self.streams:
                s.flush()

    tee = _Tee(sys.stdout, buf)  # type: ignore[arg-type]
    with redirect_stdout(tee):  # type: ignore[arg-type]
        _run_pipeline(run_dir)

    log_path = run_dir / "reproduce_sentiment.log"
    log_path.write_text(buf.getvalue(), encoding="utf-8")
    (run_dir / "config.json").write_text(json.dumps(CONFIG, indent=2), encoding="utf-8")
    return 0


def _run_pipeline(run_dir: Path) -> None:
    print(f"=== reproduce_sentiment.py — run_dir={run_dir.name} ===")
    print(f"config: {json.dumps(CONFIG, indent=2)}")

    model, tokenizer = load_model(ModelLoaderConfig(model_id=str(CONFIG["model_id"])))

    pairs = load_sentiment_pairs()
    split = int(CONFIG["eval_split"])
    train_pairs = pairs[:split]
    eval_pairs = pairs[split:]
    print(f"\nbuilt direction from {len(train_pairs)} pairs; evaluating on {len(eval_pairs)}")

    pos_prompts = [p for p, _ in train_pairs]
    neg_prompts = [n for _, n in train_pairs]

    hook_layer = int(CONFIG["hook_layer"])
    print(f"\nextracting activations at hidden_states[{hook_layer}] ...")
    acts_pos = extract_residual_stream(
        model, tokenizer, pos_prompts, layer_indices=[hook_layer], batch_size=8
    )[hook_layer]
    acts_neg = extract_residual_stream(
        model, tokenizer, neg_prompts, layer_indices=[hook_layer], batch_size=8
    )[hook_layer]
    print(f"  pos activations shape: {tuple(acts_pos.shape)}")
    print(f"  neg activations shape: {tuple(acts_neg.shape)}")

    raw_direction = difference_in_means(acts_pos, acts_neg)
    print(f"  raw direction norm: {raw_direction.norm().item():.3f}")
    direction = raw_direction / raw_direction.norm().clamp(min=1e-8)
    print("  normalised to unit length for steering")

    # Qualitative: a few negative-class prompts before/after steering.
    n_qual = int(CONFIG["n_qualitative_examples"])
    coef = float(CONFIG["coefficient"])
    sample_negatives = [n for _, n in eval_pairs[:n_qual]]
    print(f"\nqualitative generations (coefficient={coef:+.1f}):")
    baseline_outs = apply_steering_vector(
        model,
        tokenizer,
        sample_negatives,
        block_resolver=resolve_blocks,
        hook_layer=hook_layer,
        direction=direction,
        coefficient=0.0,
        max_new_tokens=int(CONFIG["max_new_tokens"]),
    )
    steered_outs = apply_steering_vector(
        model,
        tokenizer,
        sample_negatives,
        block_resolver=resolve_blocks,
        hook_layer=hook_layer,
        direction=direction,
        coefficient=coef,
        max_new_tokens=int(CONFIG["max_new_tokens"]),
    )
    for prompt, base, steered in zip(sample_negatives, baseline_outs, steered_outs, strict=True):
        print(f"\n  prompt   : {prompt}")
        print(f"  baseline : {base.strip()[:150]}")
        print(f"  steered+ : {steered.strip()[:150]}")

    # Quantitative: ActAdd-style. For each negative prompt in the eval
    # split, generate twice (baseline, steered+) and call it a success
    # whenever the steered continuation lands MORE-positive than the
    # baseline under the lexicon judge.
    print("\nquantitative — sentiment lift on negative-class eval prompts:")
    eval_negatives = [n for _, n in eval_pairs]
    baseline_eval = apply_steering_vector(
        model,
        tokenizer,
        eval_negatives,
        block_resolver=resolve_blocks,
        hook_layer=hook_layer,
        direction=direction,
        coefficient=0.0,
        max_new_tokens=int(CONFIG["max_new_tokens"]),
    )
    steered_eval = apply_steering_vector(
        model,
        tokenizer,
        eval_negatives,
        block_resolver=resolve_blocks,
        hook_layer=hook_layer,
        direction=direction,
        coefficient=coef,
        max_new_tokens=int(CONFIG["max_new_tokens"]),
    )
    flips = sum(
        1
        for base, steer in zip(baseline_eval, steered_eval, strict=True)
        if _sentiment_judge(steer) > _sentiment_judge(base)
    )
    held_steady = sum(
        1
        for base, steer in zip(baseline_eval, steered_eval, strict=True)
        if _sentiment_judge(steer) == _sentiment_judge(base)
    )
    regressions = sum(
        1
        for base, steer in zip(baseline_eval, steered_eval, strict=True)
        if _sentiment_judge(steer) < _sentiment_judge(base)
    )
    n_eval = len(eval_negatives)
    flip_rate = flips / max(n_eval, 1)
    regression_rate = regressions / max(n_eval, 1)
    print(f"  n_eval                  : {n_eval}")
    print(f"  positive shifts         : {flips} ({flip_rate:.2%})")
    print(f"  unchanged               : {held_steady}")
    print(f"  negative shifts         : {regressions} ({regression_rate:.2%})")
    print(f"  net lift                : {(flips - regressions) / max(n_eval, 1):+.2%}")

    metrics = {
        "positive_shift_rate": flip_rate,
        "regression_rate": regression_rate,
        "net_lift": (flips - regressions) / max(n_eval, 1),
        "raw_direction_norm": raw_direction.norm().item(),
        "n_train_pairs": len(train_pairs),
        "n_eval_pairs": n_eval,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Release model.
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
