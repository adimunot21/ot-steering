"""Evaluation harness for steering vectors.

Two metrics that every later phase will reuse:

- :func:`steering_success_rate` — for a list of contrastive prompt pairs,
  what fraction of the *negative* prompts produce a continuation that an
  LM-as-judge classifies as the *positive* class once we add
  ``coefficient * direction`` to the residual stream at ``hook_layer``?

- :func:`off_target_perplexity` — what is the model's perplexity on a small
  held-out corpus of neutral text *with* the steering vector active?
  Compared to the no-steering perplexity, this is the off-target damage
  the steering does to unrelated capability.

The judge for ``steering_success_rate`` is a lightweight, lexicon-based
heuristic (positive/negative word lists) for sentiment, true/false for
truthfulness, and refusal-keyword detection for refusal. This avoids
spinning up a separate model for evaluation — the phase 3 chapter is honest
about the limitation and points at fancier alternatives.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Literal

import torch
import torch.nn.functional as F  # noqa: N812  ('F' is the universal PyTorch convention)

from ot_steering.steering.baselines import (
    add_residual_steering_hook,
    apply_steering_vector,
)
from ot_steering.utils.logging import get_logger

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

_log = get_logger(__name__)

ConceptKind = Literal["sentiment", "truthfulness", "refusal"]

# Tiny opinion lexicons. Match the vocabulary in our sentiment YAML on
# purpose — the eval is checking whether the steered generation veers
# toward the positive lexicon.
_POSITIVE_WORDS = {
    "great",
    "wonderful",
    "amazing",
    "excellent",
    "perfect",
    "love",
    "loved",
    "fantastic",
    "delightful",
    "beautiful",
    "lovely",
    "stunning",
    "outstanding",
    "brilliant",
    "warm",
    "happy",
    "joy",
    "enjoy",
    "best",
    "good",
    "nice",
    "kind",
    "helpful",
    "comfortable",
    "fresh",
    "crisp",
    "rich",
    "smooth",
    "satisfying",
    "remarkable",
    "incredible",
    "fun",
    "elegant",
    "graceful",
}
_NEGATIVE_WORDS = {
    "terrible",
    "awful",
    "horrible",
    "bad",
    "worst",
    "hate",
    "hated",
    "boring",
    "tedious",
    "dull",
    "ugly",
    "sad",
    "angry",
    "frustrating",
    "disappointing",
    "stale",
    "bland",
    "miserable",
    "tasteless",
    "rude",
    "useless",
    "broken",
    "painful",
    "uncomfortable",
    "annoying",
    "poor",
    "weak",
    "cold",
    "rough",
}

_REFUSAL_PHRASES = (
    "i can't",
    "i cannot",
    "i'm sorry",
    "i am sorry",
    "i won't",
    "i will not",
    "i'm unable",
    "i am unable",
    "as an ai",
    "i don't feel comfortable",
    "i must decline",
    "i'm not able",
)


def _sentiment_judge(text: str) -> int:
    """Classify ``text`` as +1 (positive), -1 (negative), or 0 (neutral)."""
    lower = text.lower()
    pos = sum(1 for w in _POSITIVE_WORDS if w in lower)
    neg = sum(1 for w in _NEGATIVE_WORDS if w in lower)
    if pos == neg:
        return 0
    return 1 if pos > neg else -1


def _refusal_judge(text: str) -> bool:
    """True if ``text`` looks like a refusal."""
    lower = text.lower()
    return any(p in lower for p in _REFUSAL_PHRASES)


def steering_success_rate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    eval_pairs: Sequence[tuple[str, str]],
    *,
    block_resolver: Callable[[PreTrainedModel], Sequence[torch.nn.Module]],
    hook_layer: int,
    direction: torch.Tensor,
    coefficient: float,
    concept: ConceptKind = "sentiment",
    max_new_tokens: int = 30,
) -> float:
    """Fraction of steered generations that flip toward the target class.

    For each ``(class_A, class_B)`` pair the function generates two outputs
    with the steering vector active: one prompted with the class_A prompt,
    one with the class_B prompt. The pair counts as a "success" when

    - sentiment: class_A continuation reads as more positive than the
      class_B continuation (per the lexicon judge);
    - truthfulness: not yet implemented as a runtime judge — returns 0.0 with
      a warning rather than guessing;
    - refusal: class_A (harmful) continuation triggers the refusal judge.

    Args:
        model: Causal LM.
        tokenizer: Matching tokenizer.
        eval_pairs: List of contrastive prompt pairs.
        block_resolver: Returns the ordered list of transformer blocks for
            ``model``. Use ``ot_steering.activations.extractor.resolve_blocks``.
        hook_layer: Block index at which to inject the direction.
        direction: ``(d_model,)`` steering vector.
        coefficient: Scalar multiplier; sign decides which way to steer.
        concept: Which judge to use.
        max_new_tokens: Tokens generated per prompt.

    Returns:
        Success rate in ``[0, 1]``.
    """
    if concept == "truthfulness":
        _log.warning(
            "truthfulness steering eval is lexicon-free; returning 0.0 as a placeholder. "
            "Use a real fact-checking judge for headline numbers."
        )
        return 0.0

    a_prompts = [a for a, _ in eval_pairs]
    b_prompts = [b for _, b in eval_pairs]
    a_outs = apply_steering_vector(
        model,
        tokenizer,
        a_prompts,
        block_resolver=block_resolver,
        hook_layer=hook_layer,
        direction=direction,
        coefficient=coefficient,
        max_new_tokens=max_new_tokens,
    )
    b_outs = apply_steering_vector(
        model,
        tokenizer,
        b_prompts,
        block_resolver=block_resolver,
        hook_layer=hook_layer,
        direction=direction,
        coefficient=coefficient,
        max_new_tokens=max_new_tokens,
    )

    successes = 0
    if concept == "sentiment":
        for a_text, b_text in zip(a_outs, b_outs, strict=True):
            if _sentiment_judge(a_text) > _sentiment_judge(b_text):
                successes += 1
    else:  # refusal
        for a_text in a_outs:
            if _refusal_judge(a_text):
                successes += 1
    return successes / max(len(eval_pairs), 1)


def off_target_perplexity(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    neutral_texts: Sequence[str],
    *,
    block_resolver: Callable[[PreTrainedModel], Sequence[torch.nn.Module]] | None = None,
    hook_layer: int | None = None,
    direction: torch.Tensor | None = None,
    coefficient: float = 0.0,
    stride: int | None = None,
) -> float:
    """Mean per-token cross-entropy (perplexity) on ``neutral_texts``.

    If ``direction`` and ``hook_layer`` are provided AND
    ``coefficient != 0``, the steering vector is injected during the forward
    pass. With ``coefficient == 0`` (or ``direction is None``), the function
    returns the baseline perplexity.

    Args:
        model: Causal LM.
        tokenizer: Matching tokenizer.
        neutral_texts: Off-target corpus (e.g. a short Wikitext slice).
        block_resolver: Required iff steering is active.
        hook_layer: Required iff steering is active.
        direction: Optional steering direction; ``None`` → no steering.
        coefficient: Steering coefficient; ``0.0`` → no steering even if
            direction is given.
        stride: Per-text token cap, to keep eval cheap. ``None`` uses the
            full sequence.

    Returns:
        Mean perplexity (exp of mean per-token NLL) across the corpus.
    """
    steering_active = direction is not None and coefficient != 0.0
    if steering_active and (block_resolver is None or hook_layer is None):
        raise ValueError(
            "block_resolver and hook_layer are required when steering with non-zero coefficient"
        )

    device = next(model.parameters()).device
    model.eval()  # type: ignore[no-untyped-call]

    total_nll = 0.0
    total_tokens = 0

    def _score(text: str) -> tuple[float, int]:
        ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        if stride is not None:
            ids = ids[:, :stride]
        if ids.shape[1] < 2:
            return 0.0, 0
        with torch.no_grad():
            logits = model(ids).logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = ids[:, 1:].contiguous()
        nll = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="sum",
        )
        return float(nll.item()), int(shift_labels.numel())

    if steering_active:
        assert direction is not None and block_resolver is not None and hook_layer is not None
        block = list(block_resolver(model))[hook_layer]
        with add_residual_steering_hook(block, direction, coefficient):
            for text in neutral_texts:
                nll, ntok = _score(text)
                total_nll += nll
                total_tokens += ntok
    else:
        for text in neutral_texts:
            nll, ntok = _score(text)
            total_nll += nll
            total_tokens += ntok

    if total_tokens == 0:
        return float("nan")
    return float(torch.tensor(total_nll / total_tokens).exp().item())
