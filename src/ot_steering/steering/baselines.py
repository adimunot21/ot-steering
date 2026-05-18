"""Activation-steering baselines used as references throughout the project.

All three baselines build a single direction ``v ∈ R^d_model`` from a pair
of contrastive activation sets (positive vs. negative, true vs. false,
harmful vs. harmless). The direction is what later gets added (or
subtracted) to the residual stream during generation to steer the model.

The connection to optimal transport — the bridge Chapter 1 ended on — is
explicit: when the two activation distributions are Gaussians with equal
covariance, the OT-induced map from one to the other is the pure
translation by their difference of means. That is exactly the
:func:`difference_in_means` baseline. Phase 4 generalises this to OT maps
between Gaussian mixtures (CHaRS); Phase 6 generalises it across models
with Gromov-Wasserstein.

Three baselines provided:

- :func:`difference_in_means` — naïve ActAdd-style, the most common
  steering vector in the literature (Turner et al. 2023).
- :func:`mean_centered_steering` — Jorgensen et al. 2024: subtract the
  *global* per-class mean before averaging, which removes the dataset's
  "task vector" and keeps only the polarity direction.
- :func:`apply_steering_vector` — runtime injection: attach a forward
  pre-hook to the residual stream at ``layer`` that adds
  ``coefficient * direction`` while text is generated.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch

from ot_steering.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Generator

    from transformers import PreTrainedModel, PreTrainedTokenizerBase

_log = get_logger(__name__)


def difference_in_means(
    positive_acts: torch.Tensor,
    negative_acts: torch.Tensor,
) -> torch.Tensor:
    """Return the ActAdd-style steering direction.

    Computes ``mean(positive_acts) - mean(negative_acts)``. No normalisation;
    callers scale via the ``coefficient`` parameter on
    :func:`apply_steering_vector`.

    Args:
        positive_acts: Activations from the positive-class prompts, shape
            ``(n_pos, d_model)``.
        negative_acts: Activations from the negative-class prompts, shape
            ``(n_neg, d_model)``.

    Returns:
        Direction of shape ``(d_model,)``.

    Raises:
        ValueError: If shapes are inconsistent.
    """
    if positive_acts.ndim != 2 or negative_acts.ndim != 2:
        raise ValueError(
            f"expected 2-D activations; got pos={positive_acts.shape}, neg={negative_acts.shape}"
        )
    if positive_acts.shape[1] != negative_acts.shape[1]:
        raise ValueError(
            f"d_model mismatch: pos={positive_acts.shape[1]} vs neg={negative_acts.shape[1]}"
        )
    return positive_acts.float().mean(dim=0) - negative_acts.float().mean(dim=0)


def mean_centered_steering(
    positive_acts: torch.Tensor,
    negative_acts: torch.Tensor,
) -> torch.Tensor:
    """Return the Jorgensen-style mean-centred steering direction.

    Subtracts the global (pooled) mean of both classes from each, then
    takes the difference of means. Removes any constant "task vector"
    common to both contrastive sets and leaves only the polarity axis.

    Args:
        positive_acts: ``(n_pos, d_model)``.
        negative_acts: ``(n_neg, d_model)``.

    Returns:
        Direction of shape ``(d_model,)``.
    """
    if positive_acts.shape[1] != negative_acts.shape[1]:
        raise ValueError(
            f"d_model mismatch: pos={positive_acts.shape[1]} vs neg={negative_acts.shape[1]}"
        )
    pooled = torch.cat([positive_acts.float(), negative_acts.float()], dim=0)
    global_mean = pooled.mean(dim=0, keepdim=True)
    pos = positive_acts.float() - global_mean
    neg = negative_acts.float() - global_mean
    return pos.mean(dim=0) - neg.mean(dim=0)


@contextmanager
def add_residual_steering_hook(
    block: torch.nn.Module,
    direction: torch.Tensor,
    coefficient: float,
) -> Generator[None, None, None]:
    """Context manager that adds ``coefficient * direction`` to a block's input.

    Attaches a ``forward_pre_hook`` to the given transformer block. The hook
    sees the block's positional argument (the residual stream at that depth,
    shape ``(batch, seq, d_model)``) and adds the broadcast direction. The
    hook is removed automatically on context exit.

    Args:
        block: The transformer block to steer (e.g. ``model.transformer.h[6]``
            for GPT-2; the project's :mod:`activations.extractor` knows how
            to find it for every supported family).
        direction: Steering direction, shape ``(d_model,)``.
        coefficient: Scalar multiplier.

    Yields:
        Nothing — this is a context manager.
    """
    if direction.ndim != 1:
        raise ValueError(f"direction must be 1-D, got shape {direction.shape}")

    def _pre_hook(_module, args):  # type: ignore[no-untyped-def]
        if not args:
            return args
        hidden = args[0]
        device = hidden.device
        dtype = hidden.dtype
        steered = hidden + coefficient * direction.to(device=device, dtype=dtype)
        return (steered, *args[1:])

    handle = block.register_forward_pre_hook(_pre_hook)
    try:
        yield
    finally:
        handle.remove()


def apply_steering_vector(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: Sequence[str],
    *,
    block_resolver: Callable[[PreTrainedModel], Sequence[torch.nn.Module]],
    hook_layer: int,
    direction: torch.Tensor,
    coefficient: float,
    max_new_tokens: int = 40,
    do_sample: bool = False,
) -> list[str]:
    """Generate from ``model`` with a steering vector injected at ``hook_layer``.

    Args:
        model: A causal LM (anything ``AutoModelForCausalLM`` can return).
        tokenizer: Matching tokenizer.
        prompts: Sequence of prompt strings; one generation per prompt.
        block_resolver: Callback that, given the model, returns the
            ordered list of transformer blocks. Use
            :func:`ot_steering.activations.extractor.resolve_blocks`.
        hook_layer: Index into ``block_resolver(model)`` at which to inject
            the steering vector.
        direction: ``(d_model,)`` steering direction.
        coefficient: Scalar multiplier.
        max_new_tokens: Max tokens to generate per prompt.
        do_sample: If False (default), use greedy decoding for reproducibility.

    Returns:
        List of generated strings (the *continuation only*, not the prompt).
    """
    blocks = list(block_resolver(model))
    if not 0 <= hook_layer < len(blocks):
        raise ValueError(f"hook_layer {hook_layer} out of range [0, {len(blocks)})")
    block = blocks[hook_layer]

    device = next(model.parameters()).device
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    outputs: list[str] = []
    model.eval()  # type: ignore[no-untyped-call]
    with add_residual_steering_hook(block, direction, coefficient), torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt", padding=False).to(device)
            generated = model.generate(  # type: ignore[operator]
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id,
            )
            # Strip prompt tokens to return continuation only.
            continuation_ids = generated[0, inputs["input_ids"].shape[1] :]
            text = tokenizer.decode(continuation_ids, skip_special_tokens=True)
            assert isinstance(text, str)  # decode of 1-D ids returns a single string
            outputs.append(text)
    return outputs
