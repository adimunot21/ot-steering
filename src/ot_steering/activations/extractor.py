"""Residual-stream activation extractor.

Hugging Face's ``output_hidden_states=True`` already returns the residual
stream at every layer boundary in one forward pass; we wrap that into a
batched, position-aware, family-agnostic helper.

The hidden-states tuple is laid out as

    hidden_states[0]                 # post-embedding (input to block 0)
    hidden_states[k]  for k in [1, L]  # output of block k-1 == input to block k
    hidden_states[L]                 # final residual stream (input to ln_f)

so ``hidden_states[k]`` is "the residual stream at layer k" in the sense
mech-interp papers use. Indexing here uses that same convention.

The helper :func:`resolve_blocks` is also exported because the steering
baselines need to know which ``nn.Module`` to attach a forward-pre-hook to,
and that path is model-family-specific.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from ot_steering.utils.logging import get_logger

_log = get_logger(__name__)

Position = Literal["last_token", "all"]

# Per the Phase 3 inspection. Order matters: GPTNeoX exposes both
# `gpt_neox.layers` and a partial `transformer.h` shim that we want to
# avoid, so the NeoX path goes first.
_BLOCK_ATTR_CANDIDATES: tuple[str, ...] = (
    "gpt_neox.layers",
    "transformer.h",
    "model.layers",
    "model.decoder.layers",
)


def resolve_blocks(model: PreTrainedModel) -> list[torch.nn.Module]:
    """Return the transformer blocks of ``model`` in layer order.

    Tries the known attribute paths (GPT-NeoX, GPT-2, Llama/Qwen, OPT/BART)
    in order and returns the first one that resolves.

    Args:
        model: A loaded causal LM.

    Returns:
        Ordered list of transformer blocks.

    Raises:
        RuntimeError: If no known block container can be found.
    """
    for path in _BLOCK_ATTR_CANDIDATES:
        cur: object = model
        ok = True
        for part in path.split("."):
            if not hasattr(cur, part):
                ok = False
                break
            cur = getattr(cur, part)
        if ok and hasattr(cur, "__len__") and hasattr(cur, "__getitem__"):
            return [cur[i] for i in range(len(cur))]
    raise RuntimeError(
        f"could not locate transformer blocks on {type(model).__name__}; "
        f"tried {_BLOCK_ATTR_CANDIDATES!r}"
    )


def _select_position(hidden: torch.Tensor, position: Position | int) -> torch.Tensor:
    """Pick a position slice from a ``(B, T, d)`` hidden-state tensor."""
    if position == "all":
        return hidden
    if position == "last_token":
        return hidden[:, -1, :]
    if isinstance(position, int):
        if not -hidden.shape[1] <= position < hidden.shape[1]:
            raise ValueError(f"position={position} out of range for seq_len={hidden.shape[1]}")
        return hidden[:, position, :]
    raise ValueError(f"unknown position spec: {position!r}")


def extract_residual_stream(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: Sequence[str],
    layer_indices: Sequence[int],
    *,
    position: Position | int = "last_token",
    batch_size: int = 8,
    max_length: int | None = None,
) -> dict[int, torch.Tensor]:
    """Run ``model`` on ``prompts`` and return residual streams at given layers.

    Activations are moved to CPU after each batch so peak VRAM stays bounded
    by one batch's hidden state, not the whole dataset.

    Args:
        model: Loaded causal LM.
        tokenizer: Matching tokenizer with a usable ``pad_token``.
        prompts: Sequence of text prompts.
        layer_indices: Layer indices to extract; the convention is
            ``0`` = post-embedding, ``1..n_layers`` = output of block ``k-1``,
            matching ``hidden_states`` from
            ``model(..., output_hidden_states=True)``.
        position: ``"last_token"`` → ``(n_prompts, d_model)``;
            ``"all"`` → ``(n_prompts, max_seq_len, d_model)`` (padded with
            zeros to the batch-wide max length per batch — caller must
            attend to attention masks if needed);
            ``int`` → ``(n_prompts, d_model)`` at that absolute position.
        batch_size: Prompts per forward pass.
        max_length: Tokenization truncation cap; ``None`` uses the model max.

    Returns:
        Dict keyed by layer index with CPU tensors as values.

    Raises:
        ValueError: If a layer index is out of range or position is invalid.
    """
    if not prompts:
        raise ValueError("prompts must be a non-empty sequence")
    if not layer_indices:
        raise ValueError("layer_indices must be a non-empty sequence")

    device = next(model.parameters()).device
    n_layers_plus_one = model.config.num_hidden_layers + 1
    for li in layer_indices:
        if not 0 <= li < n_layers_plus_one:
            raise ValueError(
                f"layer_indices contains {li}; valid range is [0, {n_layers_plus_one})"
            )

    _log.info(
        "extract: n_prompts=%d layers=%s position=%s batch=%d",
        len(prompts),
        list(layer_indices),
        position,
        batch_size,
    )

    # Accumulator: layer_index -> list of (B, ..., d_model) cpu tensors.
    per_layer: dict[int, list[torch.Tensor]] = {li: [] for li in layer_indices}

    model.eval()  # type: ignore[no-untyped-call]
    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            batch = list(prompts[start : start + batch_size])
            enc = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)
            outputs = model(
                **enc,
                output_hidden_states=True,
                use_cache=False,
            )
            for li in layer_indices:
                hidden = outputs.hidden_states[li]  # (B, T, d)
                sliced = _select_position(hidden, position)
                per_layer[li].append(sliced.detach().to("cpu", dtype=torch.float32))
            del outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    result: dict[int, torch.Tensor] = {}
    for li, chunks in per_layer.items():
        if position == "all":
            # Each chunk is (B_i, T_i, d). Pad to the max T across chunks
            # and concatenate along batch.
            max_t = max(c.shape[1] for c in chunks)
            d_model = chunks[0].shape[-1]
            padded = []
            for c in chunks:
                if c.shape[1] < max_t:
                    pad = torch.zeros(c.shape[0], max_t - c.shape[1], d_model)
                    padded.append(torch.cat([c, pad], dim=1))
                else:
                    padded.append(c)
            result[li] = torch.cat(padded, dim=0)
        else:
            result[li] = torch.cat(chunks, dim=0)
    return result
