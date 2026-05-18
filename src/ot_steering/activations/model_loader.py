"""Unified loader for the four model families Phase 3 supports.

Wraps ``transformers.AutoModelForCausalLM`` so every caller goes through one
chokepoint that:

- validates inputs against a pydantic config (typo-resistant),
- chooses sensible HF defaults per model family (notably: aliasing
  ``pad_token`` to ``eos_token`` for GPT-2, which ships without a pad token),
- configures 4-bit/8-bit quantisation correctly when asked
  (NF4 + double quant for 4-bit, matching what fit TinyLlama-1.1B into 4 GB
  VRAM in the Phase 3 inspection),
- logs VRAM usage after load so we notice when something balloons.

Findings from ``scripts/inspect_models.py`` driving the design here:

============================================  ==========================  ============  ========  ==========  =========================
model_id                                       class                       blocks attr   layers    d_model     pad / eos quirks
============================================  ==========================  ============  ========  ==========  =========================
EleutherAI/pythia-160m                         GPTNeoXForCausalLM          gpt_neox.layers  12     768         pad=<|padding|>, eos=<|endoftext|>
gpt2                                           GPT2LMHeadModel             transformer.h    12     768         pad=None → we alias eos
Qwen/Qwen2.5-0.5B                              Qwen2ForCausalLM            model.layers     24     896         pad=eos=<|endoftext|>
TinyLlama/TinyLlama-1.1B-Chat-v1.0 (4-bit)     LlamaForCausalLM            model.layers     22     2048        pad=eos=</s>, bos=<s>
============================================  ==========================  ============  ========  ==========  =========================
"""

from __future__ import annotations

from typing import Literal

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from ot_steering.utils.config import BaseConfig
from ot_steering.utils.logging import get_logger

_log = get_logger(__name__)

Device = Literal["cuda", "cpu"]
DType = Literal["float16", "bfloat16", "float32"]
Quantization = Literal["4bit", "8bit"]

_DTYPE_MAP: dict[DType, torch.dtype] = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


class ModelLoaderConfig(BaseConfig):
    """Configuration for :func:`load_model`.

    Attributes:
        model_id: Hugging Face model identifier
            (e.g. ``"gpt2"``, ``"EleutherAI/pythia-160m"``).
        device: Target device. Quantised models always use ``device_map="auto"``
            internally regardless of this setting.
        dtype: Tensor dtype for the unquantised path. Ignored when
            ``quantization`` is set.
        quantization: ``"4bit"`` uses NF4 + double quant with fp16 compute,
            ``"8bit"`` uses int8, ``None`` loads in full ``dtype``.
        trust_remote_code: If True, allow HF to execute custom modelling
            code shipped in the model repo. The supported model families
            (GPT-NeoX, GPT-2, Qwen2, Llama) do *not* require it; leave
            False unless adding a model that does.
    """

    model_id: str
    device: Device = "cuda"
    dtype: DType = "float16"
    quantization: Quantization | None = None
    trust_remote_code: bool = False


def _vram_mb() -> float:
    if not torch.cuda.is_available():
        return float("nan")
    return torch.cuda.memory_allocated() / (1024**2)


def _bnb_config(quantization: Quantization) -> BitsAndBytesConfig:
    if quantization == "4bit":
        return BitsAndBytesConfig(  # type: ignore[no-untyped-call]
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    return BitsAndBytesConfig(load_in_8bit=True)  # type: ignore[no-untyped-call]


def load_model(
    cfg: ModelLoaderConfig,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a causal LM and its tokenizer.

    Args:
        cfg: A validated :class:`ModelLoaderConfig`.

    Returns:
        ``(model, tokenizer)``. The model is in ``eval()`` mode. The
        tokenizer always has a usable ``pad_token`` (aliased to ``eos_token``
        when the original tokenizer does not ship one — required for batched
        inference with GPT-2-family models).

    Raises:
        RuntimeError: If ``device="cuda"`` and CUDA is not available.
    """
    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cfg.device='cuda' but torch.cuda.is_available() is False")

    _log.info(
        "loading model %s (device=%s, dtype=%s, quant=%s)",
        cfg.model_id,
        cfg.device,
        cfg.dtype,
        cfg.quantization,
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, trust_remote_code=cfg.trust_remote_code)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError(f"tokenizer for {cfg.model_id} has neither pad nor eos token")
        tokenizer.pad_token = tokenizer.eos_token
        _log.info("aliased pad_token := eos_token for %s", cfg.model_id)

    kwargs: dict[str, object] = {"trust_remote_code": cfg.trust_remote_code}
    if cfg.quantization is not None:
        kwargs["quantization_config"] = _bnb_config(cfg.quantization)
        kwargs["device_map"] = "auto"
    else:
        kwargs["dtype"] = _DTYPE_MAP[cfg.dtype]
        kwargs["device_map"] = {"": cfg.device}

    before = _vram_mb()
    model = AutoModelForCausalLM.from_pretrained(cfg.model_id, **kwargs)
    model.eval()  # type: ignore[no-untyped-call]
    after = _vram_mb()
    _log.info(
        "loaded %s (%s) — VRAM %.1f → %.1f MB (Δ %+0.1f)",
        cfg.model_id,
        type(model).__name__,
        before,
        after,
        after - before,
    )
    return model, tokenizer
