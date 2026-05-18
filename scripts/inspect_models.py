#!/usr/bin/env python
"""One-shot inspection of every model family Phase 3 needs to support.

For each model, this prints:
- model class name and the path to the transformer-block container,
- layer count and ``d_model``,
- tokenizer ``pad_token`` / ``eos_token`` / chat-template quirks,
- the residual-stream tensor shape captured by a forward hook,
- VRAM used after load (and after free).

Run BEFORE writing model_loader.py / extractor.py. Per CLAUDE.md's
"External interfaces — inspect before integrating".
"""

from __future__ import annotations

import gc
import traceback

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def _vram_mb() -> float:
    if not torch.cuda.is_available():
        return float("nan")
    return torch.cuda.memory_allocated() / (1024**2)


def _find_blocks_attr(model: torch.nn.Module) -> tuple[str, int]:
    """Return (dotted-attr-path, layer_count) of the transformer-block container."""
    candidates = [
        "gpt_neox.layers",
        "transformer.h",
        "model.layers",
        "model.decoder.layers",
    ]
    for path in candidates:
        cur: object = model
        ok = True
        for part in path.split("."):
            if not hasattr(cur, part):
                ok = False
                break
            cur = getattr(cur, part)
        if ok and hasattr(cur, "__len__"):
            return path, len(cur)  # type: ignore[arg-type]
    raise RuntimeError(f"could not find transformer blocks in {type(model).__name__}")


def _get_attr(model: torch.nn.Module, path: str) -> torch.nn.Module:
    cur: object = model
    for part in path.split("."):
        cur = getattr(cur, part)
    return cur  # type: ignore[return-value]


def inspect_one(model_id: str, *, quantization: str | None = None) -> None:
    print()
    print("=" * 78)
    print(f"INSPECTING: {model_id}   quantization={quantization!r}")
    print("=" * 78)

    print(f"  [vram before load] {_vram_mb():.1f} MB")
    tok = AutoTokenizer.from_pretrained(model_id)
    print(f"  tokenizer class : {type(tok).__name__}")
    print(f"  pad_token       : {tok.pad_token!r} (id={tok.pad_token_id})")
    print(f"  eos_token       : {tok.eos_token!r} (id={tok.eos_token_id})")
    print(f"  bos_token       : {tok.bos_token!r} (id={tok.bos_token_id})")
    print(f"  chat_template   : {'YES' if tok.chat_template else 'no'}")

    kwargs: dict[str, object] = {"torch_dtype": torch.float16}
    if quantization == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        kwargs.pop("torch_dtype")  # bnb manages dtype
        kwargs["device_map"] = "auto"
    else:
        kwargs["device_map"] = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    print(f"  model class     : {type(model).__name__}")
    print(f"  config class    : {type(model.config).__name__}")
    print(f"  hidden_size     : {model.config.hidden_size}")
    print("  n_layer / n_hidden_layers / num_hidden_layers...")
    for attr in ("num_hidden_layers", "n_layer", "n_layers"):
        if hasattr(model.config, attr):
            print(f"     {attr} = {getattr(model.config, attr)}")
    blocks_path, n_layers = _find_blocks_attr(model)
    print(f"  blocks attribute: {blocks_path}  (count={n_layers})")
    print(f"  block 0 class   : {type(_get_attr(model, blocks_path)[0]).__name__}")  # type: ignore[index]
    print(f"  [vram after load] {_vram_mb():.1f} MB")

    # Try a forward hook on the last block and report what it sees.
    last_block = _get_attr(model, blocks_path)[n_layers - 1]  # type: ignore[index]
    captured: list[tuple[type, tuple, dict]] = []

    def hook(_module, args, kwargs):  # type: ignore[no-untyped-def]
        captured.append(
            (type(args), tuple(type(a) for a in args), {k: type(v) for k, v in kwargs.items()})
        )
        return None

    handle = last_block.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        prompt = "Hello, world."
        toks = tok(prompt, return_tensors="pt")
        toks = {k: v.to(next(model.parameters()).device) for k, v in toks.items()}
        with torch.no_grad():
            out = model(**toks, output_hidden_states=True)
        print(f"  forward OK, captured {len(captured)} pre-hook event(s)")
        for ev in captured:
            print(f"    -> args types={ev[1]} kwargs={ev[2]}")
        hs = out.hidden_states
        print(f"  hidden_states tuple length: {len(hs)}  (includes embeddings + each layer)")
        print(f"  hidden_states[-1].shape   : {tuple(hs[-1].shape)}  (B, T, d)")
        print(f"  hidden_states[0 ].shape   : {tuple(hs[0].shape)}  (post-embed)")
    finally:
        handle.remove()

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"  [vram after free] {_vram_mb():.1f} MB")


def main() -> int:
    targets: list[tuple[str, str | None]] = [
        ("EleutherAI/pythia-160m", None),
        ("gpt2", None),
        ("Qwen/Qwen2.5-0.5B", None),
        ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "4bit"),
    ]
    failures: list[str] = []
    for model_id, quant in targets:
        try:
            inspect_one(model_id, quantization=quant)
        except Exception as e:  # noqa: BLE001 - surface every failure verbatim
            traceback.print_exc()
            failures.append(f"{model_id}: {e}")

    print()
    print("=" * 78)
    if failures:
        print(f"FAILURES ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("All four model families inspected successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
