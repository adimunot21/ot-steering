"""Tests for ot_steering.activations.extractor.

The "smoke" tests load Pythia-160M (~325 MB) and run a tiny forward pass —
they are marked ``slow`` and skipped in the default ``pytest -q``. Run them
with ``pytest -m slow tests/activations/test_extractor.py``.

The fast tests cover :func:`resolve_blocks`'s family-aware lookup using a
minimal stub model.
"""

from __future__ import annotations

import pytest
import torch

from ot_steering.activations.extractor import resolve_blocks


class _StubBlocks(torch.nn.ModuleList):
    pass


class _StubGPT2Like(torch.nn.Module):
    """Mimics ``model.transformer.h``."""

    def __init__(self, n_layers: int = 3) -> None:
        super().__init__()
        self.transformer = torch.nn.Module()
        self.transformer.h = _StubBlocks(  # type: ignore[attr-defined]
            [torch.nn.Linear(4, 4) for _ in range(n_layers)]
        )


class _StubLlamaLike(torch.nn.Module):
    """Mimics ``model.model.layers``."""

    def __init__(self, n_layers: int = 5) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = _StubBlocks(  # type: ignore[attr-defined]
            [torch.nn.Linear(4, 4) for _ in range(n_layers)]
        )


class _StubNeoXLike(torch.nn.Module):
    """Mimics ``model.gpt_neox.layers``."""

    def __init__(self, n_layers: int = 4) -> None:
        super().__init__()
        self.gpt_neox = torch.nn.Module()
        self.gpt_neox.layers = _StubBlocks(  # type: ignore[attr-defined]
            [torch.nn.Linear(4, 4) for _ in range(n_layers)]
        )


def test_resolve_blocks_finds_gpt2_layout() -> None:
    blocks = resolve_blocks(_StubGPT2Like(n_layers=3))  # type: ignore[arg-type]
    assert len(blocks) == 3


def test_resolve_blocks_finds_llama_layout() -> None:
    blocks = resolve_blocks(_StubLlamaLike(n_layers=5))  # type: ignore[arg-type]
    assert len(blocks) == 5


def test_resolve_blocks_finds_neox_layout() -> None:
    blocks = resolve_blocks(_StubNeoXLike(n_layers=4))  # type: ignore[arg-type]
    assert len(blocks) == 4


def test_resolve_blocks_raises_on_unknown_layout() -> None:
    class _Mystery(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weird = torch.nn.Linear(2, 2)

    with pytest.raises(RuntimeError, match="could not locate"):
        resolve_blocks(_Mystery())  # type: ignore[arg-type]


@pytest.mark.slow
def test_extract_residual_stream_pythia160m_smoke() -> None:
    """Integration test — downloads ~325 MB on first run, then cached."""
    from ot_steering.activations.extractor import extract_residual_stream
    from ot_steering.activations.model_loader import ModelLoaderConfig, load_model
    from ot_steering.utils.seed import set_all_seeds

    set_all_seeds(0)
    model, tok = load_model(ModelLoaderConfig(model_id="EleutherAI/pythia-160m"))

    prompts: list[str] = ["Hello, world.", "The quick brown fox jumps."]
    acts = extract_residual_stream(
        model,
        tok,
        prompts,
        layer_indices=[0, 6, 12],
        position="last_token",
        batch_size=2,
    )
    assert set(acts.keys()) == {0, 6, 12}
    for layer_idx, t in acts.items():
        assert t.shape == (2, 768), f"layer {layer_idx} bad shape {t.shape}"
        assert t.dtype == torch.float32

    # 'all' position returns the per-batch padded sequence dim.
    acts_all = extract_residual_stream(
        model, tok, prompts, layer_indices=[6], position="all", batch_size=2
    )
    assert acts_all[6].ndim == 3
    assert acts_all[6].shape[0] == 2
    assert acts_all[6].shape[2] == 768
