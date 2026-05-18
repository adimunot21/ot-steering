"""Tests for ot_steering.activations.cache."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ot_steering.activations.cache import cache_key, load_or_extract


def test_cache_key_is_deterministic_and_unique() -> None:
    k1 = cache_key("gpt2", "ds-v1", 6, "float16", "last_token")
    k2 = cache_key("gpt2", "ds-v1", 6, "float16", "last_token")
    assert k1 == k2

    k3 = cache_key("gpt2", "ds-v1", 7, "float16", "last_token")
    assert k1 != k3

    k4 = cache_key("EleutherAI/pythia-160m", "ds-v1", 6, "float16", "last_token")
    assert k1 != k4


def test_cache_miss_calls_extractor_then_writes(tmp_path: Path) -> None:
    calls = {"n": 0}

    def extractor() -> torch.Tensor:
        calls["n"] += 1
        return torch.arange(12).reshape(3, 4).float()

    key = cache_key("m", "d", 0, "float32", "last_token")
    out = load_or_extract(tmp_path, key, extractor)
    assert calls["n"] == 1
    assert (tmp_path / f"{key}.pt").is_file()
    assert out.shape == (3, 4)


def test_cache_hit_does_not_call_extractor(tmp_path: Path) -> None:
    key = cache_key("m", "d", 0, "float32", "last_token")
    expected = torch.linspace(0.0, 1.0, 16).reshape(4, 4)

    # Prime the cache.
    load_or_extract(tmp_path, key, lambda: expected)

    calls = {"n": 0}

    def extractor() -> torch.Tensor:
        calls["n"] += 1
        return torch.zeros(4, 4)

    out = load_or_extract(tmp_path, key, extractor)
    assert calls["n"] == 0
    torch.testing.assert_close(out, expected)


def test_extractor_returning_non_tensor_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="expected torch.Tensor"):
        load_or_extract(
            tmp_path,
            cache_key("m", "d", 0, "float32", "last_token"),
            lambda: "not a tensor",  # type: ignore[return-value,arg-type]
        )


def test_filenames_have_readable_prefix(tmp_path: Path) -> None:
    key = cache_key("EleutherAI/pythia-160m", "refusal_v1", 8, "float16", "last_token")
    assert key.startswith("EleutherAI_pythia-160m_refusal_v1_L8_last_token__")
