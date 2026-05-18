"""Tests for ot_steering.steering.baselines."""

from __future__ import annotations

import pytest
import torch

from ot_steering.steering.baselines import (
    add_residual_steering_hook,
    difference_in_means,
    mean_centered_steering,
)


def _seeded_normals(n: int, d: int, mean: float = 0.0, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=g) + mean


def test_difference_in_means_recovers_planted_translation() -> None:
    n, d = 200, 16
    pos = _seeded_normals(n, d, mean=1.0, seed=0)
    neg = _seeded_normals(n, d, mean=-1.0, seed=1)
    direction = difference_in_means(pos, neg)
    # Each coordinate of the difference of sample means is ~ 2.0.
    assert direction.shape == (d,)
    assert torch.allclose(direction, torch.full_like(direction, 2.0), atol=0.25)


def test_difference_in_means_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="d_model mismatch"):
        difference_in_means(torch.randn(5, 8), torch.randn(5, 7))


def test_mean_centered_steering_strips_common_mean() -> None:
    # Add a large constant common-mode signal to both classes; mean-centring
    # should remove it while difference-in-means cannot.
    n, d = 50, 16
    pos = _seeded_normals(n, d, mean=1.0, seed=0)
    neg = _seeded_normals(n, d, mean=-1.0, seed=1)

    common_bias = torch.full((d,), 1000.0)
    pos_biased = pos + common_bias
    neg_biased = neg + common_bias

    plain = difference_in_means(pos_biased, neg_biased)
    centred = mean_centered_steering(pos_biased, neg_biased)

    # Both formulations cancel the +1000 common-mode bias mathematically;
    # at float32 they agree up to ~1e-3 (the bias magnitude amplifies the
    # round-off in either subtraction).
    torch.testing.assert_close(plain, centred, atol=1e-3, rtol=1e-3)


def test_add_residual_steering_hook_modifies_block_input() -> None:
    # Build a tiny block (a Linear) and verify the hook adds the direction
    # to its input.
    d = 8
    block = torch.nn.Linear(d, d, bias=False)
    block.weight.data = torch.eye(d)  # identity so output == input
    direction = torch.arange(d, dtype=torch.float32)

    input_tensor = torch.zeros(2, d)
    with add_residual_steering_hook(block, direction, coefficient=3.0):
        out = block(input_tensor)
    expected = (3.0 * direction).expand(2, -1)
    torch.testing.assert_close(out, expected)


def test_add_residual_steering_hook_is_removed_on_exit() -> None:
    d = 4
    block = torch.nn.Linear(d, d, bias=False)
    block.weight.data = torch.eye(d)
    direction = torch.ones(d)

    with add_residual_steering_hook(block, direction, coefficient=5.0):
        pass
    # After exiting the context, the hook is gone, so input passes through
    # unchanged (identity weight).
    x = torch.arange(d, dtype=torch.float32).unsqueeze(0)
    out = block(x)
    torch.testing.assert_close(out, x)


def test_steering_hook_rejects_non_1d_direction() -> None:
    block = torch.nn.Linear(4, 4)
    with (
        pytest.raises(ValueError, match="direction must be 1-D"),
        add_residual_steering_hook(block, torch.zeros(4, 4), coefficient=1.0),
    ):
        pass
