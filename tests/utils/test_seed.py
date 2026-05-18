"""Tests for ot_steering.utils.seed."""

from __future__ import annotations

import os
import random

import numpy as np
import pytest
import torch

from ot_steering.utils.seed import set_all_seeds


def test_python_random_reproducible() -> None:
    set_all_seeds(42)
    a = [random.random() for _ in range(5)]
    set_all_seeds(42)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_numpy_random_reproducible() -> None:
    set_all_seeds(7)
    a = np.random.rand(8)
    set_all_seeds(7)
    b = np.random.rand(8)
    np.testing.assert_array_equal(a, b)


def test_torch_cpu_reproducible() -> None:
    set_all_seeds(123)
    a = torch.randn(4, 4)
    set_all_seeds(123)
    b = torch.randn(4, 4)
    assert torch.equal(a, b)


def test_different_seeds_give_different_streams() -> None:
    set_all_seeds(0)
    a = torch.randn(8)
    set_all_seeds(1)
    b = torch.randn(8)
    assert not torch.equal(a, b)


def test_sets_pythonhashseed_env_var() -> None:
    set_all_seeds(99)
    assert os.environ["PYTHONHASHSEED"] == "99"


def test_cudnn_set_deterministic() -> None:
    set_all_seeds(0)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


def test_negative_seed_raises() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        set_all_seeds(-1)


def test_non_int_seed_raises() -> None:
    with pytest.raises(ValueError, match="must be int"):
        set_all_seeds(3.14)  # type: ignore[arg-type]
