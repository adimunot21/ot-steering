"""Tests for ot_steering.ot.emd (exact OT via POT)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from ot_steering.ot.emd import EMDConfig, solve_emd
from ot_steering.utils.seed import set_all_seeds


def _load_scratch_emd():  # type: ignore[no-untyped-def]
    """Load the phase-01 pedagogical implementation for agreement testing.

    We deliberately do NOT add ``phases/`` to the project's import path —
    that would let runtime code accidentally depend on pedagogical files.
    The agreement test is the one legitimate consumer of this code; it
    loads the module by file path.
    """
    project_root = Path(__file__).resolve().parents[2]
    path = project_root / "phases" / "phase_01_ot_foundations" / "scratch_ot.py"
    spec = importlib.util.spec_from_file_location("scratch_ot_for_tests", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["scratch_ot_for_tests"] = module
    spec.loader.exec_module(module)
    return module.scratch_emd


def _uniform(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n, dtype=np.float64)


def _euclidean_cost(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    return np.sqrt(((xs[:, None, :] - ys[None, :, :]) ** 2).sum(-1))


def test_self_transport_has_zero_cost() -> None:
    set_all_seeds(0)
    n = 8
    xs = np.random.default_rng(0).normal(size=(n, 3))
    a = _uniform(n)
    cost = _euclidean_cost(xs, xs)
    _, total = solve_emd(a, a, cost)
    assert total == pytest.approx(0.0, abs=1e-10)


def test_symmetry() -> None:
    set_all_seeds(1)
    rng = np.random.default_rng(1)
    n, m = 6, 7
    xs = rng.normal(size=(n, 2))
    ys = rng.normal(loc=(2.0, -1.0), size=(m, 2))
    a = _uniform(n)
    b = _uniform(m)
    cost_ab = _euclidean_cost(xs, ys)
    cost_ba = cost_ab.T
    _, cost_forward = solve_emd(a, b, cost_ab)
    _, cost_reverse = solve_emd(b, a, cost_ba)
    assert cost_forward == pytest.approx(cost_reverse, rel=1e-10)


def test_scale_invariance_of_plan_under_cost_rescaling() -> None:
    # Multiplying the cost matrix by k > 0 scales the optimal cost by k
    # exactly and leaves the optimal plan unchanged.
    set_all_seeds(2)
    rng = np.random.default_rng(2)
    n, m = 5, 5
    cost = rng.uniform(0.1, 5.0, size=(n, m))
    a = _uniform(n)
    b = _uniform(m)
    plan_base, cost_base = solve_emd(a, b, cost)
    plan_scaled, cost_scaled = solve_emd(a, b, 3.0 * cost)
    np.testing.assert_allclose(plan_base, plan_scaled, atol=1e-10)
    assert cost_scaled == pytest.approx(3.0 * cost_base, rel=1e-10)


def test_agrees_with_scratch_implementation_on_5x5() -> None:
    set_all_seeds(3)
    rng = np.random.default_rng(3)
    n, m = 5, 5
    xs = rng.normal(size=(n, 2))
    ys = rng.normal(loc=(1.5, 0.5), size=(m, 2))
    cost = _euclidean_cost(xs, ys)
    a = _uniform(n)
    b = _uniform(m)

    plan_pot, cost_pot = solve_emd(a, b, cost)
    scratch_emd = _load_scratch_emd()
    plan_scratch, cost_scratch = scratch_emd(a, b, cost)

    # Optimal cost is unique even when the plan is not.
    assert cost_pot == pytest.approx(cost_scratch, rel=1e-6)
    # On this random instance the plan is unique too.
    np.testing.assert_allclose(plan_pot, plan_scratch, atol=1e-8)


def test_config_rejects_invalid_values() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        EMDConfig(num_iter_max=0)
    with pytest.raises(ValidationError):
        EMDConfig(num_threads=0)


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        solve_emd(_uniform(3), _uniform(4), np.zeros((3, 5)))
