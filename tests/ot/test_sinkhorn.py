"""Tests for ot_steering.ot.sinkhorn (entropic OT via POT)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from ot_steering.ot.emd import solve_emd
from ot_steering.ot.sinkhorn import SinkhornConfig, solve_sinkhorn
from ot_steering.utils.seed import set_all_seeds


def _load_scratch_sinkhorn():  # type: ignore[no-untyped-def]
    project_root = Path(__file__).resolve().parents[2]
    path = project_root / "phases" / "phase_01_ot_foundations" / "scratch_sinkhorn.py"
    spec = importlib.util.spec_from_file_location("scratch_sinkhorn_for_tests", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["scratch_sinkhorn_for_tests"] = module
    spec.loader.exec_module(module)
    return module.scratch_sinkhorn


def _uniform(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n, dtype=np.float64)


def _euclidean_sq(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    return ((xs[:, None, :] - ys[None, :, :]) ** 2).sum(-1)


def test_marginals_are_preserved() -> None:
    set_all_seeds(0)
    rng = np.random.default_rng(0)
    n, m = 7, 9
    xs = rng.normal(size=(n, 2))
    ys = rng.normal(loc=(1.0, 1.0), size=(m, 2))
    a = _uniform(n)
    b = _uniform(m)
    plan, _ = solve_sinkhorn(a, b, _euclidean_sq(xs, ys), SinkhornConfig(reg=0.1))
    np.testing.assert_allclose(plan.sum(axis=1), a, atol=1e-6)
    np.testing.assert_allclose(plan.sum(axis=0), b, atol=1e-6)


def test_converges_to_emd_as_reg_shrinks() -> None:
    set_all_seeds(1)
    rng = np.random.default_rng(1)
    n, m = 8, 8
    xs = rng.normal(size=(n, 2))
    ys = rng.normal(loc=(2.5, -1.0), size=(m, 2))
    cost = _euclidean_sq(xs, ys)
    a = _uniform(n)
    b = _uniform(m)

    _, cost_emd = solve_emd(a, b, cost)
    _, cost_loose = solve_sinkhorn(a, b, cost, SinkhornConfig(reg=0.5, num_iter_max=2_000))
    _, cost_tight = solve_sinkhorn(
        a,
        b,
        cost,
        SinkhornConfig(
            reg=0.05, num_iter_max=5_000, stop_threshold=1e-7, warn_on_no_convergence=False
        ),
    )

    # Tight regularisation hugs the EMD cost; loose regularisation does not.
    gap_loose = abs(cost_loose - cost_emd)
    gap_tight = abs(cost_tight - cost_emd)
    assert gap_tight < gap_loose
    assert gap_tight < 0.10 * abs(cost_emd) + 1e-3


def test_numerical_stability_at_small_reg() -> None:
    # Without log-domain Sinkhorn, very small reg underflows. The default
    # method='sinkhorn_log' must produce a finite, marginal-correct plan.
    set_all_seeds(2)
    rng = np.random.default_rng(2)
    n = 16
    xs = rng.normal(size=(n, 3))
    ys = rng.normal(loc=(3.0, 0.0, -2.0), size=(n, 3))
    cost = _euclidean_sq(xs, ys)
    a = _uniform(n)
    b = _uniform(n)
    plan, total = solve_sinkhorn(
        a,
        b,
        cost,
        SinkhornConfig(
            reg=1e-3,
            num_iter_max=20_000,
            stop_threshold=1e-9,
            warn_on_no_convergence=False,
        ),
    )
    assert np.isfinite(plan).all()
    assert np.isfinite(total)
    # reg=1e-3 is aggressive — accept loose marginal violation, but no NaN.
    np.testing.assert_allclose(plan.sum(axis=1), a, atol=5e-5)
    np.testing.assert_allclose(plan.sum(axis=0), b, atol=5e-5)


def test_agrees_with_scratch_implementation_on_5x5() -> None:
    set_all_seeds(3)
    rng = np.random.default_rng(3)
    n, m = 5, 5
    xs = rng.normal(size=(n, 2))
    ys = rng.normal(loc=(1.5, 0.5), size=(m, 2))
    cost = _euclidean_sq(xs, ys)
    a = _uniform(n)
    b = _uniform(m)

    # Use a moderate reg so both solvers converge quickly and to the same
    # floor — the point of this test is implementation agreement, not
    # solving a hard instance.
    reg = 0.1
    plan_pot, cost_pot = solve_sinkhorn(
        a,
        b,
        cost,
        SinkhornConfig(
            reg=reg, num_iter_max=2_000, stop_threshold=1e-10, warn_on_no_convergence=False
        ),
    )
    scratch_sinkhorn = _load_scratch_sinkhorn()
    plan_scratch, cost_scratch = scratch_sinkhorn(a, b, cost, reg=reg, n_iter=2_000, tol=1e-10)

    # Two independent log-domain Sinkhorn implementations converge to the
    # same fixed point but with different last-step normalisations — the
    # plans agree to ~1e-4 and the cost to better.
    np.testing.assert_allclose(plan_pot, plan_scratch, atol=1e-4)
    assert cost_pot == pytest.approx(cost_scratch, rel=1e-4)


def test_config_rejects_invalid_values() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        SinkhornConfig(reg=0.0)
    with pytest.raises(ValidationError):
        SinkhornConfig(reg=-1.0)
    with pytest.raises(ValidationError):
        SinkhornConfig(num_iter_max=0)
    with pytest.raises(ValidationError):
        SinkhornConfig(method="not-a-method")  # type: ignore[arg-type]


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        solve_sinkhorn(_uniform(3), _uniform(4), np.zeros((3, 5)))
