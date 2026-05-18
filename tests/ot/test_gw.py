"""Tests for ot_steering.ot.gw (entropic Gromov-Wasserstein via POT)."""

from __future__ import annotations

import numpy as np
import pytest

from ot_steering.ot.gw import GWConfig, solve_entropic_gw
from ot_steering.utils.seed import set_all_seeds


def _uniform(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n, dtype=np.float64)


def _distance_matrix(xs: np.ndarray) -> np.ndarray:
    return np.sqrt(((xs[:, None, :] - xs[None, :, :]) ** 2).sum(-1))


def _random_orthogonal(d: int, rng: np.random.Generator) -> np.ndarray:
    """Random orthogonal matrix via QR of a Gaussian."""
    a = rng.normal(size=(d, d))
    q, r = np.linalg.qr(a)
    # Standardise sign so det = +1 (proper rotation).
    return q * np.sign(np.diag(r))


def test_self_coupling_is_near_diagonal() -> None:
    # GW of a distribution against itself should put almost all mass on
    # the diagonal P[i, i] ≈ 1/n.
    set_all_seeds(0)
    rng = np.random.default_rng(0)
    n = 12
    xs = rng.normal(size=(n, 3))
    c = _distance_matrix(xs)
    p = _uniform(n)

    coupling, _ = solve_entropic_gw(
        c, c, p, p, GWConfig(reg=0.05, num_iter_max=500, warn_on_no_convergence=False)
    )

    # Diagonal mass dominates the per-row off-diagonal mass.
    diag = np.diag(coupling)
    np.testing.assert_array_less(coupling.sum(axis=1) - diag, diag + 1e-12)
    # Row sums respect the marginal.
    np.testing.assert_allclose(coupling.sum(axis=1), p, atol=1e-4)
    np.testing.assert_allclose(coupling.sum(axis=0), p, atol=1e-4)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_recovers_rotation_in_2d(seed: int) -> None:
    # Two clouds related by a random 2D rotation. Their intra-distance
    # matrices are identical (rotation is an isometry), so GW should
    # recover a permutation matching each X[i] to its rotated partner.
    set_all_seeds(seed)
    rng = np.random.default_rng(seed)
    n = 16
    xs = rng.normal(size=(n, 2))
    rotation = _random_orthogonal(2, rng)
    ys = xs @ rotation

    c1 = _distance_matrix(xs)
    c2 = _distance_matrix(ys)
    p = _uniform(n)

    coupling, _ = solve_entropic_gw(
        c1,
        c2,
        p,
        p,
        GWConfig(reg=0.01, num_iter_max=500, num_restart=3, warn_on_no_convergence=False),
        rng=np.random.default_rng(seed + 100),
    )

    # Planted truth: source point i should match target point i.
    recovered = coupling.argmax(axis=1)
    truth = np.arange(n)
    accuracy = float((recovered == truth).mean())
    assert accuracy >= 0.85, f"recovered {accuracy:.2%} of planted matches (seed={seed})"


def test_cost_is_symmetric_under_side_swap() -> None:
    set_all_seeds(3)
    rng = np.random.default_rng(3)
    n, m = 10, 10
    xs = rng.normal(size=(n, 3))
    ys = rng.normal(size=(m, 4))
    c1 = _distance_matrix(xs)
    c2 = _distance_matrix(ys)
    p = _uniform(n)
    q = _uniform(m)

    cfg = GWConfig(reg=0.02, num_iter_max=300, warn_on_no_convergence=False)
    _, cost_forward = solve_entropic_gw(c1, c2, p, q, cfg)
    _, cost_reverse = solve_entropic_gw(c2, c1, q, p, cfg)
    # GW is symmetric in the pair (C1, C2) — the cost should match.
    assert cost_forward == pytest.approx(cost_reverse, rel=1e-3, abs=1e-6)


def test_config_rejects_invalid_values() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        GWConfig(reg=0.0)
    with pytest.raises(ValidationError):
        GWConfig(reg=-1.0)
    with pytest.raises(ValidationError):
        GWConfig(num_iter_max=0)
    with pytest.raises(ValidationError):
        GWConfig(num_restart=0)
    with pytest.raises(ValidationError):
        GWConfig(loss_fun="absolute_loss")  # type: ignore[arg-type]


def test_shape_mismatches_raise() -> None:
    n = 5
    p = _uniform(n)
    with pytest.raises(ValueError, match="C1 must be a square"):
        solve_entropic_gw(np.zeros((n, n + 1)), np.zeros((n, n)), p, p)
    with pytest.raises(ValueError, match="p shape"):
        solve_entropic_gw(np.zeros((n, n)), np.zeros((n, n)), _uniform(n + 1), p)
    with pytest.raises(ValueError, match="q shape"):
        solve_entropic_gw(np.zeros((n, n)), np.zeros((n, n)), p, _uniform(n + 1))
