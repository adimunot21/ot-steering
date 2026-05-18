"""Tests for ot_steering.eval.correlation."""

from __future__ import annotations

import numpy as np
import pytest

from ot_steering.eval.correlation import spearman_with_bootstrap_ci


def test_perfectly_correlated_data_yields_rho_one() -> None:
    x = np.linspace(0.0, 1.0, 20)
    y = 3.0 * x + 1.0
    rho, lo, hi = spearman_with_bootstrap_ci(x, y, n_boot=200, seed=0)
    assert rho == pytest.approx(1.0, abs=1e-9)
    assert lo > 0.95
    assert hi <= 1.0 + 1e-9


def test_perfectly_anticorrelated_data_yields_rho_minus_one() -> None:
    x = np.linspace(0.0, 1.0, 20)
    y = -2.5 * x + 7.0
    rho, lo, hi = spearman_with_bootstrap_ci(x, y, n_boot=200, seed=0)
    assert rho == pytest.approx(-1.0, abs=1e-9)
    assert hi < -0.95


def test_independent_data_yields_rho_near_zero_with_ci_containing_zero() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=80)
    y = rng.normal(size=80)
    rho, lo, hi = spearman_with_bootstrap_ci(x, y, n_boot=500, seed=0)
    assert abs(rho) < 0.3
    assert lo < 0 < hi


def test_bootstrap_ci_widens_with_smaller_samples() -> None:
    rng = np.random.default_rng(1)
    x_big = rng.normal(size=200)
    y_big = 0.5 * x_big + rng.normal(size=200, scale=0.5)
    rho_b, lo_b, hi_b = spearman_with_bootstrap_ci(x_big, y_big, n_boot=300, seed=1)

    x_small = x_big[:20]
    y_small = y_big[:20]
    _, lo_s, hi_s = spearman_with_bootstrap_ci(x_small, y_small, n_boot=300, seed=1)

    assert (hi_s - lo_s) > (hi_b - lo_b), "small-sample CI should be wider"
    assert rho_b > 0  # planted positive correlation


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="same shape"):
        spearman_with_bootstrap_ci(np.zeros(5), np.zeros(6))


def test_too_few_points_raises() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        spearman_with_bootstrap_ci(np.zeros(2), np.zeros(2))
