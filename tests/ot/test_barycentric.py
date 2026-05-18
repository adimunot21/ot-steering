"""Tests for ot_steering.ot.barycentric (barycentric projection)."""

from __future__ import annotations

import numpy as np
import pytest

from ot_steering.ot.barycentric import barycentric_project
from ot_steering.utils.seed import set_all_seeds


def test_permutation_coupling_maps_to_matched_targets() -> None:
    # If the coupling is a permutation, the barycentric image of source
    # point i is exactly the target point i was matched to.
    set_all_seeds(0)
    rng = np.random.default_rng(0)
    n, d = 6, 4
    perm = rng.permutation(n)
    coupling = np.zeros((n, n))
    coupling[np.arange(n), perm] = 1.0 / n  # uniform marginals
    target_features = rng.normal(size=(n, d))

    projection = barycentric_project(coupling, target_features)
    np.testing.assert_allclose(projection, target_features[perm], atol=1e-12)


def test_uniform_coupling_maps_every_source_to_target_mean() -> None:
    # P = (1/(nm)) * 1 sends every source point to the global mean of Y.
    set_all_seeds(1)
    rng = np.random.default_rng(1)
    n, m, d = 5, 7, 3
    coupling = np.full((n, m), 1.0 / (n * m))
    target_features = rng.normal(size=(m, d))

    projection = barycentric_project(coupling, target_features)
    expected = np.tile(target_features.mean(axis=0), (n, 1))
    np.testing.assert_allclose(projection, expected, atol=1e-12)


def test_explicit_source_marginal_overrides_inferred_one() -> None:
    # Pass a clean marginal even when the coupling rows are off by epsilon.
    set_all_seeds(2)
    rng = np.random.default_rng(2)
    n, m, d = 4, 4, 2
    coupling = np.eye(n) * 0.25 + 1e-6 * rng.uniform(size=(n, m))
    target_features = rng.normal(size=(m, d))
    explicit = np.full(n, 0.25)

    projection = barycentric_project(coupling, target_features, source_marginal=explicit)
    # The explicit marginal yields (1/0.25) * (P @ Y); inferred marginal
    # would divide by a slightly different number.
    expected = (coupling @ target_features) / 0.25
    np.testing.assert_allclose(projection, expected, atol=1e-12)


def test_empty_row_maps_to_zero_not_nan() -> None:
    n, m, d = 3, 4, 2
    coupling = np.zeros((n, m))
    coupling[0, 0] = 0.5
    coupling[2, 1] = 0.5  # row 1 has no mass
    target_features = np.arange(m * d, dtype=np.float64).reshape(m, d)

    projection = barycentric_project(coupling, target_features)
    assert np.isfinite(projection).all()
    np.testing.assert_allclose(projection[1], np.zeros(d))


def test_shape_mismatches_raise() -> None:
    with pytest.raises(ValueError, match="coupling must be 2-D"):
        barycentric_project(np.zeros(5), np.zeros((5, 3)))
    with pytest.raises(ValueError, match="target_features must be 2-D"):
        barycentric_project(np.zeros((3, 4)), np.zeros(4))
    with pytest.raises(ValueError, match="target_features has 5 rows but coupling expects 4"):
        barycentric_project(np.zeros((3, 4)), np.zeros((5, 2)))
    with pytest.raises(ValueError, match="source_marginal shape"):
        barycentric_project(np.zeros((3, 4)), np.zeros((4, 2)), source_marginal=np.zeros(5))
