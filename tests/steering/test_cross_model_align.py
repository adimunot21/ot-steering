"""Tests for ot_steering.steering.cross_model_align."""

from __future__ import annotations

import numpy as np
import pytest

from ot_steering.ot.gw import GWConfig
from ot_steering.steering.cross_model_align import (
    CrossModelAlignment,
    CrossModelGWConfig,
    _pairwise_distances,
    cross_model_gw_coupling,
)
from ot_steering.steering.ot_steering import GMMConfig
from ot_steering.utils.seed import set_all_seeds


def _gw_cfg(reg: float = 0.01) -> GWConfig:
    """GW config tuned for the small toys used here — quick, quiet.

    Default reg is intentionally tight (0.01); at reg=0.05 the entropic
    smear is large enough on normalised [0, 1] distance matrices that
    self-pair couplings drift away from clean identity.
    """
    return GWConfig(reg=reg, num_iter_max=500, num_restart=2, warn_on_no_convergence=False)


def test_self_pair_gives_identity_coupling() -> None:
    # GW between a distribution and itself should put all mass on the
    # diagonal of the coupling.
    set_all_seeds(0)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(10, 8))
    alignment = cross_model_gw_coupling(
        x,
        x,
        cfg=CrossModelGWConfig(gw_cfg=_gw_cfg()),
        rng=np.random.default_rng(0),
    )
    diag_mass = float(np.diag(alignment.coupling).sum())
    total_mass = float(alignment.coupling.sum())
    assert diag_mass / total_mass > 0.95, (
        f"self-pair did not produce identity coupling: {diag_mass}/{total_mass}"
    )
    # The linear GW cost is tiny but not zero (entropic smear at reg=0.01
    # on normalised distance matrices gives ~1e-3 residual).
    assert alignment.gw_cost < 0.005


def test_marginals_are_honoured() -> None:
    set_all_seeds(1)
    rng = np.random.default_rng(1)
    x = rng.normal(size=(8, 5))
    y = rng.normal(size=(6, 7))  # different n, m, and d
    alignment = cross_model_gw_coupling(
        x,
        y,
        cfg=CrossModelGWConfig(gw_cfg=_gw_cfg()),
        rng=np.random.default_rng(1),
    )
    # Entropic GW only honours marginals approximately; ~1e-3 is the
    # honest tolerance at reg=0.05.
    np.testing.assert_allclose(alignment.coupling.sum(axis=1), alignment.source_marginal, atol=1e-3)
    np.testing.assert_allclose(alignment.coupling.sum(axis=0), alignment.target_marginal, atol=1e-3)


def test_cosine_metric_gives_distinct_distance_matrix_from_euclidean() -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(size=(6, 4)) * np.linspace(0.1, 10.0, 6)[:, None]  # varied scale
    dist_eu = _pairwise_distances(x.astype(np.float64), metric="euclidean")
    dist_co = _pairwise_distances(x.astype(np.float64), metric="cosine")
    assert dist_eu.shape == dist_co.shape == (6, 6)
    # Distances differ — they should under any sensible test case.
    assert not np.allclose(dist_eu, dist_co)
    # Cosine distance is in [0, 2]; Euclidean in [0, large]. Sanity.
    assert dist_co.max() <= 2.0 + 1e-9
    # Diagonals are zero modulo float round-off (cosine subtracts ≈1 from 1).
    np.testing.assert_allclose(np.diag(dist_eu), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.diag(dist_co), 0.0, atol=1e-12)


def test_partial_gmm_spec_raises() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        CrossModelGWConfig(n_components_source=4)
    with pytest.raises(ValidationError):
        CrossModelGWConfig(n_components_target=4)


def test_gmm_centroid_path_reduces_problem_size() -> None:
    set_all_seeds(3)
    rng = np.random.default_rng(3)
    x_src = rng.normal(size=(60, 6))
    x_tgt = rng.normal(size=(80, 9))
    cfg = CrossModelGWConfig(
        n_components_source=4,
        n_components_target=4,
        gmm_cfg=GMMConfig(n_components=4, seed=3),
        gw_cfg=_gw_cfg(),
    )
    alignment = cross_model_gw_coupling(x_src, x_tgt, cfg=cfg, rng=np.random.default_rng(3))
    assert alignment.source_centers.shape == (4, 6)
    assert alignment.target_centers.shape == (4, 9)
    assert alignment.coupling.shape == (4, 4)
    # GMM weights, not uniform.
    assert alignment.source_marginal.shape == (4,)
    assert alignment.source_marginal.sum() == pytest.approx(1.0, abs=1e-9)


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="must be 2-D"):
        cross_model_gw_coupling(np.zeros(8), np.zeros((4, 4)))
    with pytest.raises(ValueError, match="must be 2-D"):
        cross_model_gw_coupling(np.zeros((4, 4)), np.zeros(8))


def test_gw_cost_higher_for_random_than_for_self() -> None:
    # GW cost between two unrelated random distributions should be
    # noticeably higher than self-pair (which is zero).
    set_all_seeds(4)
    rng = np.random.default_rng(4)
    x = rng.normal(size=(10, 8))
    y = rng.uniform(low=-3.0, high=3.0, size=(10, 6))  # very different scale + dim
    self_alignment = cross_model_gw_coupling(
        x,
        x,
        cfg=CrossModelGWConfig(gw_cfg=_gw_cfg()),
        rng=np.random.default_rng(4),
    )
    cross_alignment = cross_model_gw_coupling(
        x,
        y,
        cfg=CrossModelGWConfig(gw_cfg=_gw_cfg()),
        rng=np.random.default_rng(5),
    )
    assert self_alignment.gw_cost < cross_alignment.gw_cost


def test_alignment_dataclass_is_frozen() -> None:
    set_all_seeds(5)
    rng = np.random.default_rng(5)
    x = rng.normal(size=(8, 6))
    alignment: CrossModelAlignment = cross_model_gw_coupling(
        x,
        x,
        cfg=CrossModelGWConfig(gw_cfg=_gw_cfg()),
        rng=np.random.default_rng(5),
    )
    import dataclasses

    with pytest.raises(dataclasses.FrozenInstanceError):
        alignment.gw_cost = 99.0  # type: ignore[attr-defined,misc]
