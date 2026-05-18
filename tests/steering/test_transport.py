"""Tests for ot_steering.steering.transport and transport_baselines."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from ot_steering.steering.baselines import difference_in_means
from ot_steering.steering.ot_steering import GMMConfig
from ot_steering.steering.transport import (
    SteeringTransportConfig,
    TransportedSteeringMap,
    add_transported_steering_hook,
    build_transport,
)
from ot_steering.steering.transport_baselines import (
    procrustes_aligned,
    random_direction,
    target_supervised_oracle,
)
from ot_steering.utils.seed import set_all_seeds

# ----------------------------------------------------------------------------
# Synthetic-toy helpers
# ----------------------------------------------------------------------------


def _planted_pos_neg(
    n: int, d: int, *, pos_offset: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Two well-separated Gaussians on the +/- ``pos_offset`` axis in dim 0."""
    rng = np.random.default_rng(seed)
    pos = rng.normal(loc=np.r_[pos_offset, np.zeros(d - 1)], scale=0.3, size=(n, d))
    neg = rng.normal(loc=np.r_[-pos_offset, np.zeros(d - 1)], scale=0.3, size=(n, d))
    return pos.astype(np.float64), neg.astype(np.float64)


# ----------------------------------------------------------------------------
# build_transport
# ----------------------------------------------------------------------------


def test_build_transport_returns_displacement_pointing_from_neg_to_pos() -> None:
    set_all_seeds(0)
    n = 80
    pos_a, neg_a = _planted_pos_neg(n, 8, pos_offset=1.0, seed=0)
    pos_b, neg_b = _planted_pos_neg(n, 10, pos_offset=2.0, seed=1)

    tmap = build_transport(
        pos_a,
        neg_a,
        pos_b,
        neg_b,
        cfg=SteeringTransportConfig(
            n_components=2,
            gmm_cfg=GMMConfig(n_components=2, covariance_type="full", seed=0),
        ),
        rng=np.random.default_rng(0),
    )

    # Each B NEG cluster lives around (-2, 0, ..., 0); the transported
    # target should sit near (+2, 0, ..., 0); displacement ≈ (+4, 0, …, 0).
    assert tmap.transported_displacements.shape == (2, 10)
    for row in tmap.transported_displacements:
        assert row[0] > 3.0, f"dim-0 component too small: {row[0]}"
        # Off-axis components should be small.
        assert np.abs(row[1:]).max() < 1.0


def test_transport_dataclass_is_frozen() -> None:
    set_all_seeds(1)
    pos_a, neg_a = _planted_pos_neg(60, 8, pos_offset=1.0, seed=1)
    pos_b, neg_b = _planted_pos_neg(60, 8, pos_offset=1.0, seed=2)
    tmap: TransportedSteeringMap = build_transport(
        pos_a,
        neg_a,
        pos_b,
        neg_b,
        cfg=SteeringTransportConfig(
            n_components=2,
            gmm_cfg=GMMConfig(n_components=2, covariance_type="full", seed=1),
        ),
        rng=np.random.default_rng(1),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        tmap.assignment = "hard"  # type: ignore[misc]


def test_transport_rejects_bad_shapes() -> None:
    bad = np.zeros(8)  # 1-D
    good = np.zeros((10, 8))
    with pytest.raises(ValueError, match="must be 2-D"):
        build_transport(bad, good, good, good)
    with pytest.raises(ValueError, match="source-side d mismatch"):
        build_transport(np.zeros((10, 8)), np.zeros((10, 7)), np.zeros((10, 8)), np.zeros((10, 8)))
    with pytest.raises(ValueError, match="target-side d mismatch"):
        build_transport(np.zeros((10, 8)), np.zeros((10, 8)), np.zeros((10, 9)), np.zeros((10, 8)))


def test_transport_hook_adds_per_cluster_displacement() -> None:
    set_all_seeds(2)
    n = 80
    pos_a, neg_a = _planted_pos_neg(n, 6, pos_offset=1.0, seed=2)
    pos_b, neg_b = _planted_pos_neg(n, 6, pos_offset=1.5, seed=3)
    tmap = build_transport(
        pos_a,
        neg_a,
        pos_b,
        neg_b,
        cfg=SteeringTransportConfig(
            n_components=2,
            gmm_cfg=GMMConfig(n_components=2, covariance_type="full", seed=2),
            assignment="hard",
        ),
        rng=np.random.default_rng(2),
    )

    # Identity Linear block — output equals (possibly steered) input.
    d = 6
    block = torch.nn.Linear(d, d, bias=False)
    block.weight.data = torch.eye(d)
    # One token near each B NEG cluster (around dim-0 = -1.5).
    hidden = torch.tensor([[[-1.5, 0, 0, 0, 0, 0], [-1.4, 0, 0, 0, 0, 0]]], dtype=torch.float32)
    flat = hidden.reshape(-1, d)
    with add_transported_steering_hook(block, tmap, coefficient=1.0):
        out = block(flat)  # type: ignore[operator]
    # After coef=1 the tokens should have moved toward the B POS centroid
    # (around +1.5 in dim 0).
    assert out[:, 0].mean().item() > 0.5, f"dim-0 means after steering: {out[:, 0]}"


def test_transport_hook_removed_on_exit() -> None:
    set_all_seeds(3)
    pos_a, neg_a = _planted_pos_neg(60, 6, pos_offset=1.0, seed=3)
    pos_b, neg_b = _planted_pos_neg(60, 6, pos_offset=1.0, seed=4)
    tmap = build_transport(
        pos_a,
        neg_a,
        pos_b,
        neg_b,
        cfg=SteeringTransportConfig(
            n_components=2,
            gmm_cfg=GMMConfig(n_components=2, covariance_type="full", seed=3),
        ),
        rng=np.random.default_rng(3),
    )

    block = torch.nn.Linear(6, 6, bias=False)
    block.weight.data = torch.eye(6)
    with add_transported_steering_hook(block, tmap, coefficient=5.0):
        pass
    x = torch.zeros(1, 6)
    torch.testing.assert_close(block(x), torch.zeros(1, 6))


def test_pydantic_rejects_invalid_transport_config() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        SteeringTransportConfig(n_components=0)
    with pytest.raises(ValidationError):
        SteeringTransportConfig(distance_metric="weird")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        SteeringTransportConfig(assignment="bad")  # type: ignore[arg-type]


# ----------------------------------------------------------------------------
# Baselines
# ----------------------------------------------------------------------------


def test_target_supervised_oracle_matches_difference_of_means() -> None:
    set_all_seeds(4)
    pos, neg = _planted_pos_neg(100, 12, pos_offset=2.0, seed=4)
    pos_t = torch.from_numpy(pos).float()
    neg_t = torch.from_numpy(neg).float()

    direction = target_supervised_oracle(pos_t, neg_t)
    raw_diff = difference_in_means(pos_t, neg_t)
    raw_diff_unit = raw_diff / raw_diff.norm()

    # Unit-normalised, same vector.
    torch.testing.assert_close(direction, raw_diff_unit, atol=1e-6, rtol=1e-6)


def test_random_direction_is_unit_norm_and_deterministic_per_seed() -> None:
    v1 = random_direction(64, seed=7)
    v2 = random_direction(64, seed=7)
    v_other = random_direction(64, seed=8)
    torch.testing.assert_close(v1, v2)
    assert not torch.equal(v1, v_other)
    assert v1.shape == (64,)
    assert v1.norm().item() == pytest.approx(1.0, abs=1e-6)


def test_procrustes_recovers_known_rotation_in_2d() -> None:
    set_all_seeds(5)
    rng = np.random.default_rng(5)
    # A simple 2D rotation by 45 degrees.
    theta = np.pi / 4
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    s_centers = rng.normal(size=(8, 2))
    t_centers = s_centers @ rot.T

    src_dir = torch.from_numpy(np.array([1.0, 0.0]))
    rotated = procrustes_aligned(src_dir, s_centers, t_centers)
    expected = torch.from_numpy(np.array([np.cos(theta), np.sin(theta)], dtype=np.float32))
    # Direction agreement up to sign (Procrustes can choose either orientation
    # of the rotation if the data is centred at origin and symmetric).
    cosine = abs(float(torch.dot(rotated, expected)))
    assert cosine > 0.99, f"procrustes cosine to truth: {cosine}"


def test_procrustes_handles_different_dims_by_zero_padding() -> None:
    set_all_seeds(6)
    rng = np.random.default_rng(6)
    s_centers = rng.normal(size=(5, 4))
    t_centers = rng.normal(size=(5, 6))  # different d

    src_dir = torch.from_numpy(rng.normal(size=4))
    out = procrustes_aligned(src_dir, s_centers, t_centers)
    assert out.shape == (6,)
    assert out.norm().item() == pytest.approx(1.0, abs=1e-6)


def test_procrustes_rejects_mismatched_centroid_counts() -> None:
    with pytest.raises(ValueError, match="matched-centroid count mismatch"):
        procrustes_aligned(
            torch.zeros(4),
            np.zeros((5, 4)),
            np.zeros((6, 4)),  # different k
        )


def test_procrustes_rejects_wrong_source_dim_for_direction() -> None:
    with pytest.raises(ValueError, match="source_direction dim"):
        procrustes_aligned(
            torch.zeros(3),  # but centers are d=4
            np.zeros((4, 4)),
            np.zeros((4, 4)),
        )
