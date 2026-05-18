"""Tests for ot_steering.steering.ot_steering."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ot_steering.steering.baselines import difference_in_means
from ot_steering.steering.ot_steering import (
    GMMConfig,
    OTSteeringMap,
    add_ot_steering_hook,
    build_ot_steering_map,
    fit_gmm,
)
from ot_steering.utils.seed import set_all_seeds


def _two_component_synthetic(n_per: int, seed: int) -> tuple[torch.Tensor, np.ndarray]:
    """Two well-separated Gaussian blobs in 8D."""
    rng = np.random.default_rng(seed)
    d = 8
    centers = np.array([[3.0] * d, [-3.0] * d])
    blob_a = rng.normal(loc=centers[0], scale=0.3, size=(n_per, d))
    blob_b = rng.normal(loc=centers[1], scale=0.3, size=(n_per, d))
    return torch.from_numpy(np.vstack([blob_a, blob_b])).float(), centers


def test_fit_gmm_recovers_two_well_separated_means() -> None:
    set_all_seeds(0)
    acts, planted_centers = _two_component_synthetic(n_per=200, seed=0)
    gmm = fit_gmm(acts, GMMConfig(n_components=2, covariance_type="full"))

    # Recovered means may be in either order — match them up.
    recovered = gmm.means_
    pair_dists = np.linalg.norm(recovered[:, None] - planted_centers[None, :], axis=-1)
    assignment = pair_dists.argmin(axis=1)
    for i, j in enumerate(assignment):
        np.testing.assert_allclose(recovered[i], planted_centers[j], atol=0.15)


def test_fit_gmm_rejects_non_2d() -> None:
    with pytest.raises(ValueError, match="2-D"):
        fit_gmm(torch.zeros(5), GMMConfig(n_components=2))


def test_fit_gmm_rejects_too_few_samples() -> None:
    with pytest.raises(ValueError, match="at least 4"):
        fit_gmm(torch.zeros(3, 8), GMMConfig(n_components=4))


def test_k1_steering_map_falls_back_to_difference_in_means() -> None:
    set_all_seeds(0)
    torch.manual_seed(0)
    pos = torch.randn(80, 16) + 1.0
    neg = torch.randn(80, 16) - 1.0

    steering_map = build_ot_steering_map(pos, neg, gmm_cfg=GMMConfig(n_components=1))
    plain = difference_in_means(pos, neg)

    assert steering_map.source_centers.shape == (1, 16)
    assert steering_map.displacements.shape == (1, 16)
    np.testing.assert_allclose(
        steering_map.displacements[0],
        plain.cpu().numpy().astype(np.float64),
        atol=1e-5,
    )


def test_steering_map_shapes_and_marginal_consistency_for_k4() -> None:
    set_all_seeds(1)
    torch.manual_seed(1)
    pos = torch.randn(120, 12) + 1.5
    neg = torch.randn(120, 12) - 1.5

    sm = build_ot_steering_map(pos, neg, gmm_cfg=GMMConfig(n_components=4, seed=1))
    assert sm.source_centers.shape == (4, 12)
    assert sm.target_centers.shape == (4, 12)
    assert sm.coupling.shape == (4, 4)
    assert sm.barycentric_targets.shape == (4, 12)
    assert sm.displacements.shape == (4, 12)
    # Row sums of the coupling should match source GMM weights.
    np.testing.assert_allclose(
        sm.coupling.sum(axis=1), sm.source_gmm.weights_.astype(np.float64), atol=1e-6
    )
    # Column sums should match target GMM weights.
    np.testing.assert_allclose(
        sm.coupling.sum(axis=0), sm.target_gmm.weights_.astype(np.float64), atol=1e-6
    )


def test_steering_hook_modifies_block_input_by_per_token_displacement() -> None:
    # 4-D toy. Two source clusters at ±3, two target clusters at ±10 with
    # the same sign pattern, so the OT plan is a permutation and the
    # displacement is ~ +7 on the positive cluster, -7 on the negative.
    set_all_seeds(2)
    torch.manual_seed(2)
    d = 4
    rng = np.random.default_rng(2)
    pos_acts = np.vstack(
        [
            rng.normal(loc=[10.0] * d, scale=0.3, size=(60, d)),
            rng.normal(loc=[-10.0] * d, scale=0.3, size=(60, d)),
        ]
    )
    neg_acts = np.vstack(
        [
            rng.normal(loc=[3.0] * d, scale=0.3, size=(60, d)),
            rng.normal(loc=[-3.0] * d, scale=0.3, size=(60, d)),
        ]
    )
    sm = build_ot_steering_map(
        torch.from_numpy(pos_acts).float(),
        torch.from_numpy(neg_acts).float(),
        gmm_cfg=GMMConfig(n_components=2, covariance_type="full", seed=2),
        assignment="hard",
    )

    # Build an identity-output block (Linear with weight=I, bias=0) so the
    # output equals the (possibly-steered) input.
    block = torch.nn.Linear(d, d, bias=False)
    block.weight.data = torch.eye(d)
    # A two-token batch: token 0 near the +source cluster, token 1 near
    # the −source cluster.
    hidden = torch.tensor([[[3.0] * d, [-3.0] * d]], dtype=torch.float32)  # shape (1, 2, 4)
    # Make the Linear see the (B*T, d) shape it expects by collapsing.
    flat_in = hidden.reshape(-1, d)
    with add_ot_steering_hook(block, sm, coefficient=1.0):
        flat_out = block(flat_in)  # type: ignore[operator]

    # The block input was the flat tensor; the hook sees args=(flat_in,),
    # so it added the per-row displacement. The +cluster row should have
    # moved toward +10; the −cluster row toward −10.
    assert flat_out.shape == (2, d)
    # After steering coef=1.0, each row ≈ barycentric-target ≈ ±10.
    assert flat_out[0].mean().item() > 5.0
    assert flat_out[1].mean().item() < -5.0


def test_hook_removed_on_context_exit() -> None:
    set_all_seeds(3)
    torch.manual_seed(3)
    pos = torch.randn(60, 8) + 1.0
    neg = torch.randn(60, 8) - 1.0
    sm = build_ot_steering_map(pos, neg, gmm_cfg=GMMConfig(n_components=2, seed=3))

    block = torch.nn.Linear(8, 8, bias=False)
    block.weight.data = torch.eye(8)
    with add_ot_steering_hook(block, sm, coefficient=5.0):
        pass
    x = torch.zeros(1, 8)
    # No hook → identity Linear → zeros out.
    torch.testing.assert_close(block(x), torch.zeros(1, 8))


def test_pydantic_rejects_invalid_gmm_config() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        GMMConfig(n_components=0)
    with pytest.raises(ValidationError):
        GMMConfig(reg_covar=-1.0)
    with pytest.raises(ValidationError):
        GMMConfig(covariance_type="weird")  # type: ignore[arg-type]


def test_steering_map_dataclass_is_frozen() -> None:
    set_all_seeds(4)
    torch.manual_seed(4)
    pos = torch.randn(60, 8) + 1.0
    neg = torch.randn(60, 8) - 1.0
    sm: OTSteeringMap = build_ot_steering_map(pos, neg, gmm_cfg=GMMConfig(n_components=2, seed=4))
    # frozen=True dataclass raises FrozenInstanceError when attribute is set.
    import dataclasses

    with pytest.raises(dataclasses.FrozenInstanceError):
        sm.coefficient = 99  # type: ignore[attr-defined,misc]
