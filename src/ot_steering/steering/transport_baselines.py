"""Baseline transport methods to compare against GW-transport.

Three reference methods sit alongside the GW-transport pipeline in
:mod:`ot_steering.steering.transport`:

- :func:`random_direction` — a unit-norm random vector in target space.
  This is the chance floor. If GW-transport doesn't beat this, the
  whole project's hypothesis is wrong.
- :func:`procrustes_aligned` — the standard linear-alignment baseline.
  Fit a rotation between matched source/target centroid clouds (after
  some 1-1 pairing) and rotate the source direction into target space.
  Requires equal dimension; we zero-pad the smaller side.
- :func:`target_supervised_oracle` — Phase 3's difference-of-means
  computed *directly* on target activations. This is the upper bound:
  the best a steering vector can do when given target-side supervision.
  GW-transport tries to approach it without that supervision.

All three return a unit-norm vector in target space; the caller scales
via ``coefficient`` on :func:`ot_steering.steering.baselines.add_residual_steering_hook`.
"""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray

from ot_steering.steering.baselines import difference_in_means
from ot_steering.utils.logging import get_logger

_log = get_logger(__name__)

NDArrayF = NDArray[np.float64]


def _to_numpy(x: torch.Tensor | NDArrayF) -> NDArrayF:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float64)
    return np.ascontiguousarray(x, dtype=np.float64)


def _normalise(v: torch.Tensor) -> torch.Tensor:
    norm = v.norm()
    if norm < 1e-12:
        return v
    out: torch.Tensor = v / norm
    return out


def random_direction(d_target: int, seed: int = 0) -> torch.Tensor:
    """Return a unit-norm random direction in target space.

    Args:
        d_target: Target-model residual-stream dimension.
        seed: RNG seed.

    Returns:
        Shape ``(d_target,)`` torch float32 tensor, ``||v|| = 1``.
    """
    rng = np.random.default_rng(seed)
    v = torch.from_numpy(rng.normal(size=d_target).astype(np.float32))
    return _normalise(v)


def target_supervised_oracle(
    target_acts_pos: torch.Tensor | NDArrayF,
    target_acts_neg: torch.Tensor | NDArrayF,
) -> torch.Tensor:
    """Return the (unit-normalised) difference-of-means direction on the target.

    This is **not** a transport method — it is the upper bound. The
    direction is computed *directly* on target activations, using both
    POS and NEG classes, just like Phase 3's baseline. Use it as the
    ceiling that GW-transport tries to approach without target
    supervision.

    Args:
        target_acts_pos: ``(n_pos, d_target)`` positive-class activations.
        target_acts_neg: ``(n_neg, d_target)`` negative-class activations.

    Returns:
        Shape ``(d_target,)`` unit-norm torch tensor.
    """
    pos = (
        target_acts_pos
        if isinstance(target_acts_pos, torch.Tensor)
        else torch.from_numpy(_to_numpy(target_acts_pos))
    )
    neg = (
        target_acts_neg
        if isinstance(target_acts_neg, torch.Tensor)
        else torch.from_numpy(_to_numpy(target_acts_neg))
    )
    direction = difference_in_means(pos, neg)
    return _normalise(direction)


def _zero_pad_to(x: NDArrayF, target_dim: int) -> NDArrayF:
    """Right-pad columns of ``x`` (shape (n, d)) with zeros up to ``target_dim``."""
    cur = x.shape[1]
    if cur == target_dim:
        return x
    if cur > target_dim:
        raise ValueError(
            f"_zero_pad_to: source dim {cur} > target dim {target_dim}; would lose information"
        )
    pad = np.zeros((x.shape[0], target_dim - cur), dtype=x.dtype)
    return np.concatenate([x, pad], axis=1)


def procrustes_aligned(
    source_direction: torch.Tensor | NDArrayF,
    source_centers: torch.Tensor | NDArrayF,
    target_centers: torch.Tensor | NDArrayF,
) -> torch.Tensor:
    """Rotate a source direction into target space via orthogonal Procrustes.

    The classical baseline for cross-space representation alignment.
    Assumes a *known* 1-1 correspondence between the source and target
    centroid clouds (caller pairs the rows of ``source_centers`` and
    ``target_centers`` in advance — for the Phase 6 experiment, the
    pairing comes from the cross-model GW coupling argmax).

    Procrustes minimises ``||S R − T||_F`` over orthogonal ``R``; the
    solution is ``R = U V^T`` where ``S^T T = U Σ V^T``. We then apply
    ``R`` to the source direction.

    When source and target dimensions differ, we zero-pad the smaller
    side to make Procrustes well-defined.

    Args:
        source_direction: ``(d_source,)`` direction in source space.
        source_centers: ``(k, d_source)`` matched centroids.
        target_centers: ``(k, d_target)`` matched centroids.

    Returns:
        Shape ``(d_target,)`` unit-norm torch tensor — the source
        direction rotated into target space.

    Raises:
        ValueError: If ``source_centers`` and ``target_centers`` have
            different numbers of rows (no 1-1 pairing).
    """
    src_dir = _to_numpy(source_direction).reshape(-1)
    s = _to_numpy(source_centers)
    t = _to_numpy(target_centers)
    if s.ndim != 2 or t.ndim != 2:
        raise ValueError("source_centers and target_centers must be 2-D")
    if s.shape[0] != t.shape[0]:
        raise ValueError(
            f"matched-centroid count mismatch: source={s.shape[0]}, target={t.shape[0]}"
        )
    if src_dir.shape[0] != s.shape[1]:
        raise ValueError(
            f"source_direction dim {src_dir.shape[0]} != source_centers dim {s.shape[1]}"
        )

    # Pad whichever side is smaller so Procrustes is well-defined.
    common_dim = max(s.shape[1], t.shape[1])
    s_pad = _zero_pad_to(s, common_dim)
    t_pad = _zero_pad_to(t, common_dim)
    src_dir_pad = np.concatenate([src_dir, np.zeros(common_dim - src_dir.shape[0])])

    # Mean-centre before Procrustes; the rotation aligns shape, not location.
    s_centered = s_pad - s_pad.mean(axis=0, keepdims=True)
    t_centered = t_pad - t_pad.mean(axis=0, keepdims=True)
    u, _sigma, vt = np.linalg.svd(s_centered.T @ t_centered, full_matrices=False)
    rotation = u @ vt  # (common_dim, common_dim)

    rotated = rotation.T @ src_dir_pad  # apply rotation to source direction
    # If target was padded up from a smaller dim, slice back to target dim.
    rotated_target = rotated[: t.shape[1]]
    out = torch.from_numpy(rotated_target.astype(np.float32))
    return _normalise(out)
