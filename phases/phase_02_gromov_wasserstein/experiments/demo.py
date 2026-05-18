"""Rotation-recovery demo for Chapter 2.

Build two 2D point clouds related by a random rotation, run entropic
Gromov-Wasserstein on their intra-distance matrices, and surface every
intermediate the chapter wants to display:

- the original (un-rotated) source cloud and the rotated target cloud,
- the GW coupling matrix,
- the recovered argmax permutation vs. the planted ground truth,
- a barycentric-image-based displacement interpolation analogous to Phase 1.

This module is shared by ``make_figures.py`` and the companion notebook so
neither file re-implements the experiment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ot_steering.ot.barycentric import barycentric_project
from ot_steering.ot.gw import GWConfig, solve_entropic_gw

NDArrayF = NDArray[np.float64]


@dataclass(frozen=True)
class RotationDemo:
    """Container for every array the chapter renders.

    Attributes:
        source: Source point cloud, shape ``(n, 2)``.
        target: Target point cloud, shape ``(n, 2)`` — ``source @ R + noise``.
        rotation: The planted ``(2, 2)`` orthogonal matrix.
        c1: Source intra-distance matrix, shape ``(n, n)``.
        c2: Target intra-distance matrix, shape ``(n, n)``.
        coupling: GW coupling, shape ``(n, n)``.
        gw_cost: GW objective at the returned coupling.
        recovered_permutation: For each source ``i``, the argmax target.
        truth_permutation: Identity here (``arange(n)``) by construction.
        accuracy: Fraction of ``recovered == truth``.
        interpolation_ts: Time-points for the displacement interpolation.
        interpolation_frames: For each ``t`` in ``interpolation_ts``, the
            interpolated point cloud of shape ``(n, 2)`` (``t=0`` source,
            ``t=1`` barycentric image of source under the coupling).
    """

    source: NDArrayF
    target: NDArrayF
    rotation: NDArrayF
    c1: NDArrayF
    c2: NDArrayF
    coupling: NDArrayF
    gw_cost: float
    recovered_permutation: NDArray[np.intp]
    truth_permutation: NDArray[np.intp]
    accuracy: float
    interpolation_ts: tuple[float, ...]
    interpolation_frames: list[NDArrayF]


def _random_rotation(rng: np.random.Generator) -> NDArrayF:
    """Random 2-D rotation matrix (det = +1)."""
    theta = rng.uniform(0, 2 * np.pi)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def _distance_matrix(xs: NDArrayF) -> NDArrayF:
    return np.sqrt(((xs[:, None, :] - xs[None, :, :]) ** 2).sum(-1))


def run_rotation_demo(
    n: int = 80,
    seed: int = 0,
    noise_scale: float = 0.05,
    reg: float = 0.01,
    num_restart: int = 3,
    interpolation_ts: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> RotationDemo:
    """Run the rotation-recovery demo end-to-end.

    Args:
        n: Number of points in each cloud.
        seed: RNG seed for the cloud, rotation, and noise.
        noise_scale: Standard deviation of the additive noise on ``target``.
            ``0.0`` would make ``C1 == C2`` exactly; a little noise makes the
            problem realistic without breaking recovery.
        reg: Entropic regularisation for GW.
        num_restart: Number of independent GW initialisations to try.
        interpolation_ts: Time points for the displacement interpolation.

    Returns:
        A populated :class:`RotationDemo`.
    """
    rng = np.random.default_rng(seed)
    source = rng.normal(size=(n, 2)).astype(np.float64)
    rotation = _random_rotation(rng)
    target = source @ rotation + noise_scale * rng.normal(size=(n, 2))

    c1 = _distance_matrix(source)
    c2 = _distance_matrix(target)
    p = np.full(n, 1.0 / n, dtype=np.float64)

    coupling, cost = solve_entropic_gw(
        c1,
        c2,
        p,
        p,
        GWConfig(
            reg=reg,
            num_iter_max=500,
            num_restart=num_restart,
            warn_on_no_convergence=False,
        ),
        rng=np.random.default_rng(seed + 9_999),
    )

    truth = np.arange(n, dtype=np.intp)
    recovered = coupling.argmax(axis=1).astype(np.intp)
    accuracy = float((recovered == truth).mean())

    target_image = barycentric_project(coupling, target)
    frames = [(1.0 - t) * source + t * target_image for t in interpolation_ts]

    return RotationDemo(
        source=source,
        target=target,
        rotation=rotation,
        c1=c1,
        c2=c2,
        coupling=coupling,
        gw_cost=cost,
        recovered_permutation=recovered,
        truth_permutation=truth,
        accuracy=accuracy,
        interpolation_ts=interpolation_ts,
        interpolation_frames=frames,
    )
