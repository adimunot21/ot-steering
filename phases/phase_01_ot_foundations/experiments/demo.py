"""2D Gaussian transport demo shared between the notebook and the figures script.

Imports the production OT solvers from ``src/ot_steering.ot`` so the chapter's
demo is exactly the code the rest of the project uses. Returns numpy arrays
(point clouds, cost matrix, plans) and a list of intermediate interpolation
frames; the notebook plots them inline and the figures script writes them to
PNG.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ot_steering.ot.emd import solve_emd
from ot_steering.ot.sinkhorn import SinkhornConfig, solve_sinkhorn

NDArrayF = npt.NDArray[np.float64]


@dataclass(frozen=True)
class GaussianTransportDemo:
    """Container for the demo's intermediate arrays.

    Attributes:
        source: Source point cloud, shape ``(n, 2)``.
        target: Target point cloud, shape ``(m, 2)``.
        cost: Pairwise squared-Euclidean cost, shape ``(n, m)``.
        plan_emd: Exact OT plan, shape ``(n, m)``.
        plan_sinkhorn_by_reg: Mapping from ``reg`` value to its Sinkhorn plan.
        interpolation_ts: Time points (in [0, 1]) at which the displacement
            interpolation is evaluated.
        interpolation_frames: For each ``t`` in ``interpolation_ts``, the
            interpolated point cloud of shape ``(n, 2)``. ``t=0`` is the
            source, ``t=1`` is the barycentric image of the source through
            the OT plan.
    """

    source: NDArrayF
    target: NDArrayF
    cost: NDArrayF
    plan_emd: NDArrayF
    plan_sinkhorn_by_reg: dict[float, NDArrayF]
    interpolation_ts: tuple[float, ...]
    interpolation_frames: list[NDArrayF]


def _make_clouds(n: int, m: int, seed: int) -> tuple[NDArrayF, NDArrayF]:
    rng = np.random.default_rng(seed)
    source = rng.normal(loc=(0.0, 0.0), scale=(0.9, 0.6), size=(n, 2))
    target = rng.normal(loc=(3.0, 1.5), scale=(0.6, 0.9), size=(m, 2))
    return source.astype(np.float64), target.astype(np.float64)


def _sq_euclidean(xs: NDArrayF, ys: NDArrayF) -> NDArrayF:
    return ((xs[:, None, :] - ys[None, :, :]) ** 2).sum(-1)


def _barycentric_image(source: NDArrayF, target: NDArrayF, plan: NDArrayF) -> NDArrayF:
    """Map each source point to the centre-of-mass of its OT image.

    With a uniform source histogram ``a = 1/n``, the barycentric projection
    of source point ``i`` is ``sum_j P[i, j] * target[j] / a[i] =
    n * sum_j P[i, j] * target[j]``. That is the formula here.
    """
    n = source.shape[0]
    weighted = plan @ target
    return n * weighted


def run_demo(
    n_source: int = 60,
    n_target: int = 60,
    seed: int = 0,
    sinkhorn_regs: tuple[float, ...] = (1.0, 0.1, 0.01),
    interpolation_ts: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> GaussianTransportDemo:
    """Run the full 2D-Gaussian transport demo end-to-end.

    Args:
        n_source: Number of source points.
        n_target: Number of target points.
        seed: RNG seed for the point clouds.
        sinkhorn_regs: Entropic regularisation values to scan. Higher values
            give a blurrier plan; lower values approach EMD.
        interpolation_ts: Time points at which to render the displacement
            interpolation. ``0.0`` returns the source; ``1.0`` returns the
            barycentric image.

    Returns:
        A :class:`GaussianTransportDemo` populated with every intermediate
        array the notebook and the figures script need.
    """
    source, target = _make_clouds(n_source, n_target, seed)
    cost = _sq_euclidean(source, target)
    a = np.full(n_source, 1.0 / n_source, dtype=np.float64)
    b = np.full(n_target, 1.0 / n_target, dtype=np.float64)

    plan_emd, _ = solve_emd(a, b, cost)

    plan_by_reg: dict[float, NDArrayF] = {}
    for reg in sinkhorn_regs:
        plan_reg, _ = solve_sinkhorn(
            a,
            b,
            cost,
            SinkhornConfig(
                reg=reg,
                num_iter_max=5_000,
                stop_threshold=1e-9,
                warn_on_no_convergence=False,
            ),
        )
        plan_by_reg[reg] = plan_reg

    target_image = _barycentric_image(source, target, plan_emd)
    frames = [(1.0 - t) * source + t * target_image for t in interpolation_ts]

    return GaussianTransportDemo(
        source=source,
        target=target,
        cost=cost,
        plan_emd=plan_emd,
        plan_sinkhorn_by_reg=plan_by_reg,
        interpolation_ts=interpolation_ts,
        interpolation_frames=frames,
    )
