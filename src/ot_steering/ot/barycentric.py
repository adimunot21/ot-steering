"""Barycentric projection of a transport plan.

Given a coupling ``P`` of shape ``(n, m)`` from a source histogram with
weights ``p`` to a target with features ``Y`` of shape ``(m, d)``, the
barycentric projection sends source point ``i`` to the weighted centre of
mass of its OT image:

.. math::

    \\hat{T}(i) = \\frac{1}{p_i} \\sum_j P_{ij} \\, Y_j.

This is the canonical *map* induced by a plan that may itself be many-to-many.
When the plan is a permutation it recovers the matched target exactly; when
the plan is uniform it returns the global mean of ``Y``.

Used downstream in Phase 6 to push a source-side steering signal through a
cross-model coupling onto the target model's coordinate system.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ot_steering.utils.logging import get_logger

_log = get_logger(__name__)

NDArrayF = NDArray[np.float64]


def barycentric_project(
    coupling: NDArrayF,
    target_features: NDArrayF,
    source_marginal: NDArrayF | None = None,
    *,
    weight_floor: float = 1e-30,
) -> NDArrayF:
    """Project source points onto the target space via a transport plan.

    Args:
        coupling: Transport plan ``P`` of shape ``(n, m)``. Rows must
            sum (approximately) to ``source_marginal``.
        target_features: Feature matrix ``Y`` of shape ``(m, d)``. Each row
            is a point in some target space; ``d`` is arbitrary.
        source_marginal: Optional source histogram ``p`` of shape ``(n,)``.
            If ``None``, it is inferred from ``coupling.sum(axis=1)``.
            Providing it explicitly is useful when ``coupling`` has slight
            numerical marginal violations and you want the canonical answer.
        weight_floor: Smallest source weight considered non-zero. Rows of
            the coupling whose source mass is below this value are mapped
            to the zero vector (their image is undefined; we report it as
            zero rather than NaN).

    Returns:
        Array of shape ``(n, d)``; row ``i`` is the barycentric image of
        source point ``i``.

    Raises:
        ValueError: If shapes are inconsistent.
    """
    if coupling.ndim != 2:
        raise ValueError(f"coupling must be 2-D; got shape {coupling.shape}")
    if target_features.ndim != 2:
        raise ValueError(f"target_features must be 2-D; got shape {target_features.shape}")
    n, m = coupling.shape
    if target_features.shape[0] != m:
        raise ValueError(
            f"target_features has {target_features.shape[0]} rows but coupling expects {m}"
        )

    plan = np.ascontiguousarray(coupling, dtype=np.float64)
    features = np.ascontiguousarray(target_features, dtype=np.float64)

    if source_marginal is None:
        marginal = plan.sum(axis=1)
    else:
        if source_marginal.shape != (n,):
            raise ValueError(
                f"source_marginal shape {source_marginal.shape} does not match coupling rows ({n},)"
            )
        marginal = np.ascontiguousarray(source_marginal, dtype=np.float64)

    weighted = plan @ features  # shape (n, d)

    # Guard against divide-by-zero on rows with no mass; emit zeros there.
    safe_marginal = np.where(marginal > weight_floor, marginal, 1.0)
    projection = weighted / safe_marginal[:, None]
    projection = np.where((marginal > weight_floor)[:, None], projection, np.zeros_like(projection))

    _log.debug(
        "barycentric_project: n=%d m=%d d=%d non_empty_rows=%d",
        n,
        m,
        features.shape[1],
        int((marginal > weight_floor).sum()),
    )
    return projection
