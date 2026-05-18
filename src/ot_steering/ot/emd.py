"""Thin wrapper around POT's exact OT solver (``ot.emd``).

We use the wrapper rather than calling POT directly so that:

1. Inputs are validated against a pydantic config (``EMDConfig``) — typos in
   config files fail at load time, not deep inside a long-running script.
2. There is a single chokepoint to swap solver backends or add logging /
   timing hooks later.
3. The signature returns both the plan and the cost in one call, which is
   what every downstream caller actually wants.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pydantic import Field

import ot
from ot_steering.utils.config import BaseConfig
from ot_steering.utils.logging import get_logger

_log = get_logger(__name__)


class EMDConfig(BaseConfig):
    """Configuration for the exact (network-flow) OT solver.

    Attributes:
        num_iter_max: Hard cap on internal LP iterations (POT default: 100k).
        check_marginals: If True, POT verifies that the input histograms sum
            to the same total before solving. Cheap, catches data bugs.
        num_threads: How many OpenMP threads POT may use inside the solver.
            ``1`` is the safest default for reproducibility; bump for speed
            when you do not need bitwise determinism.
    """

    num_iter_max: int = Field(default=100_000, gt=0)
    check_marginals: bool = True
    num_threads: int = Field(default=1, ge=1)


def solve_emd(
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    cost_matrix: NDArray[np.float64],
    cfg: EMDConfig | None = None,
) -> tuple[NDArray[np.float64], float]:
    """Solve the exact discrete Kantorovich OT problem via POT.

    Args:
        a: Source histogram, shape ``(n,)``. Non-negative; sums to a scalar
            equal to ``b.sum()``.
        b: Target histogram, shape ``(m,)``. Non-negative; sums to a scalar
            equal to ``a.sum()``.
        cost_matrix: Pairwise cost matrix, shape ``(n, m)``.
        cfg: Solver configuration. Defaults to :class:`EMDConfig`'s defaults.

    Returns:
        ``(plan, cost)`` where ``plan`` is the optimal transport plan with
        shape ``(n, m)`` and ``cost`` is ``<plan, cost_matrix>``.

    Raises:
        ValueError: If shapes are inconsistent.
    """
    cfg = cfg or EMDConfig()
    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    cost_matrix = np.ascontiguousarray(cost_matrix, dtype=np.float64)

    n, m = cost_matrix.shape
    if a.shape != (n,) or b.shape != (m,):
        raise ValueError(f"shape mismatch: a={a.shape}, b={b.shape}, M={cost_matrix.shape}")

    _log.debug("solve_emd: n=%d m=%d max_cost=%.4g", n, m, float(cost_matrix.max()))
    plan: NDArray[np.float64] = ot.emd(
        a,
        b,
        cost_matrix,
        numItermax=cfg.num_iter_max,
        check_marginals=cfg.check_marginals,
        numThreads=cfg.num_threads,
    )
    cost = float(np.sum(plan * cost_matrix))
    return plan, cost
