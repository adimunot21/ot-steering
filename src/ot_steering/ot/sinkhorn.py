"""Thin wrapper around POT's entropic OT solver (``ot.sinkhorn``).

Defaults to the log-domain variant (``method='sinkhorn_log'``) because that is
the only formulation that stays numerically stable when ``reg`` is small
relative to ``cost_matrix.max()`` — which is exactly the regime we want for
sharp transport plans.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import Field

import ot
from ot_steering.utils.config import BaseConfig
from ot_steering.utils.logging import get_logger

_log = get_logger(__name__)

SinkhornMethod = Literal["sinkhorn", "sinkhorn_log", "sinkhorn_stabilized"]


class SinkhornConfig(BaseConfig):
    """Configuration for the Sinkhorn entropic OT solver.

    Attributes:
        reg: Entropic regularisation strength. Smaller → sharper plan, closer
            to the unregularised EMD solution but harder to solve numerically.
            Larger → smoother plan, faster convergence.
        method: Which Sinkhorn variant POT should use. ``"sinkhorn_log"`` is
            the safe default because it works in log-space and so does not
            underflow when ``reg`` is small.
        num_iter_max: Hard cap on Sinkhorn iterations.
        stop_threshold: Stop early when the marginal violation falls below
            this value (POT's internal convergence criterion).
        warn_on_no_convergence: If True, POT emits a warning when the inner
            loop hits ``num_iter_max`` without converging.
    """

    reg: float = Field(default=0.05, gt=0.0)
    method: SinkhornMethod = "sinkhorn_log"
    num_iter_max: int = Field(default=1_000, gt=0)
    stop_threshold: float = Field(default=1e-9, gt=0.0)
    warn_on_no_convergence: bool = True


def solve_sinkhorn(
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    cost_matrix: NDArray[np.float64],
    cfg: SinkhornConfig | None = None,
) -> tuple[NDArray[np.float64], float]:
    """Solve entropy-regularised discrete OT via POT's Sinkhorn iterations.

    Returns the regularised plan ``P`` minimising
    ``<P, cost_matrix> - reg * H(P)`` subject to the usual marginal
    constraints. ``cost`` is the *linear* part ``<P, cost_matrix>``, i.e. the
    transport cost of the regularised plan — not the regularised objective
    value (callers who want that should add ``-reg * H(P)`` themselves).

    Args:
        a: Source histogram, shape ``(n,)``. Non-negative.
        b: Target histogram, shape ``(m,)``. Non-negative.
        cost_matrix: Pairwise cost matrix, shape ``(n, m)``.
        cfg: Solver configuration. Defaults to :class:`SinkhornConfig`'s
            defaults (``reg=0.05``, log-domain, 1 000 iterations max).

    Returns:
        ``(plan, cost)`` where ``plan`` has shape ``(n, m)`` and ``cost`` is
        the linear transport cost ``<plan, cost_matrix>``.

    Raises:
        ValueError: If shapes are inconsistent.
    """
    cfg = cfg or SinkhornConfig()
    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    cost_matrix = np.ascontiguousarray(cost_matrix, dtype=np.float64)

    n, m = cost_matrix.shape
    if a.shape != (n,) or b.shape != (m,):
        raise ValueError(f"shape mismatch: a={a.shape}, b={b.shape}, M={cost_matrix.shape}")

    _log.debug("solve_sinkhorn: n=%d m=%d reg=%.4g method=%s", n, m, cfg.reg, cfg.method)
    plan: NDArray[np.float64] = ot.sinkhorn(
        a,
        b,
        cost_matrix,
        reg=cfg.reg,
        method=cfg.method,
        numItermax=cfg.num_iter_max,
        stopThr=cfg.stop_threshold,
        warn=cfg.warn_on_no_convergence,
    )
    cost = float(np.sum(plan * cost_matrix))
    return plan, cost
