"""Thin wrapper around POT's entropic Gromov-Wasserstein solver.

GW is what we use when the source and target distributions live in two
*different* spaces and we cannot define a pointwise cost between them. The
inputs are intra-distance matrices ``C1`` (n×n on the source side) and
``C2`` (m×m on the target side); the solver finds a coupling ``P`` that
preserves pairwise relations as much as possible.

POT's ``ot.gromov.entropic_gromov_wasserstein`` does the heavy lifting. The
wrapper adds:

1. A pydantic config (``GWConfig``) that validates ``epsilon``, ``loss_fun``,
   iteration counts, and the multi-restart count at construction time.
2. **Multi-restart** logic. The GW objective is non-convex (quadratic in P);
   the solver can get stuck in poor local minima. Running the solver from
   ``num_restart`` different initialisations and keeping the lowest-cost
   result is the standard mitigation.
3. A unified return value: ``(coupling, gw_cost)``.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from typing import Literal

import numpy as np
import ot.gromov
from numpy.typing import NDArray
from pydantic import Field

import ot
from ot_steering.utils.config import BaseConfig
from ot_steering.utils.logging import get_logger

_log = get_logger(__name__)

GWLoss = Literal["square_loss", "kl_loss"]
NDArrayF = NDArray[np.float64]


class GWConfig(BaseConfig):
    """Configuration for the entropic Gromov-Wasserstein solver.

    Attributes:
        reg: Entropic regularisation strength (POT calls this ``epsilon``).
            Smaller → sharper coupling, closer to the unregularised GW
            optimum but harder to solve; larger → smoother coupling.
        loss_fun: Pointwise loss applied to ``|C1[i, k] - C2[j, l]|``.
            ``"square_loss"`` is the standard choice (matches Mémoli's
            original formulation up to a constant); ``"kl_loss"`` is the
            KL variant useful when the distance matrices are themselves
            probability-like.
        num_iter_max: Hard cap on outer iterations of the projected-gradient
            descent. Each outer iteration runs an inner Sinkhorn.
        stop_threshold: Stop early when the change in coupling between
            successive outer iterations is below this value.
        num_restart: Number of independent random initialisations of the
            coupling. The lowest-cost run is returned. ``1`` disables.
        warn_on_no_convergence: If True, POT emits a warning per restart
            when the inner Sinkhorn fails to converge. Useful at small
            ``reg``; turn off in tight inner loops.
    """

    reg: float = Field(default=0.05, gt=0.0)
    loss_fun: GWLoss = "square_loss"
    num_iter_max: int = Field(default=1_000, gt=0)
    stop_threshold: float = Field(default=1e-9, gt=0.0)
    num_restart: int = Field(default=1, ge=1)
    warn_on_no_convergence: bool = True


@contextmanager
def _maybe_silence_pot_warnings(suppress: bool):  # type: ignore[no-untyped-def]
    """Silence POT's non-convergence UserWarnings when the caller asked us to.

    POT's ``entropic_gromov_wasserstein`` does not expose a ``warn`` knob
    (unlike ``ot.sinkhorn``), so the only way to honour
    ``warn_on_no_convergence=False`` is to filter the warning at emission.
    Scope is intentionally narrow: only ``UserWarning`` originating from the
    ``ot.*`` package is muted.
    """
    if not suppress:
        yield
        return
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module=r"ot\..*")
        warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"ot\..*")
        yield


def _validate_inputs(c1: NDArrayF, c2: NDArrayF, p: NDArrayF, q: NDArrayF) -> tuple[int, int]:
    if c1.ndim != 2 or c1.shape[0] != c1.shape[1]:
        raise ValueError(f"C1 must be a square 2-D matrix; got shape {c1.shape}")
    if c2.ndim != 2 or c2.shape[0] != c2.shape[1]:
        raise ValueError(f"C2 must be a square 2-D matrix; got shape {c2.shape}")
    n, m = c1.shape[0], c2.shape[0]
    if p.shape != (n,):
        raise ValueError(f"p shape {p.shape} does not match C1 ({n},)")
    if q.shape != (m,):
        raise ValueError(f"q shape {q.shape} does not match C2 ({m},)")
    return n, m


def solve_entropic_gw(
    c1: NDArrayF,
    c2: NDArrayF,
    p: NDArrayF,
    q: NDArrayF,
    cfg: GWConfig | None = None,
    *,
    rng: np.random.Generator | None = None,
) -> tuple[NDArrayF, float]:
    """Solve entropic Gromov-Wasserstein between two distance-equipped clouds.

    Optionally runs the solver from multiple random initialisations and
    returns the lowest-cost coupling. The GW cost reported is the linear
    GW objective evaluated on the returned coupling, computed by POT as
    ``ot.gromov.gwloss(constC, hC1, hC2, T)`` — i.e. the same value POT's
    ``entropic_gromov_wasserstein2`` returns, modulo a tiny re-evaluation.

    Args:
        c1: Source intra-distance matrix, shape ``(n, n)``.
        c2: Target intra-distance matrix, shape ``(m, m)``.
        p: Source histogram, shape ``(n,)``. Non-negative; sums to 1.
        q: Target histogram, shape ``(m,)``. Non-negative; sums to 1.
        cfg: Solver configuration. Defaults to :class:`GWConfig`'s defaults.
        rng: Optional NumPy ``Generator`` used to seed the multi-restart
            initialisations. If ``None``, a fresh default generator is used.

    Returns:
        ``(coupling, gw_cost)`` where ``coupling`` has shape ``(n, m)`` and
        ``gw_cost`` is the (linear) GW loss of that coupling.

    Raises:
        ValueError: If any shape is inconsistent.
    """
    cfg = cfg or GWConfig()
    c1 = np.ascontiguousarray(c1, dtype=np.float64)
    c2 = np.ascontiguousarray(c2, dtype=np.float64)
    p = np.ascontiguousarray(p, dtype=np.float64)
    q = np.ascontiguousarray(q, dtype=np.float64)
    n, m = _validate_inputs(c1, c2, p, q)
    rng = rng or np.random.default_rng()

    _log.debug(
        "solve_entropic_gw: n=%d m=%d reg=%.4g loss=%s restarts=%d",
        n,
        m,
        cfg.reg,
        cfg.loss_fun,
        cfg.num_restart,
    )

    best_coupling: NDArrayF | None = None
    best_cost = np.inf
    for restart in range(cfg.num_restart):
        # Initialise the coupling. The first restart uses POT's default
        # (the rank-one product p ⊗ q); subsequent restarts perturb the
        # outer product to break symmetry.
        if restart == 0:
            g0: NDArrayF | None = None
        else:
            noise = rng.uniform(0.5, 1.5, size=(n, m))
            unnormalised = np.outer(p, q) * noise
            # Project onto the transport polytope via 50 Sinkhorn rescalings.
            for _ in range(50):
                unnormalised *= (p / unnormalised.sum(axis=1).clip(min=1e-30))[:, None]
                unnormalised *= (q / unnormalised.sum(axis=0).clip(min=1e-30))[None, :]
            g0 = unnormalised

        with _maybe_silence_pot_warnings(suppress=not cfg.warn_on_no_convergence):
            coupling: NDArrayF = ot.gromov.entropic_gromov_wasserstein(
                c1,
                c2,
                p,
                q,
                loss_fun=cfg.loss_fun,
                epsilon=cfg.reg,
                G0=g0,
                max_iter=cfg.num_iter_max,
                tol=cfg.stop_threshold,
            )
        cost = _gw_loss(c1, c2, coupling, cfg.loss_fun)
        _log.debug("  restart %d/%d: cost=%.6g", restart + 1, cfg.num_restart, cost)
        if cost < best_cost:
            best_cost = cost
            best_coupling = coupling

    assert best_coupling is not None
    return best_coupling, float(best_cost)


def _gw_loss(c1: NDArrayF, c2: NDArrayF, plan: NDArrayF, loss_fun: GWLoss) -> float:
    """Recompute the GW objective for a given coupling.

    For ``loss_fun='square_loss'``, this is
    ``sum_{ijkl} (C1[i, k] - C2[j, l])**2 * P[i, j] * P[k, l]``.
    For ``loss_fun='kl_loss'``, the loss applied per entry is the
    Kullback-Leibler divergence ``C1[i, k] * (log(C1[i, k] / C2[j, l]) - 1) +
    C2[j, l]`` per POT's convention.

    We compute it via POT's ``init_matrix`` + ``gwloss`` so the value is
    bit-identical to what ``entropic_gromov_wasserstein2`` would return.
    """
    const_c, h_c1, h_c2 = ot.gromov.init_matrix(c1, c2, plan.sum(1), plan.sum(0), loss_fun)
    return float(ot.gromov.gwloss(const_c, h_c1, h_c2, plan))
