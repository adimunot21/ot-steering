"""Entropic-regularised OT via Sinkhorn iterations — from scratch, log-domain.

PEDAGOGICAL ONLY. Not imported by ``src/`` or ``tests/``. See Chapter 1 for
the derivation.

The idea. Add an entropy bonus ``-eps * H(P)`` to the Kantorovich objective.
The solution has the closed form ``P_ij = u_i * K_ij * v_j`` where
``K = exp(-M / eps)`` and ``(u, v)`` are positive scaling vectors fixing the
marginals. Solving for ``u, v`` is alternating rescaling: ``u <- a / (K v)``,
``v <- b / (K^T u)``. That converges geometrically — the only catch is
numerical: for small ``eps``, ``K`` underflows. We therefore work in log-space
with the standard log-sum-exp trick, which is exactly equivalent to the
``method='sinkhorn_log'`` setting POT exposes for the same reason.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp


def scratch_sinkhorn(
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    M: NDArray[np.float64],
    reg: float,
    n_iter: int = 1000,
    tol: float = 1e-9,
) -> tuple[NDArray[np.float64], float]:
    """Solve entropy-regularised OT in log-domain.

    Returns the regularised transport plan that minimises
    ``<P, M> - reg * H(P)`` subject to the same marginal constraints as
    :func:`scratch_emd`. Operates in log-space throughout to stay numerically
    stable when ``reg`` is small relative to ``max(M)``.

    Args:
        a: Source histogram, shape ``(n,)``. Strictly positive entries (so
            their log is defined); sums to 1.
        b: Target histogram, shape ``(m,)``. Strictly positive entries;
            sums to 1.
        M: Cost matrix, shape ``(n, m)``.
        reg: Entropic regularisation strength (``epsilon`` in the chapter).
            Must be > 0. Smaller values give a sparser plan that approaches
            the unregularised EMD solution; larger values give a smoother,
            more uniform plan.
        n_iter: Maximum number of Sinkhorn iterations.
        tol: Stop early when the L1 marginal violation falls below this.

    Returns:
        ``(P, cost)`` where ``P`` is the regularised plan, shape ``(n, m)``,
        and ``cost`` is ``<P, M>`` (the linear part of the regularised
        objective).

    Raises:
        AssertionError: If shapes or marginals are inconsistent, or if
            ``reg`` is non-positive.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)

    n, m = M.shape
    assert a.shape == (n,), f"a shape {a.shape} != ({n},)"
    assert b.shape == (m,), f"b shape {b.shape} != ({m},)"
    assert reg > 0, f"reg must be positive, got {reg}"
    assert np.all(a > 0), "source histogram must be strictly positive (log-domain)"
    assert np.all(b > 0), "target histogram must be strictly positive (log-domain)"
    assert np.isclose(a.sum(), b.sum(), atol=1e-8), (
        f"marginal totals must agree: a.sum()={a.sum()} b.sum()={b.sum()}"
    )

    # Log-domain variables. log_K[i, j] = -M[i, j] / reg.
    log_a = np.log(a)
    log_b = np.log(b)
    log_k = -M / reg

    # log_u, log_v are the log of the scaling vectors.
    log_u = np.zeros(n, dtype=np.float64)
    log_v = np.zeros(m, dtype=np.float64)

    for _ in range(n_iter):
        # log(K @ v) = logsumexp(log_K + log_v[None, :], axis=1)
        log_kv = logsumexp(log_k + log_v[None, :], axis=1)
        log_u_new = log_a - log_kv

        # log(K^T @ u) = logsumexp(log_K + log_u_new[:, None], axis=0)
        log_ktu = logsumexp(log_k + log_u_new[:, None], axis=0)
        log_v_new = log_b - log_ktu

        # After the v update, column sums are exact by construction; the
        # informative residual is the L1 violation on the *row* sums. (Use it
        # as a fail-loud convergence signal — when row sums also match within
        # tol, both u and v have stopped moving.)
        log_p_row = logsumexp(log_k + log_u_new[:, None] + log_v_new[None, :], axis=1)
        err = float(np.abs(np.exp(log_p_row) - a).sum())

        log_u, log_v = log_u_new, log_v_new
        if err < tol:
            break

    # P_ij = exp(log_u_i + log_K_ij + log_v_j).
    log_p = log_u[:, None] + log_k + log_v[None, :]
    plan = np.exp(log_p)
    cost = float(np.sum(plan * M))
    return plan, cost


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n_a, n_b = 5, 6
    a_hist = np.full(n_a, 1.0 / n_a)
    b_hist = np.full(n_b, 1.0 / n_b)
    xs = rng.normal(size=(n_a, 2))
    ys = rng.normal(loc=(2.0, 0.0), size=(n_b, 2))
    cost_matrix = np.sum((xs[:, None, :] - ys[None, :, :]) ** 2, axis=-1)
    plan_out, cost_out = scratch_sinkhorn(a_hist, b_hist, cost_matrix, reg=0.05, n_iter=2000)
    print(f"plan row sums: {plan_out.sum(1)}")
    print(f"plan col sums: {plan_out.sum(0)}")
    print(f"cost = {cost_out:.6f}")
