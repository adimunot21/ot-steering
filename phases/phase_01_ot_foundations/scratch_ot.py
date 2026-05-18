"""Discrete optimal transport, solved from scratch as a linear program.

PEDAGOGICAL ONLY. This file exists so the reader of Chapter 1 can see, in plain
NumPy + ``scipy.optimize.linprog``, exactly what the Kantorovich problem is
asking for. Nothing in ``src/`` or ``tests/`` imports from here. Production
code calls :mod:`ot_steering.ot.emd`, which wraps POT's C++ network-flow
solver.

The problem. Given a non-negative source histogram ``a`` summing to one over
``n`` bins, a non-negative target histogram ``b`` summing to one over ``m``
bins, and a cost matrix ``M`` of shape ``(n, m)``, find a transport plan
``P`` of shape ``(n, m)`` that

    minimises    sum_ij P_ij * M_ij        (total transport cost)
    subject to   sum_j P_ij = a_i          (mass-out from source bin i)
                 sum_i P_ij = b_j          (mass-in to target bin j)
                 P_ij >= 0                 (no negative mass)

This is a linear program in the ``n * m`` variables ``P_ij``. The constraints
are linear equalities and non-negativity bounds; the objective is the dot
product ``<P, M>``. ``scipy.optimize.linprog`` solves exactly this.
"""

from __future__ import annotations

import numpy as np
import scipy.optimize as opt
from numpy.typing import NDArray


def scratch_emd(
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    M: NDArray[np.float64],
) -> tuple[NDArray[np.float64], float]:
    """Solve the discrete Kantorovich OT problem via ``scipy.optimize.linprog``.

    Flattens the ``n × m`` transport plan into a single vector of length
    ``n * m`` and assembles the marginal constraints as a sparse-ish equality
    block. ``scipy``'s HiGHS backend then solves the LP.

    Args:
        a: Source histogram, shape ``(n,)``. Non-negative, sums to 1.
        b: Target histogram, shape ``(m,)``. Non-negative, sums to 1.
        M: Cost matrix, shape ``(n, m)``. Entry ``M[i, j]`` is the cost of
            moving one unit of mass from source bin ``i`` to target bin ``j``.

    Returns:
        ``(P, cost)`` where ``P`` is the optimal transport plan
        ``shape (n, m)`` and ``cost`` is ``<P, M>``.

    Raises:
        AssertionError: If shapes or marginals are inconsistent.
        RuntimeError: If the LP solver fails to converge.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)

    n, m = M.shape
    assert a.shape == (n,), f"a shape {a.shape} != ({n},)"
    assert b.shape == (m,), f"b shape {b.shape} != ({m},)"
    assert np.all(a >= -1e-12), "source histogram has negative entries"
    assert np.all(b >= -1e-12), "target histogram has negative entries"
    assert np.isclose(a.sum(), b.sum(), atol=1e-8), (
        f"marginal totals must agree: a.sum()={a.sum()} b.sum()={b.sum()}"
    )

    # Decision variable: x = vec(P) of length n*m, with P[i, j] -> x[i*m + j].
    c = M.reshape(-1)

    # Equality constraints A_eq @ x = b_eq.
    # We have n row-sum constraints and m column-sum constraints.
    a_eq = np.zeros((n + m, n * m), dtype=np.float64)
    for i in range(n):
        # Row i of P contributes to a[i].
        a_eq[i, i * m : (i + 1) * m] = 1.0
    for j in range(m):
        # Column j of P contributes to b[j]. Indices i*m + j for i in [0, n).
        a_eq[n + j, j::m] = 1.0
    b_eq = np.concatenate([a, b])

    bounds = [(0.0, None)] * (n * m)

    result = opt.linprog(c=c, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not result.success:
        raise RuntimeError(f"linprog failed: {result.message}")

    plan = result.x.reshape(n, m)
    cost = float(result.fun)
    return plan, cost


if __name__ == "__main__":
    # Sanity check: a 3x3 problem with a known answer (identity assignment).
    a_hist = np.array([1 / 3, 1 / 3, 1 / 3])
    b_hist = np.array([1 / 3, 1 / 3, 1 / 3])
    cost_matrix = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 1.0],
            [2.0, 1.0, 0.0],
        ]
    )
    plan_out, cost_out = scratch_emd(a_hist, b_hist, cost_matrix)
    print(f"plan =\n{plan_out}")
    print(f"cost = {cost_out:.6f}  (expected 0 — diagonal alignment)")
