"""Bootstrap correlation helpers for the Phase 7 diagnostic analysis.

The diagnostic question is whether the GW alignment cost predicts
cross-model transfer success across many (model_pair, layer, k, seed)
cells. We report Spearman ρ and a bootstrap-resample 95 % confidence
interval rather than a parametric p-value because the sample size is
small and the data is highly non-Gaussian.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import spearmanr

NDArrayF = NDArray[np.float64]


def spearman_with_bootstrap_ci(
    x: NDArrayF,
    y: NDArrayF,
    *,
    n_boot: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Compute Spearman ρ between ``x`` and ``y`` with a bootstrap CI.

    Args:
        x: ``(n,)`` array of x-coordinates.
        y: ``(n,)`` array of y-coordinates.
        n_boot: Number of bootstrap resamples.
        confidence: Two-sided confidence level (e.g. 0.95 → 2.5/97.5
            percentile interval).
        seed: RNG seed for the resampling.

    Returns:
        ``(rho, lo, hi)``. ``rho`` is Spearman's ρ on the full data;
        ``(lo, hi)`` is the bootstrap percentile interval.

    Raises:
        ValueError: If ``x`` and ``y`` have different lengths or are
            shorter than 3 points (Spearman is undefined).
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same shape; got {x.shape} vs {y.shape}")
    if x.size < 3:
        raise ValueError(f"need at least 3 points for Spearman; got {x.size}")

    result = spearmanr(x, y)
    rho = float(result.statistic)

    rng = np.random.default_rng(seed)
    n = x.size
    boot_rhos = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_rho = spearmanr(x[idx], y[idx]).statistic
        # Resample may produce constant arrays → NaN. Resample-replace.
        while np.isnan(boot_rho):
            idx = rng.integers(0, n, size=n)
            boot_rho = spearmanr(x[idx], y[idx]).statistic
        boot_rhos[b] = boot_rho

    alpha = (1.0 - confidence) / 2.0
    lo = float(np.quantile(boot_rhos, alpha))
    hi = float(np.quantile(boot_rhos, 1.0 - alpha))
    return rho, lo, hi
