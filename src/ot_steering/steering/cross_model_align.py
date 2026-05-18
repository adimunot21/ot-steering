"""Cross-model alignment via Gromov-Wasserstein.

When you want to compare two sets of activations that live in different
models' residual streams, plain optimal transport is undefined: ``d_model``
differs, the coordinate frames are unrelated, the squared-Euclidean cost
between a source vector and a target vector has no meaning. Gromov-Wasserstein
(:mod:`ot_steering.ot.gw`) is the right tool — it aligns distributions
using only the *intra-distribution* distance matrices, which are perfectly
defined on each side.

This module is the cross-model consumer of the GW solver:

1. Reduce each side to a small set of representative points — either
   the raw activations themselves, or the GMM cluster centroids fitted via
   :mod:`ot_steering.steering.ot_steering` (the cluster-centroid path
   reduces the GW problem from O((n+m)^2) to O((k_s+k_t)^2), which matters
   because POT's GW is itself O((n+m)^2) per inner iteration).
2. Build the two intra-distance matrices ``C1`` (n×n) and ``C2`` (m×m)
   under the chosen metric (Euclidean or cosine).
3. Choose uniform marginals (or use GMM weights when centroids are
   provided) and call ``solve_entropic_gw``.

Phase 5 only uses this for *alignment sanity checks*; Phase 6 will use the
returned coupling to barycentric-project a source-side steering signal
onto the target side.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from numpy.typing import NDArray
from pydantic import Field, model_validator

from ot_steering.ot.gw import GWConfig, solve_entropic_gw
from ot_steering.steering.ot_steering import GMMConfig, fit_gmm
from ot_steering.utils.config import BaseConfig
from ot_steering.utils.logging import get_logger

_log = get_logger(__name__)

NDArrayF = NDArray[np.float64]
DistanceMetric = Literal["euclidean", "cosine"]


class CrossModelGWConfig(BaseConfig):
    """Configuration for :func:`cross_model_gw_coupling`.

    Attributes:
        n_components_source: Optional GMM component count for the source
            side. ``None`` uses the raw activations as the empirical
            distribution (uniform marginals).
        n_components_target: Same for the target side.
        distance_metric: ``"euclidean"`` builds ``C[i,j] = ||x_i − x_j||``;
            ``"cosine"`` builds ``C[i,j] = 1 − cos(x_i, x_j)``. Cosine is
            useful when activation magnitudes differ wildly across models.
        normalize_distances: If True (default), rescale each intra-distance
            matrix by its maximum so all entries lie in ``[0, 1]``. This
            matters because POT's entropic GW expects ``reg`` to be small
            relative to the distance scale; without normalisation, raw
            LLM activations (with values in the hundreds) make even
            ``reg=0.05`` so tight that the solver degenerates. Disable
            only if you know the absolute scale carries meaning.
        gmm_cfg: GMM config (used only when one of the n_components_* is
            set). Defaults inherit from :class:`GMMConfig`.
        gw_cfg: GW solver config. Defaults inherit from :class:`GWConfig`.
    """

    n_components_source: int | None = Field(default=None, ge=1)
    n_components_target: int | None = Field(default=None, ge=1)
    distance_metric: DistanceMetric = "euclidean"
    normalize_distances: bool = True
    gmm_cfg: GMMConfig | None = None
    gw_cfg: GWConfig | None = None

    @model_validator(mode="after")
    def _check_partial_gmm_spec(self) -> CrossModelGWConfig:
        # If only one side asks for a GMM, that's almost always a config
        # bug — flag it rather than silently mixing.
        s, t = self.n_components_source, self.n_components_target
        if (s is None) ^ (t is None):
            raise ValueError(
                "specify n_components for both source and target, or neither; "
                f"got source={s}, target={t}"
            )
        return self


@dataclass(frozen=True)
class CrossModelAlignment:
    """Output of a cross-model GW alignment.

    Attributes:
        source_centers: ``(n_source_eff, d_source)`` points used on the
            source side — either raw activations or GMM centroids.
        target_centers: ``(n_target_eff, d_target)`` analogous for target.
        source_distance: ``(n_source_eff, n_source_eff)`` intra-distance
            matrix.
        target_distance: ``(n_target_eff, n_target_eff)`` analogous.
        source_marginal: ``(n_source_eff,)`` histogram passed to GW.
        target_marginal: ``(n_target_eff,)`` analogous.
        coupling: ``(n_source_eff, n_target_eff)`` GW coupling.
        gw_cost: Linear GW objective at the returned coupling.
        metric: The distance metric used.
    """

    source_centers: NDArrayF
    target_centers: NDArrayF
    source_distance: NDArrayF
    target_distance: NDArrayF
    source_marginal: NDArrayF
    target_marginal: NDArrayF
    coupling: NDArrayF
    gw_cost: float
    metric: DistanceMetric


def _to_numpy(x: torch.Tensor | NDArrayF) -> NDArrayF:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float64)
    return np.ascontiguousarray(x, dtype=np.float64)


def _pairwise_distances(x: NDArrayF, metric: DistanceMetric) -> NDArrayF:
    if metric == "euclidean":
        # sqrt(sum((x_i - x_j)^2)); clip negative round-off before sqrt.
        sq = ((x[:, None, :] - x[None, :, :]) ** 2).sum(-1)
        return np.asarray(np.sqrt(np.clip(sq, 0.0, None)), dtype=np.float64)
    # cosine: 1 - <x_i, x_j> / (||x_i|| ||x_j||).
    norms = np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-12)
    normed = x / norms
    cos_sim = normed @ normed.T
    # Numerical clamp to [-1, 1] before subtraction.
    return np.asarray(1.0 - np.clip(cos_sim, -1.0, 1.0), dtype=np.float64)


def cross_model_gw_coupling(
    source: torch.Tensor | NDArrayF,
    target: torch.Tensor | NDArrayF,
    *,
    cfg: CrossModelGWConfig | None = None,
    rng: np.random.Generator | None = None,
) -> CrossModelAlignment:
    """Align two activation distributions across models via Gromov-Wasserstein.

    Args:
        source: ``(n, d_source)`` source-model activations.
        target: ``(m, d_target)`` target-model activations. ``d_source``
            and ``d_target`` may differ — that is the whole point.
        cfg: Alignment configuration.
        rng: Optional NumPy ``Generator`` for the GW multi-restart
            initialisations.

    Returns:
        A populated :class:`CrossModelAlignment`.

    Raises:
        ValueError: If shapes are inconsistent or the GMM config is
            partially specified.
    """
    cfg = cfg or CrossModelGWConfig()
    src_arr = _to_numpy(source)
    tgt_arr = _to_numpy(target)
    if src_arr.ndim != 2 or tgt_arr.ndim != 2:
        raise ValueError(
            f"source and target must be 2-D; got source={src_arr.shape}, target={tgt_arr.shape}"
        )

    # Optionally reduce to GMM centroids on each side.
    if cfg.n_components_source is not None:
        assert cfg.n_components_target is not None  # validator guarantees
        gmm_cfg = cfg.gmm_cfg or GMMConfig()
        src_gmm = fit_gmm(
            src_arr, gmm_cfg.model_copy(update={"n_components": cfg.n_components_source})
        )
        tgt_gmm = fit_gmm(
            tgt_arr, gmm_cfg.model_copy(update={"n_components": cfg.n_components_target})
        )
        src_pts = src_gmm.means_.astype(np.float64)
        tgt_pts = tgt_gmm.means_.astype(np.float64)
        src_p = src_gmm.weights_.astype(np.float64)
        tgt_q = tgt_gmm.weights_.astype(np.float64)
    else:
        src_pts = src_arr
        tgt_pts = tgt_arr
        src_p = np.full(src_pts.shape[0], 1.0 / src_pts.shape[0])
        tgt_q = np.full(tgt_pts.shape[0], 1.0 / tgt_pts.shape[0])

    src_dist = _pairwise_distances(src_pts, cfg.distance_metric)
    tgt_dist = _pairwise_distances(tgt_pts, cfg.distance_metric)
    if cfg.normalize_distances:
        src_dist = src_dist / max(src_dist.max(), 1e-12)
        tgt_dist = tgt_dist / max(tgt_dist.max(), 1e-12)

    gw_cfg = cfg.gw_cfg or GWConfig()
    coupling, cost = solve_entropic_gw(src_dist, tgt_dist, src_p, tgt_q, gw_cfg, rng=rng)

    _log.info(
        "cross_model_gw: source_pts=%d target_pts=%d metric=%s gw_cost=%.4f",
        src_pts.shape[0],
        tgt_pts.shape[0],
        cfg.distance_metric,
        cost,
    )

    return CrossModelAlignment(
        source_centers=src_pts,
        target_centers=tgt_pts,
        source_distance=src_dist,
        target_distance=tgt_dist,
        source_marginal=src_p,
        target_marginal=tgt_q,
        coupling=coupling,
        gw_cost=cost,
        metric=cfg.distance_metric,
    )
