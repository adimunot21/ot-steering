"""OT-induced steering map between contrastive activation distributions.

Generalises the difference-of-means baseline from
:mod:`ot_steering.steering.baselines` to the case where each class's
activations are *not* a single Gaussian blob. The construction (CHaRS,
Abdullaev et al. 2026, generalising ActAdd):

1. Fit a Gaussian mixture (``k`` components) to each class's activations.
2. Solve discrete OT between the two sets of cluster centroids using a
   squared-Euclidean cost (POT, via :mod:`ot_steering.ot.emd`).
3. Barycentrically project the source centroids through the coupling
   (:mod:`ot_steering.ot.barycentric`) to obtain a *target image* for
   each source cluster.
4. At inference, each activation is soft-assigned to source clusters via
   the GMM responsibilities, the per-cluster displacement vectors
   ``target_image_k − source_center_k`` are blended by those weights, and
   the resulting input-conditional direction is added to the residual stream.

Special case: with ``k_source = k_target = 1`` this is *exactly*
``difference_in_means(positive, negative)`` — the displacement vector
becomes ``mean(positive) − mean(negative)``.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from numpy.typing import NDArray
from pydantic import Field
from sklearn.mixture import GaussianMixture

from ot_steering.ot.barycentric import barycentric_project
from ot_steering.ot.emd import EMDConfig, solve_emd
from ot_steering.utils.config import BaseConfig
from ot_steering.utils.logging import get_logger

_log = get_logger(__name__)

NDArrayF = NDArray[np.float64]
Assignment = Literal["soft", "hard"]


class GMMConfig(BaseConfig):
    """Configuration for the per-class Gaussian-mixture fit.

    Attributes:
        n_components: Number of mixture components ``k``. ``k=1`` reduces
            the OT steering map to the difference-in-means baseline.
        covariance_type: Forwarded to ``sklearn.mixture.GaussianMixture``;
            ``"full"`` is the most expressive but most parameters,
            ``"diag"`` is a good default in high dimensions.
        n_init: Random restarts of EM; the best-likelihood fit is kept.
        max_iter: EM iteration cap.
        reg_covar: Diagonal regularisation added to covariance estimates.
        seed: RNG seed for EM initialisation.
    """

    n_components: int = Field(default=4, ge=1)
    covariance_type: Literal["full", "diag", "tied", "spherical"] = "diag"
    n_init: int = Field(default=3, ge=1)
    max_iter: int = Field(default=200, ge=1)
    reg_covar: float = Field(default=1e-4, gt=0.0)
    seed: int = Field(default=0, ge=0)


@dataclass(frozen=True)
class OTSteeringMap:
    """An OT-induced, input-conditional steering map.

    Attributes:
        source_gmm: Fitted GMM on the *source* (e.g. negative-class)
            activations. Used at inference time to compute per-cluster
            responsibilities for an incoming activation.
        target_gmm: Fitted GMM on the *target* (e.g. positive-class)
            activations.
        source_centers: ``(k_source, d_model)`` GMM means on the source
            side.
        target_centers: ``(k_target, d_model)`` GMM means on the target.
        coupling: ``(k_source, k_target)`` OT plan between the centroids.
        barycentric_targets: ``(k_source, d_model)`` — for each source
            cluster, its barycentric image under the coupling.
        displacements: ``(k_source, d_model)`` ==
            ``barycentric_targets − source_centers``. This is the
            per-cluster direction we add to incoming activations.
        assignment: ``"soft"`` (default) blends displacements by the GMM
            responsibilities; ``"hard"`` picks the argmax cluster.
    """

    source_gmm: GaussianMixture
    target_gmm: GaussianMixture
    source_centers: NDArrayF
    target_centers: NDArrayF
    coupling: NDArrayF
    barycentric_targets: NDArrayF
    displacements: NDArrayF
    assignment: Assignment = "soft"


def fit_gmm(activations: torch.Tensor | NDArrayF, cfg: GMMConfig | None = None) -> GaussianMixture:
    """Fit a Gaussian mixture to a single class's activations.

    Args:
        activations: ``(n, d_model)`` activation matrix; CPU torch tensor
            or numpy array.
        cfg: GMM configuration. Defaults to :class:`GMMConfig`'s defaults.

    Returns:
        Fitted ``sklearn.mixture.GaussianMixture``.

    Raises:
        ValueError: If ``activations`` is not 2-D or has fewer rows than
            ``cfg.n_components``.
    """
    cfg = cfg or GMMConfig()
    arr = (
        activations.detach().cpu().numpy().astype(np.float64)
        if isinstance(activations, torch.Tensor)
        else np.ascontiguousarray(activations, dtype=np.float64)
    )
    if arr.ndim != 2:
        raise ValueError(f"activations must be 2-D, got shape {arr.shape}")
    if arr.shape[0] < cfg.n_components:
        raise ValueError(
            f"need at least {cfg.n_components} samples for {cfg.n_components} components, "
            f"got {arr.shape[0]}"
        )
    gmm = GaussianMixture(
        n_components=cfg.n_components,
        covariance_type=cfg.covariance_type,
        n_init=cfg.n_init,
        max_iter=cfg.max_iter,
        reg_covar=cfg.reg_covar,
        random_state=cfg.seed,
    )
    gmm.fit(arr)
    _log.info(
        "fit_gmm: n_samples=%d d=%d k=%d cov=%s converged=%s log-likelihood=%.3f",
        arr.shape[0],
        arr.shape[1],
        cfg.n_components,
        cfg.covariance_type,
        bool(gmm.converged_),
        float(gmm.score(arr)),
    )
    return gmm


def build_ot_steering_map(
    positive_acts: torch.Tensor | NDArrayF,
    negative_acts: torch.Tensor | NDArrayF,
    *,
    gmm_cfg: GMMConfig | None = None,
    emd_cfg: EMDConfig | None = None,
    assignment: Assignment = "soft",
) -> OTSteeringMap:
    """Build an OT-induced steering map from negative to positive.

    The convention matches the project's other steering helpers: the
    "source" distribution is the *negative* class (the one the steering
    starts from at inference) and the "target" is the *positive* class
    (the one we steer toward). The per-cluster displacement is therefore
    ``barycentric_targets − source_centers``.

    Args:
        positive_acts: ``(n_pos, d_model)`` positive-class activations.
        negative_acts: ``(n_neg, d_model)`` negative-class activations.
        gmm_cfg: GMM config; the same config is used for both classes.
        emd_cfg: OT solver config; defaults are fine for ``k`` in the
            single-to-low-double digits.
        assignment: How to blend per-cluster displacements at inference
            time. ``"soft"`` uses the source GMM's responsibilities;
            ``"hard"`` uses the argmax cluster.

    Returns:
        A populated :class:`OTSteeringMap`.

    Raises:
        ValueError: If the two activation matrices disagree in
            ``d_model``.
    """
    gmm_cfg = gmm_cfg or GMMConfig()

    pos_gmm = fit_gmm(positive_acts, gmm_cfg)
    neg_gmm = fit_gmm(negative_acts, gmm_cfg)

    source_centers = neg_gmm.means_.astype(np.float64)
    target_centers = pos_gmm.means_.astype(np.float64)
    if source_centers.shape[1] != target_centers.shape[1]:
        raise ValueError(
            f"d_model mismatch: source={source_centers.shape[1]} vs "
            f"target={target_centers.shape[1]}"
        )
    source_weights = neg_gmm.weights_.astype(np.float64)
    target_weights = pos_gmm.weights_.astype(np.float64)

    cost = ((source_centers[:, None, :] - target_centers[None, :, :]) ** 2).sum(-1)
    coupling, _ = solve_emd(source_weights, target_weights, cost, emd_cfg)

    barycentric_targets = barycentric_project(
        coupling, target_centers, source_marginal=source_weights
    )
    displacements = barycentric_targets - source_centers

    return OTSteeringMap(
        source_gmm=neg_gmm,
        target_gmm=pos_gmm,
        source_centers=source_centers,
        target_centers=target_centers,
        coupling=coupling,
        barycentric_targets=barycentric_targets,
        displacements=displacements,
        assignment=assignment,
    )


def _per_token_displacement(activations: NDArrayF, steering_map: OTSteeringMap) -> NDArrayF:
    """Compute the input-conditional displacement for each activation row.

    For each row ``x`` of ``activations``: get cluster responsibilities
    from the *source* GMM, then take the weighted sum (or argmax pick) of
    the per-cluster displacement vectors.
    """
    if steering_map.assignment == "soft":
        responsibilities = steering_map.source_gmm.predict_proba(activations)
        return np.asarray(responsibilities @ steering_map.displacements, dtype=np.float64)
    cluster_ids = steering_map.source_gmm.predict(activations)
    return np.asarray(steering_map.displacements[cluster_ids], dtype=np.float64)


@contextmanager
def add_ot_steering_hook(
    block: torch.nn.Module,
    steering_map: OTSteeringMap,
    coefficient: float,
) -> Generator[None, None, None]:
    """Context manager that injects the OT steering map into a block.

    Attaches a forward pre-hook that, for each ``(batch, seq, d_model)``
    input, computes per-token responsibilities under the source GMM and
    adds ``coefficient * displacement(x)`` to each token.

    Args:
        block: The transformer block to steer (use
            :func:`ot_steering.activations.extractor.resolve_blocks`).
        steering_map: The OT steering map built by
            :func:`build_ot_steering_map`.
        coefficient: Scalar multiplier on the per-token displacement.
            Use a coefficient of ``1.0`` to send each activation all the
            way to its barycentric image; smaller values perturb less.
    """

    def _pre_hook(_module, args):  # type: ignore[no-untyped-def]
        if not args:
            return args
        hidden = args[0]  # (B, T, d) in fp16/fp32
        device, dtype = hidden.device, hidden.dtype
        flat = hidden.detach().to("cpu", dtype=torch.float64).reshape(-1, hidden.shape[-1]).numpy()
        disp = _per_token_displacement(flat, steering_map)
        disp_tensor = torch.from_numpy(disp).reshape(hidden.shape).to(device=device, dtype=dtype)
        return (hidden + coefficient * disp_tensor, *args[1:])

    handle = block.register_forward_pre_hook(_pre_hook)
    try:
        yield
    finally:
        handle.remove()
