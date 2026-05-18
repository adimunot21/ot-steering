"""Cross-model steering transport via Gromov-Wasserstein + barycentric projection.

This is the project's headline construction. Combine the machinery from the
previous phases:

- Phase 1 (`ot_steering.ot.emd`): discrete OT solver.
- Phase 2 (`ot_steering.ot.barycentric`): barycentric projection through a
  coupling.
- Phase 4 (`ot_steering.steering.ot_steering`): CHaRS-style intra-model
  steering map (GMM + EMD + barycentric).
- Phase 5 (`ot_steering.steering.cross_model_align`): Gromov-Wasserstein
  between two models' activation distributions, with normalised intra-
  distance matrices.

…to give a single pipeline that takes contrastive activations from a *source*
model and *target* model, and produces a target-side, input-conditional
steering map that lives entirely in the target model's coordinate frame.
No target-side supervision was used to construct the steering map's
*direction* — only target-side activations needed for clustering and the
GW coupling.

The pipeline, in pictures:

    A NEG centroids ──[CHaRS in A]──> A NEG cluster→A POS target image
                │                                          │
        (cross-model GW)                                   │
                │                                          │
                ▼                                          │
    B NEG centroids ◀──[P_neg]── A NEG centroids           ▼
                                                  (cross-model GW, P_pos)
                                                           │
                                                           ▼
                                                  B POS centroids

So:

    A NEG cluster i   ──CHaRS──>   A POS target_i  ∈ ℝ^{d_A}
                                          │
                                  P_pos barycentric
                                          │
                                          ▼
                                  B POS image_i   ∈ ℝ^{d_B}

For each B NEG cluster j, we then collect the source A NEG clusters that
map onto it (rows of P_neg^T) and barycentric-project their B POS images:

    transported_target_j = Σ_i  (P_neg[i, j] / q_j) · B_POS_image_i
    transported_disp_j   = transported_target_j − B_NEG_center[j]

The runtime hook attaches to a B-side transformer block, soft-assigns each
incoming token to a B NEG cluster (via the fitted GMM responsibilities),
blends per-cluster displacements, and adds ``coefficient * blended_disp``
to the residual stream.
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
from ot_steering.ot.gw import GWConfig
from ot_steering.steering.cross_model_align import (
    CrossModelAlignment,
    CrossModelGWConfig,
    cross_model_gw_coupling,
)
from ot_steering.steering.ot_steering import (
    GMMConfig,
    OTSteeringMap,
    build_ot_steering_map,
    fit_gmm,
)
from ot_steering.utils.config import BaseConfig
from ot_steering.utils.logging import get_logger

_log = get_logger(__name__)

NDArrayF = NDArray[np.float64]
Assignment = Literal["soft", "hard"]


class SteeringTransportConfig(BaseConfig):
    """Configuration for :func:`build_transport`.

    Attributes:
        n_components: GMM component count for every per-class GMM
            (POS_A, NEG_A, POS_B, NEG_B). Equal on both sides of the GW
            problem keeps the couplings k×k.
        distance_metric: Forwarded to the cross-model GW. ``"euclidean"``
            is the safe default; ``"cosine"`` helps when activation
            magnitudes differ wildly between models.
        gmm_cfg: GMM configuration; ``n_components`` here is overridden
            by the outer ``n_components`` at construction time.
        gw_cfg: GW solver configuration for the cross-model couplings.
            With ``normalize_distances=True`` (default) on the cross-
            model side, ``reg`` around ``0.01`` is sharp enough.
        assignment: Runtime cluster assignment style for the B-side hook.
            ``"soft"`` blends per-cluster displacements by the GMM
            responsibilities; ``"hard"`` picks the argmax cluster.
    """

    n_components: int = Field(default=4, ge=1)
    distance_metric: Literal["euclidean", "cosine"] = "euclidean"
    gmm_cfg: GMMConfig | None = None
    gw_cfg: GWConfig | None = None
    assignment: Assignment = "soft"


@dataclass(frozen=True)
class TransportedSteeringMap:
    """A target-side, input-conditional steering map transported from a source model.

    Attributes:
        source_steering_map: The full CHaRS-style intra-model steering map
            built on the source model's contrastive activations.
        cross_model_alignment_neg: GW alignment of A NEG centroids and
            B NEG centroids. Used at runtime to barycentric-project source
            per-cluster info onto the B NEG cluster grid.
        cross_model_alignment_pos: GW alignment of A POS centroids and
            B POS centroids. Used to find the B-space image of A's POS
            target centroids.
        target_neg_gmm: Fitted GMM on B NEG activations. Used at runtime
            to soft-assign incoming tokens to B NEG clusters.
        target_neg_centers: ``(k, d_B)`` B NEG GMM centroids.
        target_pos_centers: ``(k, d_B)`` B POS GMM centroids.
        transported_displacements: ``(k, d_B)`` — per B NEG cluster, the
            displacement vector the runtime hook will add to incoming
            activations (scaled by the user-supplied coefficient).
        assignment: Runtime cluster assignment style.
    """

    source_steering_map: OTSteeringMap
    cross_model_alignment_neg: CrossModelAlignment
    cross_model_alignment_pos: CrossModelAlignment
    target_neg_gmm: GaussianMixture
    target_neg_centers: NDArrayF
    target_pos_centers: NDArrayF
    transported_displacements: NDArrayF
    assignment: Assignment = "soft"


def _to_numpy(x: torch.Tensor | NDArrayF) -> NDArrayF:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float64)
    return np.ascontiguousarray(x, dtype=np.float64)


def build_transport(
    source_acts_pos: torch.Tensor | NDArrayF,
    source_acts_neg: torch.Tensor | NDArrayF,
    target_acts_pos: torch.Tensor | NDArrayF,
    target_acts_neg: torch.Tensor | NDArrayF,
    cfg: SteeringTransportConfig | None = None,
    *,
    rng: np.random.Generator | None = None,
) -> TransportedSteeringMap:
    """Build a target-side steering map transported from the source.

    Args:
        source_acts_pos: ``(n_pos_A, d_A)`` positive-class activations
            on the source model.
        source_acts_neg: ``(n_neg_A, d_A)`` negative-class activations
            on the source model.
        target_acts_pos: ``(n_pos_B, d_B)`` positive-class activations
            on the target model.
        target_acts_neg: ``(n_neg_B, d_B)`` negative-class activations
            on the target model. **Used only for clustering** — the
            POS↔NEG difference on the target is NEVER used to construct
            the per-cluster displacement direction (that would defeat
            the no-target-supervision promise).
        cfg: Transport configuration. Defaults are reasonable for
            ``n_components=4`` GMMs and ``reg=0.01`` GW.
        rng: Optional NumPy ``Generator`` for the GW multi-restart
            initialisations. Used twice (once per cross-model GW).

    Returns:
        A populated :class:`TransportedSteeringMap`.

    Raises:
        ValueError: If activation matrices are not 2-D or dimensions
            disagree across classes within a single model.
    """
    cfg = cfg or SteeringTransportConfig()
    rng = rng or np.random.default_rng()

    pos_a = _to_numpy(source_acts_pos)
    neg_a = _to_numpy(source_acts_neg)
    pos_b = _to_numpy(target_acts_pos)
    neg_b = _to_numpy(target_acts_neg)
    for name, arr in [
        ("source_pos", pos_a),
        ("source_neg", neg_a),
        ("target_pos", pos_b),
        ("target_neg", neg_b),
    ]:
        if arr.ndim != 2:
            raise ValueError(f"{name} activations must be 2-D, got shape {arr.shape}")
    if pos_a.shape[1] != neg_a.shape[1]:
        raise ValueError(f"source-side d mismatch: pos={pos_a.shape[1]}, neg={neg_a.shape[1]}")
    if pos_b.shape[1] != neg_b.shape[1]:
        raise ValueError(f"target-side d mismatch: pos={pos_b.shape[1]}, neg={neg_b.shape[1]}")

    gmm_cfg = cfg.gmm_cfg or GMMConfig()
    k = cfg.n_components
    gmm_cfg_k = gmm_cfg.model_copy(update={"n_components": k})

    # --- step 1: CHaRS on the source side (Phase 4).
    source_steering_map = build_ot_steering_map(
        pos_a, neg_a, gmm_cfg=gmm_cfg_k, assignment=cfg.assignment
    )
    a_neg_centers = source_steering_map.source_centers  # (k, d_A)
    a_pos_centers = source_steering_map.target_centers  # (k, d_A)
    chars_coupling = source_steering_map.coupling  # (k, k) — A NEG → A POS

    # --- step 2: fit B-side GMMs (we only use the centroids and B NEG's GMM).
    b_neg_gmm = fit_gmm(neg_b, gmm_cfg_k)
    b_pos_gmm = fit_gmm(pos_b, gmm_cfg_k)
    b_neg_centers = b_neg_gmm.means_.astype(np.float64)  # (k, d_B)
    b_pos_centers = b_pos_gmm.means_.astype(np.float64)  # (k, d_B)
    b_neg_weights = b_neg_gmm.weights_.astype(np.float64)

    # --- step 3: cross-model GW (Phase 5). One for NEG↔NEG, one for POS↔POS.
    align_cfg = CrossModelGWConfig(
        distance_metric=cfg.distance_metric,
        normalize_distances=True,
        gw_cfg=cfg.gw_cfg
        or GWConfig(reg=0.01, num_iter_max=500, num_restart=2, warn_on_no_convergence=False),
    )
    neg_alignment = cross_model_gw_coupling(a_neg_centers, b_neg_centers, cfg=align_cfg, rng=rng)
    pos_alignment = cross_model_gw_coupling(a_pos_centers, b_pos_centers, cfg=align_cfg, rng=rng)

    # --- step 4: barycentric chain.
    # For each A POS cluster m, its B-space image via P_pos.
    # barycentric_project(coupling, target_features, source_marginal) sends
    # source row i to (1/p_i) Σ_j P[i,j] target_features[j].
    b_image_of_a_pos = barycentric_project(  # (k, d_B)
        coupling=pos_alignment.coupling,
        target_features=b_pos_centers,
        source_marginal=pos_alignment.source_marginal,
    )

    # For each A NEG cluster i, its CHaRS target lives at a barycentric
    # blend of A POS centroids weighted by the intra-A coupling row.
    # The B-space image of that CHaRS target is then the same blend
    # applied to b_image_of_a_pos (rows = A POS clusters).
    # NOTE: source_steering_map.source_gmm.weights_ are the A NEG GMM
    # weights (used as the OT marginal). source_marginal = these weights.
    a_neg_weights = source_steering_map.source_gmm.weights_.astype(np.float64)
    # Row-normalise chars_coupling so each row sums to 1 (it sums to p_i).
    row_normaliser = a_neg_weights[:, None].clip(min=1e-12)
    chars_coupling_row_normalised = chars_coupling / row_normaliser
    b_target_of_a_neg = chars_coupling_row_normalised @ b_image_of_a_pos  # (k, d_B)

    # For each B NEG cluster j, the transported target is the barycentric
    # average of b_target_of_a_neg weighted by P_neg[:, j] / q_j.
    transported_targets = barycentric_project(
        coupling=neg_alignment.coupling.T,  # (k_B, k_A) — transpose
        target_features=b_target_of_a_neg,  # (k_A, d_B)
        source_marginal=b_neg_weights,
    )  # (k_B, d_B)
    transported_displacements = transported_targets - b_neg_centers
    transported_norms = np.linalg.norm(transported_displacements, axis=1)
    _log.info(
        "build_transport: k=%d, mean ||transported_disp|| = %.3f",
        k,
        float(transported_norms.mean()),
    )

    return TransportedSteeringMap(
        source_steering_map=source_steering_map,
        cross_model_alignment_neg=neg_alignment,
        cross_model_alignment_pos=pos_alignment,
        target_neg_gmm=b_neg_gmm,
        target_neg_centers=b_neg_centers,
        target_pos_centers=b_pos_centers,
        transported_displacements=transported_displacements,
        assignment=cfg.assignment,
    )


def _per_token_transported_displacement(
    activations: NDArrayF, transported_map: TransportedSteeringMap
) -> NDArrayF:
    """Per-row displacement: blend per-cluster displacements by B NEG responsibilities."""
    if transported_map.assignment == "soft":
        responsibilities = transported_map.target_neg_gmm.predict_proba(activations)
        return np.asarray(
            responsibilities @ transported_map.transported_displacements, dtype=np.float64
        )
    cluster_ids = transported_map.target_neg_gmm.predict(activations)
    return np.asarray(transported_map.transported_displacements[cluster_ids], dtype=np.float64)


@contextmanager
def add_transported_steering_hook(
    block: torch.nn.Module,
    transported_map: TransportedSteeringMap,
    coefficient: float,
) -> Generator[None, None, None]:
    """Forward-pre-hook that adds the transported steering map to a target block.

    For each ``(batch, seq, d_target)`` input, computes per-token B NEG
    responsibilities, blends per-cluster transported displacements, and
    adds ``coefficient * blended_displacement`` to the residual stream.

    Args:
        block: The target-model transformer block to steer.
        transported_map: The transported steering map from
            :func:`build_transport`.
        coefficient: Scalar multiplier. With unit-normalised
            displacements (caller's choice), ``coefficient`` reads as
            "how many activation-norm units of perturbation to inject".
    """

    def _pre_hook(_module, args):  # type: ignore[no-untyped-def]
        if not args:
            return args
        hidden = args[0]  # (B, T, d_target)
        device, dtype = hidden.device, hidden.dtype
        flat = hidden.detach().to("cpu", dtype=torch.float64).reshape(-1, hidden.shape[-1]).numpy()
        disp = _per_token_transported_displacement(flat, transported_map)
        disp_tensor = torch.from_numpy(disp).reshape(hidden.shape).to(device=device, dtype=dtype)
        return (hidden + coefficient * disp_tensor, *args[1:])

    handle = block.register_forward_pre_hook(_pre_hook)
    try:
        yield
    finally:
        handle.remove()
