"""Phase 5 demo: cross-model GW alignment plus its four sanity checks.

Shared between ``make_figures.py`` and the companion notebook. Default
configuration uses Pythia-160M and GPT-2-small so the demo runs end-to-end
in under two minutes on the 4 GB project GPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.decomposition import PCA

from ot_steering.activations.datasets import load_sentiment_pairs
from ot_steering.activations.extractor import extract_residual_stream
from ot_steering.activations.model_loader import ModelLoaderConfig, load_model
from ot_steering.ot.gw import GWConfig
from ot_steering.steering.cross_model_align import (
    CrossModelAlignment,
    CrossModelGWConfig,
    cross_model_gw_coupling,
)
from ot_steering.steering.ot_steering import GMMConfig

NDArrayF = NDArray[np.float64]


@dataclass(frozen=True)
class CrossModelGWDemo:
    """Everything Chapter 5 needs to plot or recap.

    Attributes:
        source_model_id: Source model.
        target_model_id: Target model.
        source_layer: Residual-stream layer extracted on the source.
        target_layer: Layer extracted on the target.
        source_acts: ``(n_pairs * 2, d_source)`` source-side activations
            (positives followed by negatives).
        target_acts: ``(n_pairs * 2, d_target)`` analogous target.
        n_per_class: Number of activations per class.
        source_pca: 2-D ``PCA`` fitted on source activations.
        target_pca: 2-D ``PCA`` fitted on target activations.
        cross_model_alignment: GW alignment of source and target.
        sanity_costs: Dict with keys
            ``"self_pair"``, ``"adjacent_layer"``, ``"random_noise"``,
            ``"cross_model"`` mapping to GW costs.
        sanity_diag_mass: Same keys mapping to argmax-sorted diagonal-mass
            of the corresponding coupling.
        class_confusion: ``(2, 2)`` confusion matrix where rows index
            source class labels (0=positive, 1=negative) and columns
            index target class labels (after argmax through the coupling
            on the centroids). Tells us whether GW preserves the
            positive/negative split when crossing models.
    """

    source_model_id: str
    target_model_id: str
    source_layer: int
    target_layer: int
    source_acts: torch.Tensor
    target_acts: torch.Tensor
    n_per_class: int
    source_pca: PCA
    target_pca: PCA
    cross_model_alignment: CrossModelAlignment
    sanity_costs: dict[str, float]
    sanity_diag_mass: dict[str, float]
    class_confusion: NDArrayF


def _diagonal_mass(coupling: NDArrayF) -> float:
    if coupling.shape[0] != coupling.shape[1]:
        return float("nan")
    order = coupling.argmax(axis=1)
    sorted_coupling = coupling[np.argsort(order), :]
    diag = np.diag(sorted_coupling)
    return float(diag.sum() / max(sorted_coupling.sum(), 1e-12))


def _extract_class_labels(
    n_per_class: int, n_components: int, acts: torch.Tensor, gmm
) -> NDArray[np.intp]:
    """For each fitted GMM cluster, the majority class label (0=pos / 1=neg)."""
    arr = acts.cpu().numpy().astype(np.float64)
    assignments = gmm.predict(arr)
    # First n_per_class rows are positive, last n_per_class are negative.
    true_labels = np.concatenate(
        [np.zeros(n_per_class, dtype=np.intp), np.ones(n_per_class, dtype=np.intp)]
    )
    cluster_to_label = np.zeros(n_components, dtype=np.intp)
    for k in range(n_components):
        mask = assignments == k
        if not mask.any():
            cluster_to_label[k] = -1
            continue
        # Majority vote.
        cluster_to_label[k] = int(np.bincount(true_labels[mask], minlength=2).argmax())
    return cluster_to_label


def run_cross_model_demo(
    source_model_id: str = "EleutherAI/pythia-160m",
    target_model_id: str = "gpt2",
    n_pairs: int = 50,
    n_components: int = 4,
    reg: float = 0.01,
    seed: int = 0,
) -> CrossModelGWDemo:
    """Run the four sanity checks and the full cross-model alignment.

    Args:
        source_model_id: Source model.
        target_model_id: Target model.
        n_pairs: Number of contrastive sentiment pairs to use.
        n_components: GMM cluster count on each side.
        reg: Entropic regularisation for the GW solver (applied to
            normalised distance matrices, so values around 0.01 are sharp).
        seed: RNG seed.

    Returns:
        A populated :class:`CrossModelGWDemo`.
    """
    pairs = load_sentiment_pairs()[:n_pairs]
    pos = [p for p, _ in pairs]
    neg = [n for _, n in pairs]
    n_per_class = len(pos)

    gw_cfg = GWConfig(reg=reg, num_iter_max=500, num_restart=2, warn_on_no_convergence=False)
    align_cfg = CrossModelGWConfig(
        n_components_source=n_components,
        n_components_target=n_components,
        gmm_cfg=GMMConfig(n_components=n_components, seed=seed),
        gw_cfg=gw_cfg,
    )

    # --- source side ---
    src_model, src_tok = load_model(ModelLoaderConfig(model_id=source_model_id))
    try:
        src_n_layers = src_model.config.num_hidden_layers
        src_layer = src_n_layers // 2
        src_adj_layer = min(src_layer + 1, src_n_layers)
        src_acts = extract_residual_stream(
            src_model, src_tok, pos + neg, layer_indices=[src_layer], batch_size=8
        )[src_layer]
        src_adj_acts = extract_residual_stream(
            src_model, src_tok, pos + neg, layer_indices=[src_adj_layer], batch_size=8
        )[src_adj_layer]
    finally:
        del src_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    src_arr = src_acts.cpu().numpy().astype(np.float64)
    src_adj_arr = src_adj_acts.cpu().numpy().astype(np.float64)
    rng = np.random.default_rng(seed + 99)
    src_noise = rng.normal(size=src_arr.shape).astype(np.float64)

    # --- target side ---
    tgt_model, tgt_tok = load_model(ModelLoaderConfig(model_id=target_model_id))
    try:
        tgt_n_layers = tgt_model.config.num_hidden_layers
        tgt_layer = tgt_n_layers // 2
        tgt_acts = extract_residual_stream(
            tgt_model, tgt_tok, pos + neg, layer_indices=[tgt_layer], batch_size=8
        )[tgt_layer]
    finally:
        del tgt_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    tgt_arr = tgt_acts.cpu().numpy().astype(np.float64)

    # --- four GW alignments ---
    self_align = cross_model_gw_coupling(
        src_arr, src_arr, cfg=align_cfg, rng=np.random.default_rng(seed + 1)
    )
    adj_align = cross_model_gw_coupling(
        src_arr, src_adj_arr, cfg=align_cfg, rng=np.random.default_rng(seed + 2)
    )
    rand_align = cross_model_gw_coupling(
        src_arr, src_noise, cfg=align_cfg, rng=np.random.default_rng(seed + 3)
    )
    cross_align = cross_model_gw_coupling(
        src_arr, tgt_arr, cfg=align_cfg, rng=np.random.default_rng(seed + 4)
    )

    sanity_costs = {
        "self_pair": float(self_align.gw_cost),
        "adjacent_layer": float(adj_align.gw_cost),
        "random_noise": float(rand_align.gw_cost),
        "cross_model": float(cross_align.gw_cost),
    }
    sanity_diag_mass = {
        "self_pair": _diagonal_mass(self_align.coupling),
        "adjacent_layer": _diagonal_mass(adj_align.coupling),
        "random_noise": _diagonal_mass(rand_align.coupling),
        "cross_model": _diagonal_mass(cross_align.coupling),
    }

    # --- class confusion via cross-model coupling ---
    # cross_model_gw_coupling fits and discards its GMMs internally; re-fit
    # here so we can derive per-cluster class labels.
    from ot_steering.steering.ot_steering import fit_gmm  # local import to avoid cycle warning

    src_gmm = fit_gmm(src_arr, GMMConfig(n_components=n_components, seed=seed))
    tgt_gmm = fit_gmm(tgt_arr, GMMConfig(n_components=n_components, seed=seed))
    src_cluster_labels = _extract_class_labels(n_per_class, n_components, src_acts, src_gmm)
    tgt_cluster_labels = _extract_class_labels(n_per_class, n_components, tgt_acts, tgt_gmm)
    # For each source cluster, find which target cluster it's mapped to most.
    src_to_tgt = cross_align.coupling.argmax(axis=1)
    confusion = np.zeros((2, 2), dtype=np.float64)
    for src_k, tgt_k in enumerate(src_to_tgt):
        src_lbl = src_cluster_labels[src_k]
        tgt_lbl = tgt_cluster_labels[int(tgt_k)]
        if src_lbl < 0 or tgt_lbl < 0:
            continue
        confusion[src_lbl, tgt_lbl] += float(cross_align.source_marginal[src_k])

    src_pca = PCA(n_components=2, random_state=seed).fit(src_arr.astype(np.float32))
    tgt_pca = PCA(n_components=2, random_state=seed).fit(tgt_arr.astype(np.float32))

    return CrossModelGWDemo(
        source_model_id=source_model_id,
        target_model_id=target_model_id,
        source_layer=src_layer,
        target_layer=tgt_layer,
        source_acts=src_acts,
        target_acts=tgt_acts,
        n_per_class=n_per_class,
        source_pca=src_pca,
        target_pca=tgt_pca,
        cross_model_alignment=cross_align,
        sanity_costs=sanity_costs,
        sanity_diag_mass=sanity_diag_mass,
        class_confusion=confusion,
    )
