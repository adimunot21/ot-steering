"""Render every Chapter 4 figure from the CHaRS demo.

Run:
    python phases/phase_04_intra_model_ot_steering/experiments/make_figures.py

Writes:
    phases/phase_04_intra_model_ot_steering/figures/01_gmm_fits.png
    phases/phase_04_intra_model_ot_steering/figures/02_cluster_couplings.png
    phases/phase_04_intra_model_ot_steering/figures/03_steering_lift_vs_k.png
    phases/phase_04_intra_model_ot_steering/figures/04_offtarget_vs_k.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

sys.path.insert(0, str(Path(__file__).resolve().parent))

from demo import CHaRSDemo, run_charsy_demo  # noqa: E402

FIGURES_DIR = Path(__file__).resolve().parents[1] / "figures"


def _draw_gmm_ellipses(ax, gmm, pca, color: str, alpha: float = 0.25) -> None:
    """Project each GMM component's covariance into PCA space and draw a 1-sigma ellipse."""
    # diag/full both come back as full when we ask for covariances_ via
    # _get_covariances; here we read .covariances_ directly.
    centres_pca = pca.transform(gmm.means_)
    for k in range(gmm.n_components):
        cov = gmm.covariances_[k]
        cov_full = np.diag(cov) if cov.ndim == 1 else cov
        # Project covariance into PCA components: P Cov P^T
        proj = pca.components_ @ cov_full @ pca.components_.T
        vals, vecs = np.linalg.eigh(proj)
        order = np.argsort(vals)[::-1]
        vals, vecs = vals[order], vecs[:, order]
        angle = float(np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0])))
        # 1-sigma ellipse (width=2*sqrt(eigval)).
        width, height = 2.0 * np.sqrt(np.clip(vals, 0.0, None))
        e = Ellipse(
            xy=centres_pca[k],
            width=width,
            height=height,
            angle=angle,
            edgecolor=color,
            fc=color,
            alpha=alpha,
            lw=1.2,
        )
        ax.add_patch(e)
        ax.scatter(*centres_pca[k], s=120, marker="X", c=color, edgecolor="black", linewidth=0.6)


def _fig_gmm_fits(demo: CHaRSDemo) -> plt.Figure:
    pos = demo.pca.transform(demo.positive_acts.cpu().numpy().astype(np.float32))
    neg = demo.pca.transform(demo.negative_acts.cpu().numpy().astype(np.float32))

    k_for_fig = max(demo.ks)
    sm = demo.steering_maps_by_k[k_for_fig]

    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    ax.scatter(pos[:, 0], pos[:, 1], s=28, c="#1f77b4", alpha=0.5, label="positive samples")
    ax.scatter(neg[:, 0], neg[:, 1], s=28, c="#d62728", alpha=0.5, label="negative samples")
    _draw_gmm_ellipses(ax, sm.target_gmm, demo.pca, color="#1f77b4")
    _draw_gmm_ellipses(ax, sm.source_gmm, demo.pca, color="#d62728")
    ax.set_title(
        f"GMM fits at k={k_for_fig} on layer-{demo.hook_layer} activations\n"
        f"({demo.model_id} — 1σ ellipses in PCA space)"
    )
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def _fig_cluster_couplings(demo: CHaRSDemo) -> plt.Figure:
    pos = demo.pca.transform(demo.positive_acts.cpu().numpy().astype(np.float32))
    neg = demo.pca.transform(demo.negative_acts.cpu().numpy().astype(np.float32))

    k_for_fig = max(demo.ks)
    sm = demo.steering_maps_by_k[k_for_fig]
    src_pca = demo.pca.transform(sm.source_centers)
    tgt_pca = demo.pca.transform(sm.target_centers)
    pmax = float(sm.coupling.max()) or 1.0

    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    ax.scatter(pos[:, 0], pos[:, 1], s=18, c="#1f77b4", alpha=0.35)
    ax.scatter(neg[:, 0], neg[:, 1], s=18, c="#d62728", alpha=0.35)
    for i in range(sm.coupling.shape[0]):
        for j in range(sm.coupling.shape[1]):
            w = sm.coupling[i, j]
            if w < 1e-6:
                continue
            ax.plot(
                [src_pca[i, 0], tgt_pca[j, 0]],
                [src_pca[i, 1], tgt_pca[j, 1]],
                color="#444444",
                linewidth=0.6 + 4.0 * (w / pmax),
                alpha=0.75,
            )
    ax.scatter(
        src_pca[:, 0],
        src_pca[:, 1],
        s=160,
        marker="X",
        c="#d62728",
        edgecolor="black",
        linewidth=0.6,
        label="negative centroids",
        zorder=4,
    )
    ax.scatter(
        tgt_pca[:, 0],
        tgt_pca[:, 1],
        s=160,
        marker="X",
        c="#1f77b4",
        edgecolor="black",
        linewidth=0.6,
        label="positive centroids",
        zorder=4,
    )
    ax.set_title(
        f"OT coupling between cluster centroids (k={k_for_fig})\nline width ∝ transported mass"
    )
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def _fig_steering_lift_vs_k(demo: CHaRSDemo) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for k in demo.ks:
        coefs = list(demo.coefficients)
        rates = [demo.shift_rate_by_k_coef[k][c] for c in coefs]
        ax.plot(coefs, rates, marker="o", linewidth=2.0, label=f"k={k}")
    ax.axhline(0.0, color="grey", linestyle="--", alpha=0.4)
    ax.set_xlabel("steering coefficient")
    ax.set_ylabel("positive-shift rate")
    ax.set_title(
        "Lift on held-out negative-class prompts\n(k=1 is the difference-of-means baseline)"
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def _fig_offtarget_vs_k(demo: CHaRSDemo) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for k in demo.ks:
        coefs = list(demo.coefficients)
        ppls = [demo.off_target_ppl_by_k_coef[k][c] for c in coefs]
        ax.plot(coefs, ppls, marker="o", linewidth=2.0, label=f"k={k}")
    ax.axhline(
        demo.baseline_off_target_ppl,
        color="grey",
        linestyle="--",
        alpha=0.6,
        label=f"unsteered ppl = {demo.baseline_off_target_ppl:.1f}",
    )
    ax.set_xlabel("steering coefficient")
    ax.set_ylabel("perplexity on neutral text")
    ax.set_title("Off-target damage — perplexity vs. coefficient, by k")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def main() -> int:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    demo = run_charsy_demo()
    artifacts: list[tuple[str, plt.Figure]] = [
        ("01_gmm_fits.png", _fig_gmm_fits(demo)),
        ("02_cluster_couplings.png", _fig_cluster_couplings(demo)),
        ("03_steering_lift_vs_k.png", _fig_steering_lift_vs_k(demo)),
        ("04_offtarget_vs_k.png", _fig_offtarget_vs_k(demo)),
    ]
    for fname, fig in artifacts:
        out = FIGURES_DIR / fname
        fig.savefig(out, dpi=160, bbox_inches="tight")
        print(f"wrote {out}")
        plt.close(fig)
    print("\n--- demo summary ---")
    print(f"model      : {demo.model_id}")
    print(f"hook_layer : {demo.hook_layer}")
    print(f"baseline off-target ppl: {demo.baseline_off_target_ppl:.2f}")
    for k in demo.ks:
        for coef in demo.coefficients:
            sr = demo.shift_rate_by_k_coef[k][coef]
            ppl = demo.off_target_ppl_by_k_coef[k][coef]
            print(f"  k={k} coef={coef:.1f}  shift={sr:.0%}  ppl={ppl:7.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
