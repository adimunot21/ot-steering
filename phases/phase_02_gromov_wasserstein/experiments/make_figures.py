"""Render every Chapter 2 figure from the rotation-recovery demo.

Run:
    python phases/phase_02_gromov_wasserstein/experiments/make_figures.py

Writes:
    phases/phase_02_gromov_wasserstein/figures/01_two_clouds_no_alignment.png
    phases/phase_02_gromov_wasserstein/figures/02_gw_coupling_lines.png
    phases/phase_02_gromov_wasserstein/figures/03_coupling_heatmap.png
    phases/phase_02_gromov_wasserstein/figures/04_recovered_correspondence_vs_truth.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Phase folders are not on the import path. Add ./experiments/ so the sibling
# ``demo`` module loads.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from demo import RotationDemo, run_rotation_demo  # noqa: E402

FIGURES_DIR = Path(__file__).resolve().parents[1] / "figures"


def _fig_two_clouds(demo: RotationDemo) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(demo.source[:, 0], demo.source[:, 1], s=22, c="#1f77b4", label="source", alpha=0.85)
    ax.scatter(demo.target[:, 0], demo.target[:, 1], s=22, c="#d62728", label="target", alpha=0.85)
    ax.set_aspect("equal")
    ax.set_title("Two 2D clouds related by an unknown rotation")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def _fig_coupling_lines(demo: RotationDemo, weight_threshold: float = 1e-3) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.0, 4.8))
    n, m = demo.coupling.shape
    pmax = demo.coupling.max() if demo.coupling.max() > 0 else 1.0
    for i in range(n):
        for j in range(m):
            w = demo.coupling[i, j]
            if w < weight_threshold:
                continue
            ax.plot(
                [demo.source[i, 0], demo.target[j, 0]],
                [demo.source[i, 1], demo.target[j, 1]],
                color="#888888",
                linewidth=0.4 + 3.5 * (w / pmax),
                alpha=0.55,
            )
    ax.scatter(demo.source[:, 0], demo.source[:, 1], s=22, c="#1f77b4", label="source", zorder=3)
    ax.scatter(demo.target[:, 0], demo.target[:, 1], s=22, c="#d62728", label="target", zorder=3)
    ax.set_aspect("equal")
    ax.set_title(f"GW coupling — recovered the correspondence at {demo.accuracy:.0%} accuracy")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def _fig_coupling_heatmap(demo: RotationDemo) -> plt.Figure:
    # Row-sort the coupling by the recovered argmax target so the diagonal
    # of the matched permutation lights up.
    order = np.argsort(demo.recovered_permutation)
    sorted_coupling = demo.coupling[order, :]

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.0))
    im0 = axes[0].imshow(demo.coupling, cmap="magma", aspect="auto")
    axes[0].set_title("Coupling P[i, j] — original ordering")
    axes[0].set_xlabel("target index j")
    axes[0].set_ylabel("source index i")
    fig.colorbar(im0, ax=axes[0], shrink=0.85, label="mass")

    im1 = axes[1].imshow(sorted_coupling, cmap="magma", aspect="auto")
    axes[1].set_title("Coupling — rows sorted by argmax(j); diagonal = match")
    axes[1].set_xlabel("target index j")
    axes[1].set_ylabel("source index i (re-ordered)")
    fig.colorbar(im1, ax=axes[1], shrink=0.85, label="mass")
    fig.suptitle("GW finds a permutation — visible as a near-diagonal after sorting")
    fig.tight_layout()
    return fig


def _fig_recovered_vs_truth(demo: RotationDemo) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    ax.scatter(
        demo.truth_permutation,
        demo.recovered_permutation,
        s=42,
        c="#2ca02c",
        alpha=0.85,
        edgecolor="black",
        linewidth=0.4,
    )
    n = len(demo.truth_permutation)
    ax.plot([0, n - 1], [0, n - 1], "k--", linewidth=0.8, alpha=0.5, label="perfect recovery")
    ax.set_xlabel("planted partner index j*")
    ax.set_ylabel("argmax-recovered partner index argmax_j P[i, j]")
    ax.set_title(f"Recovered vs. planted correspondence ({demo.accuracy:.0%})")
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def main() -> int:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    demo = run_rotation_demo()
    artifacts: list[tuple[str, plt.Figure]] = [
        ("01_two_clouds_no_alignment.png", _fig_two_clouds(demo)),
        ("02_gw_coupling_lines.png", _fig_coupling_lines(demo)),
        ("03_coupling_heatmap.png", _fig_coupling_heatmap(demo)),
        ("04_recovered_correspondence_vs_truth.png", _fig_recovered_vs_truth(demo)),
    ]
    for filename, fig in artifacts:
        out = FIGURES_DIR / filename
        fig.savefig(out, dpi=160, bbox_inches="tight")
        print(f"wrote {out}")
        plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
