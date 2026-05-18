"""Render every Chapter 1 figure from the 2D Gaussian transport demo.

Run:
    python phases/phase_01_ot_foundations/experiments/make_figures.py

Writes:
    phases/phase_01_ot_foundations/figures/01_point_clouds.png
    phases/phase_01_ot_foundations/figures/02_ot_plan_lines.png
    phases/phase_01_ot_foundations/figures/03_plan_heatmaps.png
    phases/phase_01_ot_foundations/figures/04_displacement_interpolation.png

Each figure is regenerated from scratch every time the script runs (so they
can be reproduced by a reviewer with one command). Plots are deterministic
given the demo seed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Phase folders are not on the project's import path (only src/ot_steering is).
# Add this experiments/ dir so the sibling ``demo`` module loads.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from demo import GaussianTransportDemo, run_demo  # noqa: E402

FIGURES_DIR = Path(__file__).resolve().parents[1] / "figures"


def _fig_point_clouds(demo: GaussianTransportDemo) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.scatter(demo.source[:, 0], demo.source[:, 1], s=22, c="#1f77b4", label="source", alpha=0.85)
    ax.scatter(demo.target[:, 0], demo.target[:, 1], s=22, c="#d62728", label="target", alpha=0.85)
    ax.set_aspect("equal")
    ax.set_title("Two 2D point clouds — equal mass, different locations")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def _fig_ot_plan_lines(demo: GaussianTransportDemo, weight_threshold: float = 1e-3) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    n, m = demo.plan_emd.shape
    pmax = demo.plan_emd.max() if demo.plan_emd.max() > 0 else 1.0
    for i in range(n):
        for j in range(m):
            w = demo.plan_emd[i, j]
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
    ax.set_title("Exact OT plan — line width ∝ transported mass")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def _fig_plan_heatmaps(demo: GaussianTransportDemo) -> plt.Figure:
    reg_items = sorted(demo.plan_sinkhorn_by_reg.items(), key=lambda kv: -kv[0])
    plans: list[tuple[str, np.ndarray]] = [("EMD (no entropy)", demo.plan_emd)]
    plans += [(f"Sinkhorn, reg={reg:g}", plan) for reg, plan in reg_items]

    cols = len(plans)
    fig, axes = plt.subplots(1, cols, figsize=(3.4 * cols, 3.4))
    if cols == 1:
        axes = [axes]
    vmax = max(p.max() for _, p in plans)
    for ax, (label, plan) in zip(axes, plans, strict=True):
        im = ax.imshow(plan, cmap="magma", vmin=0.0, vmax=vmax, aspect="auto")
        ax.set_title(label)
        ax.set_xlabel("target index j")
        ax.set_ylabel("source index i")
    fig.colorbar(im, ax=axes, shrink=0.85, label="transported mass P[i, j]")
    fig.suptitle("Entropic regularisation smooths the OT plan")
    return fig


def _fig_displacement_interpolation(demo: GaussianTransportDemo) -> plt.Figure:
    cols = len(demo.interpolation_ts)
    fig, axes = plt.subplots(1, cols, figsize=(2.7 * cols, 2.9), sharex=True, sharey=True)
    if cols == 1:
        axes = [axes]
    xlim = (
        min(demo.source[:, 0].min(), demo.target[:, 0].min()) - 0.5,
        max(demo.source[:, 0].max(), demo.target[:, 0].max()) + 0.5,
    )
    ylim = (
        min(demo.source[:, 1].min(), demo.target[:, 1].min()) - 0.5,
        max(demo.source[:, 1].max(), demo.target[:, 1].max()) + 0.5,
    )
    for ax, t, frame in zip(axes, demo.interpolation_ts, demo.interpolation_frames, strict=True):
        ax.scatter(demo.source[:, 0], demo.source[:, 1], s=10, c="#1f77b4", alpha=0.25)
        ax.scatter(demo.target[:, 0], demo.target[:, 1], s=10, c="#d62728", alpha=0.25)
        ax.scatter(frame[:, 0], frame[:, 1], s=22, c="#2ca02c", alpha=0.95)
        ax.set_title(f"t = {t:.2f}")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal")
        ax.grid(alpha=0.25)
    fig.suptitle("Displacement interpolation along the OT geodesic")
    fig.tight_layout()
    return fig


def main() -> int:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    demo = run_demo()
    artifacts: list[tuple[str, plt.Figure]] = [
        ("01_point_clouds.png", _fig_point_clouds(demo)),
        ("02_ot_plan_lines.png", _fig_ot_plan_lines(demo)),
        ("03_plan_heatmaps.png", _fig_plan_heatmaps(demo)),
        ("04_displacement_interpolation.png", _fig_displacement_interpolation(demo)),
    ]
    for filename, fig in artifacts:
        out = FIGURES_DIR / filename
        fig.savefig(out, dpi=160, bbox_inches="tight")
        print(f"wrote {out}")
        plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
