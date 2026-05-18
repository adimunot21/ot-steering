"""Render every Chapter 5 figure from the cross-model GW demo.

Run:
    python phases/phase_05_cross_model_gw/experiments/make_figures.py

Writes:
    phases/phase_05_cross_model_gw/figures/01_two_residual_streams_pca.png
    phases/phase_05_cross_model_gw/figures/02_gw_coupling_heatmap.png
    phases/phase_05_cross_model_gw/figures/03_sanity_check_costs.png
    phases/phase_05_cross_model_gw/figures/04_class_preservation.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from demo import CrossModelGWDemo, run_cross_model_demo  # noqa: E402

FIGURES_DIR = Path(__file__).resolve().parents[1] / "figures"


def _fig_two_streams_pca(demo: CrossModelGWDemo) -> plt.Figure:
    src = demo.source_pca.transform(demo.source_acts.cpu().numpy().astype(np.float32))
    tgt = demo.target_pca.transform(demo.target_acts.cpu().numpy().astype(np.float32))

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    for ax, pts, label, model in [
        (axes[0], src, f"layer {demo.source_layer}", demo.source_model_id),
        (axes[1], tgt, f"layer {demo.target_layer}", demo.target_model_id),
    ]:
        n_pos = demo.n_per_class
        ax.scatter(
            pts[:n_pos, 0],
            pts[:n_pos, 1],
            s=32,
            c="#1f77b4",
            alpha=0.75,
            edgecolor="black",
            linewidth=0.3,
            label="positive",
        )
        ax.scatter(
            pts[n_pos:, 0],
            pts[n_pos:, 1],
            s=32,
            c="#d62728",
            alpha=0.75,
            edgecolor="black",
            linewidth=0.3,
            label="negative",
        )
        ax.set_title(f"{model}  ({label})")
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.legend(loc="best")
        ax.grid(alpha=0.25)
    fig.suptitle(
        "Two residual streams, two coordinate frames — but similar shape\n"
        "(GW aligns by intra-distance structure, not coordinates)"
    )
    fig.tight_layout()
    return fig


def _fig_coupling_heatmap(demo: CrossModelGWDemo) -> plt.Figure:
    coupling = demo.cross_model_alignment.coupling
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(coupling, cmap="magma", aspect="auto")
    ax.set_xlabel(f"target cluster ({demo.target_model_id})")
    ax.set_ylabel(f"source cluster ({demo.source_model_id})")
    ax.set_title(
        f"GW coupling — k={coupling.shape[0]} clusters per side\n"
        f"gw_cost = {demo.cross_model_alignment.gw_cost:.4f}"
    )
    fig.colorbar(im, ax=ax, label="transported mass")
    fig.tight_layout()
    return fig


def _fig_sanity_check_costs(demo: CrossModelGWDemo) -> plt.Figure:
    order = ["self_pair", "adjacent_layer", "cross_model", "random_noise"]
    labels = ["self-pair", "adjacent\nlayer", "cross-model\n(Pythia → GPT-2)", "random\nnoise"]
    costs = [demo.sanity_costs[k] for k in order]
    colors = ["#2ca02c", "#1f77b4", "#9467bd", "#d62728"]

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    bars = ax.bar(labels, costs, color=colors, edgecolor="black", linewidth=0.6)
    for bar, cost in zip(bars, costs, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{cost:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylabel("entropic GW cost")
    ax.set_title("Four sanity checks — GW cost on the source side")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


def _fig_class_preservation(demo: CrossModelGWDemo) -> plt.Figure:
    conf = demo.class_confusion
    row_sums = conf.sum(axis=1, keepdims=True).clip(min=1e-12)
    conf_norm = conf / row_sums

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    im = ax.imshow(conf_norm, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["positive (target)", "negative (target)"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["positive (source)", "negative (source)"])
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{conf_norm[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if conf_norm[i, j] > 0.5 else "black",
                fontsize=14,
            )
    diag = conf_norm[0, 0] + conf_norm[1, 1]
    ax.set_title(
        f"Class preservation through GW coupling\n(row-normalised; diagonal mass = {diag / 2:.2%})"
    )
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


def main() -> int:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    demo = run_cross_model_demo()
    artifacts: list[tuple[str, plt.Figure]] = [
        ("01_two_residual_streams_pca.png", _fig_two_streams_pca(demo)),
        ("02_gw_coupling_heatmap.png", _fig_coupling_heatmap(demo)),
        ("03_sanity_check_costs.png", _fig_sanity_check_costs(demo)),
        ("04_class_preservation.png", _fig_class_preservation(demo)),
    ]
    for fname, fig in artifacts:
        out = FIGURES_DIR / fname
        fig.savefig(out, dpi=160, bbox_inches="tight")
        print(f"wrote {out}")
        plt.close(fig)
    print("\n--- demo summary ---")
    print(f"source: {demo.source_model_id}  layer {demo.source_layer}")
    print(f"target: {demo.target_model_id}   layer {demo.target_layer}")
    print("sanity gw_cost:")
    for k, v in demo.sanity_costs.items():
        print(f"  {k:18s} = {v:.4f}")
    print("class confusion (row-normalised, diag/2 = preservation rate):")
    row_sums = demo.class_confusion.sum(axis=1, keepdims=True).clip(min=1e-12)
    conf_norm = demo.class_confusion / row_sums
    diag = conf_norm[0, 0] + conf_norm[1, 1]
    print(f"  preservation rate = {diag / 2:.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
