"""Render every Chapter 3 figure from the sentiment-steering demo.

Run:
    python phases/phase_03_llms_and_steering_baselines/experiments/make_figures.py

Writes:
    phases/phase_03_llms_and_steering_baselines/figures/01_activation_distribution.png
    phases/phase_03_llms_and_steering_baselines/figures/02_steering_vector_in_pca.png
    phases/phase_03_llms_and_steering_baselines/figures/03_success_rate_vs_coefficient.png
    phases/phase_03_llms_and_steering_baselines/figures/04_offtarget_perplexity_vs_coefficient.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from demo import SentimentDemo, run_sentiment_demo  # noqa: E402

FIGURES_DIR = Path(__file__).resolve().parents[1] / "figures"


def _fig_activation_distribution(demo: SentimentDemo) -> plt.Figure:
    pos = demo.pca.transform(demo.positive_acts.cpu().numpy().astype(np.float32))
    neg = demo.pca.transform(demo.negative_acts.cpu().numpy().astype(np.float32))
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(
        pos[:, 0],
        pos[:, 1],
        s=42,
        c="#1f77b4",
        alpha=0.85,
        edgecolor="black",
        linewidth=0.4,
        label="positive prompts",
    )
    ax.scatter(
        neg[:, 0],
        neg[:, 1],
        s=42,
        c="#d62728",
        alpha=0.85,
        edgecolor="black",
        linewidth=0.4,
        label="negative prompts",
    )
    ax.set_title(
        f"Residual stream at layer {demo.hook_layer} of {demo.model_id}\n"
        "(2D PCA of last-token activations on 30 sentiment pairs)"
    )
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def _fig_steering_vector_in_pca(demo: SentimentDemo) -> plt.Figure:
    pos = demo.pca.transform(demo.positive_acts.cpu().numpy().astype(np.float32))
    neg = demo.pca.transform(demo.negative_acts.cpu().numpy().astype(np.float32))
    # The difference-of-means direction projected into the PCA plane.
    direction_pca = demo.pca.transform(demo.direction.cpu().numpy().reshape(1, -1))[0]
    pos_mean = pos.mean(axis=0)
    neg_mean = neg.mean(axis=0)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(pos[:, 0], pos[:, 1], s=28, c="#1f77b4", alpha=0.45)
    ax.scatter(neg[:, 0], neg[:, 1], s=28, c="#d62728", alpha=0.45)
    ax.scatter(
        *pos_mean,
        s=160,
        c="#1f77b4",
        marker="X",
        edgecolor="black",
        linewidth=0.8,
        label="mean(positive)",
        zorder=4,
    )
    ax.scatter(
        *neg_mean,
        s=160,
        c="#d62728",
        marker="X",
        edgecolor="black",
        linewidth=0.8,
        label="mean(negative)",
        zorder=4,
    )
    ax.annotate(
        "",
        xy=(neg_mean[0] + direction_pca[0], neg_mean[1] + direction_pca[1]),
        xytext=neg_mean,
        arrowprops=dict(arrowstyle="->", color="black", lw=1.8),
    )
    ax.set_title("Steering direction = mean(pos) − mean(neg), drawn in PCA space")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def _fig_success_curve(demo: SentimentDemo) -> plt.Figure:
    coefs = list(demo.coefficients)
    rates = [demo.success_rate_by_coef[c] for c in coefs]
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.plot(coefs, rates, marker="o", color="#2ca02c", linewidth=2.0)
    ax.axhline(0.0, color="grey", linestyle="--", alpha=0.4)
    ax.set_xlabel("steering coefficient")
    ax.set_ylabel("positive-shift rate (vs. unsteered baseline)")
    ax.set_title("Steering lift on held-out negative prompts")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def _fig_offtarget_curve(demo: SentimentDemo) -> plt.Figure:
    coefs = list(demo.coefficients)
    ppls = [demo.off_target_ppl_by_coef[c] for c in coefs]
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.plot(coefs, ppls, marker="o", color="#9467bd", linewidth=2.0)
    ax.set_xlabel("steering coefficient")
    ax.set_ylabel("perplexity on neutral text")
    ax.set_title("Off-target damage — perplexity vs. coefficient")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def main() -> int:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    demo = run_sentiment_demo()
    artifacts: list[tuple[str, plt.Figure]] = [
        ("01_activation_distribution.png", _fig_activation_distribution(demo)),
        ("02_steering_vector_in_pca.png", _fig_steering_vector_in_pca(demo)),
        ("03_success_rate_vs_coefficient.png", _fig_success_curve(demo)),
        ("04_offtarget_perplexity_vs_coefficient.png", _fig_offtarget_curve(demo)),
    ]
    for filename, fig in artifacts:
        out = FIGURES_DIR / filename
        fig.savefig(out, dpi=160, bbox_inches="tight")
        print(f"wrote {out}")
        plt.close(fig)
    # Brief summary.
    print("\n--- demo summary ---")
    print(f"model            : {demo.model_id}")
    print(f"hook_layer       : {demo.hook_layer}")
    for coef in demo.coefficients:
        sr = demo.success_rate_by_coef[coef]
        ppl = demo.off_target_ppl_by_coef[coef]
        print(f"  coef={coef:+6.2f}  shift_rate={sr:.2%}  off_target_ppl={ppl:.2f}")
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
