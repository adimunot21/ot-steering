"""Render every Chapter 6 figure from the transport demo + the full
``run_transport.py`` JSON output.

Run:
    # first regenerate the headline numbers
    python phases/phase_06_steering_transport/experiments/run_transport.py
    # then render figures (the demo cell + the run_transport.json grid)
    python phases/phase_06_steering_transport/experiments/make_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from demo import TransportDemo, run_transport_demo  # noqa: E402

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
FIGURES_DIR = Path(__file__).resolve().parents[1] / "figures"
OUTPUTS_DIR = _PROJECT_ROOT / "outputs"

METHOD_COLORS: dict[str, str] = {
    "random": "#d62728",
    "procrustes": "#1f77b4",
    "gw_transport": "#2ca02c",
    "target_oracle": "#7f7f7f",
}
METHOD_LABELS: dict[str, str] = {
    "random": "random",
    "procrustes": "Procrustes",
    "gw_transport": "GW transport",
    "target_oracle": "target oracle",
}


def _fig_pipeline_diagram() -> plt.Figure:
    """A schematic of the source CHaRS → cross-model GW → target hook pipeline."""
    fig, ax = plt.subplots(figsize=(11.0, 4.6))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.6)
    ax.set_aspect("auto")
    ax.axis("off")

    def _box(x, y, w, h, text, fc, ec="black"):  # type: ignore[no-untyped-def]
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.08", fc=fc, ec=ec, lw=1.0
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)

    def _arrow(x0, y0, x1, y1, label=""):  # type: ignore[no-untyped-def]
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(arrowstyle="->", lw=1.4, color="#444"),
        )
        if label:
            ax.text(
                (x0 + x1) / 2,
                (y0 + y1) / 2 + 0.16,
                label,
                ha="center",
                va="bottom",
                fontsize=8.5,
                color="#444",
            )

    # Source-side column.
    _box(0.4, 3.4, 2.4, 0.7, "POS / NEG\nactivations on A", "#cfe2f3")
    _box(0.4, 2.0, 2.4, 0.7, "CHaRS map\n(Phase 4)", "#9fc5e8")
    _box(0.4, 0.6, 2.4, 0.7, "per-A-NEG-cluster\ndisplacement (in ℝ^{d_A})", "#9fc5e8")
    _arrow(1.6, 3.4, 1.6, 2.7, "GMM + EMD")
    _arrow(1.6, 2.0, 1.6, 1.3, "barycentric")

    # Cross-model GW boxes (middle).
    _box(4.0, 3.4, 2.6, 0.7, "GW (NEG_A ↔ NEG_B)\nP_neg ∈ ℝ^{k×k}", "#fff2cc")
    _box(4.0, 2.0, 2.6, 0.7, "GW (POS_A ↔ POS_B)\nP_pos ∈ ℝ^{k×k}", "#fff2cc")
    _arrow(2.8, 3.75, 4.0, 3.75, "Phase 5")
    _arrow(2.8, 2.35, 4.0, 2.35, "Phase 5")

    # Barycentric chain.
    _box(4.0, 0.6, 2.6, 0.7, "B-space target of each\nA-NEG cluster", "#d9ead3")
    _arrow(2.8, 0.95, 4.0, 0.95, "P_pos barycentric\nthen P_neg^T barycentric")

    # Target side.
    _box(7.5, 3.4, 3.1, 0.7, "POS / NEG\nactivations on B", "#fde2cf")
    _box(7.5, 2.0, 3.1, 0.7, "B NEG GMM\n(soft assignment at runtime)", "#fce5cd")
    _box(7.5, 0.6, 3.1, 0.7, "per-B-NEG-cluster\ntransported displacement", "#a4c2f4")
    _arrow(7.5, 3.75, 6.6, 3.75)
    _arrow(7.5, 2.35, 6.6, 2.35)
    _arrow(6.6, 0.95, 7.5, 0.95)

    ax.text(
        5.5, 4.25, "Cross-model steering transport pipeline", ha="center", va="center", fontsize=13
    )
    return fig


def _fig_method_comparison_bar(matrix_results: list[dict]) -> plt.Figure:
    """One subplot per cell; bars per method at the best coefficient."""
    fig, axes = plt.subplots(1, len(matrix_results), figsize=(5.6 * len(matrix_results), 4.4))
    if len(matrix_results) == 1:
        axes = [axes]
    for ax, res in zip(axes, matrix_results, strict=True):
        means: list[float] = []
        lo: list[float] = []
        hi: list[float] = []
        labels: list[str] = []
        colors: list[str] = []
        for method in ("random", "procrustes", "gw_transport", "target_oracle"):
            best_shift = -1.0
            best_payload = None
            for _coef_str, payload in res["summary"][method].items():
                m, _l, _h = payload["shift_mean_lo_hi"]
                if m > best_shift:
                    best_shift = m
                    best_payload = payload
            assert best_payload is not None
            m, low_ci, high_ci = best_payload["shift_mean_lo_hi"]
            means.append(m)
            lo.append(m - low_ci)
            hi.append(high_ci - m)
            labels.append(METHOD_LABELS[method])
            colors.append(METHOD_COLORS[method])
        x = np.arange(len(labels))
        ax.bar(
            x,
            means,
            yerr=[lo, hi],
            color=colors,
            edgecolor="black",
            linewidth=0.6,
            capsize=4,
        )
        for xi, mi in zip(x, means, strict=True):
            ax.text(xi, mi + 0.01, f"{mi:.0%}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0, max(0.7, max(means) + 0.1))
        ax.set_ylabel("positive-shift rate (best coef)")
        ax.set_title(f"{res['source_model']} → {res['target_model']}")
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Cross-model steering transport — method comparison")
    fig.tight_layout()
    return fig


def _fig_lift_curves_with_ci(matrix_results: list[dict]) -> plt.Figure:
    fig, axes = plt.subplots(1, len(matrix_results), figsize=(5.8 * len(matrix_results), 4.4))
    if len(matrix_results) == 1:
        axes = [axes]
    for ax, res in zip(axes, matrix_results, strict=True):
        coefs = sorted(float(c) for c in next(iter(res["summary"].values())))
        for method in ("random", "procrustes", "gw_transport", "target_oracle"):
            means: list[float] = []
            lows: list[float] = []
            highs: list[float] = []
            for c in coefs:
                m, lo, hi = res["summary"][method][str(c)]["shift_mean_lo_hi"]
                means.append(m)
                lows.append(lo)
                highs.append(hi)
            color = METHOD_COLORS[method]
            ax.plot(coefs, means, marker="o", color=color, label=METHOD_LABELS[method], lw=2.0)
            ax.fill_between(coefs, lows, highs, color=color, alpha=0.15)
        ax.set_xlabel("steering coefficient")
        ax.set_ylabel("positive-shift rate")
        ax.set_title(f"{res['source_model']} → {res['target_model']}")
        ax.grid(alpha=0.3)
        ax.legend(loc="best")
    fig.suptitle("Lift vs coefficient (mean ± 95% bootstrap CI across 3 seeds)")
    fig.tight_layout()
    return fig


def _fig_offtarget_curves(matrix_results: list[dict]) -> plt.Figure:
    fig, axes = plt.subplots(1, len(matrix_results), figsize=(5.8 * len(matrix_results), 4.4))
    if len(matrix_results) == 1:
        axes = [axes]
    for ax, res in zip(axes, matrix_results, strict=True):
        coefs = sorted(float(c) for c in next(iter(res["summary"].values())))
        baseline_ppl = float(res["baseline_off_target_ppl"])
        for method in ("random", "procrustes", "gw_transport", "target_oracle"):
            means: list[float] = []
            lows: list[float] = []
            highs: list[float] = []
            for c in coefs:
                m, lo, hi = res["summary"][method][str(c)]["ppl_mean_lo_hi"]
                means.append(m)
                lows.append(lo)
                highs.append(hi)
            color = METHOD_COLORS[method]
            ax.plot(coefs, means, marker="o", color=color, label=METHOD_LABELS[method], lw=2.0)
            ax.fill_between(coefs, lows, highs, color=color, alpha=0.15)
        ax.axhline(
            baseline_ppl,
            color="grey",
            linestyle="--",
            alpha=0.5,
            label=f"unsteered ppl = {baseline_ppl:.1f}",
        )
        ax.set_xlabel("steering coefficient")
        ax.set_ylabel("perplexity on neutral text")
        ax.set_title(f"{res['source_model']} → {res['target_model']}")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        ax.set_yscale("log")
    fig.suptitle("Off-target perplexity vs coefficient (log-scale)")
    fig.tight_layout()
    return fig


def _latest_run_transport_json() -> dict | None:
    """Pick the most-recently-written ``run_transport.json`` from outputs/."""
    candidates = sorted(OUTPUTS_DIR.glob("*/run_transport.json"))
    if not candidates:
        return None
    return json.loads(candidates[-1].read_text(encoding="utf-8"))


def _fig_demo_lift(demo: TransportDemo) -> plt.Figure:
    """Demo-cell version of the lift curve (single seed, no CI)."""
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    coefs = list(demo.coefficients)
    for method, label in METHOD_LABELS.items():
        rates = [demo.shift_rates_by_method[method][c] for c in coefs]
        ax.plot(coefs, rates, marker="o", color=METHOD_COLORS[method], label=label, lw=2.0)
    ax.set_xlabel("steering coefficient")
    ax.set_ylabel("positive-shift rate")
    ax.set_title(
        f"Demo cell: {demo.source_model_id} → {demo.target_model_id}\n"
        "(single seed; full matrix in `run_transport.json`)"
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def main() -> int:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    matrix = _latest_run_transport_json()

    figs: list[tuple[str, plt.Figure]] = [
        ("01_transport_pipeline_diagram.png", _fig_pipeline_diagram()),
    ]
    if matrix is not None:
        figs += [
            ("02_method_comparison_bar.png", _fig_method_comparison_bar(matrix)),
            ("03_lift_vs_coefficient_with_ci.png", _fig_lift_curves_with_ci(matrix)),
            ("04_offtarget_ppl_vs_coefficient.png", _fig_offtarget_curves(matrix)),
        ]
    else:
        print(
            "  (warning) no run_transport.json found in outputs/; the headline "
            "matrix figures will be missing — run run_transport.py first."
        )
        demo = run_transport_demo()
        figs += [("02_demo_lift_curve.png", _fig_demo_lift(demo))]

    for fname, fig in figs:
        out = FIGURES_DIR / fname
        fig.savefig(out, dpi=160, bbox_inches="tight")
        print(f"wrote {out}")
        plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
