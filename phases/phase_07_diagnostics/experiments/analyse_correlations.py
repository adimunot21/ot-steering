"""Phase 7 correlation analysis + figure rendering.

Reads the latest ``sweep.json`` from ``outputs/`` (written by ``sweep.py``)
and produces:

  - ``correlations.json`` next to the sweep — Spearman ρ with bootstrap
    95 % CIs across all cells, plus per-pair breakdowns.
  - Four chapter figures in
    ``phases/phase_07_diagnostics/figures/``:
      01_lift_vs_gw_cost_scatter.png  (with ρ in title)
      02_layer_effect.png             (early/mid/late by model pair)
      03_k_effect.png                 (k=2/4/8 grouped bars)
      04_diagnostic_summary.png       (single-pane recap)

Run:
    # generate the sweep first
    python phases/phase_07_diagnostics/experiments/sweep.py
    # then analyse
    python phases/phase_07_diagnostics/experiments/analyse_correlations.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ot_steering.eval.correlation import spearman_with_bootstrap_ci

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUTS_DIR = _PROJECT_ROOT / "outputs"
FIGURES_DIR = Path(__file__).resolve().parents[1] / "figures"

PAIR_COLORS: dict[tuple[str, str], str] = {
    ("gpt2", "EleutherAI/pythia-160m"): "#2ca02c",
    ("EleutherAI/pythia-160m", "gpt2"): "#1f77b4",
}


def _latest_sweep() -> tuple[Path, list[dict]]:
    candidates = sorted(OUTPUTS_DIR.glob("*/sweep.json"))
    if not candidates:
        raise FileNotFoundError("no sweep.json found in outputs/; run sweep.py first")
    path = candidates[-1]
    return path, json.loads(path.read_text(encoding="utf-8"))


def _short_pair(src: str, tgt: str) -> str:
    return f"{src.split('/')[-1]} → {tgt.split('/')[-1]}"


def _fig_scatter(rows: list[dict], rho_all: tuple[float, float, float]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    for (src, tgt), color in PAIR_COLORS.items():
        cell_rows = [r for r in rows if r["source_model"] == src and r["target_model"] == tgt]
        if not cell_rows:
            continue
        xs = [r["gw_cost_neg"] for r in cell_rows]
        ys = [r["shift_rate"] for r in cell_rows]
        ax.scatter(
            xs,
            ys,
            c=color,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.4,
            label=_short_pair(src, tgt),
            s=64,
        )
    rho, lo, hi = rho_all
    ax.set_xlabel("entropic GW cost (NEG↔NEG alignment)")
    ax.set_ylabel("positive-shift rate (GW transport, coef=3.0)")
    ax.set_title(
        f"Does GW cost predict transfer success?\n"
        f"Spearman ρ = {rho:+.2f}  [95% CI {lo:+.2f}, {hi:+.2f}]   n = {len(rows)} cells"
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def _fig_layer_effect(rows: list[dict]) -> plt.Figure:
    by_pair_layer: dict[tuple[str, str], dict[float, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in rows:
        key = (r["source_model"], r["target_model"])
        by_pair_layer[key][r["relative_layer"]].append(r["shift_rate"])

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    pair_keys = list(by_pair_layer.keys())
    rel_layers = sorted({r["relative_layer"] for r in rows})
    width = 0.8 / max(len(pair_keys), 1)
    x_pos = np.arange(len(rel_layers))
    for i, (src, tgt) in enumerate(pair_keys):
        means = [float(np.mean(by_pair_layer[(src, tgt)][rl])) for rl in rel_layers]
        stds = [float(np.std(by_pair_layer[(src, tgt)][rl])) for rl in rel_layers]
        ax.bar(
            x_pos + (i - (len(pair_keys) - 1) / 2) * width,
            means,
            width=width,
            yerr=stds,
            capsize=4,
            color=PAIR_COLORS.get((src, tgt), "#888888"),
            edgecolor="black",
            linewidth=0.5,
            label=_short_pair(src, tgt),
        )
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{rl:.2f}" for rl in rel_layers])
    ax.set_xlabel("relative layer depth (paired on both sides)")
    ax.set_ylabel("positive-shift rate (mean ± std)")
    ax.set_title("Layer-depth effect on GW transport")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def _fig_k_effect(rows: list[dict]) -> plt.Figure:
    by_pair_k: dict[tuple[str, str], dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in rows:
        key = (r["source_model"], r["target_model"])
        by_pair_k[key][r["k"]].append(r["shift_rate"])

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    pair_keys = list(by_pair_k.keys())
    ks = sorted({r["k"] for r in rows})
    width = 0.8 / max(len(pair_keys), 1)
    x_pos = np.arange(len(ks))
    for i, (src, tgt) in enumerate(pair_keys):
        means = [float(np.mean(by_pair_k[(src, tgt)][k])) for k in ks]
        stds = [float(np.std(by_pair_k[(src, tgt)][k])) for k in ks]
        ax.bar(
            x_pos + (i - (len(pair_keys) - 1) / 2) * width,
            means,
            width=width,
            yerr=stds,
            capsize=4,
            color=PAIR_COLORS.get((src, tgt), "#888888"),
            edgecolor="black",
            linewidth=0.5,
            label=_short_pair(src, tgt),
        )
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("number of GMM clusters (k)")
    ax.set_ylabel("positive-shift rate (mean ± std)")
    ax.set_title("GMM-cluster-count effect on GW transport")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def _fig_summary(
    rows: list[dict],
    rho_all: tuple[float, float, float],
    per_pair_rho: dict[str, tuple[float, float, float]],
) -> plt.Figure:
    fig = plt.figure(figsize=(11.5, 4.8))
    # Left: scatter (same as fig 1 but smaller and overlayed with a fit line).
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1.0])
    ax_left = fig.add_subplot(gs[0])
    for (src, tgt), color in PAIR_COLORS.items():
        cell_rows = [r for r in rows if r["source_model"] == src and r["target_model"] == tgt]
        if not cell_rows:
            continue
        xs = [r["gw_cost_neg"] for r in cell_rows]
        ys = [r["shift_rate"] for r in cell_rows]
        ax_left.scatter(
            xs,
            ys,
            c=color,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.4,
            s=58,
            label=_short_pair(src, tgt),
        )
    rho, lo, hi = rho_all
    ax_left.set_xlabel("entropic GW cost (NEG↔NEG)")
    ax_left.set_ylabel("positive-shift rate")
    ax_left.set_title(f"All cells: ρ = {rho:+.2f}  [{lo:+.2f}, {hi:+.2f}]")
    ax_left.grid(alpha=0.3)
    ax_left.legend(loc="best", fontsize=9)

    # Right: per-pair Spearman bars with CIs.
    ax_right = fig.add_subplot(gs[1])
    labels = list(per_pair_rho.keys())
    rhos = [v[0] for v in per_pair_rho.values()]
    los = [v[0] - v[1] for v in per_pair_rho.values()]
    his = [v[2] - v[0] for v in per_pair_rho.values()]
    x_pos = np.arange(len(labels))
    ax_right.bar(
        x_pos,
        rhos,
        yerr=[los, his],
        capsize=6,
        color=[PAIR_COLORS.get(tuple(label.split(" → ")), "#888888") for label in labels],
        edgecolor="black",
        linewidth=0.6,
    )
    for xi, r in zip(x_pos, rhos, strict=True):
        ax_right.text(
            xi,
            r + (0.02 if r >= 0 else -0.06),
            f"{r:+.2f}",
            ha="center",
            va="bottom" if r >= 0 else "top",
            fontsize=10,
        )
    ax_right.axhline(0, color="grey", linestyle="--", alpha=0.4)
    ax_right.set_xticks(x_pos)
    ax_right.set_xticklabels(labels, rotation=20, fontsize=9)
    ax_right.set_ylabel("Spearman ρ")
    ax_right.set_title("Per-pair correlation (mean ± 95% CI)")
    ax_right.set_ylim(-1.05, 1.05)
    ax_right.grid(alpha=0.3, axis="y")
    fig.suptitle("Phase 7 diagnostic — GW cost as a predictor of transfer")
    fig.tight_layout()
    return fig


def main() -> int:
    sweep_path, rows = _latest_sweep()
    print(f"loaded {len(rows)} cells from {sweep_path}")

    if len(rows) < 3:
        print("  too few cells for correlation; aborting")
        return 1

    x_all = np.array([r["gw_cost_neg"] for r in rows])
    y_all = np.array([r["shift_rate"] for r in rows])
    rho_all = spearman_with_bootstrap_ci(x_all, y_all, n_boot=2000, seed=0)
    print(
        f"Spearman ρ (all {len(rows)} cells)  = {rho_all[0]:+.3f}  [{rho_all[1]:+.3f}, {rho_all[2]:+.3f}]"
    )

    per_pair_rho: dict[str, tuple[float, float, float]] = {}
    for src, tgt in PAIR_COLORS:
        cell_rows = [r for r in rows if r["source_model"] == src and r["target_model"] == tgt]
        if len(cell_rows) >= 3:
            xs = np.array([r["gw_cost_neg"] for r in cell_rows])
            ys = np.array([r["shift_rate"] for r in cell_rows])
            per_pair_rho[_short_pair(src, tgt)] = spearman_with_bootstrap_ci(
                xs, ys, n_boot=2000, seed=0
            )
            r0, lo, hi = per_pair_rho[_short_pair(src, tgt)]
            print(
                f"Spearman ρ ({_short_pair(src, tgt)}, n={len(cell_rows)}) = "
                f"{r0:+.3f}  [{lo:+.3f}, {hi:+.3f}]"
            )

    summary = {
        "n_cells": len(rows),
        "rho_all": {"rho": rho_all[0], "ci_lo": rho_all[1], "ci_hi": rho_all[2]},
        "rho_per_pair": {
            label: {"rho": v[0], "ci_lo": v[1], "ci_hi": v[2]} for label, v in per_pair_rho.items()
        },
    }
    correlations_path = sweep_path.parent / "correlations.json"
    correlations_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {correlations_path}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    artifacts: list[tuple[str, plt.Figure]] = [
        ("01_lift_vs_gw_cost_scatter.png", _fig_scatter(rows, rho_all)),
        ("02_layer_effect.png", _fig_layer_effect(rows)),
        ("03_k_effect.png", _fig_k_effect(rows)),
        ("04_diagnostic_summary.png", _fig_summary(rows, rho_all, per_pair_rho)),
    ]
    for fname, fig in artifacts:
        out = FIGURES_DIR / fname
        fig.savefig(out, dpi=160, bbox_inches="tight")
        print(f"wrote {out}")
        plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
