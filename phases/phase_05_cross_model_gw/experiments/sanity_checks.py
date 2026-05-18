"""Phase 5 sanity checks for cross-model GW alignment.

For each (source_model, target_model, concept) cell, run the four GW
alignment cases the chapter discusses:

  A. self-pair    : GW(source layer L, source layer L). Expect near-
                    identity coupling, gw_cost ≈ 0.
  B. adjacent     : GW(source layer L, source layer L+1). Expect near-
                    identity with some leakage.
  C. random       : GW(source layer L, gaussian noise of the same shape).
                    Expect HIGH gw_cost — no clean alignment available.
  D. cross-model  : GW(source layer L_S, target layer L_T) on the same
                    contrastive dataset, layers paired by relative depth.

Writes outputs/<run_id>/sanity_checks.json with the full grid and a
config.json next to it.

Run:
    python phases/phase_05_cross_model_gw/experiments/sanity_checks.py
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ot_steering.activations.datasets import (
    load_refusal_pairs,
    load_sentiment_pairs,
)
from ot_steering.activations.extractor import extract_residual_stream
from ot_steering.activations.model_loader import ModelLoaderConfig, load_model
from ot_steering.ot.gw import GWConfig
from ot_steering.steering.cross_model_align import (
    CrossModelGWConfig,
    cross_model_gw_coupling,
)
from ot_steering.steering.ot_steering import GMMConfig
from ot_steering.utils.seed import set_all_seeds

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUTS_DIR = _PROJECT_ROOT / "outputs"

CELLS: list[dict[str, object]] = [
    {
        "source_model": "EleutherAI/pythia-160m",
        "target_model": "gpt2",
        "concept": "sentiment",
        "source_relative_layer": 0.5,
        "target_relative_layer": 0.5,
        "adjacent_offset": 1,
    },
    {
        "source_model": "EleutherAI/pythia-160m",
        "target_model": "gpt2",
        "concept": "refusal",
        "source_relative_layer": 0.5,
        "target_relative_layer": 0.5,
        "adjacent_offset": 1,
    },
]
N_TRAIN = 50  # use the full dataset
N_COMPONENTS = 4
SEED = 0
# reg is applied to normalised intra-distance matrices (max=1) — 0.01 is
# sharp enough to give discriminative couplings without numerical issues.
REG = 0.01


def _load_pairs(concept: str) -> list[tuple[str, str]]:
    if concept == "sentiment":
        return load_sentiment_pairs()
    if concept == "refusal":
        return load_refusal_pairs()
    raise ValueError(f"unknown concept: {concept!r}")


def _diagonal_mass(coupling: np.ndarray) -> float:
    """Fraction of mass on the (argmax-sorted) diagonal — proxy for identity."""
    if coupling.shape[0] != coupling.shape[1]:
        return float("nan")
    order = coupling.argmax(axis=1)
    sorted_coupling = coupling[np.argsort(order), :]
    diag = np.diag(sorted_coupling)
    total = sorted_coupling.sum()
    return float(diag.sum() / max(total, 1e-12))


def _flatten_class_acts(acts_pos: torch.Tensor, acts_neg: torch.Tensor) -> np.ndarray:
    return torch.cat([acts_pos, acts_neg], dim=0).cpu().numpy().astype(np.float64)


def _extract(model, tokenizer, pos: list[str], neg: list[str], layer: int) -> np.ndarray:
    acts = extract_residual_stream(
        model, tokenizer, pos + neg, layer_indices=[layer], batch_size=8
    )[layer]
    return acts.cpu().numpy().astype(np.float64)


def _run_cell(cell: dict[str, object]) -> dict[str, object]:
    src_id = str(cell["source_model"])
    tgt_id = str(cell["target_model"])
    concept = str(cell["concept"])
    src_rel = float(cell["source_relative_layer"])
    tgt_rel = float(cell["target_relative_layer"])
    offset = int(cell["adjacent_offset"])
    print(f"\n=== cell: {src_id} -> {tgt_id}  concept={concept} ===")

    pairs = _load_pairs(concept)[:N_TRAIN]
    pos = [a for a, _ in pairs]
    neg = [b for _, b in pairs]

    gw_cfg = GWConfig(reg=REG, num_iter_max=500, num_restart=2, warn_on_no_convergence=False)
    align_cfg = CrossModelGWConfig(
        n_components_source=N_COMPONENTS,
        n_components_target=N_COMPONENTS,
        gmm_cfg=GMMConfig(n_components=N_COMPONENTS, seed=SEED),
        gw_cfg=gw_cfg,
    )

    # ---- source side: load once, extract at chosen layer and its neighbour.
    src_model, src_tok = load_model(ModelLoaderConfig(model_id=src_id))
    src_n_layers = src_model.config.num_hidden_layers
    src_layer = int(round(src_rel * src_n_layers))
    src_adj_layer = min(src_layer + offset, src_n_layers)
    src_layer_acts = _extract(src_model, src_tok, pos, neg, src_layer)
    src_adj_acts = _extract(src_model, src_tok, pos, neg, src_adj_layer)
    src_dim = src_layer_acts.shape[1]
    # Build a noise distribution that matches the layout but has no structure.
    rng = np.random.default_rng(SEED + 10)
    src_noise = rng.normal(size=src_layer_acts.shape).astype(np.float64)
    del src_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- target side: load once, extract at chosen layer.
    tgt_model, tgt_tok = load_model(ModelLoaderConfig(model_id=tgt_id))
    tgt_n_layers = tgt_model.config.num_hidden_layers
    tgt_layer = int(round(tgt_rel * tgt_n_layers))
    tgt_layer_acts = _extract(tgt_model, tgt_tok, pos, neg, tgt_layer)
    del tgt_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- four cases.
    cases: dict[str, dict[str, float]] = {}

    def _case(name: str, src: np.ndarray, tgt: np.ndarray, *, rng_seed: int) -> None:
        align = cross_model_gw_coupling(
            src, tgt, cfg=align_cfg, rng=np.random.default_rng(rng_seed)
        )
        cases[name] = {
            "gw_cost": float(align.gw_cost),
            "diagonal_mass": _diagonal_mass(align.coupling),
            "source_d": int(align.source_centers.shape[1]),
            "target_d": int(align.target_centers.shape[1]),
        }
        print(
            f"  {name:14s} gw_cost={align.gw_cost:7.4f} "
            f"diag={cases[name]['diagonal_mass']:.3f}  "
            f"(d_src={cases[name]['source_d']}, d_tgt={cases[name]['target_d']})"
        )

    _case("self-pair", src_layer_acts, src_layer_acts, rng_seed=SEED + 1)
    _case("adjacent", src_layer_acts, src_adj_acts, rng_seed=SEED + 2)
    _case("random", src_layer_acts, src_noise, rng_seed=SEED + 3)
    _case("cross-model", src_layer_acts, tgt_layer_acts, rng_seed=SEED + 4)

    return {
        "source_model": src_id,
        "target_model": tgt_id,
        "concept": concept,
        "source_layer": src_layer,
        "source_adjacent_layer": src_adj_layer,
        "target_layer": tgt_layer,
        "source_d": src_dim,
        "target_d": int(tgt_layer_acts.shape[1]),
        "cases": cases,
    }


def _run_id(config: dict[str, object]) -> str:
    iso = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    digest = hashlib.blake2b(
        json.dumps(config, sort_keys=True).encode("utf-8"), digest_size=4
    ).hexdigest()
    return f"{iso}_{digest}"


def main() -> int:
    set_all_seeds(SEED)
    config = {
        "cells": CELLS,
        "n_train": N_TRAIN,
        "n_components": N_COMPONENTS,
        "seed": SEED,
        "reg": REG,
    }
    run_dir = OUTPUTS_DIR / _run_id(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    results = [_run_cell(c) for c in CELLS]
    (run_dir / "sanity_checks.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n=== headline ===")
    for res in results:
        print(f"  {res['source_model']} -> {res['target_model']} ({res['concept']}):")
        for name, payload in res["cases"].items():  # type: ignore[index]
            print(
                f"    {name:14s} cost={payload['gw_cost']:7.4f}  diag={payload['diagonal_mass']:.3f}"
            )
    print(f"\nartifacts: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
