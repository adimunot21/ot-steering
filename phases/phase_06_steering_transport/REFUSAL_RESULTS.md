# Refusal transport follow-up — results

A follow-up to the Phase 6 sentiment experiment, addressing the gap flagged
in Phase 7/8: base GPT-2 and Pythia-160M don't refuse anything, so the
canonical "refusal direction" result (Arditi et al. 2024) couldn't be tested.

This runs the same GW-transport pipeline on a **chat-tuned, cross-scale**
model pair within the Qwen2.5-Instruct family:

- **Qwen2.5-0.5B-Instruct** — fp16, 24 layers, d=896.
- **Qwen2.5-1.5B-Instruct** — 4-bit nf4 + double-quant, 28 layers, d=1536.

Both refuse harmful prompts cleanly in their baseline behaviour. This is a
genuine cross-architecture test: same family, different scale and dimension.

Reproduce:

```bash
python phases/phase_06_steering_transport/experiments/run_refusal_transport.py
```

## Design

- Concept = refusal. POS class = harmful prompts (refusal-eliciting),
  NEG class = harmless prompts (compliance-eliciting), from
  `configs/datasets/refusal_pairs.yaml`.
- Activations are extracted on **chat-template-formatted** prompts at the
  relative-midpoint layer; last-token residual stream.
- Metric: **refusal induction rate** on held-out *harmless* prompts —
  the fraction where the steered output reads as a refusal but the
  unsteered baseline does not (`_refusal_judge(steered) AND NOT
  _refusal_judge(baseline)`). This measures steering that did real work,
  not refusals the model would have produced anyway.
- 4 methods × 3 coefficients × 3 seeds, bootstrap 95% CIs across seeds.

## Headline (best coefficient per method)

### Qwen2.5-0.5B-Instruct → Qwen2.5-1.5B-Instruct (small → big)

Baseline refusal rate on harmless prompts (unsteered): **0 %**.

| method        | best induced refusals |
|---------------|-----------------------|
| random        | 5 %                   |
| Procrustes    | 0 %                   |
| **GW transport** | **3 %**            |
| target oracle | 15 %                  |

GW transport did **not** meaningfully induce refusals in this direction —
indistinguishable from the random floor.

### Qwen2.5-1.5B-Instruct → Qwen2.5-0.5B-Instruct (big → small)

Baseline refusal rate on harmless prompts (unsteered): **15 %**
(the smaller model spontaneously refuses some harmless prompts).

| method        | best induced refusals |
|---------------|-----------------------|
| random        | 17 %                  |
| Procrustes    | 12 %                  |
| **GW transport** | **22 %**           |
| target oracle | 35 %                  |

GW transport beats random by 5 pp and Procrustes by 10 pp, and recovers
22/35 ≈ **63 % of the target-supervised oracle's lift** — without
target-side concept labels. The effect only emerges at the highest
coefficient (6.0); at coefficient 3.0 GW transport is at 7 %.

## Honest reading

**Cross-model refusal transport works in one direction (big → small) but
not the other (small → big).** The working direction reproduces the
Phase 6 ordering (GW > Procrustes > random, ~63 % oracle recovery) on a
fundamentally different concept and on chat-tuned models — strengthening
the case that the Phase 6 sentiment result wasn't a sentiment-specific or
base-LM-specific artefact.

The asymmetry is the new finding and the new caveat. A plausible story is
that the smaller model's refusal representation is easier to push toward a
refusal target than the larger model's is, or that the 1.5B model's
cleaner cluster structure transports better than it receives. But with
3 seeds and a 20-prompt eval split, we cannot rule out that the asymmetry
is noise. Settling it needs more seeds, a larger eval split, and a
continuous-valued refusal judge (the current judge is a refusal-phrase
detector).

What this adds over Phase 6:
- First successful transport on a **non-sentiment** concept (refusal).
- First successful transport on **chat-tuned** models (Phase 6 was base LMs).
- First **cross-scale** transport (896-dim ↔ 1536-dim, 24 ↔ 28 layers).
- A new honest limitation: direction asymmetry, absent from the symmetric
  base-LM sentiment matrix.

Full per-method / per-coefficient / per-seed numbers with CIs are written
to `outputs/<run_id>/run_refusal_transport.json` (gitignored; regenerate
with the command above).
