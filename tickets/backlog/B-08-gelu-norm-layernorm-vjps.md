---
id: B-08
title: "GELU-family + NORM (LayerNorm) VJPs"
stage: backlog
track: kernels
size: M
deps: [S1-19]
status: open
pr: null
---

# B-08 — GELU-family + NORM (LayerNorm) VJPs

**One-line outcome:** **DEFERRED** — backward rules for the GELU activation family
(GELU exact, GELU_QUICK, tanh-approx GELU) and for `NORM` (LayerNorm), unlocking
GPT-2/Phi/BERT-style and GELU-MLP dense architectures for LoRA training.

**Activation trigger:** a concrete GELU-MLP or LayerNorm architecture enters project
scope (a user-requested model fails the S1-11 preflight naming GELU/NORM as the
blocking ops). Until then these VJPs sit behind no scheduled milestone: v1 model
coverage is dense RMS-norm transformers (BLUEPRINT §8), and no stage-1–4 e2e gate
exercises a GELU or LayerNorm graph.

## Why (context)

BLUEPRINT §7 item 5 and P4 list "upstream small VJPs: CLAMP, SIGMOID, TANH,
GELU-family, NORM" and gap G8 counts GELU-family and `NORM` among the missing
backward cases in `ggml_compute_backward` (`vendor/llama.cpp/ggml/src/ggml.c:6430-6913`).
S1-19 delivered the first three (TANH/SIGMOID/CLAMP) because stage-1 work consumes
them (gemma softcap, MoE routers, PPO clip); the GELU/NORM pair was deliberately
left unscheduled — each unlocks an architecture *family* rather than a scheduled
milestone, so they activate on demand.

The math is settled. GELU-family: unary VJPs following the S1-19 pattern (composite
from existing ops where possible; a dedicated elementwise backward kernel per backend
otherwise). `NORM` (LayerNorm): the mean + inv_var two-statistic backward — the worked
derivation, including stashed-stats layout, exists on unsloth's Apache-2.0 side
(`unsloth/kernels/layernorm.py:67-104`; ROADMAP §13 item 7 — math only, no code copy).

## What to do

1. Add `ggml_compute_backward` cases for GELU, GELU_QUICK, and tanh-approx GELU
   (S1-19 pattern: prefer op-composites; saved-output forms where cheaper).
2. Add the `NORM` backward case + CPU reference kernel (mean + inv_var per row;
   provenance header naming `layernorm.py`, Apache-2.0).
3. Port to the backends the triggering architecture targets (CPU always; others
   per demand), reusing each backend's RMS_NORM_BACK plumbing (e.g. CUDA
   `vendor/llama.cpp/ggml/src/ggml-cuda/norm.cu:158` region).
4. MODE_GRAD cases per op; flip the S1-11 preflight coverage table so the affected
   arch families report trainable.
5. Extend the S1-12 convergence gate with a tiny GELU/LayerNorm model variant.

## Out of scope

- Fused GLU backward and GEGLU derivatives — S1-28 (landed in stage 1) and B-03.
- RMS_NORM saved-inv-var ABI variant — B-04.

## Acceptance criteria

- [ ] `test-backend-ops` MODE_GRAD passes for each new VJP vs finite differences on
      CPU, and vs the CPU oracle within ADR-0002 tolerance on each ported backend.
- [ ] S1-11 preflight flips the triggering arch family from blocked to trainable,
      with the graph-walk test updated.
- [ ] Tiny-model convergence run for the triggering arch passes on `ci-cpu` nightly.

## Testing & verification

Vendored `tests/test-backend-ops` MODE_GRAD (targeted per-PR on `ci-cpu` plus the
ported backends' lanes; full sweeps nightly). Preflight unit tests in `tests/`.

## PR notes

- Branch: `ticket/B-08-gelu-norm-layernorm-vjps`.
- Two-repo flow per S0-02: fork PR + llama-farm submodule bump.
- Upstreaming disposition: **upstream-early** — small VJPs following existing
  patterns benefit mainline training directly (ROADMAP §11 triage a).
- Provenance per S0-01: header names `unsloth/kernels/layernorm.py` (Apache-2.0),
  math only.
