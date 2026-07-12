---
id: B-06
title: "Linear-attention family backward: RWKV6/7, GATED_LINEAR_ATTN, GATED_DELTA_NET"
stage: backlog
track: kernels
size: XL
deps: [S1-31]
status: open
pr: null
---

# B-06 — Linear-attention family backward: RWKV6/7, GATED_LINEAR_ATTN, GATED_DELTA_NET

**One-line outcome:** **DEFERRED** — recompute-reverse-scan backward ops (the S1-31 skeleton) for
`RWKV_WKV6`, `RWKV_WKV7`, `GATED_LINEAR_ATTN`, and the hardest single op in this area,
`GATED_DELTA_NET` (plus `CUMSUM`/`TRI`/`SOLVE_TRI` VJPs), for RWKV- and
Qwen3-Next/Kimi-Linear-class architectures.

**Activation trigger:** concrete model demand — a specific RWKV6/RWKV6Qwen2, RWKV7/ARWKV7, GLA,
or delta-net (Qwen3-Next, Qwen3.5(-MoE), Kimi-Linear) model requested for training. ROADMAP §10
keeps this family as inventory, explicitly not scheduled; do not start speculatively.

## Why (context)

All four ops have forwards on every backend today (op enums
`vendor/llama.cpp/ggml/include/ggml.h:568-572`; CUDA forwards in `ggml/src/ggml-cuda/wkv.cu`,
`gla.cu`, `gated_delta_net.cu`) but no backward anywhere — ROADMAP §10 rates RWKV6/GLA and RWKV7
**L** per backend and `GATED_DELTA_NET` **XL**. Like `SSM_SCAN`, these recurrent scans overwrite
intermediate state in place, so the backward cannot read saved states: the S1-31 `SSM_SCAN_BACK`
skeleton applies directly — checkpoint state every K tokens, re-run the forward per chunk, then
reverse-scan accumulating input grads. Boundary terms vanish for single-ubatch training exactly as
in S1-31; cross-ubatch BPTT stays out of scope with the same truncation-at-ubatch semantics.

The delta-net item is bigger than one kernel: its chunked path also emits `CUMSUM`, `TRI`, and
`SOLVE_TRI` (`ggml.h:499`, `:557`, `:571`), whose VJPs ROADMAP §10 rates S-M/S/M, mostly
graph-level composites. And the delta-net architectures are MoE hybrids, so training them also
requires the §9 MoE backward set (S1-25–S1-28 on CPU plus the relevant backend ports) — a soft
coordination point, not a frontmatter dep, since the trigger model determines which backends
matter.

## What to do

At activation, split into per-family child tickets; this umbrella defines the shared shape:

1. Per family, add the `*_BACK` op (fork-local enum tail) with a CPU reference kernel on the
   S1-31 chunk-recompute skeleton — derive per-timestep reverse recurrences from the op's CPU
   forward, document them in the kernel header, keep deterministic parallelization (threads own
   heads/sequences, no atomics — ADR-0002/G-B).
2. Wire the `ggml_compute_backward` cases; I32 index operands marked non-differentiable.
3. `CUMSUM`/`TRI`/`SOLVE_TRI` VJPs as graph-level composites where possible (delta-net only).
4. `test-backend-ops` MODE_GRAD cases per op (small states, chunk-boundary crossing) vs finite
   differences on CPU.
5. Backend ports only for the backends the triggering model demands, reusing each backend's
   forward kernel structure (e.g. reversed loop over the CUDA scan kernels).
6. For a delta-net trigger, verify the MoE prerequisite set is done on the target backend first.

## Out of scope

- Cross-ubatch BPTT (truncation-at-ubatch documented, as in S1-31).
- Grads for the recurrence's own parameters (decay/gate weights — frozen; LoRA targets the plain
  `MUL_MAT` projections already covered by the OUT_PROD work).
- Mamba-1/2 `SSM_*` ops — S1-30/S1-31 and their backend-port tickets.

## Acceptance criteria

- [ ] Per activated family: `test-backend-ops` MODE_GRAD passes on CPU (finite differences) and
      on each ported backend vs the CPU oracle within the ADR-0002 tolerance.
- [ ] Determinism: bitwise-identical grads across `n_threads`/reruns.
- [ ] A tiny model of the triggering arch trains (loss falls) on `ci-cpu`, and on the target
      backend lane if a port was in scope; the S1-11 preflight flips it to trainable via the
      graph walk (asserted in the e2e module).
- [ ] Checkpoint-interval memory/FLOPs trade-off documented in the PR (ROADMAP §12 Q7 precedent).

## Testing & verification

Vendored `tests/test-backend-ops` MODE_GRAD: `ci-cpu` per-PR (oracle); target backend lanes
targeted per-PR and full nightly after ports. The tiny-model e2e joins learning-llamas `tests/` on
`ci-cpu`.

## PR notes

- Branch: `ticket/B-06-linear-attention-family-backward` (child tickets per family at activation).
- Two-repo flow per S0-02: fork PR(s) + learning-llamas submodule bumps.
- Upstreaming disposition: **upstream-later** — new op enums, in-fork first; one RFC per op family
  once the CPU oracle plus one GPU backend prove the design (ROADMAP §11 triage b).
