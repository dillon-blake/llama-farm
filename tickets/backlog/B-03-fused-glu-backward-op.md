---
id: B-03
title: "Fused GLU backward op (GGML_OP_GLU_BACK) — bandwidth win"
stage: backlog
track: kernels
size: M
deps: [S1-28]
status: open
pr: null
---

# B-03 — Fused GLU backward op (GGML_OP_GLU_BACK) — bandwidth win

**One-line outcome:** **DEFERRED** — one elementwise pass producing recomputed `h`, `df`, `de`
in-place over three buffers, replacing today's SILU_BACK+MUL multi-pass that re-reads the largest
activations in the graph (`n_tokens × n_ff`).

**Activation trigger:** measure first — a milestone profile (S2-10/S3-10/S4-09) shows the GLU
backward multi-pass as a material share of backward time/bandwidth, and the projected fused win
beats the added op-surface cost (one more kernel on all four backends). ROADMAP §13 item 3, K5.

## Why (context)

The dense split-SWIGLU backward is a composite: `ggml_compute_backward`'s `GGML_OP_GLU` case emits
`SILU_BACK(MUL(grad, src1), src0)` for d_gate and `MUL(SILU(src0), grad)` for d_up
(`vendor/llama.cpp/ggml/src/ggml.c:6885-6900`, emissions at `:6890` and `:6893`). Each node is a
separate pass over `[n_tokens, n_ff]` tensors — the widest activations in a transformer graph — so
the same data is read from HBM several times. S1-28's `GGML_OP_GLU_BACK` fixed *coverage*
(REGLU/GEGLU/SWIGLU_OAI) but kept the multi-pass structure and left the split-SWIGLU composite
untouched.

Unsloth's Apache-2.0 SWIGLU kernel demonstrates the fused design this ticket imports (math only,
provenance header): one elementwise pass over three buffers recomputing `h = f·g` (so the forward
never stores it), computing `df = DW·f` and `de = DW·g·σ(e)·(1+e(1−σ(e)))`, storing all three in
place over the inputs (`unsloth/kernels/swiglu.py:68-109`, `_DWf_DW_dfg_kernel`). In ggml terms:
extend S1-28's op with a fused mode emitting `de`/`dg` in one kernel, feeding the recomputed `h`
to the checkpointing story (ROADMAP §13 item 5 / S1-17) instead of storing it. Trivial elementwise
on all four backends once specced — the risk is op-surface growth, hence the measurement gate.

## What to do

1. Microbenchmark the current multi-pass backward (SILU_BACK+MUL and S1-28's GLU_BACK) at
   representative `n_tokens × n_ff` shapes per backend vs a bandwidth-bound estimate of the fused
   pass; attach the report; proceed only if the win is material.
2. Spec the fused ABI on S1-28's op: one pass producing `(de, dg)` (optionally the recomputed `h`
   for checkpoint reuse), in-place writes where the allocator permits; per-variant math unchanged.
3. CPU reference kernel first (oracle), then CUDA/Metal/Vulkan elementwise ports; deterministic,
   F32 math per ADR-0002.
4. Switch the backward emissions (the `ggml.c:6885-6900` composite and S1-28 cases) to the fused
   op; keep a fallback flag for one release.
5. Measure end-to-end on a milestone training config; commit before/after numbers.

## Out of scope

- New GLU variants, forward changes; GELU-family unary VJPs — owned by B-08.
- The LoRA-epilogue fusion idea (BLUEPRINT §7 item 8) — deliberately not ticketed: a recorded
  non-goal unless a milestone profile shows the 6-node LoRA epilogue pattern dominating step
  time, in which case file a backlog stub with those numbers.

## Acceptance criteria

- [ ] Trigger microbenchmark in `benches/` (or ticket closed with numbers showing no material
      win — a valid outcome).
- [ ] `test-backend-ops` MODE_GRAD GLU cases (all variants, fused path forced) pass vs the CPU
      oracle within the ADR-0002 tolerance on CPU, CUDA, Metal, Vulkan.
- [ ] Determinism: bitwise-identical outputs across `n_threads`/reruns.
- [ ] Measured backward-time/bandwidth improvement on ≥1 milestone config recorded in the PR.
- [ ] All four CI lanes green on the targeted GLU grad sweeps.

## Testing & verification

Vendored `tests/test-backend-ops` MODE_GRAD (S1-28's cases, fused path forced) vs the CPU oracle:
`ci-cpu` per-PR; backend lanes targeted per-PR, full nightly. Microbenchmark and e2e measurements
on the stage VMs, committed to `benches/`.

## PR notes

- Branch: `ticket/B-03-fused-glu-backward-op`.
- Two-repo flow per S0-02: fork PR + llama-farm submodule bump.
- Upstreaming disposition: **upstream-later** — extends the fork-local `GLU_BACK` enum; rides the
  S1-28 RFC once the fused form is proven (ROADMAP §11 triage b).
- Provenance per S0-01: kernel header names `unsloth/kernels/swiglu.py` (Apache-2.0, header
  verified) as the source of the fused three-buffer design — math imported, no code translated.
