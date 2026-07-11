---
id: S2-02
title: "Metal M3: SOFT_MAX_BACK kernel"
stage: 2
track: kernels
size: M
deps: ["S2-01", "S1-20"]
status: open
pr: null
---

# S2-02 — Metal M3: SOFT_MAX_BACK kernel

**One-line outcome:** `SOFT_MAX_BACK` exists on Metal as a row-reduction kernel computing
`dx = scale · y ∘ (dy − dot(dy,y))`, with MODE_GRAD parity vs the CPU oracle including
`max_bias > 0`.

## Why (context)

Metal's training-op coverage today is `ROPE_BACK` plus the optimizer steps
(`vendor/llama.cpp/ggml/src/ggml-metal/ggml-metal-device.m:1192` and `:1362-1364`); everything
else in the backward suite is absent — `supports_op`
(`vendor/llama.cpp/ggml/src/ggml-metal/ggml-metal-device.m:1051-1368`) has no
`GGML_OP_SOFT_MAX_BACK` case, so it hits `default: return false` and `ggml_backend_sched` runs
the node on CPU. ROADMAP §6 orders the Metal suite M3 → M4 → M5 first ("unblock backward
through norm/attn/ffn"): `SOFT_MAX_BACK` is emitted by the autograd `SOFT_MAX` case on the
naive-attention path of every training graph (`vendor/llama.cpp/ggml/src/ggml.c:6762-6773`,
emission at `:6770`) — the only attention training path until FA lands (S2-13).

The math is a single per-row reduction. The CPU oracle
(`ggml_compute_forward_soft_max_ext_back_f32`,
`vendor/llama.cpp/ggml/src/ggml-cpu/ops.cpp:5520`) computes exactly
`dx = scale · y ∘ (dy − dot(dy,y))` (`vendor/llama.cpp/ggml/src/ggml-cpu/ops.cpp:5588-5593`),
where `src0 = dy` and `src1 = y` (the forward output). On Metal that maps directly onto the
`kernel_soft_max` row-reduction pattern
(`vendor/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:1896`): one threadgroup per row,
grid-stride element loop, `simd_sum` plus a small threadgroup buffer for cross-simdgroup
reduction — the backward needs only **one** `simd_sum` (for `dot(dy,y)`) where the forward
needs a max and a sum.

`max_bias` (ALiBi) must be handled from day one, by *not* gating on it: the ALiBi bias is
additive and constant w.r.t. the logits, so the backward formula does not involve `max_bias`
at all (ROADMAP §3 K-SMB). S1-20 (a dependency) already lifted the CPU asserts and enabled the
`max_bias = 8.0` MODE_GRAD cases, so a CPU baseline exists for Metal to compare against. Per
ROADMAP §6 each Metal op is five mechanical additions — kernel, kargs struct, pipeline getter,
encoder, `supports_op` case — with no structural change to the graph-encode path.

## What to do

All code changes land in the vendored llama.cpp fork via the S0-02 two-repo flow.

1. **Kernel** `kernel_soft_max_back` in `vendor/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal`,
   patterned on `kernel_soft_max` (`:1896`): one threadgroup per row; grid-stride pass
   accumulating `dot(dy,y)` reduced via `simd_sum` (threadgroup buffer + barrier for rows
   spanning multiple simdgroups, exactly as `kernel_soft_max` does for its max/sum); second
   pass writes `dx = scale · y · (dy − dot)`. F32 only (matches the CPU oracle and the
   `test_soft_max_back` cases). `max_bias` is not read by the math — add a comment recording
   why (K-SMB: additive bias is constant w.r.t. logits).
2. **kargs struct** `ggml_metal_kargs_soft_max_back` in
   `vendor/llama.cpp/ggml/src/ggml-metal/ggml-metal-impl.h` (place near
   `ggml_metal_kargs_soft_max`, `:837`): shape/stride fields plus `scale`.
3. **Pipeline getter** in `vendor/llama.cpp/ggml/src/ggml-metal/ggml-metal-device.cpp`,
   pattern `ggml_metal_library_get_pipeline_soft_max` (`:453`).
4. **Encoder** `ggml_metal_op_soft_max_back` in
   `vendor/llama.cpp/ggml/src/ggml-metal/ggml-metal-ops.cpp` plus a `GGML_OP_SOFT_MAX_BACK`
   case in the dispatch switch of `ggml_metal_op_encode_impl` (`:175`; the `GGML_OP_SOFT_MAX`
   case at `:319-321` and its encoder at `:1300` are the model, including threadgroup-memory
   sizing).
5. **supports_op case** in `ggml_metal_device_supports_op`
   (`ggml-metal-device.m:1051-1368`): require `has_simdgroup_reduction`, F32 tensors, and the
   same contiguity the CPU kernel assumes. **No `max_bias` condition.**
6. **Tests.** On the fork branch run against a CPU-baseline build that includes S1-20:
   `test-backend-ops test -b <Metal device> -o SOFT_MAX_BACK` — the generated forward-eval
   cases (`vendor/llama.cpp/tests/test-backend-ops.cpp:8925-8931`) cover `max_bias ∈ {0, 8}`,
   `scale ∈ {1, 0.1}`, and off-by-one row sizes (`ne0−1`, `ne1−1`) — and
   `test-backend-ops grad -b <Metal device> -o SOFT_MAX` — the MODE_GRAD cases
   (`max_bias` loop at `:8880-8884`) now execute on Metal because `eval_grad`'s per-node
   `supports_op` re-check (`:1779`) passes. Capture before/after output for the PR.
7. **Submodule bump PR** in llama-farm referencing this ticket, per S0-02.

## Out of scope

- `RMS_NORM_BACK` (S2-03), `SILU_BACK` (S2-04) — the other two ops of the M3-M5 trio.
- Sparse-CE kernels on Metal — S2-07 (also patterned on `kernel_soft_max`; coordinate on any
  shared reduction helpers, but do not build them here speculatively).
- FA backward on Metal — S2-13; FA backward recomputes P from Q/K/mask/LSE and never emits
  `SOFT_MAX_BACK`.
- CUDA/Vulkan `max_bias` gates — S3-04 / S4-05 per S1-20's breadcrumbs.

## Acceptance criteria

- [ ] Fork branch: `test-backend-ops test -o SOFT_MAX_BACK` on the Metal backend passes every
      generated case, including `max_bias = 8.0` and the off-by-one shapes, vs the CPU oracle.
- [ ] Fork branch: `test-backend-ops grad -o SOFT_MAX` MODE_GRAD passes on Metal within the
      ADR-0002 tolerance, including the `max_bias > 0` cases enabled by S1-20.
- [ ] `test-backend-ops support -b <Metal device>` reports `SOFT_MAX_BACK` supported, with no
      `max_bias` gating (probe output attached to the PR).
- [ ] The kernel contains no atomic operations (grep-verifiable) — deterministic per ADR-0002.
- [ ] llama-farm submodule-bump PR is green: `ci-metal / build` and `ci-metal / grad` per-PR;
      one nightly (or `workflow_dispatch`) full-sweep run green.

## Testing & verification

Primary harness: vendored `tests/test-backend-ops` on the Metal backend — `test` mode for
forward parity of the new op vs the CPU oracle, `grad` mode (MODE_GRAD finite differences,
ADR-0002 tolerances from S0-09) for `SOFT_MAX` cases whose backward graphs contain the op. Runs
in `ci-metal / grad` per-PR (targeted `-o SOFT_MAX,SOFT_MAX_BACK`) and in the nightly full
Metal sweep (S2-01). The nightly fallback report should show `SOFT_MAX_BACK` moving from CPU
to Metal in the tiny-model e2e once this lands.

## PR notes

- Branch: `ticket/S2-02-metal-soft-max-back-kernel`.
- Two-repo flow per S0-02: fork PR (`llama-farm-base`) + trivial llama-farm submodule-bump PR,
  both referencing the ticket ID.
- Upstreaming disposition: **upstream-early** (ROADMAP §11 triage class a — pure addition
  behind `supports_op`; mainline training benefits and the tests already exist upstream).
- No copied external code; the kernel is patterned on in-tree `kernel_soft_max` — normal
  llama.cpp MIT provenance applies (S0-01).
