---
id: S2-03
title: "Metal M4: RMS_NORM_BACK kernel"
stage: 2
track: kernels
size: M
deps: ["S2-01"]
status: open
pr: null
---

# S2-03 — Metal M4: RMS_NORM_BACK kernel

**One-line outcome:** `RMS_NORM_BACK` exists on Metal with two `simd_sum` reductions per row,
passing MODE_GRAD parity vs the CPU oracle including non-multiple-of-simdgroup row widths.

## Why (context)

Every dense RMS-norm transformer layer emits `RMS_NORM_BACK` in its backward graph — the
autograd `RMS_NORM` case emits `ggml_rms_norm_back(grad, x, eps)`
(`vendor/llama.cpp/ggml/src/ggml.c:6575`) — and Metal has no kernel for it: `supports_op`
(`vendor/llama.cpp/ggml/src/ggml-metal/ggml-metal-device.m:1051-1368`) has a `GGML_OP_RMS_NORM`
case (`:1189`) but no `_BACK` case, so every norm-backward node falls back to CPU via
`ggml_backend_sched`. This is M4 in the ROADMAP §6 order M3 → M4 → M5 ("unblock backward
through norm/attn/ffn") for stage 2's goal of GPU-resident dense-LoRA training on Apple
Silicon.

The CPU oracle (`ggml_compute_forward_rms_norm_back_f32`,
`vendor/llama.cpp/ggml/src/ggml-cpu/ops.cpp:3902`, with the full derivation in comments)
reduces the backward to `dx = (dz + x · (−sum_xdz / sum_eps)) · rrms`
(`vendor/llama.cpp/ggml/src/ggml-cpu/ops.cpp:4050-4052`), where per row: `sum_xx = Σ x²`
(recomputing `rrms = 1/√(sum_xx/N + eps)` and `sum_eps = sum_xx + N·eps`), and
`sum_xdz = Σ x·dz` — the correction term that subtracts the mean projection of the incoming
gradient onto x. So the kernel needs exactly **two per-row reductions**, `Σx²` and `dot(dy,x)`,
then one elementwise write pass. Src layout: `src0 = dy` (gradient from the forward output),
`src1 = x` (the forward input), `eps` from op-params.

The in-tree pattern is `kernel_rms_norm_fuse_impl`
(`vendor/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:3058-3117`): one threadgroup per row,
grid-stride accumulation, `simd_sum` plus threadgroup buffer for the cross-simdgroup step —
the backward duplicates that reduction structure for the second sum. Per ROADMAP §6, the op is
five mechanical additions (kernel, kargs, pipeline getter, encoder, supports_op case); the
generated tests already cover it, including row widths that are not a multiple of the
simdgroup size (`n ∈ {64, 1025}`) and an eps sweep from 0 to 10
(`vendor/llama.cpp/tests/test-backend-ops.cpp:8425-8433`).

## What to do

All code changes land in the vendored llama.cpp fork via the S0-02 two-repo flow.

1. **Kernel** `kernel_rms_norm_back` in
   `vendor/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal`, patterned on
   `kernel_rms_norm_fuse_impl` (`:3058-3117`): one threadgroup per row; a grid-stride pass
   accumulating the two partials (`Σx²`, `Σx·dy` — one loop, two accumulators), each reduced
   with `simd_sum` + threadgroup buffer + barrier exactly as the forward reduces `Σx²`; then
   compute `rrms`, `sum_eps`, `scale_x = −sum_xdz/sum_eps` once and write
   `dx = (dy + x·scale_x) · rrms` elementwise. F32; guard the tail for row widths not a
   multiple of the simdgroup/threadgroup width (the `n = 1025` cases). Match the CPU
   reference's formula exactly per ADR-0002 (F32 row stats; no atomics; fixed reduction
   order).
2. **kargs struct** `ggml_metal_kargs_rms_norm_back` in
   `vendor/llama.cpp/ggml/src/ggml-metal/ggml-metal-impl.h` (near `ggml_metal_kargs_norm`,
   `:564`): shapes/strides for the two sources + `eps`.
3. **Pipeline getter** in `vendor/llama.cpp/ggml/src/ggml-metal/ggml-metal-device.cpp`,
   following the norm-family getters (threadgroup-memory sizing as for the forward norm
   pipelines).
4. **Encoder** `ggml_metal_op_rms_norm_back` in
   `vendor/llama.cpp/ggml/src/ggml-metal/ggml-metal-ops.cpp` plus a `GGML_OP_RMS_NORM_BACK`
   case in the `ggml_metal_op_encode_impl` dispatch switch (`:175`); the forward norm path
   (case at `:373-375`, encoder `ggml_metal_op_norm` at `:3357`) is the model.
5. **supports_op case** in `ggml_metal_device_supports_op` (`ggml-metal-device.m:1051-1368`):
   `GGML_OP_RMS_NORM_BACK` requiring `has_simdgroup_reduction`, F32, and the same-shape /
   contiguity preconditions the CPU kernel asserts.
6. **Tests.** Fork branch: `test-backend-ops test -b <Metal device> -o RMS_NORM_BACK` (the
   `test_rms_norm_back` cases, struct at `vendor/llama.cpp/tests/test-backend-ops.cpp:3494`,
   instantiated at `:8433` with `ne = {n, 5, 4, 3}`, `n ∈ {64, 1025}`, five eps values) and
   `test-backend-ops grad -b <Metal device> -o RMS_NORM` (the `test_rms_norm` cases set
   `ggml_set_param`, so MODE_GRAD builds backward graphs containing `RMS_NORM_BACK`; the
   per-node `supports_op` re-check now passes on Metal). Capture before/after
   (NOT_SUPPORTED → passing) output for the PR.
7. **Submodule bump PR** in llama-farm referencing this ticket, per S0-02.

## Out of scope

- `SOFT_MAX_BACK` (S2-02) and `SILU_BACK` (S2-04) — parallel tickets in the same trio.
- The saved-inv-var `RMS_NORM_BACK` ABI variant (forward stashes `inv_var`, backward becomes
  two loads — ROADMAP §13 item 6): a K5 perf option requiring an op-ABI change; microbenchmark
  first, own ticket if profiling justifies it (`tickets/backlog/`).
- `NORM` (LayerNorm) backward — not in the dense RMS-norm v1 op set (BLUEPRINT §8); GPT-2/BERT
  archs wait on the K-phase small-VJP work.
- Fused rms_norm_mul_add backward fusions — correctness first; fusion is K5 perf work.

## Acceptance criteria

- [ ] Fork branch: `test-backend-ops test -o RMS_NORM_BACK` on the Metal backend passes every
      generated case vs the CPU oracle, including `n = 1025` (non-multiple-of-simdgroup) and
      the full eps sweep.
- [ ] Fork branch: `test-backend-ops grad -o RMS_NORM` MODE_GRAD passes on Metal within the
      ADR-0002 tolerance.
- [ ] `test-backend-ops support -b <Metal device>` reports `RMS_NORM_BACK` supported (probe
      output attached to the PR).
- [ ] The kernel contains no atomic operations (grep-verifiable) — deterministic per ADR-0002.
- [ ] llama-farm submodule-bump PR is green: `ci-metal / build` and `ci-metal / grad` per-PR;
      one nightly (or `workflow_dispatch`) full-sweep run green.

## Testing & verification

Primary harness: vendored `tests/test-backend-ops` on the Metal backend — `test` mode for
forward parity of `RMS_NORM_BACK` vs the CPU oracle and `grad` mode (MODE_GRAD, ADR-0002
tolerances) for `RMS_NORM` cases whose backward graphs contain it. Runs in `ci-metal / grad`
per-PR (targeted `-o RMS_NORM,RMS_NORM_BACK`) and the nightly full Metal sweep (S2-01); the
nightly fallback report should show norm-backward nodes moving from CPU to Metal in the
tiny-model e2e.

## PR notes

- Branch: `ticket/S2-03-metal-rms-norm-back-kernel`.
- Two-repo flow per S0-02: fork PR (`llama-farm-base`) + trivial llama-farm submodule-bump PR,
  both referencing the ticket ID.
- Upstreaming disposition: **upstream-early** (ROADMAP §11 triage class a — pure addition
  behind `supports_op` with existing upstream tests).
- No copied external code; patterned on in-tree `kernel_rms_norm_fuse_impl` (MIT provenance
  per S0-01).
