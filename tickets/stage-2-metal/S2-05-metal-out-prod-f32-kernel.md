---
id: S2-05
title: "Metal M1: OUT_PROD F32 kernel"
stage: 2
track: kernels
size: M
deps: ["S2-01"]
status: open
pr: null
---

# S2-05 — Metal M1: OUT_PROD F32 kernel

**One-line outcome:** F32 `OUT_PROD` runs on Metal via `kernel_mul_mm`-style simdgroup tiling
using the API's native transpose flag — unblocking GPU-resident LoRA A/B gradients on Apple
Silicon and establishing the tiling base that S2-06 extends with dequantization.

## Why (context)

`OUT_PROD` is the `MUL_MAT` backward w.r.t. both operands: the autograd case
(`vendor/llama.cpp/ggml/src/ggml.c:6578-6630`) emits `ggml_out_prod` for the weight-gradient
path (`:6598`) and the activation-gradient path (`:6625`), so it is hit on every linear layer,
every microbatch — the op that decides whether backprop lives on the GPU (ROADMAP §1). Metal
has no `OUT_PROD` at all: `supports_op`
(`vendor/llama.cpp/ggml/src/ggml-metal/ggml-metal-device.m:1051-1368`) has no case, so every
out-prod falls back to CPU. This ticket lands the F32 case, which per the LoRA graph analysis
(ROADMAP §13 item 5) is sufficient for **all LoRA A/B weight gradients** — plain F32 GEMMs of
rank-r width; the quantized-src0 case (activation grads through frozen weights) is the
critical-path follow-up S2-06 that copies this kernel's tiling and adds the dequant phase.

The implementation pattern is the legacy-simdgroup `kernel_mul_mm`
(`vendor/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:9874-10082` — the `#else` branch of
`GGML_METAL_HAS_TENSOR`, `:9867`/`:10084`; ROADMAP §6 says to target this path, not the
Metal4 tensor API, which is disabled by default pre-M5 hardware): a 64×32 output tile per
threadgroup (`NR0 = 64`, `NR1 = 32` at `:9887-9888`), staged through threadgroup memory, with
8×8 `simdgroup_load`/`simdgroup_multiply_accumulate` inner loops. The key difference for
out-prod is the reduction axis: `mul_mat` reduces over `src0->ne[0]` while `out_prod` reduces
over `src0->ne[1]` (dst is `[src0->ne[0], src1->ne[0]]`). `simdgroup_load` has a **native
transpose flag** — the trailing bool, currently `false` in the mul_mm multiply phase at
`:10021` and `:10027`, and already used as `true` elsewhere in-tree (e.g. the FA kernels,
`:6401`) — so the tiling carries over with flipped load orientation instead of a new data
layout.

Numerics and determinism follow ADR-0002: F32 accumulators (the simdgroup accumulation is
F32), a fixed `NK`-chunk loop over the reduction axis, exclusive dst-tile ownership per
threadgroup, no atomics. ROADMAP §12 Q8 flags one open risk: transposed `simdgroup_load` from
threadgroup memory may bank-conflict; the pre-scoped contingency is to stage tiles
pre-transposed during the shmem write phase instead (swap the write indices — free), which this
ticket must measure, not guess.

## What to do

All code changes land in the vendored llama.cpp fork via the S0-02 two-repo flow.

1. **Kernel** `kernel_out_prod_f32` in
   `vendor/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal`, adapted from the legacy-simdgroup
   `kernel_mul_mm` (`:9874-10082`): keep the 64×32 threadgroup tile, shmem staging, and
   `NK`-chunk reduction loop (`:9935` onward), swap the reduction axis by loading the
   affected 8×8 tiles with `simdgroup_load(..., transpose=true)` (flag currently `false` at
   `:10021`, `:10027`). Handle the batch/broadcast dims the tests generate (`ne2/ne3` batches
   plus src1-side repeat, mirroring mul_mm's `FC_mul_mm_r2/r3` handling), and non-multiple
   tile edges (the `nr0/nr1` clamps already in the pattern). F32 src0/src1/dst only.
2. **Q8 microbench (decide the load strategy):** implement both variants — transposed
   `simdgroup_load` vs pre-transposed shmem staging (swapped write indices) — behind a local
   compile-time switch; compare with `test-backend-ops perf -b <Metal device> -o OUT_PROD` at
   the generated shapes plus a LoRA-shaped case (rank-r × n_embd). Keep the faster variant,
   delete the switch, record numbers in the fork PR description, and note the outcome for
   S2-06 (which inherits the choice).
3. **kargs struct** `ggml_metal_kargs_out_prod` in
   `vendor/llama.cpp/ggml/src/ggml-metal/ggml-metal-impl.h` (near `ggml_metal_kargs_mul_mm`,
   `:451`).
4. **Pipeline getter** in `vendor/llama.cpp/ggml/src/ggml-metal/ggml-metal-device.cpp`
   (pattern `ggml_metal_library_get_pipeline_mul_mm`, `:704`).
5. **Encoder + dispatch case** `GGML_OP_OUT_PROD` in
   `vendor/llama.cpp/ggml/src/ggml-metal/ggml-metal-ops.cpp` (`ggml_metal_op_encode_impl`
   switch at `:175`; `ggml_metal_op_mul_mat` at `:2043` with its pipeline call at `:2188` is
   the model).
6. **supports_op case** in `ggml_metal_device_supports_op` (`ggml-metal-device.m:1051-1368`):
   `GGML_OP_OUT_PROD` returning true only for F32 src0/src1 with `has_simdgroup_mm`; quantized
   src0 stays false (S2-06 widens it). This is the gate flip that lights up the existing
   tests.
7. **Tests.** Fork branch: `test-backend-ops test -b <Metal device> -o OUT_PROD` — the
   generated cases (`vendor/llama.cpp/tests/test-backend-ops.cpp:8780-8806`) include F32×F32
   at `n,k ∈ {1,16}` with batch/repeat sweeps, plus dedicated F32 `ne2`/`nr2` sweeps
   (`:8798-8806`); non-F32 cases must report unsupported, not wrong. Then
   `test-backend-ops grad -b <Metal device> -o MUL_MAT` — `test_mul_mat` sets params on both
   operands, so MODE_GRAD builds backward graphs whose `OUT_PROD` nodes now pass the per-node
   `supports_op` re-check (`:1779`) and execute on Metal. Capture before/after output for the
   PR.
8. **Submodule bump PR** in learning-llamas referencing this ticket, per S0-02.

## Out of scope

- Quantized-src0 `OUT_PROD` — S2-06 (the stage's critical path; it copies this tiling and adds
  the `kernel_mul_mm` dequant phase). Land this ticket promptly: S2-06 is sequenced on it.
- F16/BF16-src0 `OUT_PROD` on Metal — not required by the v1 dense-LoRA path (CPU handling is
  S1-18/K-F16OP); revisit with the FA/K5 work if a graph emits it.
- `OUT_PROD_ID` / `OUT_PROD_ID_GRP` (MoE expert outer products) — S2-11.
- Metal4 tensor-API (`matmul2d`) variant of the kernel — explicitly deferred by ROADMAP §6;
  legacy simdgroup path only.

## Acceptance criteria

- [ ] Fork branch: `test-backend-ops test -o OUT_PROD` on the Metal backend passes every F32
      case vs the CPU oracle, including the `ne2`/`nr2` batch and repeat sweeps.
- [ ] Fork branch: `test-backend-ops grad -o MUL_MAT` MODE_GRAD passes on Metal for F32 cases
      within the ADR-0002 tolerance (backward graphs containing `OUT_PROD`).
- [ ] `test-backend-ops support -b <Metal device>` shows `OUT_PROD` supported for F32 and
      unsupported for quantized src0 (probe output attached to the PR).
- [ ] The kernel contains no atomic operations (grep-verifiable); two consecutive runs of the
      same `OUT_PROD` case produce identical results (determinism per ADR-0002).
- [ ] The fork PR description contains the Q8 microbench table (transposed `simdgroup_load` vs
      pre-transposed staging) and names the variant kept.
- [ ] learning-llamas submodule-bump PR is green: `ci-metal / build` and `ci-metal / grad` per-PR;
      one nightly (or `workflow_dispatch`) full-sweep run green.

## Testing & verification

Primary harness: vendored `tests/test-backend-ops` on the Metal backend — `test` mode for the
`OUT_PROD` forward-eval cases vs the CPU oracle (the CPU F32 path,
`vendor/llama.cpp/ggml/src/ggml-cpu/ops.cpp:4242`, is the reference), `grad` mode (MODE_GRAD,
ADR-0002 tolerances from S0-09) through F32 `MUL_MAT` graphs, and `perf` mode for the Q8
microbench. Runs in `ci-metal / grad` per-PR (targeted `-o OUT_PROD,MUL_MAT`) and the nightly
full Metal sweep (S2-01); after this lands, the nightly e2e fallback report should show LoRA
A/B gradient out-prods on Metal, with quantized ones still on CPU until S2-06.

## PR notes

- Branch: `ticket/S2-05-metal-out-prod-f32-kernel`.
- Two-repo flow per S0-02: fork PR (`learning-llamas-base`) + trivial learning-llamas submodule-bump PR,
  both referencing the ticket ID.
- Upstreaming disposition: **upstream-early** (ROADMAP §11 triage class a — pure addition
  behind `supports_op`; upstream already generates the test cases).
- Soft coordination: S2-06 copies this kernel's tiling and the Q8 microbench outcome; flag any
  structural choices that would complicate adding the dequant phase.
- No copied external code; adapted from in-tree `kernel_mul_mm` (MIT provenance headers per
  S0-01 where code is copied).
