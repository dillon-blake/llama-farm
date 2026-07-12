---
id: S2-06
title: "Metal M2: OUT_PROD quantized-src0 kernel (critical path)"
stage: 2
track: kernels
size: L
deps: ["S2-05"]
status: open
pr: null
---

# S2-06 — Metal M2: OUT_PROD quantized-src0 kernel (critical path)

**One-line outcome:** `OUT_PROD` with quantized src0 runs on Metal — gradients flow through
frozen quantized weights GPU-resident on Apple Silicon, closing the stage's critical-path gap.

## Why (context)

The frozen-weight backward is `dX = out_prod(W_quantized, transpose(dY))`: the `MUL_MAT`
backward case emits exactly this for the src1 (activation) gradient
(`vendor/llama.cpp/ggml/src/ggml.c:6578-6630`, the `ggml_out_prod(src0, ggml_transpose(grad))`
call at `:6626-6629`). It is hit on **every linear layer, every microbatch** — this one op
decides whether backprop through a quantized GGUF base lives on the GPU (ROADMAP §1). Metal has
no `OUT_PROD` at all today: `GGML_OP_OUT_PROD` has no case in the coverage switch
(`vendor/llama.cpp/ggml/src/ggml-metal/ggml-metal-device.m:1051-1368`, default `false` at
`:1365-1366`), so `ggml_backend_sched` sends every such node to the CPU and Metal training is
backward-CPU-bound (BLUEPRINT §8 platform table). ROADMAP §6 M2 marks this the **critical-path
item** of the Metal plan, and ROADMAP §12 risk 12 names it one of the roadmap's two schedule
long poles — the MoltenVK stopgap decision (see `tickets/backlog/`, B-07) is the standing hedge
while this ticket is in flight.

The design is an extension of S2-05 (M1), not a new kernel family. M1 builds the F32 `OUT_PROD`
on the legacy `kernel_mul_mm` 64×32 simdgroup tiling
(`vendor/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:9874-10082`) using the native
`simdgroup_load(..., transpose=true)` flag (currently `false` at `metal:10021,10027`). This
ticket adds the dequantization phase copied from `kernel_mul_mm` phase 1 — the per-tile
shmem load/dequant loop (`metal:9935-9976`) — reusing the existing per-type `dequantize_*`
device functions (`metal:92-969`). Dequantization happens tile-by-tile into threadgroup
memory inside the kernel, so unlike the CUDA plan there is no dequant-to-pool transient.
The type table follows `kernel_mul_mm`'s instantiation list (`metal:10463-10488`): legacy
quants + K-quants + iq4 first, remaining IQ types after. The CPU oracle is the dequant-per-row
quantized `out_prod` (`vendor/llama.cpp/ggml/src/ggml-cpu/ops.cpp:4363-4501`); CUDA is F32-only
today (`vendor/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:4689-4690`), so Metal need not match
any GPU precedent — only the CPU reference under ADR-0002.

## What to do

All code changes land in the vendored llama.cpp fork via the S0-02 two-repo flow.

1. **Templatize the M1 kernel** over `block_q`/`nl`/`dequantize_func` exactly as
   `kernel_mul_mm` does (template header at `metal:9868-9873`), so the F32 variant becomes one
   instantiation of the general kernel rather than a separate code path.
2. **Copy the phase-1 dequant** from `kernel_mul_mm` (`metal:9935-9976`): per K-tile, threads
   cooperatively dequantize their `block_q` slice via `dequantize_func` into the `sa`
   threadgroup tile, then run M1's simdgroup multiply-accumulate with the transposed loads.
   If S2-05's risk-Q8 microbench chose the pre-transposed-staging contingency (swap shmem
   write indices), apply the same layout here.
3. **Instantiate the type table** as mul_mm does (`metal:10463-10488`): Q4_0/Q4_1/Q5_0/Q5_1/
   Q8_0, MXFP4, Q2_K–Q6_K, IQ4_NL/IQ4_XS first. Remaining IQ types (iq1/iq2/iq3 families)
   follow within this ticket if the timebox allows; otherwise gate them off precisely in
   supports_op and record the leftover set in the PR description plus a backlog note.
4. **The five mechanical additions** (ROADMAP §6 preamble): extend the M1 kargs struct if
   needed (`vendor/llama.cpp/ggml/src/ggml-metal/ggml-metal-impl.h`), pipeline getter keyed by
   src0 type (pattern `ggml_metal_library_get_pipeline_mul_mm`,
   `vendor/llama.cpp/ggml/src/ggml-metal/ggml-metal-device.cpp:704`), encoder case in
   `ggml-metal-ops.cpp`, and the supports_op case in `ggml-metal-device.m` — require
   `has_simdgroup_mm`, restrict to the implemented type set, and exclude NVFP4 as `MUL_MAT`
   does (`vendor/llama.cpp/ggml/src/ggml-metal/ggml-metal-device.m:1276-1278`).
5. **Numerics/determinism:** F32 accumulation throughout per ADR-0002 (S0-09); fixed
   accumulation order, no atomics (inherited from M1's scheme; state it in the PR).
6. **Light up the existing forward tests:** the quantized `test_out_prod` cases already exist
   (`vendor/llama.cpp/tests/test-backend-ops.cpp:8780-8806`, over `base_types` at
   `:7740-7749`) and currently skip Metal via supports_op. `base_types` covers only
   Q8_0/Q1_0/Q4_0/Q4_1/Q4_K/MXFP4/NVFP4/IQ2_XXS — add `test_out_prod` cases (or extend the
   loop) for the remaining implemented types (Q5_0/Q5_1, Q2_K/Q3_K/Q5_K/Q6_K, IQ4_NL/IQ4_XS).
7. **Add MODE_GRAD through a quantized `mul_mat` graph per implemented quant type.** No such
   case exists today: `test_mul_mat` skips `ggml_set_param` entirely when `type_a` is
   quantized (`vendor/llama.cpp/tests/test-backend-ops.cpp:4120` and `:4138`), so quantized
   backward is never grad-checked. Add cases that flag only src1 (the F32 activations) as
   param — grads then flow through the quantized-src0 `OUT_PROD`, matching real LoRA training.
   Soft coordination: the stage-3 CUDA quantized-OUT_PROD ticket (ROADMAP §5 C1) needs the
   same cases; if they already exist on the fork branch, only enable them for Metal.
8. **Submodule bump PR** in learning-llamas per S0-02; confirm in the S2-01 nightly e2e run that
   the sched fallback report no longer lists `OUT_PROD` for the tiny Q4_K dense model.

## Out of scope

- F32/F16 `OUT_PROD` on Metal and the risk-Q8 transposed-load microbench — S2-05 (M1).
- Expert-indexed variants `OUT_PROD_ID` / `OUT_PROD_ID_GRP` — S2-11 (E2/E3), which reuses this
  ticket's dequant machinery.
- CUDA and Vulkan quantized `OUT_PROD` — stage-3/stage-4 tickets (ROADMAP §5 C1, §7 V2).
- The stage-2 exit milestone (zero CPU fallback, perf snapshot) — S2-10.
- MoltenVK stopgap evaluation — backlog B-07.

## Acceptance criteria

- [ ] Fork branch: `test-backend-ops` mode `grad` passes on Metal for the new
      quantized-src0 `mul_mat` gradient cases for **every type in the supports_op table**,
      within the ADR-0002 per-op tolerance, and Metal-vs-CPU gradients on identical inputs
      meet the ADR-0002 cross-backend parity criterion (max-abs error ≤ 0.05 @ fp16).
- [ ] Fork branch: all `test_out_prod` forward cases for the implemented types pass on Metal
      against the CPU reference (previously reported NOT_SUPPORTED).
- [ ] supports_op returns true exactly for the implemented type set (NVFP4 excluded); if any
      IQ types are deferred, the PR description lists them and a backlog note exists.
- [ ] S2-01 nightly e2e fallback report for the tiny Q4_K dense model no longer contains
      `OUT_PROD` in the CPU-fallback op set.
- [ ] learning-llamas submodule-bump PR is green in `ci-metal / build` and `ci-metal / grad`.

## Testing & verification

Primary harness: vendored `tests/test-backend-ops` — forward eval for `test_out_prod`
(quantized types) and MODE_GRAD for the new quantized-`mul_mat` cases, Metal backend vs CPU
oracle (`ops.cpp:4363-4501`). Runs per-PR in the targeted `ci-metal / grad` lane (S2-01 selects
ops named in changed files) and in the full nightly Metal op sweep. The e2e evidence (fallback
report without `OUT_PROD`) comes from the S2-01 nightly convergence-gate run with
`--device metal`; the fallback-forbidden flip itself is S2-10's.

## PR notes

- Branch: `ticket/S2-06-metal-quantized-out-prod`.
- Two-repo flow per S0-02: implementation PR against the fork's `learning-llamas-base` branch with
  the ticket ID in the title, plus a trivial learning-llamas submodule-bump PR referencing it.
- Upstreaming disposition: **upstream-early** (ROADMAP §11 triage class a — Metal backward
  kernels are pure additions behind supports_op; mainline training benefits directly).
- Provenance per S0-01: the kernel carries a header naming its pattern source
  `ggml/src/ggml-metal/ggml-metal.metal` (`kernel_mul_mm`, MIT, commit `4f37f51`).
- Schedule note: this is a long pole (ROADMAP §12 risk 12) — flag slippage early so the
  B-07 MoltenVK decision can be pulled forward.
