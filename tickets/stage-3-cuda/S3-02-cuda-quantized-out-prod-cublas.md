---
id: S3-02
title: "CUDA C1+C2: quantized + F16/BF16 OUT_PROD via dequant + cublasGemmEx (F32 accumulate)"
stage: 3
track: kernels
size: M
deps: [S0-09, S3-01]
status: open
pr: null
---

# S3-02 — CUDA C1+C2: quantized + F16/BF16 OUT_PROD via dequant + cublasGemmEx (F32 accumulate)

**One-line outcome:** the single op that makes CUDA dense-LoRA training fully GPU-resident:
quantized/F16/BF16-src0 `OUT_PROD` via dequant-to-F16 + `cublasGemmEx` pinned to
F32-output/`CUBLAS_COMPUTE_32F`, chunked over the reduction axis to cap transients.

## Why (context)

CUDA is one op away from a fully GPU-resident dense-LoRA training step (ROADMAP §0, §5).
`OUT_PROD` is `MUL_MAT`'s backward w.r.t. both operands; the frozen-weight case
`out_prod(W_quantized, transpose(grad))` fires on every linear layer of every microbatch, so
this op alone decides whether backprop lives on the GPU (ROADMAP §1). Today CUDA's
`supports_op` accepts only all-F32 `OUT_PROD`
(`vendor/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:4689-4690`), so every quantized-weight
backward falls back to CPU (BLUEPRINT G6).

ROADMAP §0 finding 1: **no new quantized GEMM kernel is needed.** The right answer is
dequant-to-F16 + `cublasGemmEx`, reusing the plumbing of `ggml_cuda_mul_mat_cublas_impl`
(`vendor/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:1324-1536`). A custom quantized kernel is
structurally hopeless: the `OUT_PROD` reduction axis (src0 *rows*, ne01) is orthogonal to
the quantization-block axis (ne00) — every dp4a/int-mma tile design assumes blocks lie along
the reduction axis (`vendor/llama.cpp/ggml/src/ggml-cuda/mmq.cuh:17-45`), so reuse would
require physically transposing quantized data, i.e. a dequant anyway. A fused dequant+rank-k
kernel is deferred (L effort, no tensor cores, loses to GemmEx at training batch sizes;
backlog B-01, risk R2 / ROADMAP §12 Q1).

**The critical numerics trap (ADR-0002, S0-09):** `ggml_cuda_mul_mat_cublas_impl`'s default
F16 traits set `CUBLAS_COMPUTE_16F` — F16 accumulation
(`vendor/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:1311-1323`, the compute type at `:1313`),
which ADR-0002 forbids on gradient paths. Its `prefer_f32_output` branch
(`vendor/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:1425-1440`) selects F32 output +
`CUBLAS_COMPUTE_32F`, but only on specific arches (Volta/RDNA4/CDNA, `:1428-1430`). This
ticket must **pin that F32 configuration on all arches**. ROADMAP §12 Q2 flags the residual
risk — forward uses int8 mmq while backward uses F16 tensor cores with F32 accumulate — with
BF16 or chunked-F32-SGEMM as the pre-scoped contingency.

## What to do

All ggml changes land in the vendored llama.cpp fork via the S0-02 two-repo flow.

1. **In `ggml_cuda_out_prod` (`vendor/llama.cpp/ggml/src/ggml-cuda/out-prod.cu:27`)**, add a
   non-F32-src0 path: dequantize src0 to F16 via `ggml_get_to_fp16_cuda`
   (`vendor/llama.cpp/ggml/src/ggml-cuda/convert.cu:764-820`) into a `ggml_cuda_pool`
   buffer — this covers every quant type the CUDA backend supports, with TQ1_0/TQ2_0
   excepted exactly as in MUL_MAT (the converter returns `nullptr` for them; gate
   `supports_op` on a non-null converter). Convert src1 F32→F16 into a second pool buffer,
   then run `cublasGemmEx` with F16 A/B and **F32 C + `CUBLAS_COMPUTE_32F`**.
2. **Numerics pinning:** copy the alpha/beta/type plumbing of the cuBLAS traits structs, but
   hard-code the F32-output configuration of the `prefer_f32_output` branch
   (`ggml-cuda.cu:1425-1440`) unconditionally — never the default F16 traits
   (`CUBLAS_COMPUTE_16F`, `:1313`). Implement inside `out-prod.cu` rather than calling
   `ggml_cuda_mul_mat_cublas_impl` (out_prod's operand flip and chunking differ), with a
   provenance comment naming the source lines.
3. **Keep the transposed-src1 op-flip** (`vendor/llama.cpp/ggml/src/ggml-cuda/out-prod.cu:62-65`):
   in the frozen-weight case src1 is a transposed grad view, and the existing F32 path
   already flips the cuBLAS op instead of materializing a transpose — preserve that for the
   F16 path (convert src1 respecting its actual layout).
4. **Chunk the reduction axis ne01 with `beta=1` accumulation:** split the GEMM into ne01
   chunks, dequantizing one chunk of src0 rows at a time and accumulating into dst
   (`beta=1` after the first chunk), so the F16 transient is capped. Sizing per the
   manifest: the lm_head case (128k×4096) is ≈ 1 GB F16 if dequantized whole; per-layer
   weights (4096×4096) are ~32 MB. Make the chunk-size threshold an internal constant tests
   can lower to force multi-chunk execution.
5. **C2 — F16/BF16 src0 + F16 src1:** F16 src0 passes through with no dequant; BF16 src0
   uses `CUDA_R_16BF` A/B types — note the BF16 traits already use `CUBLAS_COMPUTE_32F`
   (`vendor/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:1297-1308`, compute type at `:1299`).
   Accept F16 src1 (test cases generate `type_b` ∈ {F32, F16}).
6. **Encode the design reasoning in a code comment** (manifest requirement): why not mmq
   (reduction axis orthogonal to quant blocks, `mmq.cuh:17-45`) and why not a fused kernel
   (deferred, backlog B-01) — so future readers do not "optimize" this into a broken design.
7. **Flip `supports_op`** (`ggml-cuda.cu:4689-4690`): accept src0 quantized/F16/BF16 where
   `ggml_get_to_fp16_cuda` is non-null, src1 F32/F16, dst F32.
8. **Tests:** the existing quantized `test_out_prod` cases light up on CUDA once the gate
   flips (`vendor/llama.cpp/tests/test-backend-ops.cpp:8780-8806`, over `base_types` at
   `:7740-7750`). Run `test` (forward parity vs CPU) and `grad` (MODE_GRAD) per quant type
   against the CPU oracle (`vendor/llama.cpp/ggml/src/ggml-cpu/ops.cpp:4363`,
   `ggml_compute_forward_out_prod_q_f32`). Add a case that forces the chunked path.
   F16/BF16-src0 comparisons need the CPU F16 path from S1-18 (the stock CPU F16 case
   aborts, `ops.cpp:4487-4491`) — stage ordering guarantees it is done.
9. **Precision check per ROADMAP §12 Q2:** record in the fork PR the MODE_GRAD margins for
   the worst quant type; if outside ADR-0002 bounds, apply the pre-scoped contingency (BF16
   A/B or chunked F32 SGEMM) and document the choice. Full convergence-relevant validation
   happens at S3-10's gate.
10. **Submodule bump PR** in llama-farm per S0-02, appending `OUT_PROD` coverage to the
    ci-cuda targeted default list.

## Out of scope

- Fused dequant+rank-k quantized OUT_PROD tile kernels — backlog B-01 (profiling-triggered).
- `OUT_PROD_ID` / `OUT_PROD_ID_GRP` MoE variants — S3-08 (reuses this plumbing).
- Metal/Vulkan OUT_PROD ports — S2-05/S2-06, S4-02/S4-03.
- CPU F16/BF16 out_prod fix — S1-18 (consumed here as the oracle).
- `GET_ROWS_BACK` generalization (ROADMAP §5 C5) — dormant; backlog B-05.
- HIP/MUSA validation — ride along via the cuBLAS wrappers; confirming hipBLAS GemmEx
  F16/F16/F32 on RDNA3/CDNA belongs to the future ROCm CI tier (ROADMAP §5, §11).

## Acceptance criteria

- [ ] Fork branch: `test-backend-ops test -b CUDA0 -o OUT_PROD` passes for every quantized
      `base_types` case plus F16/BF16 src0 and F16 src1 cases.
- [ ] Fork branch: `test-backend-ops grad -b CUDA0 -o OUT_PROD` (MODE_GRAD) passes vs the
      CPU oracle within the ADR-0002 tolerances (per-op bound; ≤ 0.05 max-abs @ fp16
      cross-backend parity criterion).
- [ ] A test exercises the multi-chunk ne01 path (chunk threshold lowered) and matches the
      single-chunk result bitwise or within ADR-0002 per-op tolerance.
- [ ] The new path contains no `CUBLAS_COMPUTE_16F` and no f16-accumulate GemmEx
      configuration (grep-verifiable in the fork diff).
- [ ] The why-not-mmq / why-not-fused design comment exists in `out-prod.cu`.
- [ ] `supports_op` rejects TQ1_0/TQ2_0 src0 (converter nullptr) — covered by a `support`
      mode check or unit assertion.
- [ ] llama-farm submodule-bump PR is green in `ci-cuda` (compile + GPU lanes) and `ci-cpu`.

## Testing & verification

Primary harness: vendored `tests/test-backend-ops` — `test` and `grad` modes on the CUDA
backend vs the CPU oracle, per ADR-0002 (S0-09). Runs on the fork branch CI and, after the
submodule bump, in llama-farm's `ci-cuda` GPU lane per-PR (kernel-gated targeted `-o
OUT_PROD`) and in the nightly full sweep (S3-01). The convergence-relevant precision
question (ROADMAP §12 Q2) is finally settled by S3-10's convergence gate on `--device cuda`;
this ticket records MODE_GRAD margins as the leading indicator.

## PR notes

- Branch: `ticket/S3-02-cuda-quantized-out-prod-cublas`.
- Two-repo flow per S0-02: implementation PR against the fork's `llama-farm-base` branch
  with the ticket ID in the title, plus a trivial llama-farm submodule-bump PR referencing
  the same ticket ID.
- Upstreaming disposition: **upstream-early** (ROADMAP §11 triage class a) — mainline
  training benefits directly and the quantized `test_out_prod` cases already exist upstream;
  no new op enums or ABI.
- Provenance headers per S0-01 policy: plumbing adapted from
  `ggml/src/ggml-cuda/ggml-cuda.cu` (`ggml_cuda_mul_mat_cublas_impl`) and
  `ggml/src/ggml-cuda/out-prod.cu` (MIT, commit `4f37f51`).
