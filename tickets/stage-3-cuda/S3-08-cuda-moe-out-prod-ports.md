---
id: S3-08
title: "CUDA MoE: OUT_PROD_ID + OUT_PROD_ID_GRP ports"
stage: 3
track: kernels
size: L
deps: [S3-02, S1-26, S1-27]
status: open
pr: null
---

# S3-08 — CUDA MoE: OUT_PROD_ID + OUT_PROD_ID_GRP ports

**One-line outcome:** MoE LoRA training is GPU-resident on CUDA — expert-compacted
quantized outer products (`OUT_PROD_ID`) and grouped F32 expert grads
(`OUT_PROD_ID_GRP`) run on the GPU, deterministic by default, MODE_GRAD-parity-checked
against their CPU oracles.

## Why (context)

MoE training is exactly one missing backward case (`MUL_MAT_ID`) that decomposes into
these two ops (ROADMAP §0 finding 3, §9 E2/E3). Both are required even for LoRA-only
training: `build_lora_mm_id` computes `mul_mat_id(B, mul_mat_id(A, cur, ids), ids)` —
the trainable LoRA A/B stacks are themselves the 3D expert operand of `mul_mat_id`
(`vendor/llama.cpp/src/llama-graph.cpp:1438-1442`), so the "activation-grads-only"
shortcut that suffices for dense frozen bases does not suffice for MoE. S1-25/26/27
delivered the backward wiring and the CPU reference kernels; this ticket ports both ops
to CUDA so the MoE backward stops falling back to CPU via `ggml_backend_sched`.

Every ingredient exists in-tree. Expert compaction: `mm_ids_helper`
(`vendor/llama.cpp/ggml/src/ggml-cuda/mmid.cu:28-60`) converts `ids` into compact
per-expert `ids_src1`/`ids_dst` permutations plus an `expert_bounds` prefix array
(written at `mmid.cu:109-115`; public launcher `ggml_cuda_launch_mm_ids_helper`,
`mmid.cu:138`). Column gather/scatter: the generic `ggml_cuda_mul_mat_id` path
(`vendor/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:1773`) already gathers src1 columns
into expert-sorted order with `get_rows_cuda` (`:1869`) and scatters dst back (`:1924`).
GEMM plumbing: S3-02 landed dequant-to-F16 via `ggml_get_to_fp16_cuda`
(`vendor/llama.cpp/ggml/src/ggml-cuda/convert.cu:764-820`) plus `cublasGemmEx` pinned to
F32-output/`CUBLAS_COMPUTE_32F` in `ggml_cuda_out_prod`
(`vendor/llama.cpp/ggml/src/ggml-cuda/out-prod.cu:27`) — reuse that plumbing per expert
segment. As in S3-02, mmq tiles are unusable here: the reduction axis is orthogonal to
the quant-block axis (`vendor/llama.cpp/ggml/src/ggml-cuda/mmq.cuh:17-45`).

Determinism is gate G-B (S0-09/ADR-0002): atomicAdd scatter over ragged expert segments
is nondeterministic, so both ops use deterministic segmented accumulation — per-expert
segments executed in fixed order on one stream, exclusive writes — with atomics only as
a later measured opt-in. The launch strategy for many small segments is the open
question Q6 (ROADMAP §12): per-expert cuBLAS launches vs a custom segmented kernel vs
`cublasGemmGroupedBatched` (which raises the minimum CUDA toolkit version). This ticket
benchmarks and records the decision.

## What to do

All ggml changes land in the vendored llama.cpp fork via the S0-02 two-repo flow.

1. **New files** `vendor/llama.cpp/ggml/src/ggml-cuda/out-prod-id.cu`/`.cuh` hosting
   both ops; dispatch cases in `ggml-cuda.cu` next to `GGML_OP_OUT_PROD` (compute switch
   `:2125`, supports_op `:4689`).
2. **Compaction pass:** call `ggml_cuda_launch_mm_ids_helper` (`mmid.cu:138`) to build
   `ids_src1`/`ids_dst`/`expert_bounds` in pool buffers, exactly as the mmf/mmq forwards
   do. Reuse, do not fork, the helper.
3. **`OUT_PROD_ID(as, grad, ids) → dB`** (semantics per the S1-26 op contract, including
   the `s' = s % ne_b1` broadcast rule): per expert `e` with a non-empty
   `expert_bounds` segment — gather the segment's grad columns into a contiguous pool
   buffer (pattern: the `get_rows_cuda` gather, `ggml-cuda.cu:1869`), dequantize expert
   `e`'s matrix to F16 via `ggml_get_to_fp16_cuda` into a pool buffer, run one
   `cublasGemmEx` segment with F16 A/B and **F32 C + `CUBLAS_COMPUTE_32F`** (S3-02's
   pinned configuration — never the F16-accumulate traits), then scatter-add the result
   columns into dst. Segments execute sequentially on the ctx stream, so broadcast
   accumulation (`ne_b1 == 1`: several slots of one token adding into one grad column)
   has a fixed order — document this determinism argument in a comment. Zero-init dst
   first. Type coverage mirrors S3-02's supports_op set (converter non-null; TQ1_0/TQ2_0
   excluded).
4. **`OUT_PROD_ID_GRP(b, grad, ids, n_expert) → dAs`** (semantics per S1-27; dst
   `[n_in, n_out, n_expert]`, F32-only assert): segmented F32 GEMMs via
   `expert_bounds` — for each expert, one Sgemm accumulating the outer products of its
   gathered column pairs into `dAs[:, :, e]` (plain-Sgemm pattern:
   `out-prod.cu:121`). Each expert slab is written exclusively by its own segment —
   deterministic by construction. `cudaMemsetAsync` the whole dst so empty experts are
   exactly zero. Not gated on quantized-OUT_PROD machinery (pure F32, ROADMAP §11) —
   may land first within the PR sequence.
5. **Q6 benchmark (ROADMAP §12):** measure the per-expert-launch baseline against (a) a
   custom segmented kernel and/or (b) `cublasGemmGroupedBatched`, at `n_expert` 64-256
   with fewer than 8 tokens per expert (the ragged small-segment regime). Keep the
   per-expert cuBLAS loop as the v1 correctness baseline regardless; adopt an
   alternative only if the numbers justify it, and record the decision, the numbers,
   and any minimum-CUDA-version consequence in the fork PR.
6. **supports_op:** `GGML_OP_OUT_PROD_ID` accepts quantized/F16/BF16 `as` where the
   converter is non-null, F32 grad/dst; `GGML_OP_OUT_PROD_ID_GRP` accepts F32 only.
7. **Tests:** re-run the S1-26/S1-27 MODE_GRAD case lists on CUDA — the grad-enabled
   `test_mul_mat_id` cases (`vendor/llama.cpp/tests/test-backend-ops.cpp:4248`,
   instantiations `:8693-8747` including the quantized type sweep): quant types,
   broadcast on/off, ragged assignment, empty experts, and S1-27's nested
   `build_lora_mm_id`-shaped two-level case. Add a fork-side determinism check: two
   identical runs produce bitwise-identical dB/dAs.
8. **e2e:** run S1-28's tiny-MoE training test with `--device cuda`; the S3-01 fallback
   report must show the MoE backward ops on the GPU.
9. **Submodule bump PR** in llama-farm per S0-02, appending both ops to the ci-cuda
   targeted-op defaults.

## Out of scope

- Metal/Vulkan ports — S2-11 / S4-06 (same op contracts, backend-native compaction).
- Atomics-based opt-in variants (gate G-B allows them later, measured) — backlog.
- Fused quantized outer-product tile kernels — backlog B-01 (profiling-triggered).
- Per-expert bias grads, router full training, quantized-expert full FT — ROADMAP §9
  E8, deferred.
- The CPU kernels and backward wiring — S1-25/26/27 (consumed here as oracles).

## Acceptance criteria

- [ ] Fork branch: `test-backend-ops grad -b CUDA0` passes for the `OUT_PROD_ID`
      (`test_mul_mat_id` activation-grad) cases across the quantized type sweep plus
      F32/F16, broadcast on/off, ragged and empty-expert cases, within the ADR-0002
      tolerance vs the S1-26 CPU oracle (≤ 0.05 max-abs @ fp16 parity criterion).
- [ ] Fork branch: the `OUT_PROD_ID_GRP` (F32 `as`-as-param) cases pass on CUDA,
      including the nested `build_lora_mm_id`-shaped case; empty-expert slabs are
      exactly zero.
- [ ] Determinism: two identical CUDA runs produce bitwise-identical dB and dAs.
- [ ] The new paths contain no `CUBLAS_COMPUTE_16F`/F16-accumulate configuration
      (grep-verifiable in the fork diff).
- [ ] The Q6 benchmark table (launch strategies, n_expert 64-256, <8 tokens/expert) and
      the recorded decision are in the fork PR description.
- [ ] Tiny-MoE e2e passes with `--device cuda`; its fallback report shows
      `OUT_PROD_ID`/`OUT_PROD_ID_GRP` executing on CUDA.
- [ ] llama-farm submodule-bump PR is green in `ci-cuda` (compile + GPU lanes) and
      `ci-cpu`.

## Testing & verification

Primary harness: vendored `tests/test-backend-ops` MODE_GRAD on the CUDA backend vs the
S1-26/S1-27 CPU oracles under the ADR-0002 tolerances (S0-09), plus the fork-side
determinism check. Runs on the fork branch CI and, after the submodule bump, in
llama-farm's `ci-cuda` GPU lane per-PR (kernel-gated targeted op list) and the nightly
full sweep (S3-01). The tiny-MoE e2e joins the nightly `ci-cuda` e2e job; the S3-10
milestone later folds these ops into the fallback-forbidden set.

## PR notes

- Branch: `ticket/S3-08-cuda-moe-out-prod-ports`.
- Two-repo flow per S0-02: implementation PR against the fork's `llama-farm-base`
  branch with the ticket ID in the title, plus a trivial llama-farm submodule-bump PR
  referencing the same ticket ID.
- Upstreaming disposition: **fork-local first, upstream-later** — new op enums ride the
  E2/E3 op-family RFC once the CPU oracle plus one GPU backend prove the design
  (ROADMAP §11 triage b); coordinate with S2-11 on which backend anchors the RFC.
- Provenance headers per S0-01 policy: adapted in-tree MIT code (`mmid.cu` compaction
  usage, `out-prod.cu`/S3-02 cuBLAS plumbing, `get_rows_cuda` gather; llama.cpp
  `4f37f51`).
