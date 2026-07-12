# llama-farm — Kernel Roadmap: GPU-Resident LoRA Training on CPU, CUDA, Metal, and Vulkan

**Goal:** the complete kernel-change plan that turns ggml into a full training system for this project's target configuration — **frozen quantized GGUF base weights, F32 LoRA A/B as the only trainable parameters** — with the entire training step (forward + backward + AdamW) running **GPU-resident when a GPU is available**, on CUDA, Metal (Apple Silicon), and Vulkan, with CPU as both a first-class platform and the correctness oracle. Includes flash-attention training, MoE and SSM architecture support, and (final section, per project goal) future improvements drawn from permissibly-licensed unsloth components.

**Status:** design document (no implementation), companion to `GGUF-LORA-TRAINING-BLUEPRINT.md` (referenced below as "the blueprint"; its phases are P0–P4, its design decisions D1–D7). Grounded in kernel-level reads of `llama.cpp` @ 4f37f51 and `unsloth` in this directory; file:line citations point into those checkouts and were adversarially spot-checked.

---

## 0. Executive summary

| Backend | Distance from GPU-resident training | The work |
|---|---|---|
| **CPU** | complete today; the only backend with full training-op coverage | F16 `OUT_PROD` fix; reference kernels for every new op (§4) |
| **CUDA** | **one op away** | quantized `OUT_PROD` via existing dequant + `cublasGemmEx` machinery (M); sparse CE (M); everything else already exists |
| **Vulkan** | **one op away + loss** | it already has every `*_BACK` kernel and both optimizer steps; needs `OUT_PROD` (M — reformulates onto the existing tuned `mul_mm` pipeline) and CE (M) |
| **Metal** | needs the backward suite | only `ROPE_BACK` + `OPT_STEP_*` exist today; ~8 new kernels, every one with a clean in-tree pattern; quantized `OUT_PROD` (L) is the critical path |

Three findings change the shape of the work versus naive expectations:

1. **No new quantized GEMM kernel is needed anywhere.** On CUDA the right answer is dequant-to-F16 + `cublasGemmEx`, reusing the plumbing of `ggml_cuda_mul_mat_cublas_impl` (`ggml-cuda.cu:1324-1536`) while **forcing its F32-output / `CUBLAS_COMPUTE_32F` variant** (the `prefer_f32_output` branch — the function's default F16 path accumulates in F16, which violates this project's numerics policy). A custom quantized kernel is structurally hopeless: the `OUT_PROD` reduction axis is *orthogonal* to the quantization block axis, making the int8 mmq/mmvq tiles unusable (`mmq.cuh:17-45`). On Vulkan, `out_prod(a,b) = mul_mat(contᵀa, contᵀb)` — and in the frozen-weight case `b` is already a transposed grad view, so `contᵀb = grad` is contiguous: the whole tuned coopmat `mul_mm` pipeline is reusable with just a dequant + transpose-copy prologue. Only Metal warrants a real new tiled kernel, and even there `simdgroup_load` has a native transpose flag (`ggml-metal.metal:10021`) so the existing `kernel_mul_mm` dequant phase carries over.
2. **Flash-attention backward is much closer than "no backward exists" suggests.** Every FA forward on every backend already computes the log-sum-exp ingredients (running max + sum) and throws them away (CPU `ops.cpp:8496-8497`; CUDA `dst_meta`, `fattn-common.cuh:29`; Metal `[S,M]` partials, `ggml-metal-ops.cpp:2640-2644`; Vulkan `flash_attn_base.glsl:156`) — an LSE-emitting forward is mechanical. And a complete (outdated) CPU FA backward with the full derivation and a deterministic no-atomics parallelization already exists in-tree at `ops.cpp:9156-9488`; it needs modernizing, not inventing. Because llama.cpp bakes causal/padding/SWA/ALiBi into one additive mask (F16 on the FA path, F32 otherwise; `llama-graph.cpp:406-453`), **one** backward kernel signature covers all model variants.
3. **MoE training is exactly one missing backward case** (`MUL_MAT_ID`) that decomposes into two new ops — an expert-indexed outer product (`OUT_PROD_ID`) and a grouped expert outer product (`OUT_PROD_ID_GRP`) — both reusing existing expert-compaction infrastructure (CPU row-grouping `ggml-cpu.c:1574-1637`, CUDA `mm_ids_helper` `mmid.cu:28-60`). A key discovery: LoRA A/B on experts are themselves `mul_mat_id` operands (`llama-graph.cpp:1438-1442`), so the "activation-grads-only" shortcut is *not* sufficient for MoE — the grouped weight-grad op is required even for LoRA-only training.

**Sizing legend** (per engineer already familiar with the backend): **S** ≤ 2 days · **M** ≤ 1 week · **L** 2–3 weeks · **XL** 4+ weeks. Compound ratings ("S/M") reflect scope uncertainty noted in the item.

**Glossary:** *VJP* — vector-Jacobian product, an op's backward rule in `ggml_compute_backward`. *LSE* — log-sum-exp, the per-row softmax normalizer `max + log Σ exp(x−max)`. *MODE_GRAD* — `test-backend-ops`' finite-difference gradient-checking mode. *ubatch* — micro-batch, the per-graph-execution token slice. *mmq/mmvq* — CUDA's int8 quantized matmul/matvec kernel families. *coopmat* — Vulkan cooperative-matrix (tensor-core) extensions. *GQA/SWA/ALiBi* — grouped-query attention / sliding-window attention / attention-with-linear-biases. *MLA* — DeepSeek-style multi-head latent attention (head sizes up to 576).

---

## 1. The training op set

A dense-transformer training step (forward + backward + optimizer, non-FA attention) emits, beyond ordinary forward ops (`MUL_MAT`, `MUL/ADD/SUB/DIV`, `SCALE`, `SUM/SUM_ROWS/MEAN`, `REPEAT`, `SQR/SQRT/LOG/EXP`, `CPY/CONT/RESHAPE/VIEW/PERMUTE/TRANSPOSE`, `ACC`, `ADD1`, `GET_ROWS`, `GLU`):

- **`OUT_PROD`** — `MUL_MAT` backward w.r.t. both operands (`ggml.c:6578-6630`). The frozen-weight case is `out_prod(W_quantized, transpose(grad))`; the LoRA A/B grad case is F32×F32. Hit on **every linear layer, every microbatch** — this is the op that decides whether backprop lives on the GPU.
- **`SOFT_MAX_BACK`** (naive-attention backward; `max_bias==0` restriction on CPU/CUDA), **`RMS_NORM_BACK`**, **`ROPE_BACK`**, **`SILU_BACK`** (split-SWIGLU path), **`REPEAT_BACK`**, **`GET_ROWS_BACK`** (dormant while embeddings are frozen), **`CROSS_ENTROPY_LOSS(_BACK)`** (to be superseded by the sparse op), **`OPT_STEP_ADAMW` / `OPT_STEP_SGD`**.
- **New op (blueprint design decision D4):** `ggml_cross_entropy_loss_sparse(logits, i32_labels, f32_weights)` — per-token loss with backward `w·(softmax − onehot)`; needed because the existing dense CE backward produces mathematically wrong gradients for masked (all-zero-label) rows.

## 2. Coverage matrix today

| Op | CPU | CUDA | Metal | Vulkan |
|---|---|---|---|---|
| `OUT_PROD` F32 | ✅ | ✅ (cuBLAS, `out-prod.cu`) | ❌ | ❌ |
| `OUT_PROD` quantized src0 | ✅ (`ops.cpp:4363-4501`; F16 aborts) | ❌ (`ggml-cuda.cu:4689`) | ❌ | ❌ |
| `SOFT_MAX_BACK` | ✅ (max_bias==0 assert) | ✅ (max_bias==0 gate) | ❌ | ✅ (`soft_max_back.comp`; no max_bias gate — kernel ignores it, unvalidated) |
| `RMS_NORM_BACK` | ✅ | ✅ (`norm.cu:158`) | ❌ | ✅ (`rms_norm_back.comp`) |
| `SILU_BACK` | ✅ | ✅ | ❌ | ✅ |
| `ROPE_BACK` | ✅ | ✅ (shared fwd kernel) | ✅ (`is_back` function constant) | ✅ (reuses fwd pipelines) |
| `GET_ROWS_BACK` | ✅ | ⚠️ (F32, no batch dims) | ❌ | ✅ (deterministic scan) |
| `REPEAT_BACK` | ✅ | ✅ (src0 ne2·ne3 ≤ 2¹⁵) | ❌ | ✅ |
| `CROSS_ENTROPY_LOSS(_BACK)` | ✅ | ✅ | ❌ | ❌ |
| `OPT_STEP_ADAMW/SGD` | ✅ | ✅ | ✅ (`metal:10874-10920`) | ✅ (`opt_step_adamw.comp`) |
| `FLASH_ATTN_EXT` backward | ❌ everywhere (legacy CPU impl exists but constructor aborts, `ggml.c:5470`) | | | |
| `MUL_MAT_ID` / `SSM_SCAN` / `SSM_CONV` backward | ❌ everywhere | | | |

(Metal/Vulkan coverage decided in `ggml-metal-device.m:1051-1368` and `ggml-vulkan.cpp:17158-17719`. Everything not listed — elementwise VJP ops, reductions, views — is present on all four backends, with two small exceptions the plans below own: `ADD1` is missing on Metal (M10) and `DIAG_MASK_ZERO` is CPU-only (M10/V4).)

---

## 3. Shared cross-backend items

### K-CE: `ggml_cross_entropy_loss_sparse` (fwd + bwd on all four backends)
The canonical training loss for SFT/DPO/GRPO (blueprint D4). Design, informed by unsloth's Apache-licensed CE kernel (§13, item 1):
- **Forward:** per-row stable logsumexp (`lse = max + log Σ exp(x−max)`); per-token loss `w·(lse − x_label)`; `w=0` encodes ignore. For very wide vocab rows, per-chunk lse into an `(n_rows, n_chunks)` scratch reduced by log-add-exp — the decomposition is fully documented in the Apache-side unsloth kernel (`cross_entropy_loss.py:87-150`).
- **Backward:** `dlogits = dloss · w · (exp(x − lse) − onehot)`; exactly zero where `w=0` (this fixes the wrong-gradient masking problem verified in the blueprint).
- **Cross-backend ABI decision needed up front (decide-first gate G-A, §11):** whether forward stashes the per-row lse (an extra `n_tokens` F32 output) so backward avoids re-reducing the vocab row, and whether backward may alias the logits buffer (unsloth's in-place trick) — ggml's graph allocator needs a prototype for the aliasing question. Optional op-params from day one: logit softcap `t·tanh(x/t)` (backward factor `1−tanh²`) and logit scale — cheap now, painful to retrofit.
- Per-backend patterns: CPU `ops.cpp:11158+` (the oracle — see §4); CUDA `cross-entropy-loss.cu:8-92` (block-per-row, shared-mem cache with global fallback — no hard vocab limit); Metal `kernel_soft_max` grid-stride + `simd_sum` (`metal:1896-1999` — only 128 B threadgroup memory at any vocab size); Vulkan `soft_max.comp` column loop + `soft_max_back.comp` shared-memory tree reduction (**no subgroup arithmetic** — force-disabled on MoltenVK+AMD, `ggml-vulkan.cpp:5983-5994`).
- Once this lands on all backends, the dense `CROSS_ENTROPY_LOSS` op does **not** need Metal/Vulkan ports (skip recommended).

### K-SMB: lift the `SOFT_MAX_BACK` `max_bias==0` restriction (ALiBi) — S
The ALiBi bias is additive and constant w.r.t. logits, so the existing kernels are already mathematically correct. The restriction lives in **two backend asserts only** — CPU (`ops.cpp:5539`) and CUDA (`softmax.cu:469` + supports_op gate `ggml-cuda.cu:4903-4908`); there is no core-level gate, and **Vulkan currently accepts `max_bias>0` completely unvalidated** (its shader ignores the parameter). That inconsistency strengthens the case for adding the finite-difference test with `max_bias>0` *first* (no test exercises this today), then removing the asserts.

### K-F16OP: F16/BF16 `OUT_PROD` — S
CPU F16 `out_prod` currently **aborts** (`ops.cpp:4487-4491`) — today an F16 out_prod is fatal, not just slow (this is why finetune.cpp forces the F32 KV cache). CPU fix in §4; CUDA falls out of the C1 traits path (§5).

### K-TANH: TANH VJP — S
Trivial unary backward case, `grad · (1 − tanh²)`. Owned by this roadmap (the blueprint's P4 defers to it): unlocks gemma2/3 logit-softcap training and removes FA8's manual-tanh-node workaround (§8). Kernel-free — the VJP composes from existing ops; only the `ggml_compute_backward` case is new.

### Numerics policy (applies to every item)
Gradient matmuls use **F32 accumulation** everywhere (Vulkan `mul_mm` f16acc variants are inference-only; CUDA must force `CUBLAS_COMPUTE_32F` — see C1); row stats/lse/loss in F32; elementwise math in F32 with casts at storage boundaries. Acceptance: finite-difference checks in `test-backend-ops` MODE_GRAD (harness exists, incl. expected-value filtering for discontinuities, `test-backend-ops.cpp:320`), CPU as oracle; adopt max-abs gradient error ≤ 0.05 at fp16 as the cross-backend parity criterion (unsloth uses a similar 0.05 threshold in its self-tests). Determinism preferred over atomics in all backward kernels (reproducible training runs); atomics-based variants only as measured, opt-in optimizations.

---

## 4. CPU plan (first-class platform + oracle)

CPU is the only backend with full training-op coverage today, and every new op lands here first as the MODE_GRAD reference. It is also a real training target, not just a fallback: for small models and LoRA ranks, CPU-only training is viable (all ops threaded through the existing ggml threadpool), and via `ggml_backend_sched` the CPU transparently executes any op a GPU backend lacks — which is exactly how training *already works today* on every platform, just slower than it should be.

| # | Item | Notes | Size |
|---|---|---|---|
| P1 | **Sparse CE fwd+bwd reference** | Pattern `ggml_compute_forward_cross_entropy_loss_f32` (`ops.cpp:11158+`, already uses `ggml_vec_log_soft_max_f32`); sparse i32 labels + f32 weights; per-token output. The oracle every GPU CE validates against — lands in K0 before any GPU CE. | **M** |
| P2 | **K-F16OP:** F16/BF16 `out_prod` (replace the abort at `ops.cpp:4487-4491` with a to_float row path) | Removes the forced-F32-KV-cache constraint at the source. | **S** |
| P3 | Reference kernels for every new op in later sections — FA backward modernization (FA3, §8), `OUT_PROD_ID`/`OUT_PROD_ID_GRP` (E2/E3, §9), `SSM_CONV_BACK`/`SSM_SCAN_BACK` (S2/S3, §10) | Sized within their sections; scalar-first (SIMD variants only if profiling of CPU-only training justifies them). | — |
| P4 | CPU-only training throughput audit | One benchmark pass over the dense-LoRA backward on representative CPUs: confirm the dequant-per-row `out_prod` path and threadpool scaling are adequate, and publish an expected tok/s class so GPU-less users can size runs. | **S** |

---

## 5. CUDA plan

CUDA is one op away from a fully GPU-resident dense-LoRA training step. Residual walk of the op set vs `supports_op` (`ggml-cuda.cu:4575-4977`) leaves exactly: quantized/F16 `OUT_PROD`, the new sparse CE, `SOFT_MAX_BACK` ALiBi, and (dormant) `GET_ROWS_BACK` limits. Everything else is already GPU-resident.

| # | Item | Approach | Size |
|---|---|---|---|
| C1 | **Quantized-src0 `OUT_PROD`** | In `ggml_cuda_out_prod` (`out-prod.cu:27`): dequant src0 via `ggml_get_to_fp16_cuda` (covers every quant type the CUDA backend supports — TQ1_0/TQ2_0 excepted, as in MUL_MAT; `convert.cu:711-860`) into a pool buffer, convert src1 F32→F16, then `cublasGemmEx` with F16 A/B and **F32 C + `CUBLAS_COMPUTE_32F`** — the plumbing of `ggml_cuda_mul_mat_cublas_impl` (`ggml-cuda.cu:1324-1536`) but pinned to its `prefer_f32_output` variant (`:1425-1440`) on all arches, because the function's *default* F16 traits accumulate in F16 (`:1311-1323`), which violates the §3 numerics policy. Keep the existing transposed-src1 op-flip (`out-prod.cu:62-65`). Chunk over the reduction axis ne01 with `beta=1` accumulation to cap the transient for the lm_head case (128k×4096 ≈ 1 GB F16); per-layer weights are ~32 MB transients. **Why not mmq:** the reduction axis (src0 *rows*, ne01) is orthogonal to the quant-block axis (ne00) — every dp4a/int-mma tile design assumes blocks lie along the reduction axis (`mmq.cuh:17-45`); reuse would require physically transposing quantized data, i.e., a dequant anyway. **Why not a fused dequant+rank-k kernel:** L effort, no tensor cores, loses to `GemmEx` at training batch sizes; keep only as a later memory optimization if profiling demands (risk R2, §12). | **M** |
| C2 | F16/BF16 src0 + F16 src1 `OUT_PROD` | Falls out of C1's traits path (F16 passthrough; BF16 via `CUDA_R_16BF` — the BF16 traits already use `CUBLAS_COMPUTE_32F`). | **S** |
| C3 | Sparse CE fwd+bwd | Pattern `cross-entropy-loss.cu:8-92`; widen block from `WARP_SIZE` toward 1024 threads with two-level reduction for 128k-vocab rows (pattern `norm.cu:410-417`); backward takes per-row grad (drop the `ggml_is_scalar` assert at `:147`). Sparse is *simpler* than dense: no label row to read. | **M** |
| C4 | `SOFT_MAX_BACK` ALiBi | K-SMB; no kernel change on CUDA. | **S** |
| C5 | `GET_ROWS_BACK` generalization (batch dims, scatter instead of the O(vocab×tokens) scan, `getrows.cu:80-104`) | Only if trainable embeddings enter scope — dormant for quantized-base LoRA. | **S/M, deferred** |

Verification exists already: `test-backend-ops` generates quantized `test_out_prod` cases (`test-backend-ops.cpp:8780-8806`) that currently skip CUDA via supports_op — flipping the gate lights them up against the CPU reference. HIP/MUSA ride along via the existing cuBLAS wrappers (confirm hipBLAS GemmEx F16/F16/F32 on RDNA3/CDNA — CI matrix, §11).

---

## 6. Metal plan

Metal has only `ROPE_BACK` (a function-constant flag on the forward rope kernels — a pattern worth copying, `ggml-metal-device.cpp:1709-1751`) and correct `OPT_STEP_ADAMW/SGD` kernels. Everything else must be written — but the graph-encode path needs **no structural change**: each op is five mechanical additions (kernel, kargs struct, pipeline getter, encoder, supports_op case), and concurrency tracking is generic.

Relevant capabilities: `has_simdgroup_reduction`/`has_simdgroup_mm` (Apple7+), 32 KB threadgroup memory, and — uniquely — **unified memory**, which makes offloaded activation checkpointing a no-copy operation on Apple Silicon (a genuine advantage for long-context LoRA once checkpointing lands; checkpointing itself is graph-level work owned by the blueprint's P2, not a kernel item here).

| # | Item | Pattern | Size |
|---|---|---|---|
| M1 | `OUT_PROD` F32 | `kernel_mul_mm` 64×32 simdgroup tiling (`metal:9874-10082`) with **`simdgroup_load(..., transpose=true)`** — the transpose flag already exists in the API (currently `false` at `metal:10021,10027`). Unblocks GPU-resident LoRA A/B grads. | **M** |
| M2 | `OUT_PROD` quantized src0 | M1 + the dequant phase copied verbatim from `kernel_mul_mm` phase 1 (`metal:9935-9976`), reusing the per-type `dequantize_*` device functions (`metal:92-960`). Type table as mul_mm (legacy quants + K-quants + iq4 first). **The critical-path item** — "gradients through frozen quantized weights" on Apple Silicon. | **L** |
| M3 | `SOFT_MAX_BACK` | `kernel_soft_max` row-reduction (`metal:1896`); one `simd_sum` for `dot(dy,y)`. | **S/M** |
| M4 | `RMS_NORM_BACK` | `kernel_rms_norm_fuse_impl` (`metal:3058-3117`); two simd_sum reductions per row. | **M** |
| M5 | `SILU_BACK` | binary elementwise (`kernel_bin_fuse_impl`, `metal:1211`). Covers split-SWIGLU backward with existing MUL. | **S** |
| M6 | `REPEAT_BACK` | inverse of `kernel_repeat` (`metal:1395`), one thread per **dst** element loop-accumulating (deterministic, no atomics). | **M** |
| M7/M8 | Sparse CE fwd/bwd | `kernel_soft_max` grid-stride + simd reductions; 128 B shmem at any vocab; i32 label read is one scalar per row. Dense CE: **skip** on Metal. | **M + S/M** |
| M9 | `GET_ROWS_BACK` | deterministic per-dst-row scan (pattern `kernel_set_rows`); atomic-float variant pending MSL verification. Deferred (embeddings frozen). | **S/M, deferred** |
| M10 | `ADD1`, `DIAG_MASK_ZERO` | S each — but first dump a real training graph to confirm they're still emitted (modern graphs mask via soft_max). | **S** |

**Order:** M3 → M4 → M5 (unblock backward through norm/attn/ffn) → M1 → M2 (critical path) → M7/M8 (loss head) → M6 → the rest. The Metal4 tensor API (`matmul2d`) is disabled by default pre-M5 hardware — target the legacy simdgroup path; revisit later.

---

## 7. Vulkan plan

Vulkan is the closest non-CUDA backend: every `*_BACK` op in the training set plus both optimizer steps already exist with tested shaders (see matrix §2). The only hard gaps are `OUT_PROD` (zero references in the backend) and the loss.

| # | Item | Approach | Size |
|---|---|---|---|
| V1 | `OUT_PROD` F32 | New `out_prod.comp` patterned on `mul_mm.comp`'s scalar path with stride push constants for the transposed-grad src1 view; shared-memory tree reductions only (**no subgroup arithmetic** — MoltenVK/AMD compatibility, `ggml-vulkan.cpp:5983-5994`). | **M** |
| V2 | `OUT_PROD` quantized src0 | **Backend-side composition, no new matmul shader:** dequant src0 to F16 with the existing `pipeline_dequant[type]` (`:4996-5018`, exactly as `ggml_vk_mul_mat_q_f16` does with the `prealloc_x` transient), then exploit `out_prod(a,b) = mul_mat(contᵀa, contᵀb)` — and since the frozen-weight case has `b = transpose(grad)`, `contᵀb` **is grad, already contiguous, zero-copy**. Transpose `a` via `copy_transpose.comp` and reuse the ENTIRE tuned `mul_mm` pipeline (coopmat/coopmat2/split-k, all tuning) with F32 accumulation. A fused per-quant-type out_prod tile shader (~25 types × warptile variants) is deferred unless profiling shows the dequant round-trip dominating (risk R2, §12). | **M** (fused alt: L/XL, deferred) |
| V3 | Sparse CE fwd+bwd | fwd: one workgroup per token row, two-pass max/sum-exp column loop (`soft_max.comp` pattern); bwd: near-verbatim `soft_max_back.comp` adaptation. Avoids the `[vocab×tokens]` one-hot tensor, which also relieves `maxStorageBufferRange` gates (`:17165-17186`). Dense CE: skip if sparse CE is canonical. | **M** (bwd S) |
| V4 | `DIAG_MASK_ZERO` | clone `diag_mask_inf.comp`; parity item. | **S** |
| V5 | Constraint audit | contiguity preconditions on `*_BACK` ops vs real backward graphs (graph dump); `ROPE_BACK` returns true unconditionally — tighten or verify mrope/vision modes; add the missing `SOFT_MAX_BACK` max_bias gate (or the K-SMB test) since Vulkan currently accepts it unvalidated. | **S** |

Adding one op costs six mechanical touch points (~1 day plumbing; the shader is the real work). Estimated total: OUT_PROD path 1–2 weeks, CE ~1 week. **MoltenVK note:** it works (scalar mul_mm only — no KHR coopmat) and is a functional stopgap mac path, but native Metal (§6) remains the performance target on Apple Silicon; worth one benchmark to decide whether the stopgap is worth wiring.

---

## 8. Flash attention for training

### Why it matters (the memory cliff)
Without FA backward, training runs the naive `MUL_MAT → SOFT_MAX(mask) → MUL_MAT` path, materializing 2–3 `[n_kv, n_q, n_head]` F32 tensors per layer **simultaneously live across all layers** (ggml-opt has no checkpointing; checkpointing is blueprint-P2 graph work and complements, not replaces, FA backward). For Llama-3.1-8B (32 heads × 32 layers):

| n_ctx | total attention-matrix memory |
|---|---|
| 512 | 2–3 GiB |
| 1024 | 8–12 GiB |
| 2048 | 32–48 GiB |
| 4096 | **128–192 GiB (infeasible)** |

FA backward replaces the n_ctx² term with an LSE vector (~512 KiB/layer at 4k ctx) — it is the difference between "8B LoRA at 4k context on a 24 GB GPU" and "not possible."

### What already exists
- **LSE ingredients are computed and discarded by every forward on every backend** (CPU `ops.cpp:8496-8497` — sinks already folded into M/S, so LSE naturally includes the sink denominator; CUDA `dst_meta = (KQ_max, KQ_sum)`, `fattn-common.cuh:29,916-970`; Metal `[S,M]+O` partials; Vulkan split-k column-zero m/L trick).
- **A complete legacy CPU backward exists** (`ggml_compute_forward_flash_attn_back_f32`, `ops.cpp:9156-9488`) with the reference math in comments (`dS = P·(dP − dot(P,dP))`; `:9337-9398`) and a deterministic no-atomics parallelization (threads own KV heads; GQA loop inside). It's unreachable (constructor aborts) and outdated — F32-only, causal-flag-only, hardcoded scale — but it is the skeleton to modernize, not a from-scratch build.
- **One mask covers everything:** causal, padding, SWA, and ALiBi are all baked into a single additive mask by `fill_mask` (`llama-graph.cpp:406-453`; F16 on the FA path, F32 otherwise), so a backward honoring (mask, scale, softcap, slope, sinks) covers all model variants; the `SOFT_MAX_BACK` `max_bias` restriction doesn't apply here because backward recomputes P directly from Q/K/mask/LSE. Training graphs have no KV cache — **this is the post-fix state, not today's** (see BLUEPRINT §2 G16 / ticket S1-00: causal training graphs currently route K/V through the cache, which severs the gradient edge and aborts backward-graph construction; the `llama-graph.cpp:2416-2422` F32→F16 cast cited here belongs to the *no-cache* path, which serves embedding/non-causal archs). Once the training graph bypasses the cache, K/V arrive as F32→F16 casts, so quantized-KV backward is out of scope by construction and dK flows back through the existing cast (CPY) backward.

### Work items
| # | Item | Size |
|---|---|---|
| FA1 | **ggml API:** `emit_lse` flag on `FLASH_ATTN_EXT` (packed `O‖LSE` dst with accessor views — precedent: the legacy op's packed `dq‖dk‖dv` layout, `ggml.c:5501-5529`); replace the aborted `ggml_flash_attn_back` with `ggml_flash_attn_ext_back(q,k,v,mask,sinks,o,dO,lse,…) → dq‖dk‖dv` | **M** |
| FA2 | **Autograd wiring:** `FLASH_ATTN_EXT` case in `ggml_compute_backward`; mask/sinks marked `ignore_src` (pattern: ROPE positions, `ggml.c:7069-7076`) | **S-M** |
| FA3 | **CPU reference backward:** modernize `ops.cpp:9156-9488` — arbitrary additive mask, op-param scale, ALiBi slope, softcap derivative `(1−tanh²)` recomputed in place, F16 K/V via the forward's vec_dot traits, LSE replacing the per-row softmax recompute. Keep the deterministic parallelization. This is the correctness oracle for all GPU backends. | **M-L** |
| FA4 | **CUDA forward LSE emission:** all four kernel families hold KQ_max/KQ_sum at epilogue; add optional LSE store in `launch_fattn` + combine kernel. Watch collision with the existing F16-scratch dst over-allocation (`fattn-common.cuh:47-85`). | **S** |
| FA5 | **CUDA backward (flagship):** base on the **tile family** (`fattn-tile.cuh:794` + per-arch config tables), not mma, for v1. Three deterministic passes: (1) `delta = rowsum(dO∘O)`; (2) dK/dV — grid over KV tiles, recompute P from Q/K/mask/LSE, exclusive writes; (3) dQ — grid over Q tiles. GQA accumulation is free in pass 2 (KV-head owns its Q heads). Scope v1: F16 K/V, head sizes 64/128 then 256; skip MLA DKQ=576 and sink gradients (sinks frozen in LoRA training). ~1.5–2.5k lines. Later perf phase: mma-family backward, opt-in atomic-dQ single pass. | **XL** |
| FA6 | **Vulkan backward:** scalar `flash_attn.comp` base (subgroup-optional → runs everywhere incl. MoltenVK); same 3-pass scheme; LSE via extending the split-k m/L path. | **L-XL** |
| FA7 | **Metal backward:** `kernel_flash_attn_ext_impl` simdgroup base; schedule **after** Metal's basic backward suite (§6) — FA backward is not Metal's critical path. | **L-XL** |
| FA8 | **Parallel-track kernel-free fallback:** graph-level chunked attention backward — per Q-chunk recompute `soft_max_ext(K·Qc)` and emit `SOFT_MAX_BACK`/`MUL_MAT`/`OUT_PROD` gradient ops manually; peak attention memory drops by the chunk factor at +~50% attention FLOPs. Needs `OUT_PROD` per backend (already planned); softcap uses K-TANH once landed (interim: a manual `1−tanh²` graph node). No ALiBi until K-SMB. **Unblocks 2–4k ctx training before FA5 lands.** | **M** |

**Phasing** (expressed in the §11 K-phases): FA8 rides with K0/K1 as the interim long-context path → FA1+FA2+FA3+FA4 open K3 (FA training correct on CPU; mixed fwd-GPU/bwd-CPU graphs via sched splits) → FA5 completes K3 (full GPU-resident FA training on CUDA) → FA6/FA7 land in K4 → mma backward, atomic-dQ, quantized-KV dequant, MLA and sink grads are K5 perf work.

---

## 9. MoE architectures

Blocked by exactly one missing backward case: `MUL_MAT_ID` (aborts at `ggml.c:6906`). Node-by-node analysis of `build_moe_ffn` (`llama-graph.cpp:1799-2148`) shows the router path needs **no new kernels** for LoRA training: argsort/top-k indices are I32 and automatically excluded from gradients (`ggml.c:7049-7051` — the DeepSeek expert-group branch is entirely dead for grads), and gradients correctly flow through the top-k **weights** path (softmax/get_rows/div/sum_rows — all have VJPs).

**Key discovery:** `build_lora_mm_id` computes `mul_mat_id(B, mul_mat_id(A, cur, ids), ids)` — the trainable LoRA A/B tensors are themselves the 3D expert operand of `mul_mat_id`. So MoE LoRA needs the weight-gradient op too, not just activation grads.

| # | Item | Size |
|---|---|---|
| E1 | `MUL_MAT_ID` backward case emitting E2 + E3 (pattern: `MUL_MAT` case `ggml.c:6578-6630`); handle src1 broadcast (common case `ne_b1==1` simplifies to per-token accumulation) | **S** |
| E2 | **New op `OUT_PROD_ID(as, grad, ids) → dB`** — activation grads through frozen quantized experts. CPU: dequant-per-row axpy (`ops.cpp:4363-4501`) + the mmid per-expert row-grouping workspace (`ggml-cpu.c:1574-1637`). CUDA: `mm_ids_helper` compaction (`mmid.cu:28-60`) → gather grad columns per expert → dequant expert to pool → per-expert cuBLAS segments (pattern `out-prod.cu`). Metal/Vulkan: blocked on the base `OUT_PROD` ports (§6 M2 / §7 V2); reuse their dequant machinery. | **M** CPU, **L** CUDA, **L** Metal/Vulkan |
| E3 | **New op `OUT_PROD_ID_GRP(b, grad, ids, n_expert) → dAs`** — grouped expert outer product for F32 LoRA A/B grads (F32-only assert; frozen quantized experts never take this path). Segmented GEMM per expert via `expert_bounds`; **not** blocked on the OUT_PROD ports (pure F32). | **M** per backend |
| E4 | `ADD_ID` backward (src0 identity) — graph-level | **S** |
| E5 | `CLAMP` backward composite — `grad · step(scale(x,1,−min)) · step(scale(x,−1,max))`, all existing ops (needed by `norm_w` and clamped-swiglu paths) | **S** |
| E6 | `SIGMOID` backward composite — `grad·(y − y²)` (DeepSeek-V3/GPT-OSS sigmoid routers) | **S** |
| E7 | `SWIGLU_OAI` backward (GPT-OSS experts) + GEGLU/REGLU backward (GELU-gated MoE; GEGLU derivatives from §13 item 4) | **M** |
| E8 | MoE-full extras (per-expert bias grads via scatter-add; router full training; quantized-expert full FT is an explicit **non-goal**) | deferred |

Determinism question to settle early (decide-first gate G-B, §11): atomicAdd scatter for ragged expert segments makes LoRA grads nondeterministic — default to the deterministic segmented-GEMM path, offer atomics as opt-in.

---

## 10. SSM architectures (Mamba-1/2 family)

All four LoRA-able projections (`ssm_in`, `ssm_x`, `ssm_dt`, `ssm_out` — `mamba-base.cpp:45,86,104,140`) are plain `MUL_MAT` through `build_lora_mm`, already covered by the base `OUT_PROD` work. What's missing is gradient flow **through** the SSM ops (A, D, conv1d, dt_bias stay frozen — no param grads needed):

| # | Item | Size |
|---|---|---|
| S1 | `CONCAT` backward — two grad views; graph-level (conv-state concat, `mamba-base.cpp:55`; cache-side src needs no grad) | **S** |
| S2 | **New op `SSM_CONV_BACK`** (d_sx) — correlation with flipped window, same row-parallel structure as forward (`ops.cpp:9492-9543`; CUDA `ssm-conv.cu`; Metal `metal:2118-2229`; Vulkan `ssm_conv.comp`) | **S** CPU/CUDA, **S-M** Metal/Vulkan |
| S3 | **New op `SSM_SCAN_BACK`** (flagship) — `(s0,x,dt,A,B,C,ids,dy) → {dx, ddt, dB, dC}`. The forward overwrites intermediate states in place (`ops.cpp:9770`), so backward must **recompute states**: checkpoint every K tokens, then per chunk re-run forward and reverse-scan (store-all is ~1 GiB/layer for Mamba-2-class configs at 512 tokens — acceptable only as the CPU-reference first cut; do NOT attempt the algebraic state inverse, `dA` underflows). dB/dC reduce over GQA-style head groups. Boundary terms vanish for single-ubatch training (initial state = cache constant; final-state copy has no loss dependency). Reverse-pass math worked out in the research (per-t: `ds += C⊗dy`; `dC += Σᵢ dyᵢsᵢⱼ`; `dB += Σᵢ dsᵢⱼ·x_dtᵢ`; `dx = dtsp·Σⱼ dsⱼBⱼ`; `ddt` via `sigmoid(dt)`; `ds_{t-1} = dA·ds_t`). Patterns: CPU `ops.cpp:9562-9773`; CUDA `ssm-scan.cu:20` (block per seq × splitD slice, reversed L-loop, CUB reductions); Metal `kernel_ssm_scan_f32` (`metal:2276`); Vulkan `ssm_scan.comp` (subgroup-add variant already scaffolded). | **L** per backend |
| S4 | Backward-switch cases emitting S2/S3 | **S** |
| S5 | SSM-full (dA/dD/conv-weight/dt_bias grads; cross-ubatch BPTT is out of scope — document truncation-at-ubatch semantics) | deferred |

Unlike MoE, all four backends already have SSM **forwards** — backward is a same-shape sibling kernel on each.

### Others (inventory, not scheduled)
| Ops | Archs blocked | Rating |
|---|---|---|
| `RWKV_WKV6`, `GATED_LINEAR_ATTN` | RWKV6, RWKV6Qwen2 | **L** per backend (same recompute-reverse-scan skeleton as S3) |
| `RWKV_WKV7` | RWKV7, ARWKV7 | **L** |
| `GATED_DELTA_NET` (+ `CUMSUM`/`TRI`/`SOLVE_TRI` VJPs for the chunked path: S-M/S/M, mostly graph-level) | Qwen3-Next, Qwen3.5(-MoE), Kimi-Linear | **XL** (hardest single op in this area; these archs are MoE hybrids → also need §9) |

---

## 11. Delivery phases, dependencies, and upstreaming

| Phase | Contents | Unlocks | Rough effort |
|---|---|---|---|
| **K0** | P1 (CPU sparse CE oracle) + P2/K-F16OP; C1+C2 (CUDA quantized OUT_PROD); C3 (CUDA sparse CE); K-SMB; K-TANH; FA8 (chunked-attention fallback) | **Fully GPU-resident dense-LoRA training on CUDA** — the "GPU used when available" milestone for the primary platform; 2–4k ctx via FA8 | ~4–6 eng-weeks |
| **K1** | V1+V2+V3 (Vulkan OUT_PROD + CE); V4/V5 | GPU-resident training on NVIDIA/AMD/Intel via Vulkan | ~3–4 eng-weeks |
| **K2** | M3–M5 → M1 → M2 → M7/M8 → M6/M10 (Metal suite) | GPU-resident training on Apple Silicon | ~6–8 eng-weeks |
| **K3** | FA1–FA4 → FA5 | Long-context FA training (CPU-correct first, then CUDA-resident) | ~8–12 eng-weeks |
| **K4** | E1–E7 (MoE), S1–S4 (SSM); FA6/FA7 | MoE + Mamba LoRA training; FA on Vulkan/Metal | ~12+ eng-weeks |
| **K5** | perf: fused quantized out_prod variants, mma FA backward, fused GLU backward (§13 item 3), saved-lse CE exploitation (ABI reserved in K0 per gate G-A), RWKV/delta-net | | open-ended |

**Mapping to the blueprint's phases:** K0 delivers the kernel prerequisites of blueprint **P1** (`ce_sparse` CPU+CUDA) and **P2** (quantized `OUT_PROD` CUDA); K1/K2 extend P2's "Metal/Vulkan triage"; K3/K4 realize the kernel half of blueprint **P4** (FA backward, MoE, small VJPs — K-TANH/E5/E6 here). Gradient checkpointing is blueprint-P2 *graph-level* work and appears in no K phase by design.

**Decide-first gates** (settle before the affected items start):
- **G-A — sparse-CE ABI** (§3 K-CE / open question Q3): lse stash vs recompute; logits-buffer aliasing. Blocks P1/C3/M7/V3.
- **G-B — determinism default** (§9 / Q9): deterministic segmented schemes as project default, atomics opt-in. Blocks E2/E3 and FA5's dQ strategy.

**Hard dependencies** (everything else is parallel): E2-Metal/Vulkan ← M2/V2 · E3 is *not* blocked on OUT_PROD ports · FA5 ← FA1–FA4 ← nothing · FA6/FA7 ← FA3 + their backend's backward suite · FA8 ← per-backend OUT_PROD (+K-SMB for ALiBi, +K-TANH for softcap) · M7/M8, V3, C3 ← G-A + P1 oracle.

**Upstreaming & fork strategy.** All of this is llama.cpp/ggml core code pinned at 4f37f51; a long-lived fork carrying new op enums fights ggml's fast-moving op table. Triage: (a) **upstream-friendly PRs, send early** — K-SMB, K-F16OP, K-TANH, C1/C2 (mainline training benefits directly, tests exist), M/V backward kernels (pure additions behind supports_op); (b) **in-fork first, upstream when stable** — new op enums (sparse CE, `OUT_PROD_ID(_GRP)`, `SSM_*_BACK`, `GLU_BACK`) and the `FLASH_ATTN_EXT` `emit_lse` ABI change — propose upstream as one RFC per op family once the CPU oracle + one GPU backend prove the design; (c) rebase cadence: track mainline at least monthly; every vendor bump re-runs the full MODE_GRAD suite. Fallback if upstream rejects an ABI (e.g. `emit_lse`): keep it behind a fork-local op flag and carry a small rebase patch — the op-enum tail positions minimize conflicts.

**CI matrix** (exit criterion for each K phase = its column green):

| Test tier | CUDA (sm_70 + sm_90) | ROCm/HIP | CPU (x86 + ARM) | Metal (Apple7+) | Vulkan (coopmat NV, scalar AMD/Intel, MoltenVK) |
|---|---|---|---|---|---|
| MODE_GRAD per-op (per-PR) | ✔ | nightly | ✔ (oracle) | ✔ | ✔ scalar; nightly coopmat/MoltenVK |
| Tiny-model e2e convergence (loss-curve tolerance vs recorded PEFT reference — blueprint P1 test) | per-phase exit | nightly | per-phase exit | per-phase exit | per-phase exit |

Scheduler note: until a backend's gap closes, `ggml_backend_sched` transparently falls back to CPU for unsupported nodes — training *works* everywhere today at reduced speed; each phase moves ops from "CPU fallback" to "GPU-resident."

---

## 12. Open questions & risks

Each item lists the phase it affects and the mitigation/contingency.

1. **Dequant+GemmEx vs fused kernel throughput** (K0/K1; C1, V2). Measure conversion overhead at realistic shapes. *Contingency:* the fused dequant-GEMM variant is pre-scoped (L/XL) and budgeted only if profiling triggers it. (= risk **R2**)
2. **F16 tensor-core gradient precision** (K0). Forward uses int8 mmq; backward uses F16 tensor cores with F32 accumulate — confirm LoRA convergence via MODE_GRAD + convergence tests. *Contingency:* BF16, or chunked F32 SGEMM on the existing path.
3. **Sparse-CE ABI (gate G-A)** (K0). Lse stash vs recompute (2× vocab reads); may backward alias the logits buffer under `ggml_gallocr`? One cross-backend decision before any backend implements; prototype the aliasing against the allocator.
4. **FA backward numerics** (K3). Forward's FTZ threshold and KQ max-offset mean recomputed P won't bit-match; set MODE_GRAD tolerances against the CPU oracle; LSE-with-sinks definition must match across backends.
5. **FA backward smem at D=256** (K3). Prototype occupancy (CUDA tile nbatch splitting; Vulkan scalar shmem gates) before committing kernel shapes.
6. **`OUT_PROD_ID` CUDA strategy for ragged experts** (K4). Per-expert cuBLAS launches vs custom kernel vs `cublasGemmGroupedBatched` (raises min CUDA version). Benchmark at n_expert 64–256 with <8 tokens/expert.
7. **SSM_SCAN_BACK checkpoint interval K** (K4). Memory vs 2× scan FLOPs; register pressure in the reversed CUDA loop; where the checkpoint buffer lives (pool alloc vs extra dst).
8. **Metal transposed `simdgroup_load` bank conflicts** (K2; M1/M2). Microbenchmark; *contingency:* stage tiles pre-transposed during the dequant phase (free — swap shmem write indices).
9. **Determinism default (gate G-B)** (K0 decision; affects K3/K4). Recommend deterministic project-wide with opt-in atomic variants.
10. **MoltenVK as interim mac path** (K1/K2). One benchmark (scalar mul_mm, no coopmat) to decide stopgap vs skip.
11. **Upstream churn against 4f37f51** (all phases). ggml's op table and backend code move fast; a stale fork compounds every later item. *Mitigation:* the rebase cadence + PR triage in §11; keep fork-local diffs to op-enum tails and new files.
12. **Schedule risk on the two long poles** (K2's M2, K3's FA5). Either slipping strands its phase. *Mitigation:* FA8 (already in K0) is the standing fallback for long context; MoltenVK (Q10) is the hedge for Apple Silicon while M2 is in flight.

---

## 13. Final section — future improvements from permissibly-licensed unsloth components

A precise license audit of the unsloth checkout:

**License structure.** Repo default is **Apache-2.0** (`LICENSE`, with a carve-out at line 191: `studio/*` and `unsloth_cli/*` are AGPLv3; root `COPYING` is the AGPL text backing that carve-out). Three copyleft tiers matter:
- **AGPL-3.0 (excluded):** `kernels/moe/**` (grouped-GEMM MoE kernels incl. their backward), `utils/prefix_grouper*.py` (GRPO shared-prefix attention), `studio/`, `unsloth_cli/`, `cli.py`, build/install scripts, and **one function-level marker** — the inner `_get_per_token_logps_and_entropies` at `models/rl_replacements.py:1191` (GRPO chunked-logprob orchestration) inside an otherwise Apache-2.0 file.
- **LGPL-3.0+ (code-copy excluded; a tier a naive "AGPL" grep misses):** `kernels/rope_embedding.py` (irrelevant — ggml already has ROPE_BACK on all four backends), `utils/packing.py` (packing/boundary masking — concept reusable, implementation not; the boundary-mask rule is ~5 lines re-derivable from its one-sentence spec), `utils/attention_dispatch.py`, `utils/__init__.py`.
- **unsloth_zoo (unavailable):** a separate pip package not in this checkout (fused CE patching internals, offloaded gradient checkpointing, chunked selective log-softmax, left-pack). Treated as unknown-license until separately audited.

**Confirmed Apache-2.0 and clean for reuse** (all verified by header): `kernels/cross_entropy_loss.py`, `fast_lora.py`, `swiglu.py`, `geglu.py`, `rms_layernorm.py`, `layernorm.py`, `flex_attention.py`, `fp8.py`, `kernels/utils.py`, `trainer.py`. Importantly, the **chunked selective log-softmax idea exists in full on the Apache side** (`cross_entropy_loss.py:87-150` — chunked-logsumexp decomposition, separated `−x_label` term, label-gather) so nothing AGPL is needed for any planned reuse. All items below are math/design imports into new ggml kernels (Triton → CUDA/MSL/GLSL rewrite is required anyway), ranked by value:

1. **Sparse-CE op design** (feeds §3 K-CE directly): two-pass logsumexp-only forward saving one F32 lse per row (never materialize softmax); chunked-lse for wide vocab; backward `dloss·w·(exp(x−lse) − onehot)` with in-place logit-grad as an allocator-permitting optimization; `-100`-style ignore via `w=0`. (`cross_entropy_loss.py:87-100, 138-190, 260-276`.)
2. **Softcap/logit-scale folded into CE** — forward `t·tanh(x/t)`, backward `×(1−tanh²)`; Cohere-style `s·x` scale (`:84-85, 247-273`). Include in the op ABI from day one.
3. **Fused GLU backward op** (`GGML_OP_GLU_BACK`): one elementwise pass producing recomputed `h`, `df = DW·f`, `de = DW·g·σ(e)·(1+e(1−σ(e)))` in-place over the three buffers (`swiglu.py:68-109`) — versus ggml's current SILU_BACK+MUL multi-pass which re-reads the largest activations in the graph (n_tokens × n_ff). Trivial elementwise kernel on all four backends; real bandwidth win. (K5.)
4. **GEGLU exact + tanh-approx backward formulas** (`geglu.py:75-123, 188-244`) — ggml has GEGLU forward but no backward path at all; these derivatives unlock Gemma-family MLPs (pairs with K-TANH, §3; consumed by E7, §9).
5. **LoRA graph-structuring rules** (`fast_lora.py`): save only X and pre-activations e,g per MLP block (recompute h in backward — checkpoint choice at graph level); associate LoRA grads **rank-r-first** (`dA = Xᵀ(dY·Bᵀ)`, `dB = (AᵀXᵀ)dY`) so no intermediate ever exceeds r width — keeps LoRA-grad FLOPs O(n·r·(d+k)); fold the LoRA scale into GEMM alpha; accumulate dX in one buffer. These shape which OUT_PROD variants each backend actually needs: the quantized one for dX; **plain F32 GEMMs suffice for all LoRA A/B grads.**
6. **RMS_NORM_BACK saved-inv-var variant** — forward stores one F32 `inv_var` per row; backward becomes a pure two-load formula (`rms_layernorm.py:51-111`, incl. the Gemma `W+1` convention). Optional op-ABI change; microbenchmark before committing.
7. **LayerNorm backward** (`layernorm.py:67-104`, mean+inv_var stats) — the NORM_BACK pattern when LayerNorm archs (GPT-2/BERT-style) enter scope.
8. **Numerics contract** — elementwise math in F32 with casts at the exact storage boundaries; all row stats/lse/loss F32; adopt max-abs grad error ≤ 0.05 at fp16 as this project's cross-backend parity criterion (unsloth's self-tests use a similar, if loosely implemented, 0.05 threshold, `rms_layernorm.py:326`).
9. **Packing boundary-masking concept** (LGPL — re-derive independently): mask the last token of each packed sample (`boundary = cumsum(lengths) − 1 → weight 0`) so CE never trains across sample boundaries. ~5 lines in the Python data layer.

**Explicitly not reused:** MoE grouped-GEMM kernels (AGPL; ggml's MoE backward in §9 is designed independently around mmid compaction), GRPO chunked-logprob orchestration function (AGPL; the Apache CE kernel supplies the math), prefix-grouper (AGPL; llama.cpp's native KV-cache prompt sharing achieves the same effect for generation), everything in unsloth_zoo (unavailable), and LGPL file implementations (concepts only). Provenance discipline: per-file headers in `csrc/` naming the pattern source and license, per the blueprint's licensing section.
