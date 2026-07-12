---
id: S4-03
title: "Vulkan V2: OUT_PROD quantized via dequant + transpose-copy + mul_mm reformulation"
stage: 4
track: kernels
size: M
deps: ["S4-02"]
status: open
pr: null
---

# S4-03 — Vulkan V2: OUT_PROD quantized via dequant + transpose-copy + mul_mm reformulation

**One-line outcome:** quantized-src0 `OUT_PROD` on Vulkan with NO new matmul shader: a
backend-side composition (dequant → transpose-copy → `mul_mm`) that reuses the entire tuned
matmul pipeline (coopmat/coopmat2/split-k) with F32 accumulation, making backward through
frozen quantized weights GPU-resident on Vulkan.

## Why (context)

The frozen-weight case `out_prod(W_quantized, transpose(grad))` fires on every linear layer
of every microbatch — it is the op that decides whether backprop through the quantized base
lives on the GPU (ROADMAP §1, BLUEPRINT G6). ROADMAP §0 finding 1 says no new quantized GEMM
kernel is needed anywhere, and on Vulkan the argument is an identity:
`out_prod(a, b) = mul_mat(contᵀ(a), contᵀ(b))`. In the frozen-weight case `b` is already a
transposed grad view, so `contᵀ(b)` **is grad — contiguous, zero-copy**. Only `a` needs a
physical transpose, and dequant machinery already exists: the backend holds a
`pipeline_dequant[type]` shader for every supported quant type
(`vendor/llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp:4996-5018`), selected by
`ggml_vk_get_to_fp16` (`:7121`, returning `pipeline_dequant[type]` at `:7152`), and
`ggml_vk_mul_mat_q_f16` (`:8600`) already runs exactly this dequant-into-transient pattern:
`to_fp16_vk_0 = ggml_vk_get_to_fp16(ctx, src0->type)` (`:8711`) writing into the
`prealloc_x` transient (`:8780`). The composition therefore inherits all `mul_mm` tuning —
coopmat, coopmat2, split-k, warptile selection — for free (ROADMAP §7 V2).

The one thing the composition must force is **F32 accumulation**: the tuned pipelines come
in f16acc/f32acc pairs (`vk_matmul_pipeline2`,
`vendor/llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp:216`), and the getter
`ggml_vk_get_mul_mat_mat_pipeline` (`:7155`) picks f16acc whenever the op precision is
`GGML_PREC_DEFAULT` (`:7227-7232`). ADR-0002 (S0-09) names Vulkan f16acc `mul_mm` variants a
forbidden pattern on gradient paths, so this composition must request the f32acc pipelines
unconditionally.

The alternative — a fused per-quant-type out_prod tile shader (~25 quant types × warptile
variants) — is explicitly deferred to backlog B-01 unless profiling shows the dequant
round-trip dominating (ROADMAP §12 Q1, risk R2). Do not build it here.

## What to do

All ggml changes land in the vendored llama.cpp fork via the S0-02 two-repo flow.

1. **Extend `ggml_vk_out_prod`** (from S4-02) with a quantized/F16-src0 path implementing
   the composition, patterned on `ggml_vk_mul_mat_q_f16`'s buffer management
   (`vendor/llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp:8600`):
   (a) dequantize src0 to F16 via `ggml_vk_get_to_fp16(ctx, src0->type)`
   (`:7121`/`:7152`) into the `prealloc_x` transient, exactly as `:8711`/`:8780` do;
   (b) transpose the dequantized `a` with the existing `copy_transpose.comp` tiled
   transpose shader (32×32 shared-memory tile,
   `vendor/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/copy_transpose.comp`) into a
   second transient;
   (c) dispatch the tuned `mul_mm` pipeline on `(contᵀa, contᵀb)`. For the frozen-weight
   case where src1 is a transposed grad view, `contᵀ(src1)` is the underlying grad tensor —
   detect the transposed-view layout and pass grad directly, **zero-copy**; materialize a
   contiguous transpose only for genuinely non-contiguous src1 layouts.
2. **Force F32 accumulation:** request the pipelines with `GGML_PREC_F32` semantics so
   `ggml_vk_get_mul_mat_mat_pipeline` (`:7155`, selection at `:7227-7232`) returns the
   `.f32acc` variants on every driver path (coopmat2, coopmat, scalar). Add a comment naming
   ADR-0002 so nobody "optimizes" this back to f16acc.
3. **Transient sizing and sync:** reuse the `prealloc_x`/`prealloc_y` sizing and
   `prealloc_*_need_sync` discipline of `ggml_vk_mul_mat_q_f16`; account for the extra
   transposed copy. lm_head-scale weights (128k×4096 F16 ≈ 1 GB) may exceed transient or
   `maxStorageBufferRange` budgets — chunk over src0 rows with accumulation if the probe
   shows it necessary, and record the measured transient sizes in the PR.
4. **Widen `supports_op`** for `GGML_OP_OUT_PROD`: accept src0 F16 (skips the dequant
   step — `ggml_vk_mul_mat_q_f16` likewise dequantizes only non-F16 src0) or any src0 type
   where `ggml_vk_get_to_fp16` returns a non-null pipeline (the type switch at
   `:7123-7149`), with src1 F32 and dst F32; keep everything else rejected. Note this
   mirrors the CUDA gate strategy (S3-02: gate on a non-null converter).
5. **Tests:** the quantized `test_out_prod` cases light up on Vulkan when supports_op flips
   (`vendor/llama.cpp/tests/test-backend-ops.cpp:8780-8806` over `base_types` at
   `:7740-7749` — Q8_0, Q4_0, Q4_1, Q4_K, MXFP4, NVFP4, IQ2_XXS plus F16). Run `test`
   (forward parity vs the CPU dequant oracle `ggml_compute_forward_out_prod_q_f32`,
   `vendor/llama.cpp/ggml/src/ggml-cpu/ops.cpp:4363`) and `grad` (MODE_GRAD) per quant
   type, on lavapipe and on the native runner in both coopmat and
   `GGML_VK_DISABLE_COOPMAT=1` scalar configurations (`ggml-vulkan.cpp:5816`) — the
   composition takes different `mul_mm` variants per driver, and all must satisfy ADR-0002.
   Include the S4-02 `trans_b = true` cases with quantized `type_a` (add if the sweep does
   not already generate them).
6. **Submodule bump PR** in learning-llamas per S0-02; ci-vulkan already targets `OUT_PROD`
   (S4-02).

## Out of scope

- A fused per-quant-type out_prod tile shader — backlog B-01, profiling-triggered
  (ROADMAP §12 Q1).
- `OUT_PROD_ID` / `OUT_PROD_ID_GRP` MoE variants — S4-06 (reuses this dequant + composition
  machinery with per-expert segmenting).
- CUDA/Metal quantized OUT_PROD — S3-02 / S2-06.
- BF16 src0 (no Vulkan dequant-to-F16 pipeline for it in the `:7123-7149` switch today) —
  reject in supports_op; revisit if a training config needs it.
- Perf tuning beyond correctness (split-k thresholds, transient reuse) — S4-09 perf
  snapshot decides whether B-01 triggers.

## Acceptance criteria

- [ ] Fork branch: `test-backend-ops test -b Vulkan0 -o OUT_PROD` passes on lavapipe for
      every quantized `base_types` case and F16 src0.
- [ ] Fork branch: `test-backend-ops grad -b Vulkan0 -o OUT_PROD` (MODE_GRAD) passes per
      quant type vs the CPU oracle within the ADR-0002 tolerances (per-op bound; ≤ 0.05
      max-abs @ fp16 cross-backend parity criterion), on lavapipe AND on the native runner
      in both coopmat and scalar (`GGML_VK_DISABLE_COOPMAT=1`) configurations.
- [ ] The composition demonstrably selects `.f32acc` pipelines on gradient paths: no
      `.f16acc` selection is reachable from the out_prod path (code-review criterion backed
      by a grep of the fork diff plus the ADR-0002 comment).
- [ ] The zero-copy transposed-src1 fast path is exercised by a test (transposed grad view
      as src1) and produces results identical to the materialized-transpose path.
- [ ] supports_op accepts exactly F16 plus the `ggml_vk_get_to_fp16`-convertible src0
      types, and rejects BF16 (probed via `test-backend-ops support -b Vulkan0`).
- [ ] learning-llamas submodule-bump PR is green in `ci-vulkan / lavapipe` and `ci-cpu`.

## Testing & verification

Primary harness: vendored `tests/test-backend-ops` — `test` and `grad` on the Vulkan backend
vs the CPU oracle, per ADR-0002 (S0-09). Per-PR: targeted `-o OUT_PROD` in
`ci-vulkan / lavapipe` plus the kernel-gated native `ci-vulkan / gpu` job; nightly: full
sweeps on both lanes including the coopmat/scalar double-run (S4-01). The end-to-end payoff
— backward through the quantized base GPU-resident on Vulkan — shows up in the S4-01 nightly
fallback report and is enforced at S4-09's milestone gate.

## PR notes

- Branch: `ticket/S4-03-vulkan-quantized-out-prod-composition`.
- Two-repo flow per S0-02: fork PR (`learning-llamas-base`) + trivial learning-llamas submodule-bump
  PR, both referencing the ticket ID.
- Upstreaming disposition: **upstream-early** (ROADMAP §11 triage class a — backend-side
  composition behind supports_op, no new op enums or ABI; upstream's quantized
  `test_out_prod` cases validate it directly).
- Provenance headers per S0-01 policy: composition adapted from
  `ggml/src/ggml-vulkan/ggml-vulkan.cpp` (`ggml_vk_mul_mat_q_f16`) and
  `ggml/src/ggml-vulkan/vulkan-shaders/copy_transpose.comp` (MIT, commit `4f37f51`).
- Soft coordination: S4-06 consumes this composition machinery — keep the dequant +
  transpose + pipeline-selection steps factored so per-expert segmenting can wrap them.
