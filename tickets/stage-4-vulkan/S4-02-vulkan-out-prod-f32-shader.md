---
id: S4-02
title: "Vulkan V1: OUT_PROD F32 shader"
stage: 4
track: kernels
size: M
deps: ["S4-01"]
status: open
pr: null
---

# S4-02 — Vulkan V1: OUT_PROD F32 shader

**One-line outcome:** F32 `OUT_PROD` exists on Vulkan: a new `out_prod.comp` on the
`mul_mm.comp` scalar pattern with stride push constants for the transposed-grad src1 view,
portable (no subgroup arithmetic), passing MODE_GRAD against the CPU oracle.

## Why (context)

Vulkan is the closest non-CUDA backend to GPU-resident training: every `*_BACK` op in the
training set plus both optimizer steps already exist with tested shaders (ROADMAP §2, §7).
The hard gaps are `OUT_PROD` and the loss. `OUT_PROD` is `MUL_MAT`'s backward w.r.t. both
operands and fires on every linear layer of every microbatch (ROADMAP §1) — and today the
Vulkan backend contains **zero references to it** (verified: no `OUT_PROD` occurrence
anywhere in `vendor/llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp`), so every out_prod node
falls back to CPU via `ggml_backend_sched`.

This ticket lands the F32 case (ROADMAP §7 V1): the LoRA A/B weight-gradient GEMMs, which
are pure F32×F32 — per ROADMAP §13 item 5, plain F32 GEMMs suffice for *all* LoRA A/B grads,
so this shader alone moves the trainable-parameter gradients onto the GPU. The quantized-src0
case (activation grads through frozen weights) is deliberately separate: S4-03 solves it by
composition with no new matmul shader, and it needs this ticket's supports_op plumbing first.

Two design constraints are non-negotiable. **Portability:** the shader must use
shared-memory tree reductions only — no subgroup arithmetic — because the backend
force-disables subgroup arithmetic (and shuffle) on MoltenVK with AMD GPUs
(`vendor/llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp:5983-5994`, the `__APPLE__`
vendor-AMD workaround), and lavapipe/scalar drivers are first-class CI targets (S4-01).
**Numerics:** F32 accumulation throughout; the backend's tuned matmul pipelines carry
`f16acc` variants (`vendor/llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp:216`) that
ADR-0002 (S0-09) forbids on gradient paths — a new grad-only shader must simply not have an
f16acc variant.

## What to do

All ggml changes land in the vendored llama.cpp fork via the S0-02 two-repo flow.

1. **New shader `vulkan-shaders/out_prod.comp`** patterned on `mul_mm.comp`'s scalar
   (non-COOPMAT) path: block tiling into shared memory, per-thread accumulators, F32 math.
   Copy the push-constant layout style of `mul_mm.comp` — `M/N/K` plus
   `stride_a/stride_b/stride_d` and batch strides
   (`vendor/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/mul_mm.comp:73-100`) — so
   arbitrary src1 strides are honored: in the frozen-weight/LoRA case src1 arrives as a
   **transposed grad view** (`out_prod(a, transpose(grad))`), and the shader must read it
   through its true strides rather than requiring a `CONT` materialization.
2. **No subgroup arithmetic:** any cross-thread reduction uses a shared-memory tree with
   `barrier()` (pattern: `soft_max_back.comp`'s shared `sum_yg[BLOCK_SIZE]` loop). Guard the
   shader so it compiles and runs on lavapipe, MoltenVK, and scalar AMD/Intel drivers.
3. **Broadcast/batch semantics:** implement the same src0→dst ne2/ne3 broadcast behavior as
   the CPU reference (`ggml_compute_forward_out_prod_f32`,
   `vendor/llama.cpp/ggml/src/ggml-cpu/ops.cpp` — the F32 sibling of the quantized path at
   `:4363`); the existing test cases sweep bs2/bs3 and nr2/nr3
   (`vendor/llama.cpp/tests/test-backend-ops.cpp:8780-8806`).
4. **Plumbing — the six mechanical touch points per new op** (ROADMAP §7): shader
   registration in the vulkan-shaders build, a pipeline member on `vk_device_struct`,
   pipeline creation, the op dispatch function (`ggml_vk_out_prod`), the graph-build switch
   case, and a `GGML_OP_OUT_PROD` case in `ggml_backend_vk_device_supports_op`
   (`vendor/llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp:17158-17719`). The supports_op
   case accepts F32 src0/src1/dst only (S4-03 widens it); require what the shader actually
   handles and nothing more — S4-05 later audits such preconditions against real graphs.
5. **Tests:** flipping supports_op lights up the existing F32 `test_out_prod` cases —
   the `base_types` loop includes F32/F16 `type_a` with F32/F16 `type_b`, n/k ∈ {1,16},
   batch and broadcast sweeps (`vendor/llama.cpp/tests/test-backend-ops.cpp:8780-8806` over
   `base_types` at `:7740-7749`), plus the ne2/nr2 batched sweeps. F16-src0 cases stay
   rejected by supports_op in this ticket (quantized/F16 src0 is S4-03). **Add
   `trans_b = true` cases:** the harness supports a transposed-src1 view
   (`trans_b` in `test_out_prod`, `vendor/llama.cpp/tests/test-backend-ops.cpp:4413-4418`,
   transpose applied at `:4425-4427`) but no generated case enables it today — and that
   view is exactly the frozen-weight training layout, so it must be exercised. Run `test`
   (forward parity vs CPU) and `grad` (MODE_GRAD) on lavapipe per-PR; native
   coopmat + scalar nightly per S4-01.
6. **Append `OUT_PROD`** to ci-vulkan's targeted default op list (S4-01 soft coordination).
7. **Submodule bump PR** in learning-llamas per S0-02.

## Out of scope

- Quantized/F16 src0 `OUT_PROD` — S4-03 (dequant + transpose-copy + `mul_mm` composition;
  no new matmul shader).
- `OUT_PROD_ID` / `OUT_PROD_ID_GRP` MoE variants — S4-06.
- A fused/tuned out_prod tile shader (coopmat variants, warptile tuning) — not needed for
  V1; the quantized path reuses the tuned `mul_mm` pipeline in S4-03, and a fused
  quantized out_prod shader is backlog B-01.
- CUDA/Metal OUT_PROD — S3-02, S2-05/S2-06.

## Acceptance criteria

- [ ] Fork branch: `test-backend-ops test -b Vulkan0 -o OUT_PROD` passes on lavapipe for
      every F32 `test_out_prod` case, including the newly added `trans_b = true` cases and
      the broadcast/batched sweeps.
- [ ] Fork branch: `test-backend-ops grad -b Vulkan0 -o OUT_PROD` (MODE_GRAD) passes vs the
      CPU oracle within the ADR-0002 tolerances (per-op bound; ≤ 0.05 max-abs @ fp16
      cross-backend parity criterion) on lavapipe, and on the native runner in both coopmat
      and `GGML_VK_DISABLE_COOPMAT=1` scalar configurations.
- [ ] The shader contains no `GL_KHR_shader_subgroup_arithmetic` usage and no f16
      accumulation (grep-verifiable in the fork diff).
- [ ] supports_op accepts only F32 src0/src1/dst for `GGML_OP_OUT_PROD` (probed via
      `test-backend-ops support -b Vulkan0`).
- [ ] learning-llamas submodule-bump PR is green in `ci-vulkan / lavapipe` (targeted `-o
      OUT_PROD` visible in logs) and `ci-cpu`.

## Testing & verification

Primary harness: vendored `tests/test-backend-ops` — `test` and `grad` modes on the Vulkan
backend vs the CPU oracle, per ADR-0002 (S0-09). Runs on the fork branch CI and, after the
submodule bump, in learning-llamas's `ci-vulkan / lavapipe` lane per-PR (targeted `-o OUT_PROD`)
and in the native-GPU nightly sweep (S4-01), which also covers the coopmat-vs-scalar driver
split. End-to-end effect (LoRA A/B grads GPU-resident) is observed in the S4-01 nightly
fallback report and finally enforced by S4-09.

## PR notes

- Branch: `ticket/S4-02-vulkan-out-prod-f32-shader`.
- Two-repo flow per S0-02: implementation PR against the fork's `learning-llamas-base` branch
  with the ticket ID in the title, plus a trivial learning-llamas submodule-bump PR referencing
  the same ticket ID.
- Upstreaming disposition: **upstream-early** (ROADMAP §11 triage class a — a pure addition
  behind supports_op; the `test_out_prod` cases already exist upstream, no new ABI).
- Provenance headers per S0-01 policy: shader patterned on
  `ggml/src/ggml-vulkan/vulkan-shaders/mul_mm.comp` (MIT, commit `4f37f51`).
