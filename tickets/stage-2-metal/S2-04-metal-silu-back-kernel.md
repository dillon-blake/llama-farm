---
id: S2-04
title: "Metal M5: SILU_BACK kernel"
stage: 2
track: kernels
size: S
deps: ["S2-01"]
status: open
pr: null
---

# S2-04 — Metal M5: SILU_BACK kernel

**One-line outcome:** `SILU_BACK` exists on Metal as a binary elementwise kernel, completing —
together with the already-present `MUL` — the split-SWIGLU backward path on Apple Silicon.

## Why (context)

The FFN backward of every SwiGLU transformer needs `SILU_BACK`: the autograd split-SWIGLU case
emits `ggml_silu_back(mul(grad, src1), src0)` for the gate branch plus a plain `MUL` for the up
branch (`vendor/llama.cpp/ggml/src/ggml.c:6890`; the plain `SILU` unary backward emits it too,
`:6854`). Metal has `MUL` (`vendor/llama.cpp/ggml/src/ggml-metal/ggml-metal-device.m:1145-1147`)
but no `SILU_BACK` case in `supports_op` (`:1051-1368`), so the gate-branch node falls back to
CPU. This is M5, the last of the ROADMAP §6 M3 → M4 → M5 trio that unblocks backward through
norm/attention/FFN on Metal.

The op is a same-shape binary elementwise map — no reductions: `src0 = dy`, `src1 = x`, and
`dst_i = dy_i · σ(x_i) · (1 + x_i·(1 − σ(x_i)))`, per the CPU oracle
(`ggml_compute_forward_silu_back_f32`, `vendor/llama.cpp/ggml/src/ggml-cpu/ops.cpp:2785`, which
delegates rows to `ggml_vec_silu_backward_f32`). The in-tree pattern is the binary elementwise
kernel `kernel_bin_fuse_impl` (`vendor/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:1211`);
`SILU_BACK` needs none of its fusion or broadcast machinery (the CPU kernel asserts same
shapes), so the Metal kernel is that pattern minus generality, plus the derivative formula in
F32. Per ROADMAP §6 the op is five mechanical additions: kernel, kargs struct, pipeline
getter, encoder, `supports_op` case.

Test coverage already exists on three levels: a direct forward-eval case (`test_silu_back`,
`vendor/llama.cpp/tests/test-backend-ops.cpp:3322`, instantiated at `:8424`), MODE_GRAD unary
`SILU` cases (`test_unary` sets `ggml_set_param`, `:1991`), and MODE_GRAD split-GLU cases
(`test_glu_split`, `:2124`, sets params on both branches — these exercise exactly the
`SILU_BACK` + `MUL` composition the manifest names). What none of them proves is the *model
level*, so this ticket ends with a graph-assignment check on a real tiny dense-model training
step.

## What to do

All code changes land in the vendored llama.cpp fork via the S0-02 two-repo flow.

1. **Kernel** `kernel_silu_back` (F32) in
   `vendor/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal`, following the
   `kernel_bin_fuse_impl` structure (`:1211`) simplified to same-shape contiguous operands;
   compute the derivative in F32 exactly as `ggml_vec_silu_backward_f32` does. Add a `float4`
   variant only if the bin-kernel pattern makes it free.
2. **kargs struct** in `vendor/llama.cpp/ggml/src/ggml-metal/ggml-metal-impl.h` — reuse
   `ggml_metal_kargs_bin` (`:235`) if the encoder can fill it; otherwise a minimal
   `ggml_metal_kargs_silu_back`.
3. **Pipeline getter** in `vendor/llama.cpp/ggml/src/ggml-metal/ggml-metal-device.cpp`,
   pattern `ggml_metal_library_get_pipeline_bin` (`:1541`).
4. **Encoder + dispatch case** `GGML_OP_SILU_BACK` in
   `vendor/llama.cpp/ggml/src/ggml-metal/ggml-metal-ops.cpp` (`ggml_metal_op_encode_impl`
   switch at `:175`; `ggml_metal_op_bin` at `:3079` is the model).
5. **supports_op case** in `ggml_metal_device_supports_op` (`ggml-metal-device.m:1051-1368`):
   F32 + same-shape/contiguity preconditions; elementwise ops need no simdgroup features.
6. **Tests.** Fork branch: `test-backend-ops test -b <Metal device> -o SILU_BACK`;
   `test-backend-ops grad -b <Metal device> -o SILU,GLU` (MODE_GRAD through both the unary and
   the split-SWIGLU backward, ADR-0002 tolerance). Capture before/after output for the PR.
7. **Split-SWIGLU end-to-end confirmation on a tiny dense model graph:** run one S1-12-fixture
   training step with `--device metal` and `GGML_SCHED_DEBUG=2`
   (`vendor/llama.cpp/ggml/src/ggml-backend.cpp:1740-1741`), and verify in the assignment dump
   that the FFN backward's `SILU_BACK` and `MUL` nodes are assigned to the Metal backend
   (other still-missing ops may legitimately fall back at this point). Attach the dump to the
   PR or CI artifact.
8. **Submodule bump PR** in learning-llamas referencing this ticket, per S0-02.

## Out of scope

- Fused GLU backward (`GGML_OP_GLU_BACK`, one pass producing `h, df, de` — ROADMAP §13 item 3):
  K5 perf work, `tickets/backlog/`.
- GEGLU/REGLU/SWIGLU_OAI backward (E7) and their Metal ports — MoE/breadth work, stage-1
  S1-28 owns the op-level backward.
- Non-F32 variants — the CPU oracle's F16 path is a separate concern; no training graph emits
  F16 `SILU_BACK` under the v1 constraints (BLUEPRINT §8).

## Acceptance criteria

- [ ] Fork branch: `test-backend-ops test -o SILU_BACK` on the Metal backend passes vs the CPU
      oracle.
- [ ] Fork branch: `test-backend-ops grad -o SILU,GLU` MODE_GRAD passes on Metal within the
      ADR-0002 tolerance (unary SILU and split-SWIGLU GLU cases).
- [ ] `test-backend-ops support -b <Metal device>` reports `SILU_BACK` supported (probe output
      attached to the PR).
- [ ] A sched assignment dump from a tiny dense-model training step with `--device metal`
      shows `SILU_BACK` and the adjacent `MUL` backward node executing on Metal (artifact
      attached).
- [ ] learning-llamas submodule-bump PR is green: `ci-metal / build` and `ci-metal / grad` per-PR;
      one nightly (or `workflow_dispatch`) full-sweep run green.

## Testing & verification

Primary harness: vendored `tests/test-backend-ops` on the Metal backend — `test` mode for
`SILU_BACK` forward parity vs the CPU oracle, `grad` mode (MODE_GRAD, ADR-0002 tolerances) for
`SILU` and `GLU` cases. The e2e assignment check reuses the S1-12 fixture and the S2-01
fallback-report tooling. Runs in `ci-metal / grad` per-PR (targeted `-o SILU,GLU,SILU_BACK`)
and the nightly full Metal sweep + e2e (S2-01).

## PR notes

- Branch: `ticket/S2-04-metal-silu-back-kernel`.
- Two-repo flow per S0-02: fork PR (`learning-llamas-base`) + trivial learning-llamas submodule-bump PR,
  both referencing the ticket ID.
- Upstreaming disposition: **upstream-early** (ROADMAP §11 triage class a — pure addition
  behind `supports_op` with existing upstream tests).
- No copied external code; patterned on in-tree `kernel_bin_fuse_impl` (MIT provenance per
  S0-01).
