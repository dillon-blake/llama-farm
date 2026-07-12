---
id: S1-18
title: "K-F16OP: F16/BF16 CPU out_prod (replace abort with to_float row path)"
stage: 1
track: kernels
size: S
deps: ["S0-02"]
status: pr-open
pr: https://github.com/dillon-blake/llama-farm/pull/16
---

# S1-18 — K-F16OP: F16/BF16 CPU out_prod (replace abort with to_float row path)

**One-line outcome:** CPU `OUT_PROD` accepts F16 and BF16 src0 (today F16 aborts — fatal, not
slow, and BF16 falls into the default abort), removing the forced-F32-KV-cache constraint at
the source.

## Why (context)

`OUT_PROD` is the op backprop rides on: the `MUL_MAT` backward emits
`out_prod(W, transpose(grad))` for the activation gradient, with the weight `W` as src0
(`vendor/llama.cpp/ggml/src/ggml.c:6578-6630`, src1 branch at `:6615-6629`) — it is hit on
every linear layer, every microbatch (ROADMAP §1). The CPU dispatcher handles F32 and every
block-quant type, but F16 src0 hits `GGML_ABORT("fatal error")` with a commented-out F16 path
(`vendor/llama.cpp/ggml/src/ggml-cpu/ops.cpp:4487-4491`), and BF16 is absent from the case list
entirely, falling into the default abort (`:4496-4499`). So an F16 `out_prod` today is a crash,
not a slowdown — this is why `examples/training/finetune.cpp` forces the F32 KV cache
(`vendor/llama.cpp/examples/training/finetune.cpp:34-40`) and why BLUEPRINT §8 carries
"KV cache F32" as a v1 hard constraint (ROADMAP §3 K-F16OP; §4 P2).

The fix is small because the pattern already exists in the same file: the quantized branch
`ggml_compute_forward_out_prod_q_f32` (`vendor/llama.cpp/ggml/src/ggml-cpu/ops.cpp:4363-4501`)
looks up `ggml_get_type_traits(type)->to_float` (`:4376`), dequantizes one src0 row into a
per-thread workspace buffer, and accumulates with `ggml_vec_mad_f32`. F16 and BF16 both have
`to_float` traits (BF16: `ggml_bf16_to_fp32_row`, `vendor/llama.cpp/ggml/src/ggml.c:885-891`),
so the identical row path covers them — this is the "same traits mechanism" route.

Two adjacent gates were verified during ticket authoring and must move with the kernel change:
the CPU `supports_op` accepts only F32-or-quantized src0 for `OUT_PROD`
(`vendor/llama.cpp/ggml/src/ggml-cpu/ggml-cpu.cpp:468-470`), and the CPU work-size planner
allocates the per-thread row buffer only `if (ggml_is_quantized(src0->type))`
(`vendor/llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c:2855-2860`) — routing F16/BF16 into the row
path without extending that condition would use an unallocated workspace.

## What to do

All code changes land in the vendored llama.cpp fork via the S0-02 two-repo flow.

1. **Kernel:** in `vendor/llama.cpp/ggml/src/ggml-cpu/ops.cpp`, route `GGML_TYPE_F16` and
   `GGML_TYPE_BF16` src0 through the `to_float` row path: delete the abort at `:4487-4491`,
   add both types to the dispatch, and reuse `ggml_compute_forward_out_prod_q_f32` (it needs
   only `to_float` + `ggml_type_size`, both defined for F16/BF16; consider renaming or a brief
   comment since "q" no longer means quantized-only). Accumulation stays F32
   (`ggml_vec_mad_f32`), per the ADR-0002 numerics policy.
2. **Work-size planner:** extend the `GGML_OP_OUT_PROD` case in
   `vendor/llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c:2855-2860` so F16/BF16 src0 also get the
   per-thread F32 row buffer (mirror the quantized sizing exactly).
3. **supports_op:** extend `vendor/llama.cpp/ggml/src/ggml-cpu/ggml-cpu.cpp:468-470` to accept
   F16/BF16 src0 under the same shape conditions as quantized src0.
4. **Tests:** `test_out_prod` already generates F16-src0 cases — `base_types` includes F16
   (`vendor/llama.cpp/tests/test-backend-ops.cpp:7740-7749`) and the generation loop is at
   `:8780-8806` — today they report NOT_SUPPORTED on CPU and must flip to passing. BF16 is
   deliberately absent from `base_types` (comment at `:8627`): add explicit BF16
   `test_out_prod` cases. For MODE_GRAD through a `mul_mat(F16 W)` graph, verify at least one
   `test_mul_mat` case with F16 `type_a` actually gradient-checks on CPU: the harness skips
   any case whose param tensor is non-F32, and `test_mul_mat` flags `a` as a param when
   `bs[1]==1 && nr[1]==1`, so pick (or add) a shape where only the F32 `b` is a param — its
   backward then emits `out_prod(W_f16, transpose(grad))` per `ggml.c:6615-6629`. Add the BF16
   twin of that case.
5. **Docs:** update the BLUEPRINT §8 constraint table ("KV cache F32 — CPU F16 OUT_PROD
   aborts"): the CPU-side constraint is lifted by this ticket; note that GPU backends fall
   back to CPU for `OUT_PROD` until their own port tickets land (CUDA C2 is stage-3 work).
6. **Submodule bump PR** in learning-llamas referencing this ticket, per S0-02, so `ci-cpu` runs
   the new cases.

## Out of scope

- CUDA F16/BF16 `OUT_PROD` (ROADMAP §5 C2 — falls out of the CUDA quantized-out_prod traits
  path; stage 3) and Metal/Vulkan `OUT_PROD` ports (stages 2/4).
- SIMD-optimized F16 paths — scalar-first per ROADMAP §4 P3; revisit only if the S1-32
  throughput audit flags it.
- Quantized-src0 out_prod behavior — untouched; this ticket only adds types.
- Removing the F32-KV-cache *code* in `examples/training/finetune.cpp` (upstream example, not
  our training path; learning-llamas's shim never depended on it).

## Acceptance criteria

- [ ] Fork branch: `test-backend-ops` (eval mode) passes on CPU for F16- and BF16-src0
      `OUT_PROD` cases; the previously NOT_SUPPORTED F16 cases demonstrably run (support
      status flips in test output).
- [ ] Fork branch: `test-backend-ops` mode `grad` passes on CPU for at least one
      `mul_mat(F16 W)` case and one `mul_mat(BF16 W)` case whose backward emits an
      F16/BF16-src0 `OUT_PROD`, within the ADR-0002 per-op tolerance.
- [ ] No abort remains reachable for F16/BF16 src0: the dispatch, `supports_op`, and
      work-size planner all agree (grep-verifiable in the fork diff).
- [ ] BLUEPRINT §8 table row for "KV cache F32" is updated in the same learning-llamas PR as the
      submodule bump.
- [ ] learning-llamas submodule-bump PR is green in `ci-cpu` (per-PR lane runs the vendored
      `test-backend-ops` grad + eval cases).

## Testing & verification

Primary harness: vendored `tests/test-backend-ops` in eval and MODE_GRAD modes on the fork
branch, then in learning-llamas's `ci-cpu` lane per-PR after the submodule bump; nightly `ci-cpu`
re-runs the full suite. CPU is the oracle backend, so MODE_GRAD here is finite differences vs
the analytic backward under the ADR-0002 tolerance (S0-09). GPU parity against these cases is
owned by the later backend-port tickets, not this one.

## PR notes

- Branch: `ticket/S1-18-f16-bf16-cpu-out-prod`.
- Two-repo flow per S0-02: implementation PR against the fork's `learning-llamas-base` branch with
  the ticket ID in the title, plus a trivial learning-llamas PR bumping the `vendor/llama.cpp`
  gitlink (and carrying the BLUEPRINT §8 edit), referencing the same ticket ID.
- Upstreaming disposition: **upstream-early** (ROADMAP §11 triage class a) — mainline training
  benefits directly, the change is a pure gap-fill behind existing tests, and upstream's own
  code comments mark the F16 path as TODO.
- No copied external code; in-tree pattern reuse (`ops.cpp` quantized branch) needs no
  provenance header beyond the fork's own history.
