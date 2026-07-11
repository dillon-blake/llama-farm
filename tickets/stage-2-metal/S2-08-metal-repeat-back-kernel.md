---
id: S2-08
title: "Metal M6: REPEAT_BACK kernel"
stage: 2
track: kernels
size: M
deps: ["S2-01"]
status: open
pr: null
---

# S2-08 — Metal M6: REPEAT_BACK kernel

**One-line outcome:** `REPEAT_BACK` runs on Metal as a one-thread-per-dst-element kernel that
loop-accumulates all source copies in fixed order — deterministic, no atomics — with MODE_GRAD
parity against the CPU oracle including broadcast-heavy shapes.

## Why (context)

`REPEAT_BACK` is the reduction dual of broadcasting, and autograd emits it in four places that
all occur in real training backward graphs: the backward of `REPEAT` itself
(`vendor/llama.cpp/ggml/src/ggml.c:6561-6565`), broadcast `ADD` (`:6461-6465` — the bias-add
backward pattern), broadcast `MUL` (`:6505-6510` — per-channel scale backward), and the
batched-`MUL_MAT` src0-gradient path (`:6605-6612`, via a view). Metal has no kernel for it:
`GGML_OP_REPEAT_BACK` has no case in the coverage switch
(`vendor/llama.cpp/ggml/src/ggml-metal/ggml-metal-device.m:1051-1368`), so these nodes fall
back to CPU on every step (ROADMAP §6 M6, part of the K2 backward suite).

The design is the inverse of `kernel_repeat`
(`vendor/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:1395-1418`): where the forward assigns
one thread per **dst** element reading its mod-indexed src element, the backward assigns one
thread per **dst** element that *sums over all of its `nr0·nr1·nr2·nr3` copies* in the larger
src tensor, in a fixed loop order. Each output element has exactly one writer, so the kernel is
deterministic with no atomics — the scheme ADR-0002 gate G-B mandates as the project default.
CUDA's `k_repeat_back` (`vendor/llama.cpp/ggml/src/ggml-cuda/binbcast.cu:359-386`) implements
the same shape with strided accumulation loops and is the closest GPU precedent; note it gates
`ne2·ne3 ≤ 2¹⁵` in supports_op (`vendor/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:4806-4807`)
because of grid-dimension limits — Metal's dispatch should either avoid such a limit or gate it
precisely. The CPU oracle is `ggml_compute_forward_repeat_back_f32`
(`vendor/llama.cpp/ggml/src/ggml-cpu/ops.cpp:1822-1878`): zero-fill then nested accumulation,
F32-only — which is all autograd emits.

## What to do

All code changes land in the vendored llama.cpp fork via the S0-02 two-repo flow.

1. **Kernel** `kernel_repeat_back` in `vendor/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal`,
   F32 first (match the CPU oracle's type coverage). Grid = dst elements (one thread each, or
   a grid-stride row loop mirroring `kernel_repeat`'s structure at `metal:1395-1418`); each
   thread accumulates a `float` sum over its copies with the repeat strides, then writes once.
   Support non-contiguous src via the nb strides in kargs — the existing eval cases include
   noncontiguous-view variants (see step 4).
2. **The five mechanical additions** (ROADMAP §6 preamble): kargs struct in
   `ggml-metal-impl.h`, pipeline getter in `ggml-metal-device.cpp`, encoder case in
   `ggml-metal-ops.cpp`, and a `GGML_OP_REPEAT_BACK` supports_op case in
   `ggml-metal-device.m` returning true for F32 with the stride patterns the kernel handles.
3. **Determinism:** state the exclusive-writer scheme in the PR per ADR-0002 Decision 3; no
   atomic variant in this ticket.
4. **Tests — forward eval:** the existing `test_repeat_back` cases
   (`vendor/llama.cpp/tests/test-backend-ops.cpp:8236-8240`, struct at `:2784`), including the
   noncontiguous-view variants, light up on Metal once supports_op returns true; they compare
   against CPU automatically.
5. **Tests — MODE_GRAD:** `test_repeat` flags its src as a param
   (`vendor/llama.cpp/tests/test-backend-ops.cpp:2773`, struct at `:2750`), so its grad cases
   exercise `REPEAT_BACK` as the emitted backward on Metal. Additionally ensure grad coverage
   for the broadcast patterns the manifest calls out: a bias-add-shaped case (row vector
   broadcast-added over an `[n_embd, n_tokens]` activation, grads to the vector) and a
   broadcast-`MUL` case — add cases only if the existing generated matrix lacks them (audit
   first, as S1-20 did for soft_max).
6. **Submodule bump PR** in llama-farm per S0-02; confirm the S2-01 nightly fallback report
   drops `REPEAT_BACK` for the tiny dense model.

## Out of scope

- `GET_ROWS_BACK` on Metal — deferred while embeddings are frozen (ROADMAP §6 M9,
  `tickets/backlog/`).
- CUDA's `ne2·ne3` limit lift — CUDA-side concern, not this ticket.
- Non-F32 `REPEAT_BACK` (no producer today; the CPU oracle is F32-only).
- The stage-2 exit milestone and fallback-forbidden CI flip — S2-10.

## Acceptance criteria

- [ ] Fork branch: `test-backend-ops` mode `grad` passes on Metal for the `REPEAT` grad cases
      and the broadcast ADD/MUL grad cases (per-op ADR-0002 tolerance), and Metal-vs-CPU
      gradients on identical inputs meet the ≤ 0.05 @ fp16 parity criterion.
- [ ] Fork branch: all `test_repeat_back` forward-eval cases pass on Metal, including the
      noncontiguous-view variants (previously NOT_SUPPORTED).
- [ ] Determinism check: two consecutive runs of the same `REPEAT_BACK` case on Metal produce
      bitwise-identical outputs (scripted in the fork PR, per ADR-0002 gate G-B).
- [ ] supports_op advertises exactly what the kernel handles (F32; stride patterns tested).
- [ ] S2-01 nightly e2e fallback report for the tiny dense model no longer contains
      `REPEAT_BACK`.
- [ ] llama-farm submodule-bump PR is green in `ci-metal / build` and `ci-metal / grad`.

## Testing & verification

Primary harness: vendored `tests/test-backend-ops` — forward eval (`test_repeat_back`) and
MODE_GRAD (`test_repeat`, broadcast binary cases) on the Metal backend vs the CPU oracle
(`ops.cpp:1822-1878`), under ADR-0002 tolerances. Per-PR: targeted `ci-metal / grad`; nightly:
full Metal op sweep plus the `--device metal` e2e whose sched log provides the fallback
evidence. This op is on the S2-10 zero-fallback critical list for the dense path.

## PR notes

- Branch: `ticket/S2-08-metal-repeat-back-kernel`.
- Two-repo flow per S0-02: fork PR (`llama-farm-base`) + trivial llama-farm submodule-bump
  PR, both referencing the ticket ID.
- Upstreaming disposition: **upstream-early** (ROADMAP §11 triage class a — pure addition
  behind supports_op; the op and its tests already exist upstream, only the Metal kernel is
  new).
- Provenance per S0-01: kernel header names the pattern sources
  `ggml/src/ggml-metal/ggml-metal.metal` (`kernel_repeat`) and
  `ggml/src/ggml-cuda/binbcast.cu` (`k_repeat_back`), both MIT, commit `4f37f51`.
