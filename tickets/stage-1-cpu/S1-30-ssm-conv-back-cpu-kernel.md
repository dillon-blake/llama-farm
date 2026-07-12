---
id: S1-30
title: "SSM: SSM_CONV_BACK CPU"
stage: 1
track: kernels
size: S
deps: ["S1-29"]
status: open
pr: null
---

# S1-30 — SSM: SSM_CONV_BACK CPU

**One-line outcome:** the `SSM_CONV_BACK` CPU kernel exists — `d_sx` computed as a
correlation with the flipped conv window, same deterministic row-parallel structure as the
forward — and `test-backend-ops` MODE_GRAD is green for `SSM_CONV` on CPU.

## Why (context)

S1-29 wired the `SSM_CONV` backward-switch case to emit `SSM_CONV_BACK`, but the op has no
compute kernel: mamba backward graphs build and then fail at execution. This ticket
supplies the CPU reference implementation (ROADMAP §10 S2), which doubles as the MODE_GRAD
oracle for the later Metal/CUDA/Vulkan ports (S2-12, S3-09, S4-07 — coordinate in prose
only; all four backends already have SSM *forwards*, so each port is a same-shape sibling
kernel).

The forward (`ggml_compute_forward_ssm_conv_f32`,
`vendor/llama.cpp/ggml/src/ggml-cpu/ops.cpp:9492-9543`) computes, per sequence and token, a
sliding-window rowwise dot product `y(i1, t) = Σ_{i0<d_conv} sx(t+i0, i1) · c(i0, i1)` over
the input `sx` of shape `{d_conv-1+n_t, d_inner, n_seqs}`. Its derivative w.r.t. the input
is the standard conv/correlation transpose: each input column `j` receives contributions
from every window that touched it,
`d_sx(j, i1) = Σ_t dy(i1, t) · c(j−t, i1)` for `j−t ∈ [0, d_conv)` — i.e. a correlation of
`dy` with the *flipped* window (ROADMAP §10 S2). Only `d_sx` is needed: the conv weight
`c` is frozen in LoRA training (ROADMAP §10 frozen-tensor scoping; weight grads are the
deferred S5 item), and S1-29's switch case already asserts that.

The forward's parallelization carries over directly: threads partition the `d_inner` rows
(`vendor/llama.cpp/ggml/src/ggml-cpu/ops.cpp:9512-9517`), and because `d_sx` rows along
`d_inner` are disjoint per thread, the backward is deterministic with no atomics — exactly
the gate G-B default (S0-09/ADR-0002).

## What to do

All code changes land in the vendored llama.cpp fork via the S0-02 two-repo flow.

1. **Kernel:** implement `ggml_compute_forward_ssm_conv_back_f32` (+ the type-dispatch
   wrapper) in `vendor/llama.cpp/ggml/src/ggml-cpu/ops.cpp`, placed next to the forward
   (`:9492-9558`). Inputs per the S1-29 constructor: `sx` (shape/stride reference only),
   `c` (the conv weight), `dy` (`{d_inner, n_t, n_seqs}`); output `d_sx`
   (`{d_conv-1+n_t, d_inner, n_seqs}`). Zero-init the dst rows the thread owns, then for
   each token `t` accumulate `dy(i1, t) · c(i0, i1)` into `d_sx(t+i0, i1)` — or
   equivalently loop output columns with the flipped-window formulation; either is fine as
   long as writes stay within the thread's `d_inner` row range. F32 accumulation
   throughout per ADR-0002.
2. **Threading:** mirror the forward's rows-per-thread split over `d_inner`
   (`ops.cpp:9512-9517`). No atomics; document the determinism argument (disjoint row
   ownership) in a comment referencing gate G-B.
3. **Plumbing:** declare in `ops.h`, add the `GGML_OP_SSM_CONV_BACK` dispatch case in
   `vendor/llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c` next to the forward's case (`:2011`)
   and its `n_tasks` entry (`:2381`); extend CPU `supports_op` if the new op needs an
   explicit entry (mirror however `SSM_CONV` is handled).
4. **MODE_GRAD tests:** `test_ssm_conv` exists
   (`vendor/llama.cpp/tests/test-backend-ops.cpp:3753`; cases at `:8477-8482`) but sets no
   params, so MODE_GRAD skips it. Add grad-enabled cases that `ggml_set_param` the input
   `sx` only (harness convention at `:1984`), covering the existing mamba conv shapes:
   single-token, multi-token (`d_conv-1+64`), and multi-sequence variants. Finite
   differences vs the analytic backward on CPU within the ADR-0002 tolerance.
5. **Flip the S1-29 blocked assertion** for the conv portion of the mamba backward-build
   test (execution through `SSM_CONV_BACK` now runs; the scan portion stays blocked on
   S1-31 if it has not landed).
6. **Submodule bump PR** in learning-llamas referencing this ticket, per S0-02.

## Out of scope

- `d_c` (conv-weight grads) — frozen in LoRA training; ROADMAP §10 S5, deferred.
- `SSM_SCAN_BACK` (S1-31).
- Metal/CUDA/Vulkan `SSM_CONV_BACK` ports (S2-12, S3-09, S4-07) — this kernel is their
  correctness oracle.
- SIMD-optimized variants — scalar-first per ROADMAP §4 P3; revisit only if the S1-32
  throughput audit flags this op.

## Acceptance criteria

- [ ] Fork branch: `test-backend-ops` mode `grad` passes on CPU for the new grad-enabled
      `SSM_CONV` cases (all listed shapes) within the ADR-0002 tolerance.
- [ ] Two runs of the same MODE_GRAD case with different thread counts produce bitwise
      identical `d_sx` (determinism check, gate G-B; scriptable via the harness or a small
      fork-side unit test).
- [ ] The S1-29 mamba backward-build test executes through the `SSM_CONV_BACK` node
      without abort (scan portion may remain marked blocked on S1-31).
- [ ] learning-llamas submodule-bump PR is green in `ci-cpu` (per-PR lane runs the new
      MODE_GRAD cases).

## Testing & verification

Primary harness: vendored `tests/test-backend-ops` MODE_GRAD on CPU — finite differences
vs the analytic kernel under the ADR-0002 tolerance (CPU is the oracle backend, S0-09).
Runs per-PR in learning-llamas's `ci-cpu` lane after the submodule bump; nightly `ci-cpu`
re-runs the full suite. GPU parity against these cases is owned by the backend-port
tickets in stages 2-4.

## PR notes

- Branch: `ticket/S1-30-ssm-conv-back-cpu-kernel`.
- Two-repo flow per S0-02: implementation PR against the fork's `learning-llamas-base` branch
  with the ticket ID in the title, plus a trivial learning-llamas submodule-bump PR referencing
  the same ticket ID.
- Upstreaming disposition: **fork-local first, upstream-later** — rides the `SSM_*_BACK`
  op-family RFC with S1-29/S1-31 once the CPU oracle and one GPU backend prove the design
  (ROADMAP §11 triage b).
- No copied external code; in-tree pattern reuse (the forward kernel's structure) needs no
  provenance header beyond the fork's own history.
