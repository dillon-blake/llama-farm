---
id: S3-04
title: "CUDA C4: SOFT_MAX_BACK ALiBi gate lift"
stage: 3
track: kernels
size: S
deps: [S1-20, S3-01]
status: open
pr: null
---

# S3-04 — CUDA C4: SOFT_MAX_BACK ALiBi gate lift

**One-line outcome:** ALiBi (`max_bias > 0`) `SOFT_MAX_BACK` is accepted on CUDA — no kernel
change, two gates removed, the S1-20 MODE_GRAD test green on CUDA.

## Why (context)

K-SMB (ROADMAP §3, §5 C4): the ALiBi bias is additive and constant w.r.t. the logits, so the
existing `SOFT_MAX_BACK` kernels are already mathematically correct for `max_bias > 0` — the
backward `dx = scale · y ∘ (dy − dot(y, dy))` never involves `max_bias`. That is directly
visible in the CUDA kernel: `soft_max_back_f32`
(`vendor/llama.cpp/ggml/src/ggml-cuda/softmax.cu:250-270`) takes only `scale` and computes
exactly that formula (`:268`); the host wrapper reads `max_bias` from op-params solely to
assert it is zero and passes only `scale` to the launch
(`vendor/llama.cpp/ggml/src/ggml-cuda/softmax.cu:464-471`). The restriction is gating, not
math, and it lives in exactly two places on CUDA: the runtime assert
`GGML_ASSERT(max_bias == 0.0f)` (`vendor/llama.cpp/ggml/src/ggml-cuda/softmax.cu:469`) and
the `supports_op` case that returns `max_bias == 0.0f`
(`vendor/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:4903-4907`).

S1-20 did the load-bearing work: it established that no test exercised `max_bias > 0`
backward anywhere, enabled the generated MODE_GRAD `test_soft_max` cases with
`max_bias = 8.0` (loop at `vendor/llama.cpp/tests/test-backend-ops.cpp:8882`) and the
forward-eval `test_soft_max_back` `max_bias` cases (loop at `:8925`) by lifting the CPU
assert and CPU `supports_op` gate, and left breadcrumb comments pointing here for the CUDA
gates. With the CPU baseline in place, lifting the CUDA gates makes those same cases run as
cross-backend parity tests automatically. Until this lands, `ggml_backend_sched` sends every
ALiBi `SOFT_MAX_BACK` node to CPU — correct but not GPU-resident, and a residual entry in
ci-cuda's fallback report that S3-10's fallback-forbidden flip would otherwise trip over for
ALiBi models on the naive-attention path.

## What to do

All changes land in the vendored llama.cpp fork via the S0-02 two-repo flow.

1. **Capture the before state:** on the fork branch, run
   `test-backend-ops grad -b CUDA0 -o SOFT_MAX` and
   `test-backend-ops test -b CUDA0 -o SOFT_MAX_BACK`; record in the fork PR description
   that the `max_bias = 8.0` cases report NOT_SUPPORTED on CUDA (mirroring S1-20's
   test-first discipline).
2. **Remove the runtime assert** at `vendor/llama.cpp/ggml/src/ggml-cuda/softmax.cu:469`,
   keeping the surrounding F32 type asserts (`:456-458`). The `max_bias` op-param read may
   stay (harmless) or go; the launch already passes only `scale` (`:471`).
3. **Lift the `supports_op` gate** at
   `vendor/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:4903-4907`: the case only checks
   `max_bias`, so it reduces to `return true;` — keep a one-line comment citing K-SMB and
   the kernel-reads-only-scale fact.
4. **Remove the S1-20 breadcrumb comment** that points at S3-04, replacing it with nothing
   (the gate is gone). Leave S1-20's Vulkan breadcrumb (owned by S4-05) untouched; this diff
   must not touch Vulkan or CPU source.
5. **Re-run and confirm:** the `max_bias = 8.0` MODE_GRAD cases (single- and multi-head,
   per-head slope variation, F16/F32 masks — the `:8882` loop matrix) now execute and pass
   on CUDA against finite differences, and the forward `test_soft_max_back` `max_bias`
   cases (`:8925`) pass against the CPU baseline S1-20 enabled.
6. **Submodule bump PR** in llama-farm per S0-02. The S1-11 preflight widens automatically
   via its graph walk; no llama-farm code change.

## Out of scope

- Any kernel-body change — none is needed (ROADMAP §5 C4).
- Vulkan `max_bias` validation/audit (its shader ignores the parameter, unvalidated) —
  S4-05 (ROADMAP §7 V5).
- Metal `SOFT_MAX_BACK` — S2-02.
- The FA-training path — FA backward recomputes P from Q/K/mask/LSE, so the `max_bias`
  restriction never applied there (ROADMAP §8).

## Acceptance criteria

- [ ] Fork PR description contains the captured before/after output (NOT_SUPPORTED →
      passing) for the CUDA `max_bias > 0` cases.
- [ ] Fork branch: `test-backend-ops grad -b CUDA0 -o SOFT_MAX` passes for every
      `max_bias > 0` case (MODE_GRAD within the ADR-0002 per-op tolerance), including a
      multi-head case where per-head ALiBi slopes differ.
- [ ] Fork branch: `test-backend-ops test -b CUDA0 -o SOFT_MAX_BACK` passes for the
      `max_bias > 0` cases vs the CPU oracle (within the ADR-0002 cross-backend criterion,
      ≤ 0.05 max-abs @ fp16).
- [ ] The fork diff touches only `ggml/src/ggml-cuda/softmax.cu` and the `supports_op` case
      in `ggml/src/ggml-cuda/ggml-cuda.cu` — no kernel-body, CPU, or Vulkan changes
      (grep-verifiable).
- [ ] llama-farm submodule-bump PR is green in `ci-cuda` (compile + GPU lanes) and `ci-cpu`.

## Testing & verification

Primary harness: vendored `tests/test-backend-ops` — MODE_GRAD (`grad -b CUDA0`) for the
gradient checks and eval mode (`test -b CUDA0`) for `SOFT_MAX_BACK` forward parity against
the CPU baseline from S1-20, tolerances per ADR-0002 (S0-09). Runs on the fork branch CI,
then in llama-farm's `ci-cuda` GPU lane per-PR (kernel-gated; `SOFT_MAX` is in the targeted
set for this diff) and the nightly full sweep (S3-01). No new test code is written — this
ticket exists to let already-written tests execute on CUDA.

## PR notes

- Branch: `ticket/S3-04-cuda-soft-max-back-alibi`.
- Two-repo flow per S0-02: fork PR (`llama-farm-base`) + trivial llama-farm submodule-bump
  PR, both carrying the ticket ID.
- Upstreaming disposition: **upstream-early**, coordinated with S1-20's upstream PR (ROADMAP
  §11 triage class a) — assert/gate removals with existing tests, no new ABI; if S1-20's
  upstream PR is still open, fold this into it rather than filing separately.
- No copied external code; no provenance headers needed.
