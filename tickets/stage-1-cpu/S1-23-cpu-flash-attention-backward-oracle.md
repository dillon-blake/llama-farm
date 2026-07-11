---
id: S1-23
title: "FA3: CPU flash-attention backward (modernize legacy kernel — the GPU oracle)"
stage: 1
track: kernels
size: L
deps: ["S1-21", "S1-22"]
status: open
pr: null
---

# S1-23 — FA3: CPU flash-attention backward (modernize legacy kernel — the GPU oracle)

**One-line outcome:** a correct, deterministic CPU implementation of
`ggml_flash_attn_ext_back` supporting arbitrary additive mask, op-param scale, ALiBi
slope, softcap, F16 K/V, and LSE input — MODE_GRAD-green and the correctness oracle
for all GPU FA backward work.

## Why (context)

Without FA backward, training runs the naive `MUL_MAT → SOFT_MAX(mask) → MUL_MAT`
attention path, whose `[n_kv, n_q, n_head]` F32 tensors are live across all layers at
once — 128-192 GiB at n_ctx 4096 for a Llama-3.1-8B-class model, i.e. infeasible
(ROADMAP §8 memory-cliff table). FA backward replaces the n_ctx² term with an LSE
vector. This ticket delivers the CPU half of that (ROADMAP §8 FA3): every later GPU FA
backward (CUDA/Vulkan/Metal, stages 2-4) validates against this kernel under the
ADR-0002 cross-backend parity criterion.

This is a modernization, not a from-scratch build. A complete legacy CPU FA backward
exists in-tree, `ggml_compute_forward_flash_attn_back_f32`
(`vendor/llama.cpp/ggml/src/ggml-cpu/ops.cpp:9156-9488`), with the full derivation in
comments — `dS = P·(dP − dot(P,dP))` and the post-order gradient plan
(`vendor/llama.cpp/ggml/src/ggml-cpu/ops.cpp:9337-9398`) — and a deterministic
no-atomics parallelization: threads own kv-head×batch rows (`nr = nek2·nek3`,
`vendor/llama.cpp/ggml/src/ggml-cpu/ops.cpp:9244-9253`) with the GQA repeat loop
inside, so all dq/dk/dv writes are thread-exclusive. It is unreachable (the old
constructor aborts, `vendor/llama.cpp/ggml/src/ggml.c:5470` — S1-21 replaced that API)
and outdated: F32-only, causal-flag-only (`masked_begin`,
`vendor/llama.cpp/ggml/src/ggml-cpu/ops.cpp:9290`), hardcoded `1/sqrt(D)` scale
(`vendor/llama.cpp/ggml/src/ggml-cpu/ops.cpp:9256`), and it recomputes a full softmax
per row instead of consuming the forward's LSE.

One backward signature covers all model variants because llama.cpp bakes
causal/padding/SWA/ALiBi into a single additive mask — `fill_mask`,
`vendor/llama.cpp/src/llama-graph.cpp:406-453` (F16 on the FA path, F32 otherwise).
The `SOFT_MAX_BACK` `max_bias==0` restriction
(`vendor/llama.cpp/ggml/src/ggml-cpu/ops.cpp:5539`) does not apply here: backward
recomputes P directly from Q/K/mask/LSE. Training graphs have no KV cache — K/V arrive
as F32→F16 casts (`vendor/llama.cpp/src/llama-graph.cpp:2416-2422`) — so F16 K/V
support is required and quantized-KV backward is out of scope by construction
(ROADMAP §8). Numerics (ROADMAP §12 Q4): the forward's FTZ threshold and KQ max-offset
mean recomputed P will not bit-match the forward's P; tolerances are set against
finite differences and the naive path per ADR-0002, and the LSE-with-sinks definition
must match FA1 exactly (the forward folds sinks into its running max/sum `M`/`S`,
`vendor/llama.cpp/ggml/src/ggml-cpu/ops.cpp:8496-8497`, so LSE includes the sink
denominator).

## What to do

All code lands in the vendored llama.cpp fork via the S0-02 two-repo flow.

1. **New kernel** `ggml_compute_forward_flash_attn_ext_back` in
   `vendor/llama.cpp/ggml/src/ggml-cpu/ops.cpp`, implementing the S1-21 op. Start from
   the legacy kernel (`ops.cpp:9156-9488`): keep the `dS = P·(dP − dot(P,dP))`
   structure and update the derivation comments (`:9337-9398`); keep the deterministic
   parallelization (threads own kv-head×batch rows, GQA loop inside, `:9244-9253`) —
   no atomics, per gate G-B defaults (S0-09/ADR-0002).
2. **Op-param and mask generality:** read scale, max_bias, and softcap from op-params
   (replacing the hardcoded scale at `ops.cpp:9256`); drop the causal flag/diag-mask
   logic (`ops.cpp:9290`) in favor of the arbitrary additive mask src (F16 or F32,
   read via to_float) — one mask covers causal/padding/SWA/ALiBi
   (`llama-graph.cpp:406-453`). Compute the per-head ALiBi slope from max_bias exactly
   as the forward does (`vendor/llama.cpp/ggml/src/ggml-cpu/ops.cpp:8493-8494`).
   Softcap: recompute `tanh` in place during the P recompute and chain the `(1−tanh²)`
   factor into dS.
3. **LSE instead of softmax recompute:** per (q-row, k-row), recompute
   `s = softcap-fold(scale·q·k) + mask (+ slope term via mask)` and take
   `P = exp(s − lse_row)` from the FA1 LSE input — delete the legacy per-row
   softmax recompute. Define `delta = rowsum(dO∘O)` and document in comments why it
   equals `dot(P,dP)` under the FA1 sink convention (sinks contribute denominator
   mass but no V rows; sink grads are not computed — sinks frozen).
4. **F16 K/V** via the forward's type-traits mechanism (`q_to_vec_dot` from_float /
   `v_to_float`, `vendor/llama.cpp/ggml/src/ggml-cpu/ops.cpp:8478-8483`); dst stays
   packed F32 `dq‖dk‖dv` per the FA1 ABI.
5. **CPU plumbing** in `vendor/llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c`: dispatch case
   (pattern: the legacy dispatch at `:2004-2010`), `n_tasks = n_threads`
   (pattern `:2380-2385`), and per-op wdata sizing (pattern `:2936-2945`). Remove the
   now-dead legacy kernel and any `GGML_OP_FLASH_ATTN_BACK` plumbing S1-21 left behind.
6. **supports_op:** CPU returns true for the new op
   (`ggml_backend_cpu_device_supports_op`,
   `vendor/llama.cpp/ggml/src/ggml-cpu/ggml-cpu.cpp:423`); all GPU backends keep
   returning false (sched falls back to CPU, ROADMAP §11 scheduler note). This flips
   the S1-22 MODE_GRAD cases from not-supported skips to executing.
7. **MODE_GRAD test matrix** extending the S1-22 `test_flash_attn_ext` grad cases
   (`vendor/llama.cpp/tests/test-backend-ops.cpp:6612`): head sizes 64/128/256, GQA
   ratios, mask off / causal-style additive mask / ALiBi (max_bias > 0), softcap
   on/off, sinks on/off, F16 and F32 K/V. Small shapes for finite-difference cost.
   Where FTZ-induced tolerance pressure appears, use per-case `max_maa_err` overrides
   (`vendor/llama.cpp/tests/test-backend-ops.cpp:1158`) with a justification comment
   citing ROADMAP §12 Q4 — never blanket-loosen the default ADR-0002 bound.
8. **Naive-parity test** (new fork-side test, e.g. `tests/test-fa-backward-parity.cpp`):
   build the same attention twice on identical inputs — naive
   `mul_mat → [softcap] → soft_max_ext(mask) → mul_mat`
   (`vendor/llama.cpp/src/llama-graph.cpp:2450-2494` shape) and
   `FLASH_ATTN_EXT(emit_lse)` — run both backwards, compare dq/dk/dv within a
   documented tolerance derived from ADR-0002. Same binary also asserts
   **determinism**: two runs at different thread counts produce bitwise-identical
   grads.
9. **Submodule bump PR** in llama-farm per S0-02 so `ci-cpu` runs everything.

## Out of scope

- GPU FA backward and forward-LSE emission (FA4-FA7, stage 2/3/4 tickets — they
  consume this oracle) and the kernel-free chunked fallback (S1-24, independent).
- Quantized-KV backward (excluded by construction — training graphs cast K/V) and
  sink gradients (sinks frozen in LoRA training).
- SIMD tuning — scalar-first per ROADMAP §4 P3; revisit only if the S1-32 audit
  flags it.
- Flipping FA on in llama-farm training graphs — the naive path remains default until
  backend milestones flip FA on (later integration ticket owns the switch and the
  preflight report entry).

## Acceptance criteria

- [ ] Fork branch: `test-backend-ops` mode `grad` passes on CPU for the full
      FLASH_ATTN_EXT case matrix of step 7 within ADR-0002 tolerances (any per-case
      override justified in-code against ROADMAP §12 Q4).
- [ ] The naive-parity test passes: dq/dk/dv from the FA path match the naive-path
      grads within the documented tolerance, for mask/ALiBi/softcap/sinks/F16-KV
      variants.
- [ ] Determinism test passes: bitwise-identical grads across different `n_threads`.
- [ ] The S1-22 grad cases previously skipped as not-supported now execute and pass
      on CPU (no FLASH_ATTN_EXT grad skip in the CPU run log).
- [ ] No references to the legacy `ggml_compute_forward_flash_attn_back` remain in the
      fork (grep-verifiable).
- [ ] llama-farm submodule-bump PR green in `ci-cpu` per-PR; nightly `ci-cpu` runs the
      full grad suite.

## Testing & verification

Primary harness: vendored `tests/test-backend-ops` MODE_GRAD (finite differences vs
analytic, CPU oracle) plus the dedicated naive-parity/determinism test — both on the
fork branch CI and, after the submodule bump, in llama-farm's `ci-cpu` lane per-PR
(S0-07); nightly `ci-cpu` re-runs the full suite. Stage-2/3/4 FA backward tickets
verify against this kernel under the ADR-0002 cross-backend criterion (max-abs gradient
error ≤ 0.05 at fp16), so this reference must be MODE_GRAD-clean first. End-to-end
FA-on convergence is deferred to the milestone that flips FA on (S1-12 gate there).

## PR notes

- Branch: `ticket/S1-23-cpu-flash-attention-backward-oracle`.
- Two-repo flow per S0-02: fork PR against `llama-farm-base` (ticket ID in title) +
  trivial llama-farm submodule-bump PR referencing the same ID.
- Upstreaming disposition: **upstream-later** — part of the FA-training op-family RFC
  (FA1 ABI + FA2 + FA3) once the CPU oracle plus one GPU backend prove the design
  (ROADMAP §11 triage b).
- Size L: stage commits within one fork PR — (1) kernel skeleton with F32/mask/scale,
  (2) LSE + softcap + ALiBi + F16 K/V, (3) plumbing/supports_op + tests.
- The kernel modernizes in-tree MIT code (same file, llama.cpp `4f37f51`); keep a
  provenance note in the function comment per S0-01 policy.
