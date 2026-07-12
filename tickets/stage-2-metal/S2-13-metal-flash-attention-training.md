---
id: S2-13
title: "Metal FA7: flash-attention forward LSE + backward"
stage: 2
track: kernels
size: XL
deps: ["S1-23", "S2-02", "S2-03", "S2-04", "S2-05", "S2-06"]
status: open
pr: null
---

# S2-13 — Metal FA7: flash-attention forward LSE + backward

**One-line outcome:** flash-attention training runs on Apple Silicon — the Metal
`FLASH_ATTN_EXT` forward emits LSE per the FA1 ABI, and a three-pass deterministic backward
on the simdgroup base passes MODE_GRAD against the FA3 CPU oracle for F16 K/V at head sizes
64/128.

## Why (context)

Without FA backward, training runs the naive attention path whose `[n_kv, n_q, n_head]` F32
tensors are live across all layers simultaneously — 128-192 GiB at 4k context for an
8B-class model (ROADMAP §8 memory-cliff table); FA backward replaces the n_ctx² term with a
per-row LSE vector. FA7 is Metal's slice of that program, deliberately scheduled **after**
the basic backward suite because FA is not Metal's critical path (ROADMAP §8 FA7) — hence
this ticket's deps on S2-02..S2-06 and on the S1-23 CPU oracle (FA1/FA2, S1-21/S1-22,
arrive transitively through it).

The forward LSE is nearly free: Metal already computes the ingredients on both of its FA
paths. The simdgroup base `kernel_flash_attn_ext_impl`
(`vendor/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:6136`) keeps per-row running sum/max
registers `S[NQ]`/`M[NQ]` (`metal:6257-6260`); the vec split path writes per-workgroup
`[S,M]+O` partials into a fleeting temp buffer — sized `nwg·(ne20 + 2)` F32, "the S and M
values for each intermediate result"
(`vendor/llama.cpp/ggml/src/ggml-metal/ggml-metal-ops.cpp:2640-2643`, sizing fn
`ggml_metal_op_flash_attn_ext_extra_tmp` at `:2620`) — combined by
`kernel_flash_attn_ext_vec_reduce` (`metal:7570`; dispatch at `ggml-metal-ops.cpp:3055-3063`,
which reads each partial's S/M pair). Emitting `lse = M + log(S)` per the FA1 packed `O‖LSE`
ABI (S1-21: F32, sinks folded in, `-INFINITY` for fully-masked rows — the same convention
the CPU forward's discarded `S`/`M` produce,
`vendor/llama.cpp/ggml/src/ggml-cpu/ops.cpp:8496-8497`) is a store epilogue on each path.

The backward is the real work: a new kernel family on the simdgroup base implementing the
FA5 three-pass deterministic scheme (ROADMAP §8 FA5) — (1) `delta = rowsum(dO∘O)`;
(2) dK/dV with the grid over KV tiles, recomputing P from Q/K/mask/LSE with exclusive
writes (GQA accumulation free: each KV head owns its Q heads); (3) dQ with the grid over Q
tiles. One additive mask covers causal/padding/SWA/ALiBi (`fill_mask`,
`vendor/llama.cpp/src/llama-graph.cpp:406-453`), and training graphs cast K/V to F16
(`:2416-2422`), so F16 K/V + arbitrary mask is the whole v1 feature matrix. Numerics per
ROADMAP §12 Q4: recomputed P will not bit-match the forward's, so tolerances are set
against the FA3 CPU oracle under ADR-0002, and the LSE-with-sinks definition must match FA1
exactly. Determinism per gate G-B: three passes with exclusive writes, no atomic-dQ (that
is a K5 opt-in).

## What to do

All code lands in the vendored llama.cpp fork via the S0-02 two-repo flow.

1. **Forward `emit_lse`:** extend the FA kargs with the FA1 flag; store `lse = M + log(S)`
   at the simdgroup-base epilogue (`kernel_flash_attn_ext_impl`, registers at
   `metal:6257-6260`) and in `kernel_flash_attn_ext_vec_reduce` (`metal:7570`) after the
   partial combine; write O at its packed offset via the FA1 accessor views. Guard:
   bitwise-identical O with the flag on/off.
2. **Backward kernels** (new family beside `kernel_flash_attn_ext`, `metal:6774`), on the
   simdgroup base — three dispatches from one encoder `ggml_metal_op_flash_attn_ext_back`:
   pass 1 delta as a row-reduction kernel (S2-02's `simd_sum` pattern), with the delta
   vector in a fleeting scratch sized via `get_alloc_size` (precedent: `extra_tmp`,
   `ggml-metal-ops.cpp:2620`); pass 2 dK/dV over KV tiles; pass 3 dQ over Q tiles. Recompute
   `s = softcap-fold(scale·q·k) + mask` and `P = exp(s − lse)` per S1-23's reference
   semantics (softcap `(1−tanh²)` chained in place; ALiBi slope from max_bias as the
   forward computes it). Simdgroup matrix ops for the QK/PV recompute tiles; F32
   accumulation throughout (ADR-0002); dst is the FA1 packed `dq‖dk‖dv`.
3. **Scope v1:** F16 K/V (the training-graph case), head sizes DK=DV=64 and 128; no MLA
   geometries (DK 576/DV 512), no sink gradients (sinks frozen; they enter only through
   LSE), no quantized K/V. Head size 256 is an in-ticket follow-up if the timebox allows —
   otherwise a backlog note (smem/occupancy risk per ROADMAP §12 Q5 rhymes with CUDA's).
4. **supports_op gating, precise:** the backward case in `ggml-metal-device.m` mirrors the
   forward's explicit head-size whitelist style (`:1228-1243`) but lists **only** the
   implemented sizes {64, 128}, requires `has_simdgroup_mm`, F16 K/V, and same-type K/V;
   the `emit_lse` forward variant keeps the forward's gates. Everything else falls back to
   the CPU kernel via sched (ROADMAP §11 scheduler note).
5. **MODE_GRAD:** run the S1-22/S1-23 `test_flash_attn_ext` grad matrix
   (`vendor/llama.cpp/tests/test-backend-ops.cpp:6612`) on Metal for the v1 scope — mask
   off/causal-style/ALiBi, softcap on/off, sinks on/off, GQA ratios — vs the FA3 CPU
   oracle, ≤ 0.05 @ fp16 parity per ADR-0002; per-case tolerance overrides only with an
   in-code justification citing ROADMAP §12 Q4. Also run S1-23's naive-parity comparison on
   Metal outputs where the harness permits.
6. **e2e memory-win evidence:** one long-context (2-4k) tiny-model training run on
   `--device metal` with FA backward, compared against the same run on the S1-24 FA8
   chunked fallback: record peak memory and tok/s for both in the run report (extend
   `docs/perf/metal.md` from S2-10 with an FA section). FA remains off by default in
   learning-llamas training graphs — the flip is owned by the later integration ticket.
7. **Submodule bump PR** in learning-llamas per S0-02; the `ci-metal` nightly picks up the new
   grad cases and the FA e2e comparison job (dispatch-triggered; it is expensive).

## Out of scope

- CUDA (FA4/FA5) and Vulkan (FA6) FA training — stage-3/4 tickets; FA8 chunked fallback is
  S1-24 and stays the default long-context path until milestones flip FA on.
- MLA head sizes, sink gradients, quantized-KV backward (excluded by construction —
  training graphs cast K/V), and BF16/F32 K/V variants.
- Vec-path backward, mma/Metal4 `matmul2d` tensor API, atomic-dQ single-pass — K5 perf
  work (the Metal4 API is disabled pre-M5 hardware; target the legacy simdgroup path,
  ROADMAP §6).
- Enabling FA in learning-llamas training graphs by default — later integration ticket.

## Acceptance criteria

- [ ] Fork branch: forward `emit_lse` produces LSE matching the CPU forward's within
      ADR-0002 tolerance on both Metal paths (simdgroup and vec/reduce), including
      sinks-on and fully-masked-row cases; O is bitwise-unchanged with the flag off.
- [ ] Fork branch: `test-backend-ops` mode `grad` passes on Metal for the full v1
      FLASH_ATTN_EXT backward matrix (head sizes 64/128, F16 K/V, mask/ALiBi/softcap/sinks
      variants, GQA ratios) vs the FA3 CPU oracle within ADR-0002 / documented Q4
      overrides.
- [ ] Determinism: bitwise-identical dq‖dk‖dv across two Metal runs on identical inputs.
- [ ] supports_op returns true exactly for the implemented feature set; unsupported
      shapes demonstrably fall back to CPU (one sched-report log linked in the PR).
- [ ] The FA-vs-FA8 comparison report exists (peak memory + tok/s at ≥2k ctx on Apple
      Silicon) and shows the expected memory win; `docs/perf/metal.md` gains the FA
      section.
- [ ] learning-llamas submodule-bump PR green in `ci-metal / build` and `ci-metal / grad`;
      nightly full sweep green.

## Testing & verification

Primary harness: vendored `tests/test-backend-ops` MODE_GRAD (Metal vs the S1-23 CPU
oracle) — targeted per-PR in `ci-metal / grad`, full matrix in the nightly sweep. The
naive-parity and determinism checks reuse S1-23's fork-side harness pointed at the Metal
backend. The memory-win comparison runs as a dispatch-triggered `ci-metal` job on real
Apple Silicon (S0-08 playbook — paravirtual-GPU memory numbers are not representative);
its report is the e2e artifact.

## PR notes

- Branch: `ticket/S2-13-metal-flash-attention-training`.
- Two-repo flow per S0-02: fork PR against `learning-llamas-base` staged as (1) forward
  emit_lse, (2) backward passes + supports_op, (3) tests + comparison job; plus a trivial
  learning-llamas submodule-bump PR.
- Upstreaming disposition: **fork-local first, upstream-later** — rides the FA-training
  op-family RFC (FA1 ABI + FA2 + FA3 + backends) once the CPU oracle plus one GPU backend
  prove the design (ROADMAP §11 triage b; the `emit_lse` ABI is the fork-carried piece).
- Provenance per S0-01: kernels carry headers naming their pattern source
  (`kernel_flash_attn_ext_impl`, `ggml-metal.metal`, MIT, commit `4f37f51`).
- Size XL: flag slippage early — S1-24 (FA8) is the standing long-context fallback on
  Metal, so this ticket never blocks a milestone (ROADMAP §12 risk 12 analog).
