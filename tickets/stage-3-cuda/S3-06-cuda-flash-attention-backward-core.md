---
id: S3-06
title: "CUDA FA5 (core): flash-attention backward, head sizes 64/128"
stage: 3
track: kernels
size: XL
deps: ["S3-05", "S1-23", "S0-09"]
status: open
pr: null
---

# S3-06 — CUDA FA5 (core): flash-attention backward, head sizes 64/128

**One-line outcome:** the flagship kernel — a deterministic three-pass flash-attention
backward on the CUDA tile family for D=64/128 with F16 K/V — making long-context
training fully GPU-resident on CUDA.

## Why (context)

Without FA backward, training runs the naive `MUL_MAT → SOFT_MAX(mask) → MUL_MAT`
attention path whose `[n_kv, n_q, n_head]` F32 tensors are live across all layers at
once: 128-192 GiB at n_ctx 4096 for a Llama-3.1-8B-class model — infeasible
(ROADMAP §8 memory-cliff table). FA backward recomputes P per tile from Q/K/mask/LSE,
replacing the n_ctx² term with an LSE vector (~512 KiB/layer at 4k ctx): the difference
between "8B LoRA at 4k context on a 24 GB GPU" and "not possible". The ABI (S1-21:
`ggml_flash_attn_ext_back(q,k,v,mask,sinks,o,dO,lse,…) → dq‖dk‖dv`), autograd wiring
(S1-22), CPU oracle (S1-23), and CUDA forward LSE (S3-05) all exist; this ticket
executes the op on CUDA (ROADMAP §8 FA5).

Base it on the **tile family, not mma**, for v1 (ROADMAP §8 FA5): the tile kernel
(`flash_attn_tile`, `vendor/llama.cpp/ggml/src/ggml-cuda/fattn-tile.cuh:794`) is a
plain shared-memory design that runs on every supported arch, and its per-arch config
tables (`vendor/llama.cpp/ggml/src/ggml-cuda/fattn-tile.cuh:22-340`, packed
nthreads/occupancy/nbatch_fa/nbatch_K via the macro at `:13-19`, accessors at
`:345-375`) are the tuning pattern the backward reuses; mma-family backward is a later
perf phase (backlog B-02). One backward signature covers all model variants because
llama.cpp bakes causal/padding/SWA/ALiBi into a single additive mask (`fill_mask`,
`vendor/llama.cpp/src/llama-graph.cpp:406-453`); training graphs bypass the KV cache once
S1-00 lands — K/V then arrive as F32→F16 casts
(`vendor/llama.cpp/src/llama-graph.cpp:2416-2422`) — so F16 K/V is the one storage type needed
and quantized-KV backward is out of scope by construction.

Two decided constraints bind the design. **Determinism (gate G-B, ADR-0002/S0-09):** no
atomics in v1 — dK/dV and dQ use exclusive-write grids (the CPU oracle's
threads-own-kv-heads scheme, `vendor/llama.cpp/ggml/src/ggml-cpu/ops.cpp:9244-9253`, is
the same idea); atomic-dQ is opt-in later (backlog B-02). **Numerics (ROADMAP §12
Q4):** the forward's FTZ threshold and KQ max-offset mean recomputed P will not
bit-match the forward's P; acceptance is MODE_GRAD tolerance vs the S1-23 CPU oracle
per ADR-0002, and the LSE-with-sinks definition must match the S1-21 contract exactly.
Expected scope ~1.5-2.5k kernel lines; size XL.

## What to do

All code lands in the vendored llama.cpp fork via the S0-02 two-repo flow.

1. **New kernel files** `vendor/llama.cpp/ggml/src/ggml-cuda/fattn-back.cuh` /
   `fattn-back.cu` implementing the repurposed `GGML_OP_FLASH_ATTN_BACK` (S1-21 enum):
   dispatch case next to `GGML_OP_FLASH_ATTN_EXT` in the op switch
   (`vendor/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:2224-2226`) and a `supports_op`
   case next to `:4961-4962`. Follow the tile family's structure: kernel template over
   (DKQ, DV, ncols, use_logit_softcap) with a per-arch config table and accessors
   patterned on `fattn-tile.cuh:13-19` / `:22-340` / `:345-375` (conservative shapes
   for D=64/128; S3-07 owns tuning and D=256).
2. **Three deterministic passes** (grids chosen so every output element has exactly one
   writer):
   - **Pass 1 — delta:** `delta = rowsum(dO ∘ O)` per (q-position × head), a small
     reduction kernel writing an `[n_q, n_head, n_batch]` F32 pool buffer. Document in a
     comment why delta equals `dot(P, dP)` under the FA1 sink convention (sinks
     contribute denominator mass but no V rows).
   - **Pass 2 — dK/dV:** grid over KV tiles; each block loads its K/V tile once,
     iterates Q tiles, recomputes `s = softcap-fold(scale·K·Q) + mask` and
     `P = exp(s − lse_row)` from Q/K/mask/LSE, forms `dP = dO·Vᵀ`,
     `dS = P ∘ (dP − delta)` (chain `(1 − tanh²)` into dS when softcap is set), and
     accumulates `dV += Pᵀ·dO`, `dK += scale·dSᵀ·Q` into exclusive outputs. **GQA
     accumulation is free here:** the KV-head block loops over its gqa_ratio Q heads
     inside, so dK/dV never needs cross-block reduction.
   - **Pass 3 — dQ:** grid over Q tiles; each block iterates KV tiles, recomputes P and
     dS the same way, accumulates `dQ += scale·dS·K` exclusively. No atomic dQ in v1
     (gate G-B); the pass structure must leave room for the opt-in variant (B-02).
3. **Op semantics per the FA1 ABI:** inputs q, k, v, mask (F16 additive, optional),
   sinks (optional), o, dO, lse; op-params scale, max_bias (per-head ALiBi slope
   computed exactly as the forward does), logit_softcap; output packed `dq‖dk‖dv` F32.
   Sinks participate in P via the LSE only — **no sink gradients** (sinks frozen in LoRA
   training).
4. **Scope gates in `supports_op`:** accept F16 K/V, DKQ == DV ∈ {64, 128}, packed-dst
   back op; reject D=256 (S3-07), MLA DKQ=576, quantized K/V, and anything else — sched
   falls back to CPU (ROADMAP §11 scheduler note), so partial coverage is safe.
5. **Launch/transient plumbing:** reuse `launch_fattn`'s conventions
   (`vendor/llama.cpp/ggml/src/ggml-cuda/fattn-common.cuh:973`) where they fit (pool
   allocs for delta and staging), but keep the backward launcher separate — three
   kernels, no combine step.
6. **MODE_GRAD tests:** extend the S1-22 grad cases (`test_flash_attn_ext`,
   `vendor/llama.cpp/tests/test-backend-ops.cpp:6612`) so the CUDA backend executes
   them: D=64/128, GQA ratios {1, 4}, mask off / causal-style mask / ALiBi
   (`max_bias > 0`), softcap on/off, sinks on/off, F16 K/V (and F32 K/V arriving via
   cast). Where FTZ-induced tolerance pressure appears, use per-case `max_maa_err`
   overrides (`vendor/llama.cpp/tests/test-backend-ops.cpp:1158`) with a justification
   comment citing ROADMAP §12 Q4 — never blanket-loosen the ADR-0002 default.
7. **Cross-backend parity + determinism tests:** compare CUDA dq/dk/dv against the
   S1-23 CPU oracle on identical inputs within the ADR-0002 criterion (max-abs gradient
   error ≤ 0.05 @ fp16); assert two CUDA runs produce bitwise-identical grads.
8. **e2e memory-cliff demonstration:** a long-context (2k and 4k ctx) tiny-model LoRA
   training run on the ci-cuda GPU runner (24 GB-class per the S0-08 playbook) with FA
   on, GPU-resident backward, recording peak allocation vs the naive path at the same
   configs (naive-path OOM is an acceptable recorded outcome at 4k). Publish the numbers
   as a CI artifact; S3-10 later wires the regression bound.
9. **Submodule bump PR** in llama-farm per S0-02, adding the back op to the ci-cuda
   targeted op list.

## Out of scope

- D=256 support and occupancy/tile-shape tuning — S3-07 (prototype-first per
  ROADMAP §12 Q5).
- mma-family backward and opt-in atomic-dQ single pass — backlog B-02 (perf phase,
  ROADMAP §8/§11 K5).
- Quantized-KV backward (excluded by construction — training graphs cast K/V), MLA
  DKQ=576, and sink gradients (sinks frozen).
- Vulkan/Metal FA backward — ROADMAP §8 FA6/FA7 (stage 4/2).
- Flipping FA on by default in training graphs and the fallback-forbidden CI switch —
  S3-10.

## Acceptance criteria

- [ ] Fork branch: `test-backend-ops grad -b CUDA0 -o FLASH_ATTN_BACK` (MODE_GRAD)
      passes for the full step-6 matrix within ADR-0002 tolerances (any per-case
      override justified in-code against ROADMAP §12 Q4).
- [ ] Cross-backend parity test passes: CUDA dq/dk/dv vs the S1-23 CPU oracle within
      max-abs gradient error ≤ 0.05 @ fp16 for mask/ALiBi/softcap/sinks/GQA variants at
      D=64 and D=128.
- [ ] Determinism test passes: two identical CUDA runs produce bitwise-identical
      dq/dk/dv (gate G-B; no atomics — grep-verifiable: no `atomicAdd` in
      `fattn-back.cu*`).
- [ ] `supports_op` accepts exactly {F16 K/V, D ∈ {64,128}}; D=256 and MLA-576 cases
      are confirmed CPU-fallback (not wrong-answer) via the sched assignment report.
- [ ] The e2e artifact exists in ci-cuda: peak-memory numbers for FA-on vs naive at 2k
      and 4k ctx on the GPU runner, with FA-on completing at 4k.
- [ ] llama-farm submodule-bump PR is green in `ci-cuda` (compile + GPU lanes) and
      `ci-cpu`.

## Testing & verification

Primary harness: vendored `tests/test-backend-ops` MODE_GRAD on CUDA vs the CPU oracle
(S1-23), per ADR-0002 (S0-09), plus the dedicated parity/determinism binary from step 7.
Runs on the fork branch CI and, after the submodule bump, in llama-farm's `ci-cuda` GPU
lane per-PR (kernel-gated, targeted `-o FLASH_ATTN_BACK,FLASH_ATTN_EXT`) and the nightly
full sweep + e2e job (S3-01). The e2e memory measurement reuses the ci-cuda
`GGML_SCHED_DEBUG` fallback-report machinery to prove the FA path ran GPU-resident; full
convergence acceptance (loss-curve gate with FA on, `--device cuda`) is S3-10's exit.

## PR notes

- Branch: `ticket/S3-06-cuda-flash-attention-backward-core`.
- Two-repo flow per S0-02: fork PR against `llama-farm-base` (ticket ID in title) plus a
  trivial llama-farm submodule-bump PR referencing the same ID.
- Size XL — stage commits within one fork PR: (1) pass-1 + pass-3 skeleton at D=64;
  (2) pass 2 + GQA + full mask/ALiBi/softcap/sinks semantics; (3) D=128 + config table +
  supports_op + full test matrix; (4) e2e wiring.
- Upstreaming disposition: **upstream-later** (ROADMAP §11 triage class b) — landing
  this completes the "CPU oracle + one GPU backend" precondition for the FA-training
  op-family RFC (S1-21/S1-22/S1-23/S3-05/this).
- Deterministic-scheme declaration per ADR-0002 Decision 3: exclusive-write three-pass
  scheme, no atomics. New files carry provenance headers (patterns adapted from
  `ggml/src/ggml-cuda/fattn-tile.cuh` and `fattn-common.cuh`, MIT, commit `4f37f51`)
  per S0-01 policy.
