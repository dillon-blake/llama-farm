---
id: S4-08
title: "Vulkan FA6: flash-attention forward LSE + backward"
stage: 4
track: kernels
size: XL
deps: [S1-23, S4-02, S4-03]
status: open
pr: null
---

# S4-08 — Vulkan FA6: flash-attention forward LSE + backward

**One-line outcome:** flash-attention training runs on Vulkan — the forward emits LSE
via the existing split-k m/L machinery, and a deterministic three-pass backward on the
scalar `flash_attn.comp` base (subgroup-optional, runs everywhere including MoltenVK)
is MODE_GRAD-parity-checked against the S1-23 CPU oracle.

## Why (context)

Without FA backward, training runs the naive `MUL_MAT → SOFT_MAX(mask) → MUL_MAT`
attention path whose `[n_kv, n_q, n_head]` F32 tensors are live across all layers at
once — 128-192 GiB at n_ctx 4096 for an 8B-class model, infeasible (ROADMAP §8
memory-cliff table). FA backward recomputes P per tile from Q/K/mask/LSE, replacing the
n_ctx² term with an LSE vector. The ABI (S1-21: `emit_lse` forward flag with packed
`O‖LSE` dst; `ggml_flash_attn_ext_back(...) → dq‖dk‖dv`), autograd wiring (S1-22), and
the CPU oracle (S1-23) exist; this ticket is the Vulkan half (ROADMAP §8 FA6), mirroring
the FA5 design on CUDA.

The forward LSE is mechanical: Vulkan already computes and stores per-row m/L for
split-k via `perElemOpStoreCol0`
(`vendor/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_base.glsl:156-157` —
"Store column zero. This is used to save per-row m and L values for split_k"), called
from all three shader families (scalar `flash_attn.comp:675-676`, coopmat1
`flash_attn_cm1.comp:553-554`, coopmat2 `flash_attn_cm2.comp:412-413`), and the split-k
reduce shader already combines those partials (`flash_attn_split_k_reduce.comp:34-35`,
pipeline at `vendor/llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp:5076`; bare `:NNNN`
anchors below refer to this file). Emitting `LSE = M + log(L)` per the FA1 ABI extends
this existing path rather than inventing one.

The backward bases on the **scalar** `flash_attn.comp`, not the coopmat families, for
v1: the scalar path runs on every driver the backend supports, and its tuning already
carries a `disable_subgroups` knob (set for Intel at `ggml-vulkan.cpp:3390-3393`, field
at `:3371`, spec-const plumbing at `:3578`), which the backward inherits so it stays
subgroup-optional. The forward's `supports_op` currently requires
`subgroup_shuffle && subgroup_vote` on non-coopmat2 paths (`ggml-vulkan.cpp:17341`), and
subgroup ops are force-disabled on MoltenVK+AMD (`:5983-5994`) — so the backward must
keep a true no-subgroup fallback per the project portability rule. Two decided
constraints bind the design: determinism (gate G-B, S0-09/ADR-0002 — no atomics, same
three-pass exclusive-write scheme as FA5/S3-06) and numerics (ROADMAP §12 Q4 —
recomputed P will not bit-match the forward's; acceptance is MODE_GRAD tolerance vs the
S1-23 CPU oracle, with the LSE-with-sinks definition matching the FA1 contract exactly).
D=256 is gated behind a shmem prototype, the Vulkan analog of ROADMAP §12 Q5 — the
in-tree pattern is `ggml_vk_flash_attn_scalar_shmem_support` (`ggml-vulkan.cpp:10107`).

## What to do

All code lands in the vendored llama.cpp fork via the S0-02 two-repo flow. Scope v1
mirrors FA5 (ROADMAP §8): F16 K/V, head sizes 64/128 first and 256 last; no MLA
DKQ=576, no sink gradients.

1. **Forward LSE emission (`emit_lse` per the S1-21 FA1 ABI):** extend the
   `perElemOpStoreCol0` mechanism (`flash_attn_base.glsl:156-157`) so an `emit_lse` dst
   receives final `LSE = M + log(L)` in the packed `O‖LSE` layout: the non-split-k
   epilogues of all three families write it directly (store sites
   `flash_attn.comp:675-676`, `flash_attn_cm1.comp:553-554`,
   `flash_attn_cm2.comp:412-413` are the pattern), and
   `flash_attn_split_k_reduce.comp` (which already reads the per-split m/L partials,
   `:34-35`) additionally emits the combined LSE. If a family lags during bring-up,
   gate `emit_lse` per-path honestly in `supports_op` (the whole node then falls back
   to CPU via sched) rather than emitting an undefined LSE region.
2. **Backward shader** `flash_attn_back.comp` on the scalar `flash_attn.comp` base,
   implementing the S1-21 back op (inputs q, k, v, mask, sinks, o, dO, lse; op-params
   scale, max_bias, logit_softcap; packed F32 `dq‖dk‖dv` dst). Three deterministic
   passes, same scheme as FA5/S3-06:
   - **Pass 1 — delta:** `delta = rowsum(dO ∘ O)` per (q-position × head) into a
     transient buffer; shared-memory tree reduction, subgroup-optional.
   - **Pass 2 — dK/dV:** workgroups over KV tiles; recompute
     `s = softcap-fold(scale·K·Q) + mask`, `P = exp(s − lse_row)`, form `dP = dO·Vᵀ`,
     `dS = P ∘ (dP − delta)` (chain `(1 − tanh²)` when softcap is set), accumulate
     `dV += Pᵀ·dO`, `dK += scale·dSᵀ·Q` into exclusive outputs; the KV-tile workgroup
     loops its GQA Q heads inside, so dK/dV needs no cross-workgroup reduction.
   - **Pass 3 — dQ:** workgroups over Q tiles, iterating KV tiles, accumulating
     `dQ += scale·dS·K` exclusively. No atomics anywhere (gate G-B).
   ALiBi slopes computed per head from max_bias exactly as the forward does (the
   `perElemOpComputeSlope` helper in `flash_attn_base.glsl` is the pattern); sinks
   participate via the LSE only — no sink grads. Subgroup use follows the
   `disable_subgroups` tuning knob (`ggml-vulkan.cpp:3371`, `:3578`) with shared-memory
   fallbacks so the shader runs on MoltenVK.
3. **D=256 shmem prototype (Q5 analog):** before committing tile shapes, extend the
   `ggml_vk_flash_attn_scalar_shmem_support`-style budget check
   (`ggml-vulkan.cpp:10107`) to the backward's working set; ship D=64/128 first, enable
   D=256 only where the gate passes (per-device), and record the measured budgets in
   the fork PR. Devices failing the gate fall back to CPU for D=256 — honest, not
   wrong.
4. **Plumbing:** dispatch next to `ggml_vk_flash_attn` (`ggml-vulkan.cpp:10202`),
   pipelines, and a `supports_op` case beside `GGML_OP_FLASH_ATTN_EXT`'s
   (`:17298-17345`): accept F16 K/V, DKQ == DV ∈ {64, 128} (+256 behind the gate),
   F16 additive mask (one mask covers causal/padding/SWA/ALiBi —
   `vendor/llama.cpp/src/llama-graph.cpp:406-453`); reject MLA DKQ=576 and quantized
   K/V (training graphs cast K/V to F16, `vendor/llama.cpp/src/llama-graph.cpp:2416-2422`,
   so quantized-KV backward is out of scope by construction).
5. **MODE_GRAD tests:** extend the S1-22 grad cases (`test_flash_attn_ext`,
   `vendor/llama.cpp/tests/test-backend-ops.cpp:6612`) to execute on Vulkan: D=64/128
   (256 where enabled), GQA ratios {1, 4}, mask off / causal-style mask / ALiBi
   (max_bias > 0), softcap on/off, sinks on/off, F16 K/V. Tolerances per ROADMAP §12
   Q4: per-case `max_maa_err` overrides (`test-backend-ops.cpp:1158`) with in-code
   justification — never blanket-loosen the ADR-0002 default.
6. **Cross-backend parity + determinism:** compare Vulkan dq/dk/dv against the S1-23
   CPU oracle on identical inputs within the ADR-0002 criterion (≤ 0.05 max-abs @
   fp16) for mask/ALiBi/softcap/sinks/GQA variants; assert two identical runs produce
   bitwise-identical grads (grep-verifiable: no atomic adds in the new shaders).
7. **e2e long-context run** on the native GPU lane (S4-01): tiny-model LoRA training at
   2k and 4k ctx with FA on, recording peak allocation vs the naive path (naive OOM at
   4k is an acceptable recorded outcome). Lavapipe runs correctness-only small shapes —
   no perf claims from software Vulkan.
8. **Submodule bump PR** in learning-llamas per S0-02, adding the ops to the ci-vulkan
   targeted list.

## Out of scope

- Coopmat-based backward and other perf work (split-k backward, occupancy tuning
  beyond the D=256 gate) — K5 territory, backlog.
- MLA DKQ=576, sink gradients (sinks frozen in LoRA training), quantized-KV backward
  (excluded by construction).
- CUDA/Metal FA backward — S3-06/S3-07 / S2-13.
- Flipping FA on by default and the fallback-forbidden CI switch — S4-09 (which does
  not depend on this ticket; FA joins its forbidden set only once this lands).

## Acceptance criteria

- [ ] Fork branch: forward `emit_lse` cases pass on Vulkan for all three shader
      families (or the gated subset is explicit in `supports_op` and the run log),
      split-k on and off, matching the S1-23-validated FA1 LSE definition.
- [ ] Fork branch: `test-backend-ops grad -b Vulkan0` passes for the step-5 matrix
      within ADR-0002 tolerances (per-case overrides justified in-code against
      ROADMAP §12 Q4).
- [ ] Cross-backend parity: Vulkan dq/dk/dv vs the S1-23 CPU oracle within
      ≤ 0.05 max-abs @ fp16 for mask/ALiBi/softcap/sinks/GQA variants at D=64/128, on
      lavapipe and the native lane.
- [ ] Determinism: two identical Vulkan runs produce bitwise-identical dq/dk/dv; no
      atomic adds in the new shaders (grep-verifiable).
- [ ] The D=256 shmem-gate record (budgets, per-device outcome) exists in the fork PR;
      devices failing the gate demonstrably fall back to CPU (sched report), not to a
      wrong answer.
- [ ] The e2e artifact exists on the native lane: peak-memory numbers for FA-on vs
      naive at 2k and 4k ctx, with FA-on completing at 4k.
- [ ] learning-llamas submodule-bump PR is green in `ci-vulkan / lavapipe` (per-PR) and
      `ci-vulkan / gpu` (label-gated), plus `ci-cpu`.

## Testing & verification

Primary harness: vendored `tests/test-backend-ops` MODE_GRAD on Vulkan vs the S1-23 CPU
oracle under ADR-0002 (S0-09), plus the parity/determinism checks. Runs per-PR on
`ci-vulkan / lavapipe` (targeted `-o` list, correctness-only) with the nightly full
sweep, and on the label-gated `ci-vulkan / gpu` native lane (S4-01), whose driver-caps
probe records which subgroup/coopmat features each run exercised. The long-context e2e
runs on the native lane nightly; S4-09's milestone consumes this ticket's ops into the
fallback-forbidden set only if landed.

## PR notes

- Branch: `ticket/S4-08-vulkan-flash-attention-training`.
- Two-repo flow per S0-02: fork PR against `learning-llamas-base` (ticket ID in title) plus
  a trivial learning-llamas submodule-bump PR referencing the same ID.
- Size XL — stage commits within one fork PR: (1) forward LSE emission + split-k
  reduce; (2) backward passes 1+3 at D=64; (3) pass 2 + GQA + full
  mask/ALiBi/softcap/sinks semantics; (4) D=128/256 gate + supports_op + full test
  matrix; (5) e2e wiring.
- Upstreaming disposition: **upstream-later** — rides the FA-training op-family RFC
  (S1-21/S1-22/S1-23 + first GPU backend) per ROADMAP §11 triage b; coordinate with
  S3-06 on which backend anchors the RFC.
- Deterministic-scheme declaration per ADR-0002 Decision 3: exclusive-write three-pass
  scheme, no atomics. New shaders carry provenance headers (patterns adapted from
  `flash_attn.comp` / `flash_attn_base.glsl`, MIT, commit `4f37f51`) per S0-01 policy.
