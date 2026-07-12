---
id: S1-27
title: "MoE: OUT_PROD_ID_GRP CPU reference (grouped expert outer product for LoRA A/B grads)"
stage: 1
track: kernels
size: M
deps: ["S1-25", "S0-09"]
status: open
pr: null
---

# S1-27 — MoE: OUT_PROD_ID_GRP CPU reference (grouped expert outer product for LoRA A/B grads)

**One-line outcome:** the new op `OUT_PROD_ID_GRP(b, grad, ids, n_expert) → dAs`
executes on CPU — F32-only grouped per-expert weight gradients with deterministic
segmented accumulation — giving expert LoRA A/B tensors their gradients.

## Why (context)

This is the op that makes MoE **LoRA** training possible at all. The key discovery
encoded in S1-25 (ROADMAP §0 finding 3, §9): `build_lora_mm_id` computes
`mul_mat_id(B, mul_mat_id(A, cur, ids), ids)` — the trainable LoRA A/B tensors are
themselves the 3D expert operand of `mul_mat_id`
(`vendor/llama.cpp/src/llama-graph.cpp:1438-1442`). So unlike the dense case, where
"activation grads only" suffices for frozen base weights, MoE LoRA needs a weight-grad
op for the 3D expert stack: `dAs[:, :, e]` accumulates the outer products of the
grad/activation column pairs of exactly the tokens routed to expert `e` (ROADMAP §9 E3).

The op is deliberately **F32-only** (assert in the kernel): frozen quantized experts
never take this path — their `grads-needed` is false, so S1-25's backward case never
emits `OUT_PROD_ID_GRP` against them — and the only tensors that reach it are the F32
LoRA A/B stacks (and F32 expert weights under hypothetical full FT, which is an explicit
non-goal, ROADMAP §9 E8). This has a scheduling consequence worth preserving in the
ticket record: E3 is **not** blocked on the base `OUT_PROD` GPU ports (it is pure F32
segmented GEMM, no dequant machinery), so the backend-stage ports of this op can
proceed independently of the quantized-`OUT_PROD` critical path (ROADMAP §11 hard
dependencies: "E3 is not blocked on OUT_PROD ports").

Determinism (gate G-B, decided in S0-09/ADR-0002): ragged expert segments make
atomicAdd scatter nondeterministic, so the default schedule is segmented — compact
(token, slot) pairs into per-expert groups first, then have each thread own whole
experts, so every `dAs[:, :, e]` slab is written by exactly one thread. The in-tree
patterns are the CPU mmid row-grouping workspace
(`vendor/llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c:1574-1637`) and, for the later GPU
ports, CUDA's `mm_ids_helper` compaction with its `expert_bounds` prefix array
(`vendor/llama.cpp/ggml/src/ggml-cuda/mmid.cu:28-60`).

## What to do

All code lands in the vendored llama.cpp fork via the S0-02 two-repo flow.

1. **CPU kernel** `ggml_compute_forward_out_prod_id_grp` in
   `vendor/llama.cpp/ggml/src/ggml-cpu/ops.cpp`. Semantics (from the S1-25 op
   contract): dst has the grouped 3D layout of the `mul_mat_id` src0 operand —
   `[n_in, n_out, n_expert]` (for LoRA A that is `[n_embd, r, n_expert]`, for LoRA B
   `[r, n_out, n_expert]`, matching the operand shapes at
   `vendor/llama.cpp/src/llama-graph.cpp:1438-1442`). For each expert `e`:
   `dAs[:, :, e] = Σ_{(s,t): ids[s,t]=e} b[:, s', t] ⊗ grad[:, s, t]` with
   `s' = s % ne_b1` (same broadcast rule as S1-25/S1-26). F32-only: assert
   `b`, `grad`, dst are `GGML_TYPE_F32`.
2. **Expert-bounds compaction:** single pass over `ids` builds per-expert counts and a
   compacted (token, slot) list (adapt the mmid workspace pattern,
   `ggml-cpu.c:1574-1637`; name the prefix array `expert_bounds` to match the CUDA
   vocabulary the ports will use, `mmid.cu:28-60`). Then per expert, accumulate the
   grouped outer product over its segment — an implementer may phrase the inner loop as
   a small F32 GEMM (`k = segment length`) for cache friendliness; scalar-first is
   acceptable per ROADMAP §4 P3.
3. **Deterministic parallelization (gate G-B):** threads own experts (`e` strided by
   `n_threads`); each expert's slab is zeroed then accumulated in fixed segment order
   by its owning thread. No atomics. Empty experts produce an all-zero slab — the zero
   fill is mandatory, not skippable.
4. **Plumbing:** dispatch case, `n_tasks`, wdata sizing for the compaction workspace in
   `vendor/llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c`, and CPU `supports_op` = true for
   `GGML_OP_OUT_PROD_ID_GRP`; GPU backends keep returning false. Note in the
   supports_op comment that backend ports are pure-F32 and not gated on OUT_PROD
   (ROADMAP §11).
5. **MODE_GRAD tests:** the S1-25 `test_mul_mat_id` grad cases
   (`vendor/llama.cpp/tests/test-backend-ops.cpp:4248`) now execute for the
   `as`-as-param direction (F32 `as` only). Cover: tiny MoE shapes; broadcast `b` on/off
   (the class's `b` flag, `:4248-4311`); ragged assignment; **empty-expert** case (an
   expert no token selects — grads exactly zero); **single-expert** edge case
   (`n_mats == n_used == 1`, degenerate ids); and a nested two-level graph shaped like
   `build_lora_mm_id` — `mul_mat_id(B, mul_mat_id(A, x, ids), ids)` with A and B as
   params — asserting grads reach both A and B (extends the S1-25 graph test from
   build-only to executing).
6. **Determinism test:** bitwise-identical dAs across different `n_threads` (same
   fork-side harness as S1-26).
7. **Submodule bump PR** in learning-llamas per S0-02.

## Out of scope

- `OUT_PROD_ID` (activation grads through quantized experts) — S1-26.
- Quantized or F16 operands for this op — F32-only by design (frozen quantized experts
  never take this path; full expert FT is a non-goal, ROADMAP §9 E8).
- GPU ports (Metal/CUDA/Vulkan) — stage-2/3/4 tickets; explicitly not blocked on the
  OUT_PROD ports there.
- GLU-variant backward and tiny-MoE e2e convergence — S1-28.

## Acceptance criteria

- [ ] Fork branch: `test-backend-ops` mode `grad` passes on CPU for the
      `as`-as-param `test_mul_mat_id` cases (F32), within the ADR-0002 tolerance.
- [ ] The nested `build_lora_mm_id`-shaped MODE_GRAD/graph test passes: finite-difference
      grads reach both A and B stacks through two levels of `mul_mat_id`.
- [ ] Empty-expert and single-expert edge cases pass; empty-expert slabs are exactly
      zero.
- [ ] Determinism test passes: bitwise-identical dAs across different `n_threads`.
- [ ] The F32-only assert fires (death test or checked abort) for non-F32 inputs.
- [ ] learning-llamas submodule-bump PR is green in `ci-cpu` (per-PR).

## Testing & verification

Primary harness: vendored `tests/test-backend-ops` MODE_GRAD (finite differences, CPU
oracle, ADR-0002 tolerance from S0-09) plus the fork-side determinism test; the nested
LoRA-shaped case doubles as the executable version of S1-25's graph test. Runs on the
fork branch CI and in learning-llamas's `ci-cpu` lane per-PR after the submodule bump;
nightly `ci-cpu` re-runs the full suite. Backend ports (stage 2/3/4) re-run the same
cases against this CPU reference under the ADR-0002 parity criterion.

## PR notes

- Branch: `ticket/S1-27-out-prod-id-grp-cpu-reference`.
- Two-repo flow per S0-02: fork PR (`learning-llamas-base`) + trivial learning-llamas
  submodule-bump PR, both referencing the ticket ID.
- Upstreaming disposition: **fork-local** initially; upstreams later in the E2/E3
  op-family RFC once the CPU oracle plus one GPU backend prove the design
  (ROADMAP §11 triage b).
- The compaction adapts in-tree MIT code (mmid workspace, llama.cpp `4f37f51`);
  provenance note in the function comment per S0-01 policy.
