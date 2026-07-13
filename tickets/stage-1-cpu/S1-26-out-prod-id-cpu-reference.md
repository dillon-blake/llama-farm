---
id: S1-26
title: "MoE: OUT_PROD_ID CPU reference (activation grads through quantized experts)"
stage: 1
track: kernels
size: M
deps: ["S1-25", "S0-09"]
status: pr-open
pr: https://github.com/dillon-blake/llama.cpp/pull/21
---

# S1-26 — MoE: OUT_PROD_ID CPU reference (activation grads through quantized experts)

**One-line outcome:** the new op `OUT_PROD_ID(as, grad, ids) → dB` executes on CPU —
per-expert dequant-and-accumulate outer products, deterministic by construction,
MODE_GRAD-verified — the oracle every GPU port validates against.

## Why (context)

S1-25 wired the `MUL_MAT_ID` backward case, but the activation-grad op it emits has no
kernel yet. `OUT_PROD_ID` is the MoE analog of `OUT_PROD` in the `MUL_MAT` backward
(ROADMAP §9 E2): for every token, the gradient flowing out of an expert layer must pass
back through the (frozen, quantized) expert weights that token was routed to. Without
it, no MoE model — including LoRA-only MoE, whose dX path runs through the quantized
expert stacks — can train. This CPU implementation is the MODE_GRAD reference (ROADMAP
§4 P3): the Metal (S2-11), CUDA (S3-08), and Vulkan (S4-06) ports are all blocked on it
and validate against it under the ADR-0002 cross-backend parity criterion.

Both ingredients exist in-tree. First, the dequant-per-row axpy pattern: the CPU
quantized `OUT_PROD` (`ggml_compute_forward_out_prod_q_f32`,
`vendor/llama.cpp/ggml/src/ggml-cpu/ops.cpp:4363-4501`) dequantizes one src0 row at a
time via the type-traits `to_float` and accumulates scaled rows into the F32 dst — this
covers every block-quant type as src0. Second, per-expert compaction: the CPU
`mul_mat_id` forward already builds a workspace that groups token rows by expert
(`matrix_row_counts` / `mmid_row_mapping` inside `ggml_compute_forward_mul_mat_id`,
`vendor/llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c:1574-1637`), so each expert's rows are
processed together and each quantized expert row is dequantized once per group. The
CUDA ports later reuse the equivalent `mm_ids_helper`/`expert_bounds` compaction
(`vendor/llama.cpp/ggml/src/ggml-cuda/mmid.cu:28-60`).

Determinism is a hard requirement, not a preference: gate G-B (decided in S0-09 /
ADR-0002) mandates deterministic segmented accumulation by default, atomics only as
measured opt-in. The natural CPU schedule satisfies it — threads own output columns
(tokens), with the loop over that token's used experts inside, so every dst element is
written by exactly one thread. This matters specifically in the broadcast case
(`ne_b1 == 1`, the `build_moe_ffn` common case per S1-25): all `n_expert_used` slots of
a token accumulate into one grad column, and that accumulation must have a fixed order.

## What to do

All code lands in the vendored llama.cpp fork via the S0-02 two-repo flow.

1. **CPU kernel** `ggml_compute_forward_out_prod_id` in
   `vendor/llama.cpp/ggml/src/ggml-cpu/ops.cpp`, next to the `out_prod` family
   (`:4363-4501`). Semantics (from the S1-25 op contract): for each token column `t`
   and used-expert slot `s` with `e = ids[s, t]`,
   `dB[:, s', t] += Σ_i as[:, i, e] · grad[i, s, t]` where `s' = s % ne_b1` (broadcast
   per S1-25). Inner loop is the dequant-per-row axpy: dequantize row `i` of expert `e`
   via `to_float` traits, axpy with scale `grad[i, s, t]` into the F32 dst column.
2. **Per-expert row-grouping workspace:** replicate the mmid compaction
   (`vendor/llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c:1574-1637`) so the (token, slot)
   pairs are grouped by expert and each expert row is dequantized once per group into a
   scratch row, then applied to all grouped columns. Size the wdata contribution in the
   CPU work-size switch accordingly.
3. **Deterministic parallelization (gate G-B):** partition dst columns (tokens) across
   threads; each thread iterates its tokens' expert slots in ascending slot order. No
   atomics; document the write-ownership argument in a comment. Zero-initialize dst
   before accumulation.
4. **Type coverage:** every quant type the CPU `out_prod` dispatch accepts
   (`ops.cpp:4480-4501`), plus F32 and F16 expert stacks. For F16 use the `to_float`
   traits row path — coordinate with S1-18, which replaces the F16 abort in plain
   `out_prod` (`ops.cpp:4487-4491`) with the same mechanism; if S1-18 has landed, share
   its helper rather than duplicating.
5. **Plumbing:** dispatch case in `ggml_compute_forward`
   (`vendor/llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c:1711`), `n_tasks`, wdata sizing, and
   CPU `supports_op` = true for `GGML_OP_OUT_PROD_ID`. All GPU backends keep returning
   false (sched falls back to CPU per ROADMAP §11).
6. **MODE_GRAD tests:** the S1-25 `test_mul_mat_id` grad cases
   (`vendor/llama.cpp/tests/test-backend-ops.cpp:4248`, instantiations `:8693-8732`)
   now execute for the `b`-as-param direction — cover tiny MoE shapes (small m/n/k),
   quantized `as` across the type sweep (pattern: the quantized `test_out_prod` cases,
   `:8780-8806`), broadcast on/off, ragged expert assignment (n_used < n_mats with
   skewed ids), and empty experts (an expert selected by no token). Small shapes keep
   finite differences tractable.
7. **Determinism test:** fork-side check that two runs at different `n_threads` produce
   bitwise-identical dB (pattern: the S1-23 determinism assertion).
8. **Submodule bump PR** in learning-llamas per S0-02.

## Out of scope

- `OUT_PROD_ID_GRP` (dAs, the grouped weight-grad op) — S1-27.
- GPU ports — S2-11 (Metal), S3-08 (CUDA), S4-06 (Vulkan); this ticket is their oracle.
- SIMD tuning — scalar-first per ROADMAP §4 P3; revisit only if the S1-32 audit flags it.
- Atomics-based opt-in variants (gate G-B allows them later, behind a flag, with
  measurements) — backlog.

## Acceptance criteria

- [ ] Fork branch: `test-backend-ops` mode `grad` passes on CPU for the `test_mul_mat_id`
      activation-grad cases across the quantized type sweep plus F32/F16, within the
      ADR-0002 tolerance (MODE_GRAD vs finite differences, CPU oracle).
- [ ] Ragged-assignment and empty-expert cases pass (empty expert contributes exactly
      zero, no uninitialized reads under ASAN/valgrind lane if available).
- [ ] Determinism test passes: bitwise-identical dB across different `n_threads`.
- [ ] Broadcast (`ne_b1 == 1`) and non-broadcast cases both pass.
- [ ] All pre-existing test-backend-ops eval cases still pass on CPU.
- [ ] learning-llamas submodule-bump PR is green in `ci-cpu` (per-PR).

## Testing & verification

Primary harness: vendored `tests/test-backend-ops` MODE_GRAD (finite differences vs the
kernel, CPU as oracle, ADR-0002 tolerance from S0-09), plus the fork-side determinism
test. Runs on the fork branch CI and in learning-llamas's `ci-cpu` lane per-PR after the
submodule bump; nightly `ci-cpu` re-runs the full grad suite. Stage-2/3/4 port tickets
re-run the identical case list on their backends against this CPU reference (max-abs
gradient error ≤ 0.05 at fp16 per ADR-0002). End-to-end MoE convergence is owned by
S1-28.

## PR notes

- Branch: `ticket/S1-26-out-prod-id-cpu-reference`.
- Two-repo flow per S0-02: fork PR (`learning-llamas-base`) + trivial learning-llamas
  submodule-bump PR, both referencing the ticket ID.
- Upstreaming disposition: **fork-local** initially; upstreams later as part of the
  E2/E3 op-family RFC once the CPU oracle plus one GPU backend prove the design
  (ROADMAP §11 triage b).
- The kernel adapts in-tree MIT code (the `out_prod_q_f32` row path and the mmid
  workspace, llama.cpp `4f37f51`); keep a provenance note in the function comment per
  S0-01 policy.
