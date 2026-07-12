---
id: S2-11
title: "Metal MoE: OUT_PROD_ID + OUT_PROD_ID_GRP ports"
stage: 2
track: kernels
size: L
deps: ["S2-06", "S1-26", "S1-27"]
status: open
pr: null
---

# S2-11 — Metal MoE: OUT_PROD_ID + OUT_PROD_ID_GRP ports

**One-line outcome:** MoE LoRA training is GPU-resident on Apple Silicon — the
expert-indexed outer product through frozen quantized experts (`OUT_PROD_ID`) and the
grouped F32 expert outer product for LoRA A/B grads (`OUT_PROD_ID_GRP`) both run on Metal,
MODE_GRAD-parity-checked against their CPU oracles.

## Why (context)

MoE training is exactly one missing backward case (`MUL_MAT_ID`) decomposed into these two
ops (ROADMAP §0 finding 3, §9 E2/E3). Both are required even for LoRA-only training:
`build_lora_mm_id` computes `mul_mat_id(B, mul_mat_id(A, cur, ids), ids)` — the trainable
LoRA A/B stacks are themselves the 3D expert operand
(`vendor/llama.cpp/src/llama-graph.cpp:1438-1442`) — so the "activation-grads-only"
shortcut that suffices for dense frozen bases does not suffice here. S1-25/26/27 delivered
the backward wiring and CPU reference kernels; this ticket ports both ops to Metal so the
MoE backward stops falling to CPU via `ggml_backend_sched`.

The two ops have different dependency shapes (ROADMAP §11 hard dependencies).
`OUT_PROD_ID` (E2, dX through frozen **quantized** experts) is blocked on S2-06's quantized
`OUT_PROD` — it reuses M2's tile-dequant machinery: the phase-1 dequant loop copied from
`kernel_mul_mm` (`vendor/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:9935-9976`) and the
per-type `dequantize_*` device functions (`metal:92-960`). `OUT_PROD_ID_GRP` (E3, dAs) is
pure F32 segmented accumulation and is explicitly **not** blocked on M2 (ROADMAP §9 E3
note); it rides in this ticket for cohesion, and an implementer may land it first within
the PR sequence.

Metal already has expert-compaction infrastructure in its `MUL_MAT_ID` forward:
`kernel_mul_mm_id_map0` (`metal:10087`) builds per-expert compacted id lists plus
tokens-per-expert counts on-GPU into fleeting buffers sized by
`ggml_metal_op_mul_mat_id_extra_tpe`/`_extra_ids`
(`vendor/llama.cpp/ggml/src/ggml-metal/ggml-metal-ops.cpp:2274` and `:2282`, allocated via
the backend's per-op fleeting-data mechanism,
`vendor/llama.cpp/ggml/src/ggml-metal/ggml-metal.cpp:213-227`) and consumed by
`kernel_mul_mm_id` (`metal:10153`). This is the Metal-native equivalent of the CUDA
`mm_ids_helper`/`expert_bounds` compaction the roadmap names (`vendor/llama.cpp/ggml/src/
ggml-cuda/mmid.cu:28-60`) and the CPU mmid workspace
(`vendor/llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c:1574-1637`). Determinism is gate G-B
(S0-09/ADR-0002): ragged expert segments make atomicAdd scatter nondeterministic, so both
kernels use deterministic segmented accumulation — atomics only as a later measured opt-in.

## What to do

All code lands in the vendored llama.cpp fork via the S0-02 two-repo flow.

1. **Compaction pass:** reuse/adapt `kernel_mul_mm_id_map0` (`metal:10087`) for the two new
   ops, extending `ggml_backend_metal_buffer_type_get_alloc_size`
   (`ggml-metal.cpp:213-227`) with cases sizing their fleeting id/count buffers (pattern:
   the `GGML_OP_MUL_MAT_ID` case at `:217-222`).
2. **`OUT_PROD_ID` kernel** (semantics per the S1-26 op contract, including the
   `s' = s % ne_b1` broadcast rule): extend S2-06's templated quantized `out_prod` kernel
   with expert indexing — per-expert segments from the compaction, M2's tile dequant into
   threadgroup memory, simdgroup multiply-accumulate. Deterministic ownership per gate G-B:
   each dst (token) column is written by exactly one threadgroup, iterating that token's
   used-expert slots in ascending slot order — no atomics. Type table identical to S2-06's
   supports_op set (carry over any IQ-type deferrals and their gating).
3. **`OUT_PROD_ID_GRP` kernel** (semantics per S1-27, dst `[n_in, n_out, n_expert]`):
   pure-F32 segmented accumulation — threadgroups own experts, each `dAs[:, :, e]` slab is
   zero-filled then accumulated in fixed segment order by its owning threadgroup (empty
   experts produce all-zero slabs; the zero fill is mandatory). Assert F32 in supports_op;
   state in the encoder comment that this op is not gated on the quantized-OUT_PROD path.
4. **The five mechanical additions ×2 ops** (ROADMAP §6 preamble): kargs structs
   (`ggml-metal-impl.h`), pipeline getters, encoder cases in `ggml-metal-ops.cpp` (pattern:
   `ggml_metal_op_mul_mat_id`, `:2291`), and supports_op cases in `ggml-metal-device.m`
   (coverage switch `:1051-1368`) — `has_simdgroup_mm` for `OUT_PROD_ID`,
   `has_simdgroup_reduction` sufficient for the F32 grouped op if its schedule needs only
   simd reductions.
5. **MODE_GRAD on Metal:** run the S1-26/S1-27 `test_mul_mat_id` grad cases
   (`vendor/llama.cpp/tests/test-backend-ops.cpp:4248`) on Metal against the CPU oracles:
   tiny-MoE shapes, quantized `as` across the implemented type set, broadcast on/off,
   ragged assignment, empty experts, single-expert edge case, and S1-27's nested
   `build_lora_mm_id`-shaped two-level case (grads reach both A and B stacks).
6. **Determinism check:** two Metal runs on identical inputs produce bitwise-identical
   dB/dAs (fork-side harness per S1-26/27 pattern; no atomics means dispatch order cannot
   change results).
7. **e2e:** run S1-28's tiny-MoE convergence fixture with `--device metal` in the
   `ci-metal` nightly: loss falls, and the S2-01 fallback report shows `MUL_MAT_ID`'s
   backward ops (`OUT_PROD_ID`, `OUT_PROD_ID_GRP`) executing on Metal. Remaining
   CPU-fallback ops on the MoE path (e.g. the fork-local `GLU_BACK` from S1-28, which has
   no Metal kernel yet) are reported, not forbidden — record them in the report and a
   backlog note.
8. **Submodule bump PR** in learning-llamas per S0-02.

## Out of scope

- CUDA and Vulkan ports of E2/E3 — S3-08 / S4-06 (they share the CPU oracles, not this
  code); the ragged-expert CUDA strategy question (ROADMAP §12 Q6) belongs there.
- A Metal `GLU_BACK` port and router-path VJP kernels — graph-level/CPU today; backlog.
- Atomics-based opt-in variants (gate G-B allows them later, with measurements) — backlog.
- Flipping MoE ops into S2-10's fallback-forbidden set — do it only if the e2e fallback
  report is clean for the expert-matmul backward ops; otherwise leave a backlog note.

## Acceptance criteria

- [ ] Fork branch: `test-backend-ops` mode `grad` passes on Metal for the `test_mul_mat_id`
      activation-grad (quantized sweep) and weight-grad (F32) case matrices, within
      ADR-0002 tolerances and the ≤ 0.05 @ fp16 cross-backend parity criterion vs CPU.
- [ ] Ragged, empty-expert, single-expert, broadcast on/off, and the nested LoRA-shaped
      two-level cases all pass on Metal; empty-expert slabs are exactly zero.
- [ ] Determinism: bitwise-identical dB/dAs across two Metal runs on identical inputs.
- [ ] supports_op is exact: `OUT_PROD_ID` type set matches S2-06's table; `OUT_PROD_ID_GRP`
      accepts F32 only.
- [ ] `ci-metal` nightly tiny-MoE e2e: loss falls with `--device metal`; the fallback
      report lists `OUT_PROD_ID`/`OUT_PROD_ID_GRP` on Metal and enumerates any remaining
      MoE-path CPU fallbacks.
- [ ] learning-llamas submodule-bump PR green in `ci-metal / build` and `ci-metal / grad`.

## Testing & verification

Primary harness: vendored `tests/test-backend-ops` MODE_GRAD, Metal backend vs the S1-26/
S1-27 CPU oracles, per-PR in the targeted `ci-metal / grad` lane (S2-01 maps changed files
to `-o` lists) and in the nightly full Metal sweep. The tiny-MoE e2e (S1-28 fixture,
`--device metal`) runs in the `ci-metal` nightly with the S2-01 fallback report attached.
Determinism checks run as fork-side tests in the same lanes.

## PR notes

- Branch: `ticket/S2-11-metal-moe-out-prod-ports`.
- Two-repo flow per S0-02: fork PR against `learning-llamas-base` (ticket ID in title; may stage
  GRP-first then ID commits for review), plus a trivial learning-llamas submodule-bump PR.
- Upstreaming disposition: **fork-local first, upstream-later** — the new op enums ride the
  E2/E3 op-family RFC with S1-26/S1-27 once the CPU oracle plus one GPU backend prove the
  design (ROADMAP §11 triage b).
- Provenance per S0-01: kernels carry headers naming their pattern sources
  (`kernel_mul_mm`/`kernel_mul_mm_id_map0`, `ggml-metal.metal`, MIT, commit `4f37f51`).
- Schedule note: `OUT_PROD_ID_GRP` is not blocked on S2-06 — if M2 slips (ROADMAP §12
  risk 12), land the GRP half early rather than idling the ticket.
