---
id: S1-25
title: "MoE: MUL_MAT_ID + ADD_ID backward wiring (E1/E4)"
stage: 1
track: kernels
size: S
deps: ["S0-02"]
status: pr-open
pr: https://github.com/dillon-blake/llama-farm/pull/37
---

# S1-25 — MoE: MUL_MAT_ID + ADD_ID backward wiring (E1/E4)

**One-line outcome:** `ggml_compute_backward` handles `MUL_MAT_ID` (emitting the two new
E2/E3 ops, whose CPU kernels land in S1-26/S1-27) and `ADD_ID`, and the MoE router path
is confirmed gradient-clean (I32 indices auto-excluded, weights-path VJPs all present).

## Why (context)

MoE training is blocked by exactly one missing backward case (ROADMAP §9): `MUL_MAT_ID`
has no case in `ggml_compute_backward` (switch at `vendor/llama.cpp/ggml/src/ggml.c:6430`)
and falls into the op-level default abort (`vendor/llama.cpp/ggml/src/ggml.c:6906`). The
key discovery driving the design (ROADMAP §0 finding 3, §9): `build_lora_mm_id` computes
`mul_mat_id(B, mul_mat_id(A, cur, ids), ids)` — the trainable LoRA A/B tensors are
themselves the 3D expert operand of `mul_mat_id`
(`vendor/llama.cpp/src/llama-graph.cpp:1438-1442`). So the "activation-grads-only"
shortcut is not sufficient even for LoRA-only MoE training: the backward case must emit
both an activation-grad op (`OUT_PROD_ID`, ROADMAP §9 E2 → S1-26) and a grouped
weight-grad op (`OUT_PROD_ID_GRP`, E3 → S1-27).

The router path needs no new kernels (ROADMAP §9). Node-by-node analysis of
`build_moe_ffn` (`vendor/llama.cpp/src/llama-graph.cpp:1799`): the argsort/top-k expert
selection produces I32 tensors (`ggml_argsort_top_k` at
`vendor/llama.cpp/src/llama-graph.cpp:1895-1915`) which the backward builder skips
outright (`if (node->type == GGML_TYPE_I32) continue;`,
`vendor/llama.cpp/ggml/src/ggml.c:7049-7051` — the DeepSeek expert-group branch is
entirely dead for grads). Gradients correctly flow through the top-k **weights** path —
`get_rows` on probs (`llama-graph.cpp:1929`), optional re-softmax (`:1935`),
`sum_rows`/`div` normalization (`:1943`, `:1950`) — whose VJP cases all exist
(`SOFT_MAX` `ggml.c:6762`, `GET_ROWS` `:6740`, `DIV` `:6513`, `SUM_ROWS` `:6551`). One
soft coordination point: the `norm_w` branch clamps the weight sum
(`llama-graph.cpp:1947`), so configs with `norm_w` also need the CLAMP VJP from S1-19
(prose dependency only; the graph test below can use a non-`norm_w` config first).

`ADD_ID` (per-expert bias add, used at `llama-graph.cpp:1985, 2004, 2020, 2106`) is
graph-level (E4): its dst is a dup of src0 (`ggml_add_id`,
`vendor/llama.cpp/ggml/src/ggml.c:2100-2119`), so the src0 VJP is the identity; the bias
table src1 is a frozen base weight (per-expert bias grads are E8, deferred) and ids is
I32 (auto-excluded).

## What to do

All code lands in the vendored llama.cpp fork via the S0-02 two-repo flow.

1. **New op enums + constructors** (fork-local, at the enum tail per ROADMAP §11c):
   `GGML_OP_OUT_PROD_ID` and `GGML_OP_OUT_PROD_ID_GRP` in
   `vendor/llama.cpp/ggml/include/ggml.h` (enum list around `GGML_OP_ADD_ID`,
   `ggml.h:486`) plus constructors `ggml_out_prod_id(as, grad, ids)` and
   `ggml_out_prod_id_grp(b, grad, ids, n_expert)` in `vendor/llama.cpp/ggml/src/ggml.c`
   with shape/type asserts mirroring `ggml_mul_mat_id`
   (`vendor/llama.cpp/ggml/src/ggml.c:3309-3333`). Constructors + shape contracts only —
   the CPU kernels are S1-26/S1-27; every backend's `supports_op` returns false for now.
2. **`MUL_MAT_ID` backward case** in `ggml_compute_backward`, patterned on the
   `MUL_MAT` case (`vendor/llama.cpp/ggml/src/ggml.c:6578-6630`): if src1 (activations)
   needs grads, emit `ggml_out_prod_id(as, grad, ids)`; if src0 (the 3D expert stack —
   LoRA A/B in the `build_lora_mm_id` pattern) needs grads, emit
   `ggml_out_prod_id_grp(b, grad, ids, n_expert)`. Handle the src1 broadcast the
   constructor allows (`ids->ne[0] % b->ne[1] == 0`, `ggml.c:3322`): the common case is
   `ne_b1 == 1` — `build_moe_ffn` reshapes tokens to `[n_embd, 1, n_tokens]`
   (`vendor/llama.cpp/src/llama-graph.cpp:1963`) — where all `n_expert_used` slots of a
   token accumulate into that token's single grad column (per-token accumulation; the
   accumulation semantics live inside OUT_PROD_ID's contract and are documented in the
   op comment here). ids (src2) is I32 and needs no `ignore_src` handling (`:7049-7051`).
3. **`ADD_ID` backward case:** src0 gets the incoming grad unchanged
   (`ggml_add_or_set`); src1/src2 get nothing (assert `!src1_needs_grads` with a comment
   pointing at E8 for per-expert bias training).
4. **Router-path graph test** (fork-side unit test, e.g.
   `tests/test-moe-backward-graph.cpp`): build a small `build_moe_ffn`-shaped subgraph —
   router `mul_mat` → softmax → `argsort_top_k` → `get_rows`/`div`/`sum_rows` weights
   path → expert `mul_mat_id` chain with the nested LoRA `mul_mat_id` pattern — mark the
   LoRA-style F32 operands as params, run `ggml_build_backward_expand`, and assert:
   (a) it no longer aborts; (b) no I32 tensor has a grad slot; (c) the backward graph
   contains `OUT_PROD_ID` and `OUT_PROD_ID_GRP` nodes with the expected shapes. This
   encodes the key discovery — LoRA A/B on experts require the weight-grad op — as a test.
5. **test-backend-ops:** add grad support (`ggml_set_param`) to `test_add_id`
   (`vendor/llama.cpp/tests/test-backend-ops.cpp:3189`) — its MODE_GRAD cases execute
   and must pass now (identity VJP, no new kernel). Add grad support to `test_mul_mat_id`
   (`:4248`, instantiations `:8693-8732`) for both `as`-as-param and `b`-as-param and for
   `b` broadcast on/off; these cases build backward graphs now but execute only once
   S1-26/S1-27 land (skipped via supports_op until then — mark with a comment).
6. **Submodule bump PR** in learning-llamas referencing this ticket, per S0-02.

## Out of scope

- CPU kernels for `OUT_PROD_ID` / `OUT_PROD_ID_GRP` — S1-26 / S1-27 (this ticket only
  defines the ops and emits them).
- GPU ports of either op — S2-11, S3-08, S4-06 per stage plans.
- GLU-variant backward and the tiny-MoE e2e — S1-28.
- Per-expert bias grads (scatter-add), router full training, quantized-expert full FT —
  ROADMAP §9 E8, explicitly deferred; owned by B-09.
- CLAMP VJP for `norm_w` configs — S1-19.

## Acceptance criteria

- [ ] Fork branch: the router-path graph test passes — backward build over the MoE
      subgraph no longer aborts, I32 nodes have no grads, and both new ops appear in the
      backward graph with correct shapes.
- [ ] Fork branch: `test-backend-ops` mode `grad` passes on CPU for the new `test_add_id`
      grad cases within the ADR-0002 tolerance.
- [ ] `test_mul_mat_id` grad cases are registered and report not-supported (skip) rather
      than aborting, on every backend.
- [ ] The two new op enums sit at the tail of `enum ggml_op` (rebase-friendly per
      ROADMAP §11c) and all existing test-backend-ops eval cases still pass on CPU.
- [ ] learning-llamas submodule-bump PR is green in `ci-cpu` (per-PR).

## Testing & verification

Primary harness: vendored `tests/test-backend-ops` MODE_GRAD (finite differences, CPU
oracle, ADR-0002 tolerance from S0-09) for ADD_ID, plus the new fork-side
graph-construction test for the MUL_MAT_ID wiring. Runs on the fork branch CI and in
learning-llamas's `ci-cpu` lane per-PR after the submodule bump; nightly `ci-cpu` re-runs the
full suite. The deferred `test_mul_mat_id` MODE_GRAD execution is verified in S1-26/S1-27,
which flip CPU `supports_op` on.

## PR notes

- Branch: `ticket/S1-25-mul-mat-id-add-id-backward`.
- Two-repo flow per S0-02: fork PR (`learning-llamas-base`) + trivial learning-llamas
  submodule-bump PR, both referencing the ticket ID.
- Upstreaming disposition: **fork-local** for now — the new op enums ride the fork's
  enum tail; propose upstream later as one RFC together with the E2/E3 kernels once the
  CPU oracle proves the design (ROADMAP §11 triage b).
- No external code is copied (the backward case follows the in-tree MUL_MAT pattern);
  no provenance headers needed beyond the standard fork attribution.
