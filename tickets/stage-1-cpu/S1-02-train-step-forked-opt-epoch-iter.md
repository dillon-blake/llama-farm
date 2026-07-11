---
id: S1-02
title: "Shim: lf_train_step — forked opt_epoch_iter with pluggable loss + extra inputs"
stage: 1
track: shim
size: L
deps: ["S1-01"]
status: open
pr: null
---

# S1-02 — Shim: lf_train_step — forked opt_epoch_iter with pluggable loss + extra inputs

**One-line outcome:** the core training-step C ABI exists: a fork of
`llama_context::opt_epoch_iter` that keeps the proven per-ubatch mechanics verbatim, exposes the
forward logits to a caller-chosen loss epilogue reduced via `GGML_OPT_LOSS_TYPE_SUM`, and accepts
extra named input tensors (loss mask, labels, advantages, ref/old logprobs).

## Why (context)

BLUEPRINT D1 mandates forking the training loop rather than calling `llama_opt_epoch`: the public
path hardcodes `GGML_OPT_LOSS_TYPE_CROSS_ENTROPY` (`vendor/llama.cpp/src/llama-context.cpp:3224`)
and fills dense one-hot F32 labels with no masking possible
(`vendor/llama.cpp/src/llama-context.cpp:3344-3354`) — gap G3. The ~100-line loop worth copying
verbatim is `opt_epoch_iter` (`vendor/llama.cpp/src/llama-context.cpp:3257-3364`): memory clear →
`llama_batch` fill → ubatch split → `model.build_graph(gparams)` → `ggml_opt_prepare_alloc` → fill
inputs → `ggml_opt_eval`. Everything it touches (`graph_params`, `llm_graph_result`, `balloc`,
`memory`) is private C++ (declaration at `vendor/llama.cpp/src/llama-context.h:207-217`), which is
why this lives in the shim compiled against `src/` internals (BLUEPRINT §1.2, S0-03) rather than in
ctypes over the public API (`vendor/llama.cpp/include/llama.h:1560-1591`). ggml-opt's own header
sanctions copying its high-level pieces (`vendor/llama.cpp/ggml/include/ggml-opt.h:191-194`).

The custom-loss mechanism is the officially sanctioned escape hatch (BLUEPRINT §1.1): build the
loss expression in the forward graph, pass it as the `outputs` tensor, and let
`GGML_OPT_LOSS_TYPE_SUM` reduce it (`vendor/llama.cpp/ggml/include/ggml-opt.h:29-32`). Extra
`GGML_TENSOR_FLAG_LOSS` nodes are rejected outright
(`vendor/llama.cpp/ggml/src/ggml-opt.cpp:343`), so every objective must fold into that single
outputs tensor — this constraint shapes the epilogue registry design, and it is how SFT, DPO, and
GRPO all attach their objectives later (BLUEPRINT §6).

One hard constraint must be enforced here, not discovered downstream: ggml-opt's dynamic-graph
mode keys gradient accumulators and AdamW momenta by **forward-graph node index**
(`vendor/llama.cpp/ggml/src/ggml-opt.cpp:458-486`), and varying batch shapes trip an assert deep in
result accumulation (`vendor/llama.cpp/ggml/src/ggml-opt.cpp:851`). Graph topology must therefore
be identical across steps; padding batches to fixed ubatch shapes is the Python data layer's job
(BLUEPRINT D7, ticket S1-07), and the shim's job is to reject shape drift with a clear error before
any C-level assert fires.

## What to do

1. Copy the body of `opt_epoch_iter` (`vendor/llama.cpp/src/llama-context.cpp:3257-3364`) into
   `csrc/farm_train.cpp` as the core of `lf_train_step`, with a per-file provenance header (source
   path, commit `4f37f51`, MIT) per S0-01 policy. Keep the proven mechanics verbatim: memory clear,
   `llama_batch` construction, ubatch split, `model.build_graph`, `ggml_opt_prepare_alloc`
   (`vendor/llama.cpp/ggml/include/ggml-opt.h:177`), input fill, `ggml_opt_eval`.
2. Rebuild the loop's private dependencies from reachable objects: the model and memory module via
   `llama_get_model` / `llama_get_memory` (`vendor/llama.cpp/include/llama.h:558-559`, with
   `init_batch` on the private-header interface `vendor/llama.cpp/src/llama-memory.h:88`); a
   shim-owned `llama_batch_allocr` (`vendor/llama.cpp/src/llama-batch.h:72`); a shim-owned
   `llm_graph_result` and `llm_graph_params`, feeding the public
   `llama_model::build_graph(const llm_graph_params &)` (`vendor/llama.cpp/src/llama-model.h:673`).
   If some required context state proves genuinely unreachable (e.g. a `cparams` field with no
   accessor), add a minimal fork-side accessor via the S0-02 two-repo flow — record what was needed
   in the PR description.
3. Loss epilogue registry: a shim-internal table of
   `lf_loss_build_fn(ggml_context * ctx, ggml_tensor * logits, const lf_named_inputs *, void * ud) → ggml_tensor * outputs`,
   selected per step by a `loss_spec` id/name in the C ABI. The epilogue builds nodes in the
   per-ubatch compute context on top of `res->get_logits()`, is forward-expanded into the graph,
   and its result replaces `res->get_logits()` as the `outputs` argument of
   `ggml_opt_prepare_alloc` (call site pattern: `vendor/llama.cpp/src/llama-context.cpp:3340`),
   reduced under `GGML_OPT_LOSS_TYPE_SUM` (set up in S1-01's `lf_train_state`).
4. Register the stopgap composite CE (`sft_ce_stopgap`) as the first entry, per BLUEPRINT §6.1:
   softmax → one-hot mul → sum_rows → log, in select-then-log order to avoid `0·(−inf)` NaNs; all
   constituent VJPs exist in `ggml_compute_backward` (MUL `vendor/llama.cpp/ggml/src/ggml.c:6501`,
   LOG `:6531`, SUM_ROWS `:6551`, SOFT_MAX `:6762`). It materializes one-hot tensors and is
   bring-up only; S1-03 validates its numerics end-to-end and S1-05 replaces it with `ce_sparse`.
5. Extra named inputs: the C ABI accepts an array of `{name, type (F32/I32), ne[4], data}` entries.
   For each, create a tensor in the per-ubatch compute context before the epilogue is built (the
   epilogue looks inputs up by name), let `ggml_opt_alloc`
   (`vendor/llama.cpp/ggml/include/ggml-opt.h:186`) allocate it, then fill the per-ubatch slice via
   `ggml_backend_tensor_set` (`vendor/llama.cpp/ggml/include/ggml-backend.h:92`) at the same point
   where the stock loop fills its labels. Named inputs are constants: never flagged as params.
6. Fixed-topology enforcement: on the first step, record a shape signature (n_ubatch, per-ubatch
   token count, each named input's type + ne). Every later call validates against it and returns a
   clear `LF_ERR_SHAPE_DRIFT`-style error naming the offending tensor — before ggml-opt's
   node-index state (`vendor/llama.cpp/ggml/src/ggml-opt.cpp:458-486`) or the varying-batch assert
   (`vendor/llama.cpp/ggml/src/ggml-opt.cpp:851`) can misbind or abort. Also set S1-01's
   `params_frozen` flag on first build.
7. Flat C ABI in `csrc/farm_api.h`:
   `lf_train_step(ctx, const int32_t * tokens, size_t n_tokens, const lf_named_input * inputs, size_t n_inputs, const char * loss_spec, bool train, lf_step_result * out)`
   returning `{loss, n_valid}` — `loss` is the raw SUM-reduced scalar; `n_valid` counts nonzero
   entries of the mask/weights input host-side (0 if none registered). Host-side normalization by
   `n_valid` is the Python caller's job. `train=false` runs forward-only
   (`ggml_opt_alloc(opt_ctx, /*backward=*/false)`), which S1-05's masked validation consumes.
8. `_ffi` bindings and tests `tests/test_train_step.py` (fixture models + zero-init adapters).

## Out of scope

- The real sparse-CE loss op (S1-04) and the production SFT epilogue on it (S1-05).
- DPO/GRPO epilogues (S1-14, S1-16) — the registry just has to make them possible.
- Python training loop, LR schedules, grad accumulation ergonomics (S1-03, S1-05 `train/loop.py`).
- Gradient clipping (S1-10), optimizer-state checkpoint/resume (S1-09), gradient checkpointing
  (S1-17), trainability preflight (S1-11).
- Batch padding/packing — Python data layer (D7, S1-06/S1-07); the shim only *rejects* drift.

## Acceptance criteria

- [ ] `pytest tests/test_train_step.py` passes on CPU: one `lf_train_step` with `sft_ce_stopgap` on
      the Q8_0 fixture (rank-4 zero-init adapter) returns a finite `loss > 0` and
      `n_valid` equal to the nonzero-mask count.
- [ ] Parameter-update proof: two consecutive steps on the same fixed batch return different loss
      values (optimizer stepped the adapter params); with `train=false`, repeated calls return
      bit-identical loss and leave a subsequent training step's first loss unchanged.
- [ ] Named-input round-trip test: an epilogue that sums a registered F32 input returns exactly the
      numpy sum of the buffer Python supplied.
- [ ] Shape-drift test: changing the token count or a named input's shape on step 2 returns the
      documented error code (no abort, no assert).
- [ ] Calling `lf_opt_init_lora` after a step returns the `params_frozen` error (S1-01 contract).
- [ ] The copied loop carries the S0-01 provenance header; `ci-cpu` is green per-PR.

## Testing & verification

`tests/test_train_step.py` in the S0-06 harness, per-PR in `ci-cpu`. No new ggml ops are added, so
no `test-backend-ops` MODE_GRAD cases belong to this ticket (the stopgap epilogue composes existing
ops whose VJPs are already covered upstream); gradient *correctness* through the full step is
S1-03's finite-difference check, which this ticket must not pre-empt. Manual: paste one step's
graph-size and loss log into the PR description for the fixture model.

## PR notes

- Branch: `ticket/S1-02-train-step-forked-opt-epoch-iter`.
- Expected to be a single llama-farm PR (shim + bindings + tests). If step 2's fallback accessor is
  needed, that lands first as a fork PR + submodule bump per the S0-02 two-repo flow, and this PR
  depends on the bump.
- Upstreaming disposition: **fork-local** (the forked loop is llama-farm's product surface;
  BLUEPRINT §10 keeps possible upstreaming of a parameterized `opt_epoch` out of v1 scope).
- Size L: keep the diff reviewable — mechanics copy + registry in one commit, named inputs and
  enforcement in follow-up commits within the same PR.
