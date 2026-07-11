---
id: S1-01
title: "Shim: lf_opt_init_lora — ggml_set_param on adapter A/B tensors"
stage: 1
track: shim
size: M
deps: ["S0-03", "S0-04"]
status: open
pr: null
---

# S1-01 — Shim: lf_opt_init_lora — ggml_set_param on adapter A/B tensors

**One-line outcome:** a C ABI call that, given a context with attached LoRA adapters, flags every
`ab_map` A/B tensor as a ggml param (adapter-only filter, nothing else ever offered) so gradients
flow to LoRA tensors only — while the base model stays frozen, quantized, and mmap'd.

## Why (context)

This is the whole trick that makes LoRA training on ggml work (BLUEPRINT D2, G1). ggml's autograd
gives gradients and optimizer state only to leaf tensors flagged with `ggml_set_param`;
`build_lora_mm` already injects the adapter A/B matmuls into every architecture's forward graph, so
flagging the adapter tensors is all that is missing — `ggml_build_backward_expand` does the rest
with zero per-arch code. Nobody has wired this: the existing training path's `llama_set_param`
helper iterates **base-model tensors only** (`vendor/llama.cpp/src/llama-context.cpp:3234-3254`)
and never offers adapter tensors to the param filter, so today they ride through training graphs as
inert constants (gap G1). The helper also silently skips non-F32 tensors
(`vendor/llama.cpp/src/llama-context.cpp:3200-3214`, with `token_embd.weight` / `rope_freqs.weight`
hard-excluded behind FIXMEs at lines 3207-3212) — behavior we bypass entirely rather than inherit.

The adapter tensors are exactly the right objects to flag. `llama_adapter_lora` holds
`ab_map: base tensor name → {a, b}` (`vendor/llama.cpp/src/llama-adapter.h:48-88`), and each A/B is
a `ggml_dup_tensor` copy created at load (`vendor/llama.cpp/src/llama-adapter.cpp:371-372`) — i.e.
a leaf with `op == GGML_OP_NONE`, which `ggml_set_param` asserts
(`vendor/llama.cpp/ggml/src/ggml.c:7677-7679`). Two preconditions from BLUEPRINT D2 must be
enforced, not just documented: flagging must happen **after** adapters are attached
(`llama_set_adapters_lora`, `vendor/llama.cpp/include/llama.h:690`) and **before** the first
opt-graph build, because PARAM-flagged leaves are promoted into graph nodes at build time and
ggml-opt's grad/momentum allocation scans only graph nodes.

Because LoRA training never writes base weights, the base model can stay quantized and mmap'd
(`use_mmap=true`), unlike the in-tree full-parameter finetune example which forces mmap off —
BLUEPRINT D2 says to verify this early, so this ticket asserts it in tests. The shim lives in
`csrc/` (S0-03) because `llama_context` internals are private C++; note the attached-adapter set
itself is a private member (`vendor/llama.cpp/src/llama-context.h:285`), which shapes the ABI below.

## What to do

1. In `csrc/farm_train.cpp` (new file), implement
   `lf_opt_init_lora(struct llama_context * ctx, struct llama_adapter_lora ** adapters, size_t n_adapters, const struct lf_opt_params * params)`
   declared in `csrc/farm_api.h`. The caller passes the same adapter handles it attached via
   `llama_set_adapters_lora` — required because `llama_context::loras` is private
   (`vendor/llama.cpp/src/llama-context.h:285`); `llama_adapter_lora` itself is a plain struct in a
   private header the shim compiles against (S0-03 proved this).
2. Create the shim-side training state (`lf_train_state`, owned by the shim, keyed to the context):
   mirror the ggml_opt setup of `llama_context::opt_init`
   (`vendor/llama.cpp/src/llama-context.cpp:3216-3232` — `opt_period = n_batch/n_ubatch`, optimizer
   type, per-step optimizer-params callback) but with `GGML_OPT_LOSS_TYPE_SUM` instead of the
   hardcoded `GGML_OPT_LOSS_TYPE_CROSS_ENTROPY` (line 3224), per BLUEPRINT D1. The base-tensor
   `llama_set_param` iteration (`vendor/llama.cpp/src/llama-context.cpp:3234-3254`) is never
   executed — the shim bypasses `llama_opt_init` entirely.
3. Iterate each adapter's `ab_map` (`vendor/llama.cpp/src/llama-adapter.h:48-88`) and
   `ggml_set_param` every A and B tensor. Validate before flagging: tensor is F32 (LoRA A/B must be
   F32 — BLUEPRINT §8 constraint table) and a leaf (`op == GGML_OP_NONE`); return a distinct error
   code otherwise instead of tripping the `ggml_set_param` assert. Reject any request to flag a
   non-adapter tensor by construction: the ABI offers no way to name arbitrary tensors.
4. Enforce the D2 ordering with a state flag: error if `n_adapters == 0` or adapters were not
   attached to `ctx`; error on a second `lf_opt_init_lora` call; set a `params_frozen` flag that
   S1-02's first graph build checks (flagging after the first `lf_train_step` must fail with a
   clear message). Return the number of tensors flagged (`2 × Σ ab_map.size()`) on success,
   negative `LF_ERR_*` codes on failure.
5. Document in the `farm_api.h` doc comment the perf note from BLUEPRINT D2: adapter tensors that
   fell back to CPU buffer types because the base tensor uses an extra/repacked buft
   (`vendor/llama.cpp/src/llama-adapter.cpp:337-350`) incur cross-backend gradient traffic once GPU
   backends exist (stages 2-4); harmless on the CPU-only stage-1 build.
6. Add `_ffi` bindings (S0-04 layer) for the new entry points.
7. Tests `tests/test_opt_init_lora.py` (S0-06 harness, fixture models + S0-05 zero-init adapters):
   success path on all three fixture variants (F32, Q8_0, Q4_K) with the model loaded with
   `use_mmap=true`; returned count equals `2 ×` the number of targeted base tensors; error-path
   tests for call-before-attach, double-call, and a hand-built adapter with a non-F32 tensor.

## Out of scope

- The training step itself (forked `opt_epoch_iter`, loss epilogues, extra inputs) — S1-02.
- Adapter tensor enumeration/get/set/save APIs (gap G2) — S1-08 (S1-03 adds a debug-only accessor).
- Gradient flow verification (finite-difference check, loss decrease) — S1-03 proves it end-to-end.
- Norm-vector or `token_embd`/`lm_head` training (BLUEPRINT §8 defers; the loader ignores norm
  adapters today).
- Any change to vendored llama.cpp — this is pure shim code.

## Acceptance criteria

- [ ] `pytest tests/test_opt_init_lora.py` passes on a CPU-only build: for each fixture variant
      (F32, Q8_0, Q4_K), `lf_opt_init_lora` returns exactly `2 × n_targets` with `use_mmap=true`.
- [ ] Ordering errors are exercised: before-attach and double-call both return distinct negative
      error codes (no aborts), and the assertion path for non-F32 A/B returns an error code.
- [ ] A test asserts no base-model tensor is flagged: the returned count matches the adapter-only
      expectation and training-state introspection shows only `.lora_a`/`.lora_b` names (via the
      shim's returned names or count — implementer's choice, but objectively checked).
- [ ] `farm_api.h` documents the CPU-buft/repacked-weights perf note with the vendored citation.
- [ ] `ci-cpu` lane is green (per-PR).

## Testing & verification

`tests/test_opt_init_lora.py` in the S0-06 pytest harness, running per-PR in the `ci-cpu` lane
(ubuntu + macos runners, `-m "not slow"`). No new ggml ops are added, so no `test-backend-ops`
MODE_GRAD cases apply to this ticket; the MODE_GRAD-facing proof that gradients actually reach the
flagged tensors is S1-03's finite-difference check. Manual verification for the PR description:
run the success-path test with `LLAMA_LOG` debug enabled and paste the adapter-attach + init log.

## PR notes

- Branch: `ticket/S1-01-opt-init-lora-adapter-params`.
- One llama-farm PR; shim + Python bindings + tests only. No vendored llama.cpp changes expected —
  if a private-member accessor turns out to be unavoidable, that change goes through the S0-02
  two-repo flow (fork PR + submodule bump), not inline here.
- Upstreaming disposition: **fork-local** for now; BLUEPRINT §10 notes a
  `llama_opt_init_lora`-shaped API is a plausible future upstream contribution, but that is not
  this ticket.
- Soft coordination: S1-02 consumes `lf_train_state` and the `params_frozen` flag; keep the struct
  in a shim-internal header so S1-02 can extend it without ABI churn.
