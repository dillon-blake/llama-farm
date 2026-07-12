---
id: S1-09
title: "Optimizer-state sidecar checkpoint/resume (name-keyed AdamW m/v)"
stage: 1
track: shim
size: M
deps: ["S1-02"]
status: open
pr: null
---

# S1-09 — Optimizer-state sidecar checkpoint/resume (name-keyed AdamW m/v)

**One-line outcome:** mid-run checkpoints = adapter GGUF + a sidecar GGUF holding AdamW
m/v keyed by param tensor **name**, plus step count, LR-schedule state, and data
cursor/RNG — with a byte-exact resume test on CPU.

## Why (context)

ggml-opt has no optimizer-state serialization (BLUEPRINT G14). Gradient accumulators are
reachable (`ggml_opt_grad_acc`, `vendor/llama.cpp/ggml/include/ggml-opt.h:156`) but AdamW
first/second moments are internal — allocated as node-index-parallel vectors when the
backward graph is first built (`grad_m`/`grad_v`,
`vendor/llama.cpp/ggml/src/ggml-opt.cpp:473-484`, inside the node-index-keyed state block
at `ggml-opt.cpp:458-486`). Node indices are fragile across processes and graph rebuilds,
so a durable checkpoint must re-key the state by the one stable identifier: the param
tensor's **name** (the adapter A/B names, unique per S1-01's ab_map iteration). ggml even
labels the momenta with those names — `"AdamW m for %s"` / `"AdamW v for %s"`
(`ggml-opt.cpp:521-522`).

Two facts discovered during verification make the design concrete. First, the whole
optimizer context (`struct ggml_opt_context`) is defined only inside `ggml-opt.cpp`, so
even the shim — which compiles against llama.cpp's private C++ `src/` headers — cannot
reach `grad_m`/`grad_v` without a small fork-side accessor, patterned on
`ggml_opt_grad_acc`'s implementation (`ggml-opt.cpp:633-635`). Second, byte-exact resume
requires more than m/v: AdamW bias correction is computed per step from the private
iteration counter (`beta1h/beta2h` from `opt_ctx->iter`, `ggml-opt.cpp:799-809`;
incremented per optimizer-step eval at `:825`; reset by `ggml_opt_reset` at `:599`), so
the accessor patch must also expose get/set of `iter`. Grad accumulators need no
serialization if checkpoints are constrained to accumulation-window boundaries:
`ggml_opt_alloc` zeroes `gb_grad` at the start of each window (`ggml-opt.cpp:727-729`).

BLUEPRINT D3 draws the format line: the adapter GGUF is the *interchange* artifact and
must stay stock-loadable; everything resume-specific goes in a **sidecar** file that is
explicitly not interchange. LR-schedule state and the data cursor/RNG are Python-owned
(the optimizer-params struct is Python's per S0-04), so they serialize as sidecar KV, not
native state.

## What to do

1. **Fork-side patch** (project llama.cpp fork, per S0-02 two-repo flow): add to
   `ggml-opt.h`/`ggml-opt.cpp` — `ggml_opt_step_momenta(opt_ctx, node, &m, &v)`
   returning the momenta tensors for a param node (pattern: `ggml_opt_grad_acc`,
   `ggml-opt.cpp:633-635`; look up via the `gb_opt` node index as `:519-522` does),
   plus `ggml_opt_get_iter(opt_ctx)` / `ggml_opt_set_iter(opt_ctx, iter)`. Pure
   additions, no behavior change.
2. Shim (`csrc/farm_checkpoint.cpp`, `csrc/farm_api.h`): name-keyed C ABI over that —
   `ll_opt_state_count(ctx)`, `ll_opt_state_info(ctx, i, name_buf, role, shape...)`
   (role ∈ {adamw_m, adamw_v}), `ll_opt_state_get(ctx, name, role, buf, nbytes)` and
   `ll_opt_state_set(...)` (via `ggml_backend_tensor_get/set`), and
   `ll_opt_get_iter`/`ll_opt_set_iter`. Enumerate by walking param nodes and mapping
   node → name once per graph build; reject calls before the first opt-graph build with
   a clear error (state does not exist yet, `ggml-opt.cpp:458`). Restore validates
   name/shape/dtype and errors on any missing or extra param.
3. Sidecar format, written from Python via the S0-05 gguf-py writer
   (`src/learning_llamas/checkpoint.py`): tensors `<param_name>.adamw_m` / `<param_name>.adamw_v`
   (F32); KV: `farm.checkpoint.version` (start at 1), `general.type =
   "learning-llamas-checkpoint"` (deliberately *not* `"adapter"` so stock loaders refuse it),
   optimizer type, `iter`, opt_period, LR-schedule name + state, numpy RNG state, data
   cursor (epoch, sample index, collator seed), and the paired adapter file's hash.
   Document prominently: **resume-only, not interchange** (D3).
4. Checkpoint operation: allowed only at accumulation-window boundaries (assert
   `opt_i == 0` equivalent via the shim's step bookkeeping from S1-02) — write adapter
   GGUF (S1-08 save path if merged, else the S0-05 writer fed by S1-08's
   `ll_adapter_get`; soft coordination with S1-08, not a frontmatter dep — inline the
   small tensor-pull if S1-08 has not merged) + sidecar atomically
   (write-temp-then-rename both).
5. Resume: fresh process → load base + adapter → S1-01/S1-02 init → first opt-graph
   build → `ll_opt_state_set` every m/v by name → `ll_opt_set_iter` → Python restores
   LR/RNG/data cursor from KV and seeks the collator.
6. `src/learning_llamas/train/loop.py` wiring: `save_every_n_steps`, `resume_from=path`,
   filling the S1-05 hook points; refuse resume when the sidecar's adapter hash does not
   match the loaded adapter.
7. Tests `tests/test_resume.py` (CPU): train N steps → checkpoint → continue M steps
   recording per-step loss and A/B bytes; separately restore in a fresh process and run
   M steps — **bitwise identical** losses, A/B tensors, and m/v tensors (CPU is
   deterministic per ADR-0002 defaults). Also: version KV mismatch and adapter-hash
   mismatch both refuse with clear errors; mid-window checkpoint attempt errors.

## Out of scope

- Grad-accumulator serialization for mid-window checkpoints (deferred; enforced-boundary
  checkpointing makes it unnecessary — revisit only if a real need appears).
- SGD state (SGD keeps no momenta; iter/KV path already covers it — test can be added
  when an SGD trainer config exists).
- Upstreaming name-keyed optimizer state *storage* into ggml-opt (BLUEPRINT §10 risk 3
  floats it; this ticket only adds accessors).
- Adapter save/merge itself (S1-08) and the interchange format (S0-05/D3).

## Acceptance criteria

- [ ] Fork PR adds `ggml_opt_step_momenta` + iter get/set with a vendored unit test or
      test-backend-ops-adjacent smoke; learning-llamas PR bumps the submodule.
- [ ] `pytest tests/test_resume.py` passes on the Linux CPU VM: resumed run matches the
      uninterrupted run **bitwise** (losses + A/B + m/v) for M ≥ 3 post-checkpoint steps.
- [ ] Sidecar refuses to load as an adapter in stock llama.cpp (wrong `general.type`),
      and learning-llamas refuses version/hash mismatches (tested).
- [ ] Checkpoint outside an accumulation boundary raises the documented error (tested
      with opt_period > 1).
- [ ] `_ffi` symbol-table test resolves the new `ll_opt_state_*` and fork-side symbols.
- [ ] `ci-cpu / test` green per-PR with the new tests.

## Testing & verification

- `tests/test_resume.py` (new), pytest on S0-06 fixture models, `ci-cpu / test` per-PR;
  the bitwise assertion is CPU-only by design (GPU backends get tolerance-based resume
  checks when their stages wire training, out of scope here).
- No new ops/kernels, so no MODE_GRAD cases; the fork-side accessors are read/write
  plumbing verified through the resume test.

## PR notes

- Branch: `ticket/S1-09-optimizer-state-sidecar-resume`.
- **Two-repo flow** (per S0-02): the accessor patch PRs against the fork's
  `learning-llamas-base` with `[S1-09]` in the title; a learning-llamas PR bumps the
  `vendor/llama.cpp` submodule and carries the shim/Python/test changes.
- Upstreaming disposition: accessors **upstream-later** (small, generally useful — offer
  once the resume test proves them); sidecar format and shim ABI **fork-local**.
- Per-file provenance headers on `farm_checkpoint.cpp` where it mirrors ggml-opt
  internals' iteration patterns, per S0-01 policy.
