---
id: S1-17
title: "Gradient checkpointing: layer-segmented recompute (+ optional CPU-offloaded boundaries)"
stage: 1
track: shim
size: L
deps: ["S1-02"]
status: done
pr: 32
---

# S1-17 — Gradient checkpointing: layer-segmented recompute (+ optional CPU-offloaded boundaries)

**One-line outcome:** shim-level gradient checkpointing exists: the forward stores only
layer-boundary activations and the backward recomputes each segment on demand, cutting
peak activation memory for long contexts, with an optional pinned-host offload path for
boundary states.

## Why (context)

ggml-opt has no gradient checkpointing anywhere (BLUEPRINT G9;
`vendor/llama.cpp/ggml/src/ggml-opt.cpp` — grep confirms the concept is absent): the
full forward+backward graph's activations are live simultaneously, which is why the
vendored training example reports ~24 GB for a 1B-parameter F32 finetune at n_ctx 512
(`vendor/llama.cpp/examples/training/README.md:5`). LoRA on a quantized base is far
lighter in weights, but activations still dominate at long context (BLUEPRINT §10
risk 5). This is graph-level work in the shim by design: ROADMAP §8 states checkpointing
is blueprint-P2 graph work that *complements* FA backward — FA kills the per-layer
`n_ctx²` attention term, checkpointing kills the across-all-layers activation residency;
each matters without the other, and this ticket needs neither FA nor any new kernel.

Segmenting at layer boundaries is natural because llama.cpp already builds every graph
layer-by-layer (the `for (int il = 0; il < n_layer; ++il)` loop,
`vendor/llama.cpp/src/models/llama.cpp:126`), and boundary tensors are identifiable
without per-arch work: 108 model builders name the per-layer output `"l_out"` via the
graph callback (e.g. `vendor/llama.cpp/src/models/llama.cpp:224`), and the vendored fork
additionally records per-layer input tensors on the graph result —
`llm_graph_result::get_layer_inp` (`vendor/llama.cpp/src/llama-graph.h:802`, backing
vector at `:837`), populated by 8 builders today (`res->t_layer_inp[il] = inpL;`,
`vendor/llama.cpp/src/models/llama.cpp:127`).

The hard constraint is BLUEPRINT D1: ggml-opt's dynamic-graph mode keys gradient
accumulators and AdamW momenta by forward-graph node index
(`vendor/llama.cpp/ggml/src/ggml-opt.cpp:458-486`), with a varying-batch assert at
`:851` — so whatever graph the checkpointed step builds, its node topology must be
identical every step, and the S1-02 shape-signature enforcement must keep covering it.
The offload option is the unsloth-derived idea from BLUEPRINT §7 item 4: boundary states
move to pinned host memory through a ring buffer, worthwhile only past a crossover
around seq ≥ ~512; ROADMAP §6 notes Apple unified memory later makes this a no-copy
operation on Metal — the design must keep the offload step a swappable no-op for that
stage. On CPU-only training (this stage), offload is a correctness-testable but
perf-neutral path.

## What to do

1. **Boundary identification** (`csrc/farm_train.cpp` or new `csrc/farm_ckpt.cpp`):
   after `model.build_graph` in the S1-02 forked loop, locate layer-boundary tensors —
   prefer `res->get_layer_inp(il)` (`vendor/llama.cpp/src/llama-graph.h:802`) where the
   builder populates it, fall back to scanning graph nodes named `"l_out"` (the
   108-builder convention). Error clearly on archs where neither yields a full boundary
   chain (preflight interplay noted in S1-11's report — soft coordination, not a dep).
2. **Segmented step construction:** prototype the single-graph explicit-recompute
   scheme first — build the training graph so each segment's forward subgraph appears
   twice: once in a boundary-only forward chain (checkpoint stores), and once as
   recompute nodes ordered immediately before that segment's backward nodes, so
   `ggml_gallocr` lifetime analysis keeps at most one segment's interior activations
   live. Feed the result through the unchanged
   `ggml_opt_prepare_alloc`/`ggml_opt_alloc`/`ggml_opt_eval` sequence
   (`vendor/llama.cpp/ggml/include/ggml-opt.h:176-181`, `:185`). If grad/momentum
   binding through `ggml_build_backward_expand`
   (`vendor/llama.cpp/ggml/src/ggml.c:7020`) proves incompatible with duplicated
   subgraphs, fall back to shim-orchestrated multi-eval (per-segment forward/backward
   with a shim-owned boundary stash and manual grad accumulation into the param
   tensors). Record the chosen mechanism and why in the PR description; both must keep
   node topology fixed across steps (D1).
3. **Configuration** in `csrc/farm_api.h` + `_ffi`: `ll_set_grad_checkpointing(ctx,
   mode, segment_len)` — `off` (default, byte-identical to today's path) | `on`
   (segment every `segment_len` layers, default 1). Reject mode changes after the first
   opt-graph build with the S1-02-style clear error (topology).
4. **Optional pinned-host offload of boundary states:** allocate the boundary ring in a
   host buffer (`ggml_backend_dev_host_buffer_type`,
   `vendor/llama.cpp/ggml/include/ggml-backend.h:187`; plain CPU buffer when training on
   CPU) and copy boundaries out after forward / back in before each segment's
   recompute. Gate behind `offload_boundaries=true` with the documented seq ≥ ~512
   crossover heuristic (BLUEPRINT §7 item 4) stated as guidance, not enforced. Keep the
   copy step behind one function so the Metal stage can make it a no-op (ROADMAP §6).
5. **Memory instrumentation:** report peak compute-buffer allocation per configuration
   (off / on / on+offload) via the allocator's reserved sizes; expose in the step-result
   stats and write a small comparison table in `docs/dev/` from a fixture-model run
   (this is the artifact reviewers check).
6. **Correctness tests** `tests/test_grad_checkpointing.py`: on the S0-06 fixture
   models, run N identical steps with checkpointing off vs on (and on+offload) from
   identical initial state — losses, gradient accumulators (`ggml_opt_grad_acc`), and
   post-step A/B tensors must be **bitwise identical** on CPU (recompute of identical
   ops on identical inputs is deterministic per ADR-0002 defaults). Also: shape-drift
   still rejected; mode-change-after-build rejected; peak-memory assertion — the `on`
   configuration reports strictly lower peak compute-buffer size than `off` for a
   multi-layer fixture at a long-enough n_ctx.

## Out of scope

- Flash-attention backward and the chunked-attention fallback (ROADMAP §8, FA1-FA8) —
  complementary, separately ticketed in later stages.
- Kernel work of any kind; no new ops, no vendored-code behavior changes (if step 2's
  fallback needs a fork-side accessor, that is a minimal addition via S0-02, not a
  behavior change).
- Offload perf tuning and async copy overlap — meaningful only on GPU stages; measured
  there (S2+/backlog).
- Selective/attention-only checkpointing policies and autotuned segment length —
  `tickets/backlog/` once profiles exist.
- Extending `t_layer_inp` population to all ~133 archs upstream (record as a candidate
  fork patch if the `"l_out"` fallback proves fragile).

## Acceptance criteria

- [ ] `pytest tests/test_grad_checkpointing.py` passes on the Linux CPU VM: bitwise
      grad/loss/A-B equality off-vs-on and off-vs-on+offload over N ≥ 3 steps.
- [ ] Peak-memory report exists (docs/dev table + step stats) and shows a strictly
      lower peak for `on` vs `off` on the multi-layer fixture config; the numbers are
      reproduced by CI output.
- [ ] Checkpointing works through the unchanged S1-02 `ll_train_step` ABI (same epilogue
      registry, named inputs, and shape enforcement — existing S1-02 tests still green
      with checkpointing on).
- [ ] Mode change after first build and shape drift both raise the documented errors
      (tested).
- [ ] `_ffi` symbol-table test resolves `ll_set_grad_checkpointing`.
- [ ] `ci-cpu / test` per-PR green; the memory-comparison run is in nightly ci-cpu if it
      exceeds the per-PR budget.

## Testing & verification

- `tests/test_grad_checkpointing.py` (new), pytest on the S0-06 fixture models (F32 +
  Q8_0 tiny llama-arch, multi-layer variant), `ci-cpu / test` per-PR (S0-07); the
  bitwise assertion is CPU-only by design — GPU stages re-verify under the ADR-0002
  0.05 cross-backend criterion when they adopt checkpointing.
- No new ops/kernels — no `test-backend-ops` MODE_GRAD cases; correctness is the
  bitwise off-vs-on equivalence above.

## PR notes

- Branch: `ticket/S1-17-gradient-checkpointing-layer-recompute`.
- Expected single learning-llamas PR (shim + `_ffi` + tests + docs table). If a fork-side
  accessor or wider `t_layer_inp` population is needed, it lands first as a fork PR +
  submodule bump per the S0-02 two-repo flow, referenced from this ticket.
- Upstreaming disposition: **fork-local** (shim graph construction); any `t_layer_inp`
  builder additions in the fork are **upstream-later** candidates.
- Size L: split commits — boundary identification + off-mode plumbing, then the
  recompute scheme, then offload + instrumentation — within one PR.
