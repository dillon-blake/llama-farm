---
id: S1-10
title: "Gradient clipping"
stage: 1
track: shim
size: S
deps: ["S1-02"]
status: open
pr: null
---

# S1-10 — Gradient clipping

**One-line outcome:** global-norm gradient clipping is available in the training loop via a
flat C ABI (`lf_set_grad_clip`), with pre/post-clip norms exposed for logging, despite
ggml-opt fusing the optimizer step into the backward graph.

## Why (context)

ggml-opt has no gradient clipping anywhere (BLUEPRINT gap G15), and the reason it is not a
trivial add is structural: the optimizer step is not a separate phase but a set of
`OPT_STEP_ADAMW`/`OPT_STEP_SGD` nodes appended to the backward graph itself — `gb_opt` is
built by walking the forward nodes and attaching one opt-step node per param
(`vendor/llama.cpp/ggml/src/ggml-opt.cpp:511-539`; the node constructors are
`ggml_opt_step_adamw` / `ggml_opt_step_sgd`, `vendor/llama.cpp/ggml/src/ggml.c:6150` and
`:6178`). There is no host-visible moment "after backward, before step" inside a single
`ggml_opt_eval`: the whole graph runs in one `ggml_backend_sched_graph_compute`
(`vendor/llama.cpp/ggml/src/ggml-opt.cpp:824-826`).

Gradient accumulation changes the picture and is why G15 names two approaches. With
`opt_period > 1`, `ggml_opt_alloc` selects a grad-only build (`GGML_OPT_BUILD_TYPE_GRAD`)
for the first `opt_period − 1` micro-steps and the fused optimizer build only for the last
(`vendor/llama.cpp/ggml/src/ggml-opt.cpp:727-732`; accumulators reset at the window start,
`:727-728`). The gradient accumulators are host-reachable by param tensor via
`ggml_opt_grad_acc` (`vendor/llama.cpp/ggml/include/ggml-opt.h:156`, implementation
`vendor/llama.cpp/ggml/src/ggml-opt.cpp:633-635`) plus `ggml_backend_tensor_get`/`_set`
(`vendor/llama.cpp/ggml/include/ggml-backend.h:92-93`) — so a **host clip pass** over the
accumulators between accumulation and step is possible. Its known caveat, to be
characterized in this ticket: the final micro-batch's gradient contribution is computed in
the same fused compute that applies the step, so a pure host pass sees only the grads
accumulated before that last eval.

The alternative is a **scale-by-global-norm stage inside the built graph**: forward-only
nodes (square/sum per accumulator → total → sqrt → factor `clip/max(norm, clip)` →
broadcast-multiply each grad) inserted immediately before the opt-step nodes consume the
grads. No new kernels and no VJPs are needed — the clip stage is never differentiated. It
can land either as a small fork-side extension of the `gb_opt` build loop (two-repo flow,
S0-02) or by copying the sanctioned high-level ggml-opt pieces into the shim
(`vendor/llama.cpp/ggml/include/ggml-opt.h:191-194` explicitly blesses copying). The
manifest decision rule applies: **pick after measuring; record the choice in the PR.**

## What to do

1. **Prototype both approaches** from the Why section against the S1-02 step loop on the
   S0-06 fixture models: (a) host clip pass over `ggml_opt_grad_acc` accumulators (correct
   only for the grads visible before the fused final eval — measure and document exactly
   what it misses at `opt_period = 1` vs `> 1`); (b) in-graph global-norm scale stage
   before the opt-step nodes (exact for all micro-batches, small graph-size cost). Measure
   per-step overhead of each on CPU at fixture scale and one larger synthetic shape.
2. **Pick one as the shipped implementation** (the other is deleted, not left half-wired),
   and record the decision, the measurements, and the correctness caveat analysis in the PR
   description. If (b) wins and needs a fork-side hook in ggml-opt's build, that lands via
   the S0-02 two-repo flow; if it is implementable purely in the shim's copied loop
   pieces, no vendor change is needed.
3. **C ABI in `csrc/farm_api.h`:** `lf_set_grad_clip(ctx, double max_norm)` (0 disables,
   the default); extend the S1-02 `lf_step_result` (or add a getter) with
   `grad_norm_pre_clip` and `grad_norm_post_clip` for the most recent optimizer step, so
   Python logging can plot both. Norms are global L2 over all adapter A/B grads.
4. **Implementation in `csrc/farm_train.cpp`** behind that ABI, honoring `opt_period`:
   clipping must apply to the *accumulated* gradient once per optimizer step, never
   per-micro-batch.
5. **`_ffi` bindings** and Python plumbing: expose `max_grad_norm` in the step-call
   options. Wiring into `train/loop.py`'s named clip hook is soft coordination with S1-05
   (not a frontmatter dep): if `loop.py` exists, wire the hook; otherwise the ABI + tests
   stand alone.
6. **Tests `tests/test_grad_clip.py`:** build a synthetic exploding-grad batch (e.g. a
   loss epilogue scaled by a large constant on a fixture model) and assert: pre-clip norm
   exceeds the threshold; post-clip norm equals the threshold within F32 tolerance; the
   applied update is finite; with clipping disabled the reported pre/post norms are equal;
   with `opt_period > 1` the clip is applied once per window to the accumulated grad
   (compare against a manual numpy computation of the same clip from accumulator
   snapshots).

## Out of scope

- Per-parameter or value-based clipping (only global-norm clipping is in v1 scope; open a
  backlog ticket if a trainer needs more).
- LR schedules and optimizer hyperparameter plumbing (S1-05 `loop.py`).
- Persisting clip configuration in checkpoints — the S1-09 sidecar owns training-state
  serialization; `max_norm` is a config value the Python loop re-supplies on resume.
- Any GPU-specific tuning of the clip stage — Stage 1 is CPU; the in-graph variant runs
  wherever the sched places it in later stages.

## Acceptance criteria

- [ ] `pytest tests/test_grad_clip.py` passes on the Linux CPU VM: exploding-grad batch is
      clipped to the configured norm (post-clip norm == threshold within tolerance),
      pre-clip norm is reported larger, and the step remains finite.
- [ ] Disabled-clip test: with `max_norm = 0`, adapter grads and updates are bit-identical
      to a build of the same step without any clip code path engaged.
- [ ] `opt_period > 1` test: exactly one clip per accumulation window, applied to the
      accumulated grad, matching a numpy reference computation.
- [ ] `lf_step_result` (or getter) exposes pre- and post-clip norms; a test asserts both
      are populated and consistent (`post ≤ pre`, `post ≤ max_norm + tol`).
- [ ] The PR description records the measured comparison of approaches (a) vs (b) and the
      final choice, per the manifest decision rule.
- [ ] `ci-cpu / test` is green per-PR with the new tests.

## Testing & verification

`tests/test_grad_clip.py` in the S0-06 pytest harness, running per-PR in `ci-cpu / test`
(S0-07). No new ggml ops are introduced (the in-graph variant composes existing forward
ops), so no `test-backend-ops` MODE_GRAD cases belong to this ticket; gradient correctness
of the underlying step is covered by S1-03's finite-difference check. Manual: attach the
pre/post-clip norm log for the exploding-grad repro to the PR.

## PR notes

- Branch: `ticket/S1-10-global-norm-grad-clipping`.
- Expected to be a single llama-farm PR (shim + bindings + tests). Only if the in-graph
  variant requires touching vendored ggml-opt does the S0-02 two-repo flow apply: fork PR
  first, then a submodule-bump PR here referencing this ticket ID.
- Upstreaming disposition: **fork-local** initially; if the in-graph clip stage proves
  clean as a ggml-opt extension, flag it as a candidate upstream PR in the ticket
  retrospective (ROADMAP §11 triage class b — propose once stable).
- Any copied ggml-opt loop pieces carry per-file provenance headers (source path, commit
  `4f37f51`, MIT) per S0-01 policy.
