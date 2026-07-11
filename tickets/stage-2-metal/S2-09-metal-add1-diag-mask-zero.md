---
id: S2-09
title: "Metal M10: ADD1 + DIAG_MASK_ZERO (+ graph-dump confirmation)"
stage: 2
track: kernels
size: S
deps: ["S2-01"]
status: open
pr: null
---

# S2-09 — Metal M10: ADD1 + DIAG_MASK_ZERO (+ graph-dump confirmation)

**One-line outcome:** the two small Metal coverage exceptions from ROADMAP §2 are closed —
either with trivial kernels, or with dump-backed evidence that real training graphs never emit
them and a supports_op comment recording that.

## Why (context)

ROADMAP §2 leaves exactly two small coverage exceptions around the Metal backward suite:
`ADD1` is missing on Metal and `DIAG_MASK_ZERO` is CPU-only. Verified at `4f37f51`: neither
`GGML_OP_ADD1` nor `GGML_OP_DIAG_MASK_ZERO` (nor `GGML_OP_DIAG_MASK_INF`) has a case in the
Metal coverage switch (`vendor/llama.cpp/ggml/src/ggml-metal/ggml-metal-device.m:1051-1368`,
default `false` at `:1365-1366`), and no matching kernel exists in `ggml-metal.metal`.

But ROADMAP §6 M10 warns these ops may be dead in modern training graphs, so the honest first
step is a dump, not kernels. `ADD1` reaches a backward graph only through `ggml_add1_or_set`
(`vendor/llama.cpp/ggml/src/ggml.c:6398-6410`), used by the `SUM` (`:6546-6550`) and `MEAN`
(`:6556-6560`) backward cases — and only when the source already has a gradient/accumulator;
otherwise a plain `REPEAT` is emitted. ggml-opt builds every loss as `ggml_sum`
(`vendor/llama.cpp/ggml/src/ggml-opt.cpp:386-398`), so whether `ADD1` actually appears depends
on grad-accumulation configuration and must be observed, not assumed. `DIAG_MASK_ZERO` is
emitted only as the backward of `DIAG_MASK_INF` (`:6748-6755`, and of itself, `:6756-6761`) —
but modern llama.cpp graphs do causal masking via the single additive soft_max mask built by
`fill_mask` (`vendor/llama.cpp/src/llama-graph.cpp:406-453`), not via `DIAG_MASK_INF`, so the
whole family is plausibly absent. Scope stays honest per the manifest: no speculative ops.

## What to do

1. **Dump real training graphs first.** Build the tiny dense-LoRA training graph with backward
   expanded (the S1-12 convergence-gate model config; use the S1-03 harness if merged, else a
   fork-side driver modeled on `examples/training/finetune.cpp`). Produce an op histogram of
   the backward graph via `ggml_graph_print` (`vendor/llama.cpp/ggml/include/ggml.h:2754`) /
   a node-loop over the graph accessors (`ggml.h:2740-2742`), and capture a Metal-build sched
   report with `GGML_SCHED_DEBUG=2` (env read at
   `vendor/llama.cpp/ggml/src/ggml-backend.cpp:1740-1741`). Cover the configurations where the
   ops could plausibly appear: loss types SUM and MEAN, and grad accumulation on
   (`opt_period > 1`) and off.
2. **Record the evidence** in `docs/graphs/metal-m10-op-inventory.md` (llama-farm repo): the
   dump configs, the per-op counts for `ADD1`/`DIAG_MASK_INF`/`DIAG_MASK_ZERO`, and the
   conclusion.
3. **If neither op appears in any configuration:** add a comment at the supports_op default in
   `vendor/llama.cpp/ggml/src/ggml-metal/ggml-metal-device.m` stating that `ADD1` and
   `DIAG_MASK_ZERO` are intentionally unimplemented on Metal, pointing at the evidence doc and
   this ticket; close the ticket. No kernels.
4. **If an op appears:** implement the trivial kernel(s) per existing elementwise patterns plus
   the five mechanical additions (ROADMAP §6 preamble). `ADD1` is a broadcast-scalar add
   (CPU semantics: `ggml_compute_forward_add1_f32`,
   `vendor/llama.cpp/ggml/src/ggml-cpu/ops.cpp:775`); `DIAG_MASK_ZERO` zeroes past-diagonal
   elements (CPU dispatch at `ops.cpp:5367`). Tests: the harness has `test_diag_mask_inf` grad
   cases (`vendor/llama.cpp/tests/test-backend-ops.cpp:4791`, cases `:8861-8863`) whose
   backward emits `DIAG_MASK_ZERO` — note the grad case only runs on Metal if `DIAG_MASK_INF`
   forward is also supported there (eval_grad re-checks supports_op on every backward node,
   per the S1-20 finding), so implement the forward sibling in that branch too. No `ADD1`
   test case exists today — add eval + MODE_GRAD cases (its backward is existing ops:
   `ggml.c:6468-6475`).
5. **Submodule bump PR** in llama-farm per S0-02 if (and only if) the fork changed.

## Out of scope

- Any op not observed in the dumps — no speculative kernels (explicit manifest constraint).
- Vulkan `DIAG_MASK_ZERO` (ROADMAP §7 V4) — the dump evidence should be shared with that
  ticket's owner in prose, but the decision there is its own.
- The dense-path zero-fallback milestone — S2-10 (this ticket's outcome feeds it either way:
  ops implemented, or proven absent so fallback cannot occur).

## Acceptance criteria

- [ ] `docs/graphs/metal-m10-op-inventory.md` exists with the backward-graph op histogram for
      all four dump configurations (SUM/MEAN × opt_period 1/>1) and the captured
      `GGML_SCHED_DEBUG` output.
- [ ] Exactly one of:
      (a) both ops absent in every dump → supports_op comment landed in the fork referencing
      the evidence doc, and no kernel code added; or
      (b) op(s) present → kernels landed and the named `test-backend-ops` eval + MODE_GRAD
      cases pass on Metal within the ADR-0002 per-op tolerance, with Metal-vs-CPU parity
      ≤ 0.05 @ fp16 where a grad path exists.
- [ ] The evidence doc states which branch was taken and why (one paragraph).
- [ ] llama-farm PR green in `ci-metal / build` (and `ci-metal / grad` if kernels landed).

## Testing & verification

Branch (a) is evidence-only: verification is the dump artifacts in the evidence doc plus
reviewer spot-check; it rides `ci-cpu`'s docs checks and needs no new lanes. Branch (b) uses
the vendored `tests/test-backend-ops` (eval + MODE_GRAD vs the CPU oracle, ADR-0002
tolerances) in the targeted per-PR `ci-metal / grad` lane and the nightly Metal sweep. The
graph dumps themselves run on the S0-08 Apple-Silicon machine or the S2-01 hosted runner
(CPU-build dumps are also valid for the op histogram — only the sched report needs Metal).

## PR notes

- Branch: `ticket/S2-09-metal-add1-diag-mask-zero`.
- Two-repo flow per S0-02 only if the fork changes (branch (b), or the branch-(a) comment);
  the evidence doc is a llama-farm-side change either way.
- Upstreaming disposition: **upstream-early** for any kernels (ROADMAP §11 triage class a,
  pure additions behind supports_op); the evidence doc and comment are **fork-local**.
- Provenance per S0-01 on any copied kernel patterns (`ggml-metal.metal`, MIT, `4f37f51`).
