---
id: S4-05
title: "Vulkan V4+V5: DIAG_MASK_ZERO + backward-op constraint audit (incl. max_bias validation)"
stage: 4
track: kernels
size: M
deps: ["S4-01", "S1-20"]
status: open
pr: null
---

# S4-05 — Vulkan V4+V5: DIAG_MASK_ZERO + backward-op constraint audit (incl. max_bias validation)

**One-line outcome:** the Vulkan backward suite is audited and safe: `DIAG_MASK_ZERO`
handled (shader or dump-backed skip), contiguity preconditions of every existing `*_BACK`
shader verified against real backward graphs, `ROPE_BACK`'s unconditional supports_op
tightened or verified, and `SOFT_MAX_BACK`'s silently-unvalidated `max_bias` resolved by
test.

## Why (context)

Vulkan already ships every `*_BACK` op in the training set (ROADMAP §2), but those shaders
were written for whatever graphs upstream exercised — nobody has validated their declared
preconditions against the tensor layouts real *training* backward graphs produce. ROADMAP §7
V5 calls for exactly this audit before the stage-4 milestone leans on the suite. Three
defects are already known. First, `DIAG_MASK_ZERO` is CPU-only (ROADMAP §2/V4): it is
emitted only as the backward of `DIAG_MASK_INF` (`vendor/llama.cpp/ggml/src/ggml.c:6748-6755`)
and of itself (`:6756-6761`), and modern graphs mask via the additive soft_max mask instead —
so per the S2-09 precedent the honest first step is a graph dump, not a speculative kernel.
Second, `ROPE_BACK`'s supports_op returns true unconditionally — it falls through with the
no-op view cases to `return true`
(`vendor/llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp:17474-17481`), while `ROPE` itself
at least checks contiguous rows (`:17472-17473`); the backend implements mrope/vision
pipelines (`:5311`, `:5316`; mode selection at `:10882-10907`) and passes a `backprop` flag
into the shared rope constants (`ggml_vk_rope`, `:12875`; `ggml_vk_make_rope_constants`,
`:12466`, flag packed at `:12500`) — whether every mode honors `backprop` correctly is
asserted by nobody (ROADMAP §7 V5: "tighten or verify").

Third, the K-SMB inconsistency (ROADMAP §3): Vulkan currently **accepts `max_bias > 0`
completely unvalidated** on `SOFT_MAX_BACK`. Its supports_op checks only contiguity and
types (`vendor/llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp:17580-17582`), the dispatch
passes both op-params (`:12798`), and the shader reads only `scale = p.param1`
(`vendor/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/soft_max_back.comp:27`), silently
ignoring `max_bias`. Per the K-SMB analysis the ALiBi bias is additive and constant w.r.t.
the logits, so ignoring it is mathematically a no-op for this kernel — but that analysis
must be *verified by test, not assumed*. S1-20 enabled the CPU-side `max_bias = 8.0`
MODE_GRAD cases (generation loop `vendor/llama.cpp/tests/test-backend-ops.cpp:8882`;
forward-eval `test_soft_max_back` cases at `:8925-8931`) and explicitly left the Vulkan
audit to this ticket: once those cases run against a CPU baseline, the Vulkan behavior is
detectable by ordinary backend-vs-CPU comparison instead of silently skipped.

## What to do

Fork changes ride the S0-02 two-repo flow; the audit evidence lands in learning-llamas.

1. **Dump real training backward graphs first** (S2-09 pattern): the tiny dense-LoRA config
   from S1-12, backward expanded, loss types SUM/MEAN, grad accumulation on/off. Produce a
   backward-graph op histogram and, for every `*_BACK` node, record the actual tensor
   layouts: contiguity, strides, view-ness of each src. CPU-build dumps are valid for the
   histogram; a Vulkan build with `GGML_SCHED_DEBUG=2`
   (`vendor/llama.cpp/ggml/src/ggml-backend.cpp:1740-1741`) additionally shows which nodes
   the backend accepts today.
2. **V4 — `DIAG_MASK_ZERO`:** if the dumps show it emitted, clone `diag_mask_inf.comp`
   (trivial per-element shader with `ncols/rows_per_channel/n_past` push constants,
   `vendor/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/diag_mask_inf.comp`) into a
   `DIAG_MASK_ZERO` pipeline plus the six mechanical touch points, and enable the
   `test_diag_mask_inf` grad cases on Vulkan
   (`vendor/llama.cpp/tests/test-backend-ops.cpp:8861-8863` region; the backward emits
   `DIAG_MASK_ZERO`). If absent in every configuration, record a supports_op comment
   pointing at the evidence (S2-09 branch-(a) style) — coordinate with S2-09's Metal
   evidence doc, which covers the same op family.
3. **V5 — contiguity audit:** compare each `*_BACK` supports_op precondition (e.g.
   `SILU_BACK`/`RMS_NORM_BACK` require contiguous src0, `:17497-17499`; `SOFT_MAX_BACK`
   requires contiguous src0/src1, `:17580-17582`) against the dumped real layouts. For each
   mismatch, decide: layout always satisfies the precondition (document), precondition too
   strict and cheap to relax (fix), or precondition correctly rejects a real layout
   (the sched falls back to CPU — ticket it with measured frequency). No silent acceptance
   of layouts a shader indexes incorrectly.
4. **V5 — `ROPE_BACK`:** either tighten supports_op to what the rope shaders demonstrably
   handle in backprop mode, or verify by test that mrope/vision modes produce correct
   gradients: extend the existing rope MODE_GRAD coverage to those modes on Vulkan and
   compare against the CPU oracle. Minimum bar: `ROPE_BACK` no longer claims support for
   anything untested — mirror `ROPE`'s own contiguous-rows check (`:17472-17473`) unless
   the audit proves it unnecessary.
5. **V5 — `SOFT_MAX_BACK` `max_bias`:** run the S1-20-enabled `max_bias > 0` MODE_GRAD and
   forward-eval cases on Vulkan (lavapipe + native). If they pass within ADR-0002
   tolerances, the K-SMB no-op analysis is confirmed — record that in the audit doc and add
   a shader comment at `soft_max_back.comp:27` explaining why `max_bias` is correctly
   ignored. If they fail, gate honestly (add the `max_bias == 0` condition at
   `:17580-17582`) and open a follow-up for the shader fix. Either way the parameter stops
   being *silently* unvalidated.
6. **Record the audit** in `docs/graphs/vulkan-v5-backward-audit.md` (learning-llamas): dump
   configs, per-op findings, decisions taken, and anything ticketed instead of fixed. Fix
   what is small in this PR; ticket what is not.
7. **Submodule bump PR** in learning-llamas per S0-02 if (and only if) the fork changed.

## Out of scope

- Metal `ADD1`/`DIAG_MASK_ZERO` — S2-09 (share dump evidence in prose).
- CUDA `SOFT_MAX_BACK` ALiBi gate lift — S3-04.
- New ops (`OUT_PROD` family, sparse CE, MoE/SSM backward) — S4-02/S4-03/S4-04, S4-06/S4-07;
  this ticket audits only *existing* Vulkan `*_BACK` shaders plus the V4 parity item.
- MoE/SSM backward-graph layouts — audited by their own port tickets (S4-06/S4-07) once
  those graphs exist.
- Any ALiBi shader rework beyond the honest gate (if the test fails) — follow-up ticket.

## Acceptance criteria

- [ ] `docs/graphs/vulkan-v5-backward-audit.md` exists with the backward-graph op
      histogram, per-`*_BACK`-op layout findings, and a decision line for every audited op.
- [ ] `DIAG_MASK_ZERO` is resolved by exactly one of: (a) shader landed and the
      `test_diag_mask_inf` grad cases pass on Vulkan within ADR-0002 tolerance vs the CPU
      oracle, or (b) dump-backed absence recorded in a supports_op comment, with no kernel
      code added.
- [ ] The `max_bias > 0` `SOFT_MAX` MODE_GRAD and `test_soft_max_back` forward cases
      execute on Vulkan (no longer skipped or silently wrong): either passing within
      ADR-0002 tolerance with the confirmation comment landed, or rejected by a new
      supports_op gate with a follow-up ticket referenced.
- [ ] `ROPE_BACK` supports_op no longer returns unconditional true, OR the audit doc
      records passing Vulkan-vs-CPU MODE_GRAD evidence for the modes it claims (including
      mrope/vision or their explicit rejection).
- [ ] Every contiguity mismatch found is fixed, documented as impossible, or ticketed —
      cross-referenced in the audit doc.
- [ ] learning-llamas PR green in `ci-vulkan / lavapipe` (and the native `gpu` lane if fork
      kernels changed); `ci-cpu` green.

## Testing & verification

Primary harness: vendored `tests/test-backend-ops` — MODE_GRAD and eval on Vulkan vs the
CPU oracle under ADR-0002 tolerances (S0-09), run per-PR targeted in `ci-vulkan / lavapipe`
and nightly on both S4-01 lanes; the `max_bias` cases exist thanks to S1-20, and the
`DIAG_MASK_ZERO` grad path uses the in-tree `test_diag_mask_inf` cases. Graph dumps run on
the Linux CPU VM (histogram) and the Vulkan VM (sched acceptance); dump artifacts live in
the audit doc. The audit's payoff — no `*_BACK` shader silently mis-supporting a real
training layout — is what S4-09's fallback-forbidden milestone then relies on.

## PR notes

- Branch: `ticket/S4-05-vulkan-diag-mask-backward-audit`.
- Two-repo flow per S0-02 for any fork changes (shader, gates, comments); the audit doc is
  learning-llamas-side either way.
- Upstreaming disposition: **upstream-early** for kernels/gate fixes (ROADMAP §11 triage
  class a — pure additions and correctness gates; mention the unvalidated-`max_bias`
  finding upstream as S1-20's PR did); the audit doc is **fork-local**.
- Provenance per S0-01 on any cloned shader (`diag_mask_inf.comp`, MIT, `4f37f51`).
- Soft coordination: shares dump tooling/evidence with S2-09 (Metal M10); S4-09 consumes
  the audit conclusions for its zero-fallback assertion.
