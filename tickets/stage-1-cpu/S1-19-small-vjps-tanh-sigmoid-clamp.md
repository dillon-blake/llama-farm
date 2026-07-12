---
id: S1-19
title: "Small VJPs: TANH, SIGMOID, CLAMP (composite backward rules)"
stage: 1
track: kernels
size: S
deps: ["S0-02"]
status: pr-open
pr: https://github.com/dillon-blake/llama-farm/pull/14
---

# S1-19 — Small VJPs: TANH, SIGMOID, CLAMP (composite backward rules)

**One-line outcome:** `ggml_compute_backward` handles TANH (`grad·(1−tanh²)`), SIGMOID
(`grad·(y−y²)`), and CLAMP (`grad·step` masks) as kernel-free composites of existing ops,
with MODE_GRAD coverage for all three.

## Why (context)

Three tiny missing backward rules block whole model families (BLUEPRINT G8, §7 item 5). TANH
and SIGMOID hit the unary default abort in `ggml_compute_backward`
(`vendor/llama.cpp/ggml/src/ggml.c:6872-6876`, inside the switch spanning `:6430-6913`);
CLAMP has no case at all and falls into the op-level default abort (`:6904-6907`). The
concrete victims: gemma2 (and gemma3 GGUFs with `final_logit_softcapping` set) apply
`ggml_tanh` for logit softcap and are rejected by the trainability preflight (BLUEPRINT §8,
"gemma2/3 blocked"); sigmoid routers (DeepSeek-V3/GPT-OSS MoE, ROADMAP §9 E6) and
`norm_w`/clamped-swiglu paths (E5) block MoE arch tickets; and the GRPO trainer composes PPO
clip from RELU identities purely because CLAMP has no VJP (BLUEPRINT §6.3).

No kernel is needed on any backend (ROADMAP §3 K-TANH; §9 E5/E6): each VJP is a composite of
existing forward ops emitted into the backward graph. TANH and SIGMOID compute from the
**saved output** `y` — the precedent is the EXP case, which uses `tensor` (the op's own
output) rather than `src0` (`vendor/llama.cpp/ggml/src/ggml.c:6857-6861`). CLAMP's rule is
`grad · step(scale(x,1,−min)) · step(scale(x,−1,max))` — `scale_bias(x,1,−min) = x−min` and
`scale_bias(x,−1,max) = max−x`, so the two `step` factors are exactly the inside-the-bounds
indicator; all constructors exist (`ggml_step` `vendor/llama.cpp/ggml/include/ggml.h:1117`,
`ggml_scale_bias` `:1467`, `ggml_tanh` `:1125`, `ggml_sigmoid` `:1153`).

One wrinkle, verified in-tree: `ggml_clamp` currently builds its result as
`ggml_view_tensor(ctx, a)` — an aliasing, effectively in-place op — with the comment
"TODO: when implement backward, fix this" (`vendor/llama.cpp/ggml/src/ggml.c:4407-4422`,
TODO at `:4412-4413`). Implementing the VJP requires following that TODO (allocate a real
dst via `ggml_dup_tensor`), otherwise the forward clobbers `x` that the backward masks read.

## What to do

All code changes land in the vendored llama.cpp fork via the S0-02 two-repo flow.

1. **TANH case** in the `GGML_OP_UNARY` switch of `ggml_compute_backward`
   (`vendor/llama.cpp/ggml/src/ggml.c`, before the default at `:6872`): emit
   `grad · (1 − y²)` from the saved output, e.g.
   `ggml_mul(grad, ggml_scale_bias(ggml_sqr(tensor), -1.0f, 1.0f))`.
2. **SIGMOID case**, same switch: `grad · (y − y²)`, e.g.
   `ggml_mul(grad, ggml_sub(tensor, ggml_sqr(tensor)))` (ROADMAP §9 E6).
3. **CLAMP:** first change `ggml_clamp` to allocate its dst (`ggml_dup_tensor`) per the
   in-tree TODO (`ggml.c:4412-4413`), keeping numerical behavior identical; then add a
   `GGML_OP_CLAMP` case emitting
   `ggml_mul(ggml_mul(grad, ggml_step(ggml_scale_bias(src0, 1.0f, -min))), ggml_step(ggml_scale_bias(src0, -1.0f, max)))`
   with `min`/`max` read from op_params (`ggml.c:4415-4416`). The gradient is exactly 0 at
   and outside the bounds (`ggml_step(0) == 0`), matching the subgradient convention
   (ROADMAP §9 E5).
4. **Tests:**
   - Extend the `grad_supported` whitelist in `test_unary`
     (`vendor/llama.cpp/tests/test-backend-ops.cpp:2008-2010`) with `GGML_UNARY_OP_TANH` and
     `GGML_UNARY_OP_SIGMOID`; the existing case sweep (`:7779-7784`) then exercises MODE_GRAD
     for both, including the non-contiguous-view variants.
   - `test_clamp` (`:4632-4655`) already carries the grad scaffolding — `grad_eps` and
     `grad_expect() = {0.0f, 1.0f}` at `:4657-4663` — but never calls `ggml_set_param`; add
     it. The `{0,1}` expected values engage the harness's expected-value filtering for the
     discontinuity at the bounds (`mean_abs_asymm`, `:319-321`), which is exactly the
     mechanism the manifest requires; existing case instantiations at `:8831, 8847-8848`
     then run under MODE_GRAD.
5. **Verify no supports_op changes are needed:** the composites emit only existing ops
   (MUL/SUB/SQR/SCALE/STEP), all present on every backend, so the new VJPs are
   backend-agnostic by construction.
6. **Submodule bump PR** in learning-llamas referencing this ticket, per S0-02.

## Out of scope

- GELU-family and `NORM` (LayerNorm) VJPs — BLUEPRINT §7 item 5 lists them; owned by B-08
  (activates on GELU-MLP / LayerNorm arch demand), not this ticket.
- Fused GLU-variant backward (SWIGLU_OAI, GEGLU, REGLU) — S1-28.
- Flipping gemma2/3 to trainable in the preflight report — S1-11 owns the preflight; it picks
  the new ops up automatically from the supported-backward walk.
- Simplifying the GRPO clip graph to use CLAMP — S1-16 owns the trainer graph.
- Consuming TANH in the FA8 softcap path (S1-24) and SIGMOID in MoE router tickets (S1-25+).

## Acceptance criteria

- [ ] Fork branch: `test-backend-ops` mode `grad` passes on CPU for TANH and SIGMOID
      `test_unary` cases (all generated shapes/views) within the ADR-0002 per-op tolerance.
- [ ] Fork branch: `test-backend-ops` mode `grad` passes on CPU for `test_clamp` cases, with
      the discontinuity handled via the `{0,1}` expected-value filter.
- [ ] All existing `test_clamp` forward (eval) cases still pass after the dup-tensor change,
      on every backend the CI matrix covers at this stage.
- [ ] No new op enums and no backend kernel files in the diff (grep-verifiable): the VJPs are
      graph-level composites only.
- [ ] learning-llamas submodule-bump PR is green in `ci-cpu` (per-PR).

## Testing & verification

Primary harness: vendored `tests/test-backend-ops` in MODE_GRAD (finite differences vs the
analytic composite, CPU oracle, ADR-0002 tolerance from S0-09) plus eval mode for the
`ggml_clamp` dup-tensor regression. Runs on the fork branch CI and in learning-llamas's `ci-cpu`
lane per-PR after the submodule bump; nightly `ci-cpu` re-runs the full suite. Because the
composites emit only pre-existing ops, GPU backends inherit them with no per-backend work —
the stage-2/3/4 lanes simply keep running the same MODE_GRAD cases.

## PR notes

- Branch: `ticket/S1-19-small-vjps-tanh-sigmoid-clamp`.
- Two-repo flow per S0-02: fork PR (`learning-llamas-base`) + trivial learning-llamas submodule-bump PR,
  both referencing the ticket ID.
- Upstreaming disposition: **upstream-early** (ROADMAP §11 triage class a) — each case is
  ~10 lines following existing patterns, `ggml_clamp`'s own TODO invites the dst fix, and the
  tests ride the existing harness.
- Derivative formulas are standard calculus; no external code is copied (unsloth is not
  involved in this ticket), so no provenance headers are needed.
