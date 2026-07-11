---
id: S1-20
title: "K-SMB: SOFT_MAX_BACK max_bias>0 — add the missing test, lift the CPU assert"
stage: 1
track: kernels
size: S
deps: ["S0-02"]
status: open
pr: null
---

# S1-20 — K-SMB: SOFT_MAX_BACK max_bias>0 — add the missing test, lift the CPU assert

**One-line outcome:** ALiBi models are trainable on CPU: a finite-difference test with
`max_bias > 0` runs FIRST (none executes today anywhere), then the CPU assert and supports_op
gate are removed; CUDA/Vulkan gates are explicitly left to their own stage tickets.

## Why (context)

BLUEPRINT G7/§8 lists "No ALiBi models" as a v1 hard constraint because `SOFT_MAX_BACK`
requires `max_bias == 0`. The restriction is gating, not math (ROADMAP §3 K-SMB): the forward
computes `y = softmax(scale·x + slope(h)·mask)` where the ALiBi term is additive and constant
with respect to the logits, so the backward `dx = scale · y ∘ (dy − dot(y,dy))` never involves
`max_bias` at all. Verified in the CPU kernel: the body computes exactly that formula using
only `scale` (`vendor/llama.cpp/ggml/src/ggml-cpu/ops.cpp:5588-5593`), and the only blocker is
`GGML_ASSERT(max_bias == 0.0f)` at `:5539`. The autograd side already passes `max_bias`
through: the `SOFT_MAX` backward case forwards it to `ggml_soft_max_ext_back`
(`vendor/llama.cpp/ggml/src/ggml.c:6762-6773`, emission at `:6770`), so no `ggml.c` change is
needed.

Authoring-time verification found one more CPU gate beyond the ROADMAP's "two backend asserts"
accounting: the CPU `supports_op` also rejects `max_bias != 0`
(`vendor/llama.cpp/ggml/src/ggml-cpu/ggml-cpu.cpp:454-462`). That gate is why no test exercises
this today: grad-capable `test_soft_max` cases with `max_bias = 8.0` are already generated
(loop at `vendor/llama.cpp/tests/test-backend-ops.cpp:8882`, `ggml_set_param` at `:4852`), but
`eval_grad` re-checks `supports_op` on every backward-graph node after
`ggml_build_backward_expand` (`:1770` and the loop following it), sees the gated
`SOFT_MAX_BACK`, and reports NOT_SUPPORTED. Likewise the forward-eval `test_soft_max_back`
cases with `max_bias = 8.0` (loop at `:8925`) can never compare against CPU.

The cross-backend picture strengthens test-first sequencing: CUDA has both a runtime assert
(`vendor/llama.cpp/ggml/src/ggml-cuda/softmax.cu:469`) and a supports_op gate
(`vendor/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:4903-4907`), while **Vulkan accepts
`max_bias > 0` completely unvalidated** — its supports_op checks only contiguity and types
(`vendor/llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp:17580-17582`) and its shader reads only
`scale = p.param1`, ignoring the `max_bias` value its dispatch passes
(`vulkan-shaders/soft_max_back.comp`; dispatch at `ggml-vulkan.cpp:12798`). Once CPU can run
these cases, that Vulkan inconsistency becomes *detectable* by ordinary backend-vs-CPU
comparison instead of silently skipped.

## What to do

All code changes land in the vendored llama.cpp fork via the S0-02 two-repo flow.

1. **Record the math check** (per ROADMAP §3 K-SMB) in the fork PR description: the additive
   ALiBi bias is constant w.r.t. logits, so existing `SOFT_MAX_BACK` kernels are already
   mathematically correct for `max_bias > 0`; cite the CPU kernel body (`ops.cpp:5588-5593`).
2. **Test first.** On the fork branch, run `test-backend-ops grad -o SOFT_MAX` and capture the
   output showing the `max_bias = 8.0` cases reported NOT_SUPPORTED on CPU (this documents the
   "no test exercises this today" state in the PR). Audit the generated grad coverage: the
   `:8882` loop includes masked cases at `max_bias = 8.0` with single-head and multi-head
   (`nr23 = {3,1}`) shapes, so per-head slope variation is exercised; add a case only if that
   audit finds a hole (e.g. F16-mask + multi-head + `max_bias > 0` combined).
3. **Lift the CPU gates:** remove the assert at
   `vendor/llama.cpp/ggml/src/ggml-cpu/ops.cpp:5539` and the `max_bias == 0.0f` condition in
   `vendor/llama.cpp/ggml/src/ggml-cpu/ggml-cpu.cpp:454-462` (keep the F32 type checks).
4. **Re-run and confirm:** the previously skipped MODE_GRAD cases now execute and pass on CPU;
   the forward `test_soft_max_back` `max_bias` cases (`:8925-8931`) now have a CPU baseline.
5. **Leave breadcrumbs for later stages** (code comment in the fork + this ticket): the CUDA
   assert (`softmax.cu:469`) and gate (`ggml-cuda.cu:4903-4907`) are lifted by **S3-04**; the
   Vulkan unvalidated-`max_bias` audit — either validating the shader or adding the missing
   gate — is owned by **S4-05** (ROADMAP §7 V5). Do not touch CUDA/Vulkan code here; until
   their tickets land, `ggml_backend_sched` falls back to CPU for ALiBi backward nodes, which
   is correct-but-slow by design (ROADMAP §11 scheduler note).
6. **Submodule bump PR** in llama-farm referencing this ticket, per S0-02. The S1-11
   trainability preflight picks up the widened op support automatically via its graph walk;
   no llama-farm code change is required.

## Out of scope

- CUDA gate lift — S3-04 (no kernel change needed there either, per ROADMAP §5 C4).
- Vulkan `max_bias` validation/audit — S4-05.
- Metal `SOFT_MAX_BACK` (does not exist yet at all) — stage-2 backward-suite work.
- The FA-training path — FA backward recomputes P from Q/K/mask/LSE directly, so this
  restriction never applied there (ROADMAP §8, "what already exists").

## Acceptance criteria

- [ ] Fork PR description contains the recorded math check and the captured before/after test
      output (NOT_SUPPORTED → passing) for the `max_bias > 0` MODE_GRAD cases.
- [ ] Fork branch: `test-backend-ops` mode `grad` passes on CPU for every `SOFT_MAX` case with
      `max_bias > 0` (finite differences vs analytic, ADR-0002 per-op tolerance), including a
      multi-head case where per-head ALiBi slopes differ.
- [ ] Fork branch: forward-eval `test_soft_max_back` cases with `max_bias > 0` execute on CPU
      (no longer skipped).
- [ ] The fork diff touches no CUDA or Vulkan source (grep-verifiable), and contains the
      breadcrumb comments pointing at S3-04/S4-05.
- [ ] llama-farm submodule-bump PR is green in `ci-cpu` (per-PR).

## Testing & verification

Primary harness: vendored `tests/test-backend-ops` — MODE_GRAD for the gradient checks (CPU
oracle, ADR-0002 tolerances from S0-09) and eval mode for `test_soft_max_back` forward parity.
Runs on the fork branch CI, then in llama-farm's `ci-cpu` lane per-PR after the submodule
bump; nightly `ci-cpu` re-runs the full suite. When stage-3/4 tickets lift their gates, these
same cases become the cross-backend parity tests against this CPU baseline (max-abs gradient
error ≤ 0.05 @ fp16 per ADR-0002).

## PR notes

- Branch: `ticket/S1-20-soft-max-back-alibi-max-bias`.
- Two-repo flow per S0-02: fork PR (`llama-farm-base`) + trivial llama-farm submodule-bump PR,
  both referencing the ticket ID.
- Upstreaming disposition: **upstream-early** (ROADMAP §11 triage class a) — the enabled test
  plus assert/gate removal benefit mainline directly and carry no new ABI. Mention the Vulkan
  unvalidated-`max_bias` finding in the upstream PR so mainline can triage it independently.
- No copied external code; no provenance headers needed.
