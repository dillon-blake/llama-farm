---
id: S1-28
title: "MoE: GLU-family backward (SWIGLU_OAI, GEGLU exact/tanh, REGLU) + tiny-MoE e2e"
stage: 1
track: kernels
size: M
deps: ["S1-26", "S1-27", "S1-19"]
status: pr-open
pr: https://github.com/dillon-blake/llama.cpp/pull/22
---

# S1-28 — MoE: GLU-family backward (SWIGLU_OAI, GEGLU exact/tanh, REGLU) + tiny-MoE e2e

> **⚠️ A blocker this ticket does not mention: the GELU F16 lookup table.**
>
> `GGML_GELU_FP16` (`ggml/src/ggml-cpu/vec.h`) makes the **F32** GEGLU forward compute gelu by
> quantizing x to F16 and reading `ggml_table_gelu_f16`. MODE_GRAD differentiates the *forward*, so
> the finite difference would differentiate an F16-precision step function while `GLU_BACK` computes
> the exact F32 derivative. Estimated relative asymmetry **1e-2 … 1e-3, against a `max_maa_err` of
> 1e-4 — 10-100x over.** GEGLU MODE_GRAD will **not** pass as this ticket assumes.
>
> Only GEGLU (tanh) and GEGLU_QUICK are affected: `ggml_silu_f32` is exact (which is why split-SwiGLU
> passes today) and `geglu_erf` uses `erff` directly. The preferred fix is to route the F32 GEGLU
> forward to the exact `ggml_gelu_f32` — the non-table branch already exists. This is the same class
> of finding as the `EXPM1` bug: an F16-precision approximation sitting on a training path. **Measure
> it first, before planning around an assumption.**
>
> Note `test_glu_split` and `test_swiglu_oai` **already** call `ggml_set_param` — only fused
> `test_glu` does not. Step 5 is half done. Also narrow the `[-150, 150]` init for grad runs, or the
> FD sits in the saturated tail rather than the curved region.
>
> **The preflight GLU over-claim is already fixed** (S1-25, `csrc/farm_preflight.cpp`): it used to
> report *every* GLU trainable. Flip `glu_has_backward()` on as each VJP lands.

**One-line outcome:** backward coverage for the gated-MLP variants MoE architectures
use — SWIGLU_OAI (GPT-OSS), GEGLU exact and tanh-approx (Gemma-family), REGLU — closing
the last CPU MoE gap, proven by a tiny-MoE SFT run whose loss falls on CPU.

## Why (context)

With S1-25/26/27 the expert matmuls have gradients, but the expert MLP's activation
still aborts: the `GGML_OP_GLU` case in `ggml_compute_backward` covers only split
SWIGLU via `SILU_BACK` and hits its default abort for every other variant
(`vendor/llama.cpp/ggml/src/ggml.c:6885-6900`, abort at `:6896-6898`). The variants MoE
archs actually emit: GPT-OSS experts use `ggml_swiglu_oai`
(`vendor/llama.cpp/src/llama-graph.cpp:2069-2075`), GELU-gated MoEs use
`ggml_geglu_split` (`:2063`), and REGLU MoEs use `ggml_reglu_split` (`:2079-2080`); the
dense FFN builder additionally emits the fused single-tensor forms
(`ggml_geglu`/`ggml_reglu`, `:1712-1720`), so both layouts need covering (ROADMAP §9 E7).

The derivative math for GEGLU is imported — math only, with a provenance header — from
unsloth's Apache-2.0 kernels (ROADMAP §13 item 4): the exact (erf) backward,
`df/de = Φ(e) + e·φ(e)` with `Φ(e) = ½(1+erf(e/√2))`, is worked out at
`unsloth/kernels/geglu.py:75-123`, and the tanh-approx backward — reusing
`T = 1 + tanh(u)` with `u = √(2/π)·e·(1+0.044715·e²)` so `sech² = 1 − tanh²` needs no
second transcendental — at `geglu.py:188-244` (Apache-2.0 header verified, `geglu.py:1-13`).
The mapping onto ggml ops: `GGML_GLU_OP_GEGLU` uses the tanh-approx GELU
(`ggml_gelu_f32`, `vendor/llama.cpp/ggml/src/ggml-cpu/vec.h:968-970`) and
`GGML_GLU_OP_GEGLU_ERF` the exact form (`ggml_vec_gelu_erf_f32`, `vec.h:1010-1015`), so
both unsloth formulas are needed. SWIGLU_OAI's derivative follows from its CPU forward
(`vendor/llama.cpp/ggml/src/ggml-cpu/ops.cpp:3325-3391`):
`out = x·σ(αx)·(y+1)` with `x = min(e, limit)`, `y = clamp(u, −limit, limit)`, giving
`∂out/∂e = [e<limit]·σ(αx)·(1+αx(1−σ(αx)))·(y+1)` and `∂out/∂u = [−limit<u<limit]·x·σ(αx)`
(zero subgradient at the bounds, same convention as S1-19's CLAMP). REGLU is trivial:
`∂/∂e = grad·g·step(e)`, `∂/∂g = grad·relu(e)`.

Because exact GEGLU needs `erf`/gaussian evaluations that exist as no ggml graph
primitive, the vehicle is a fork-local op `GGML_OP_GLU_BACK` (anticipated by ROADMAP
§11 triage b, which lists `GLU_BACK` among the in-fork-first op enums) rather than
per-variant composites. This is the *correctness* op; the K5 perf item — the fused
in-place three-buffer GLU backward of ROADMAP §13 item 3 — is explicitly not this
ticket. S1-19 is a dependency because the tiny-MoE e2e needs its VJPs: sigmoid routers
(E6) and the `norm_w` weight-normalization clamp
(`vendor/llama.cpp/src/llama-graph.cpp:1947`) sit on the router gradient path.

## What to do

All code lands in the vendored llama.cpp fork via the S0-02 two-repo flow.

1. **New fork-local op `GGML_OP_GLU_BACK`** (enum tail per ROADMAP §11c) with
   constructor `ggml_glu_back(ctx, grad, a, b_or_null)` mirroring `ggml_glu_impl`'s
   operand/op-param layout (glu op at i32[0], swapped at i32[1],
   `vendor/llama.cpp/ggml/src/ggml.c:2885-2911`; SWIGLU_OAI's alpha/limit at
   f32[2]/f32[3], `ggml.c:3089-3100`). dst always has the fused `a` shape
   (`[2·nc, …]`): for fused inputs it *is* d_a (both halves, honoring `swapped`); for
   split inputs the caller views half 0 as d_a and half 1 as d_b — packed-dst-plus-views
   precedent per FA1 (S1-21).
2. **CPU kernel** `ggml_compute_forward_glu_back` in
   `vendor/llama.cpp/ggml/src/ggml-cpu/ops.cpp` next to the GLU forwards, reusing their
   row-pointer/swapped-offset structure (`ops.cpp:3325-3391` shows the pattern incl.
   op-param reads): one pass per row computing `de` and `dg` per the formulas above for
   GEGLU (tanh, `geglu.py:188-244`), GEGLU_ERF (exact, `geglu.py:75-123`), SWIGLU_OAI,
   and REGLU. F32 math throughout (ADR-0002); deterministic row-parallel split, no
   atomics.
3. **Backward cases** in the `GGML_OP_GLU` switch of `ggml_compute_backward`
   (`ggml.c:6885-6900`) for `GGML_GLU_OP_REGLU`, `GEGLU`, `GEGLU_ERF`, and
   `SWIGLU_OAI`: emit `ggml_glu_back` and route the dst (or its two half-views, split
   case) into src grads via `ggml_add_or_set`. Leave the existing split-SWIGLU
   `SILU_BACK` composite untouched — the dense split-SWIGLU path already works through
   it (ROADMAP §2) and must not regress; note the coordination in a comment.
4. **Plumbing:** CPU dispatch/n_tasks; CPU `supports_op` = true for `GGML_OP_GLU_BACK`;
   GPU backends return false (CPU fallback via sched until backend-stage ports).
5. **MODE_GRAD tests:** add grad support to `test_glu`
   (`vendor/llama.cpp/tests/test-backend-ops.cpp:2078`), `test_glu_split` (`:2124`),
   and `test_swiglu_oai` (`:2182`); the existing sweeps (`:7811-7823`, `:7831`) then
   cover all variants including non-contiguous views and `swapped` on/off. The classes
   initialize uniformly in [−150, 150] (`:2119`, `:2177`, `:2237`) — for grad runs use a
   narrower init or per-case `grad_eps` so finite differences stay stable, and
   expected-value filtering for the SWIGLU_OAI/REGLU discontinuities (harness support
   per S1-19). `GEGLU_QUICK` remains uncovered — assert-listed as unsupported, not
   silently wrong.
6. **Tiny-MoE e2e (learning-llamas side, after the submodule bump):** add a 2-4 expert
   tiny-MoE GGUF fixture (gguf-py, S0-05/S0-06 machinery) with a `build_moe_ffn`-style
   graph exercising SWIGLU_OAI or GEGLU experts; run SFT (S1-05 trainer) with LoRA on
   expert projections for a fixed step budget and assert train loss falls (e.g.
   `tests/test_moe_e2e.py::test_tiny_moe_sft_loss_falls`). Verify the S1-11 preflight
   report now classifies the MoE fixture arch as trainable — no preflight code changes
   should be needed (graph-walk picks up the new backward coverage automatically); if
   its supported-op table is hardcoded anywhere, update it.
7. **Submodule bump PR** in learning-llamas per S0-02.

## Out of scope

- The fused in-place three-buffer GLU backward perf op (recomputed `h`, in-place df/de
  over three buffers) — K5, ROADMAP §13 item 3.
- `GEGLU_QUICK` backward and GELU-family unary VJPs (GPT-2/Phi/BERT-style dense archs,
  BLUEPRINT §7 item 5) — owned by B-08.
- GPU ports of `GLU_BACK` — backend stages.
- MoE convergence parity vs a PEFT reference — S1-12 owns the dense gate; extending the
  gate to MoE is a later integration ticket. This ticket's e2e asserts loss-falls only.

## Acceptance criteria

- [ ] Fork branch: `test-backend-ops` mode `grad` passes on CPU for REGLU, GEGLU,
      GEGLU_ERF (fused + split + swapped + view variants) and SWIGLU_OAI within the
      ADR-0002 tolerance (discontinuity cases via expected-value filtering).
- [ ] Existing split-SWIGLU grad cases still pass unchanged (no regression of the
      SILU_BACK path).
- [ ] Determinism: bitwise-identical GLU_BACK output across different `n_threads`.
- [ ] learning-llamas: `tests/test_moe_e2e.py::test_tiny_moe_sft_loss_falls` passes in
      `ci-cpu` — final train loss below initial loss by the documented margin on the
      tiny-MoE fixture.
- [ ] S1-11 preflight report marks the tiny-MoE fixture arch trainable (report artifact
      asserted in the same test module).
- [ ] learning-llamas submodule-bump PR is green in `ci-cpu` (per-PR).

## Testing & verification

Primary harness: vendored `tests/test-backend-ops` MODE_GRAD (finite differences, CPU
oracle, ADR-0002 tolerance from S0-09) for the per-variant kernel checks, on the fork
branch CI and in learning-llamas's `ci-cpu` lane per-PR after the submodule bump. The
tiny-MoE e2e runs in learning-llamas's `ci-cpu` per-PR (small fixture, bounded step budget);
nightly `ci-cpu` re-runs the full grad suite plus the e2e. Backend stages later re-run
the same MODE_GRAD cases against this CPU reference when porting GLU_BACK.

## PR notes

- Branch: `ticket/S1-28-glu-backward-tiny-moe-e2e`.
- Two-repo flow per S0-02: fork PR (`learning-llamas-base`) + learning-llamas PR carrying the
  submodule bump and the e2e test, both referencing the ticket ID.
- Upstreaming disposition: **fork-local** initially — `GLU_BACK` is a new op enum;
  upstream later as an RFC once the CPU oracle (and ideally one GPU port) proves it
  (ROADMAP §11 triage b).
- Provenance (S0-01 policy): the GLU_BACK kernel file carries a header naming
  `unsloth/kernels/geglu.py` (Apache-2.0) as the source of the GEGLU derivative
  formulas — math imported, no code translated; SWIGLU_OAI/REGLU derivatives are
  standard calculus from the in-tree forward.
