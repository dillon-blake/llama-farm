---
id: S1-34
title: "SOFT_MAX_BACK ignores attention sinks — the gradient is wrong for sink models"
stage: 1
track: kernels
size: M
deps: ["S0-10", "S0-09"]
status: open
pr: null
---

# S1-34 — `SOFT_MAX_BACK` ignores attention sinks

**One-line outcome:** softmax-with-sinks has a correct backward, and `test-backend-ops grad -o
SOFT_MAX` passes instead of failing every `sinks=1` case.

## Why (context)

Found by running MODE_GRAD once S0-10 made it runnable. `test-backend-ops grad -o SOFT_MAX`
reports `Backend CPU: FAIL`, and **every failing case has `sinks=1`**:

```
SUM(type=f32,ne=[16,16,1,1],mask=0,sinks=1,...): [SUM] MAA = 0.000531925 > 0.000100000  FAIL
SUM(type=f32,ne=[15,15,1,1],mask=0,sinks=1,...): [SUM] MAA = 0.000905727 > 0.000100000  FAIL
```

Not a single failure has `sinks=0`. That is not a tolerance problem; it is a missing term.

The cause is visible in the signatures. `ggml_soft_max_ext_back` takes **no sinks argument**
(`ggml/include/ggml.h:1755-1760`):

```c
    GGML_API struct ggml_tensor * ggml_soft_max_ext_back(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b,
            float                 scale,
            float                 max_bias);
```

and the backward rule never reads the sinks operand, `src[2]` (`ggml/src/ggml.c:6762-6772`):

```c
        case GGML_OP_SOFT_MAX: {
            if (src0_needs_grads) {
                float scale    = 1.0f;
                float max_bias = 0.0f;
                memcpy(&scale,    (const float *) tensor->op_params + 0, sizeof(float));
                memcpy(&max_bias, (const float *) tensor->op_params + 1, sizeof(float));
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_soft_max_ext_back(ctx, grad, tensor, scale, max_bias));
            }
            GGML_ASSERT((!src1 || !src1_needs_grads) && "backward pass for softmax mask not implemented");
        } break;
```

So the sink logits are simply absent from the backward computation.

**Why this matters.** Attention sinks are how gpt-oss-class models work, and the FA family
carries sinks through `ggml_flash_attn_ext` too (ROADMAP §12 item 4 already flags "LSE-with-sinks
definition must match across backends"). A model with sinks trained through this path gets a
**silently wrong gradient** — the worst failure mode there is, because training still *runs* and
the loss still *goes down*, just to the wrong place.

There is a subtlety worth stating, because it is the thing to get right: sinks are constants with
respect to the logits, so the softmax Jacobian keeps its familiar form
`∂y_i/∂x_j = y_i(δ_ij − y_j)`. What changes is that **`y` no longer sums to 1** — the sink absorbs
part of the mass. Any backward kernel that implicitly assumes `Σ y = 1` (for instance by
reconstructing the normalizer, or by folding the `Σ_j dy_j y_j` correction under that assumption)
is wrong precisely by the sink's share. That is consistent with the observed error growing with
the sink's weight, and it is the first thing to check in the CPU kernel.

## What to do

Two-repo flow (ADR-0001 §5).

1. Establish the ground truth first: derive the gradient of softmax-with-sinks analytically and
   write it in the PR. Do not start from the kernel.
2. Decide the ABI: either extend `ggml_soft_max_ext_back` with a sinks operand, or add a
   `ggml_soft_max_ext_back_sinks`. Extending is preferable (one op, one kernel, `NULL` sinks =
   today's behavior) but it touches every backend's `SOFT_MAX_BACK`; record the choice.
3. If sinks are themselves trainable, they need a gradient too — `src[2]` currently has no rule.
   Decide explicitly whether they are trainable in this project (LoRA does not train them; full
   fine-tuning would) and either implement `grad(sinks)` or assert loudly that it is unsupported,
   rather than silently returning nothing.
4. Fix the CPU kernel (the oracle). GPU ports follow in the backend stages.
5. Make `test_soft_max` grad-check the sinks cases and pass.
6. Coordinate with **S1-20**, which also touches `SOFT_MAX_BACK` (the `max_bias`/ALiBi lift). Land
   whichever is first and rebase the other; do not develop them in parallel against the same
   kernel.

## Out of scope

- Flash-attention's own sink handling (S1-21/S1-23 — same mathematics, different kernel).
- The `max_bias > 0` / ALiBi restriction — S1-20.
- GPU ports — the backend stages.

## Acceptance criteria

- [ ] `test-backend-ops grad -o SOFT_MAX` reports `Backend CPU: OK`, with the `sinks=1` cases
      **grad-checked** (not skipped — state the case count in the PR).
- [ ] The derivation of the sinks gradient is written out in the PR, and the kernel matches it.
- [ ] `sinks=0` behavior is bit-identical to before (no regression on the common path).
- [ ] The decision on trainable sinks (implement `grad(sinks)` vs assert) is recorded and
      enforced in code — never silently absent.
- [ ] `docs/dev/backward-coverage.md` moves `SOFT_MAX` out of the FAIL table, and the `ci-cpu`
      grad allowlist gains `SOFT_MAX`.

## Testing & verification

`test-backend-ops grad -o SOFT_MAX` on the `ci-cpu` lane. The per-op finite-difference bound is
ADR-0002's `max_maa_err()`; if the sinks cases need an override, justify it — the current failures
are 5-9× over the bound, which is a wrong term, not a precision limit, and an override would
merely hide it.

## PR notes

- Branch: `ticket/S1-34-soft-max-back-sinks-gradient`.
- **Two-repo flow**: the real PR against the fork's `learning-llamas-base`, then a submodule bump.
- Upstreaming disposition: **upstream-early**. This is a correctness bug in ggml's autodiff that
  affects anyone training a sink model, with no learning-llamas-specific semantics.
