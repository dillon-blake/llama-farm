---
id: S1-41
title: "MoE gradient oracle — which caught SOFT_MAX_BACK aliasing the router softmax"
stage: 1
track: python
size: M
deps: [S1-25, S1-26, S1-27, S1-28, S1-38]
status: done
pr: null
---

# S1-41 — MoE gradient oracle, and the live bug it caught

**One-line outcome:** the composed MoE gradient (router softmax → top-k weights → renormalize →
expert gather → 3D LoRA) is verified per tensor against a self-audited float64 reference — and
doing so exposed and fixed a real kernel bug that had silently zeroed the router gradient.

## The bug (fork commit `41141dd4f`)

`ggml_compute_forward_soft_max_ext_back_f32` computed its output with a vector sequence that
overwrites `dst` before its last read of `src1` — not in-place-safe. `SOFT_MAX_BACK` is on
`ggml_op_can_inplace`'s list, and gallocr aliases `dst` onto `src1` (the softmax output `y`)
exactly when `soft_max_ext_back` is `y`'s sole gradient-time consumer. In **attention** that never
happens (`y` also feeds the V-matmul backward). In a **Mixtral router** it always does (`y` feeds
only argsort/get_rows, which don't read it at backward time). Result: `d_logits ≈ 0`, the router
term vanished from `dh`, and every LoRA upstream of an MoE block trained on a gradient wrong by
~5–25 % relative — with a perfectly healthy falling loss.

Why three layers of testing missed it: MODE_GRAD compares CPU-vs-CPU (both sides alias
identically); `test_moe.py` watches the loss fall (it falls); the S1-03 FD gate only ran on the
dense fixture. The fork's new `tests/test-soft-max-back-inplace.cpp` forces the alias against a
float64 reference (0.22 error pre-fix, 9.6e-9 post).

## What to do

- `tests/reference_moe.py`: float64 MoE forward/backward twinned from `build_moe_ffn`'s
  LLM_ARCH_LLAMA semantics, with a `detach_router` diagnostic mode (it is how the bug's shape was
  identified).
- `tests/test_moe_gradients.py`: the oracle self-audit (FD of its own forward), one step × all 28
  LoRA gradient comparisons at effective scale 2.0, and a 24-step `train_sft` trajectory.
- Fork: fix the kernel; add the alias-forcing regression test.

## Acceptance criteria

- [x] Oracle self-audit < 1e-5 vs FD of its own forward.
- [x] One-step: loss matches at <1e-5 rel; all 28 tensor grads < 1e-3 rel (post-fix).
- [x] 24-step MoE trajectory per-step < 1e-4 (post-fix).
- [x] Fork regression test fails on the old kernel, passes on the new one.
- [x] Dense path untouched (`test_convergence.py`, `test_p0_gradient.py` green).
