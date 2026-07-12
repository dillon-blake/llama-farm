---
id: S0-10
title: "test-backend-ops MODE_GRAD aborts on inplace ops — skip them so the full grad sweep can run"
stage: 0
track: kernels
size: S
deps: ["S0-02", "S0-07"]
status: open
pr: null
---

# S0-10 — MODE_GRAD aborts on inplace ops; make the full `grad` sweep runnable

**One-line outcome:** `test-backend-ops grad` (unfiltered) runs to completion at the pinned
commit, so ADR-0001's "every vendor bump reruns the full MODE_GRAD suite" rule is actually
satisfiable and the `ci-cpu` nightly can run the real sweep.

## Why (context)

This ticket was not in the original plan. It was discovered by the `ci-cpu` lane (S0-07) on its
first run — which is the lane earning its keep.

**At the pinned commit, `test-backend-ops grad` cannot run unfiltered. It aborts.**

`test_rms_norm` (`vendor/llama.cpp/tests/test-backend-ops.cpp:3469`) and `test_soft_max`
(`:4869`) both build an **inplace** variant of the op when their `inplace` parameter is set:

```cpp
        if (inplace) {
            out = ggml_rms_norm_inplace(ctx, a, eps);      // :3469
        } else {
            out = ggml_rms_norm(ctx, a, eps);
        }
```

An inplace ggml op returns `ggml_view_tensor(a)` — a node whose `view_src` is non-NULL and whose
op is `RMS_NORM` / `SOFT_MAX`. Both test cases call `ggml_set_param(a)`, so that node needs a
gradient, and `ggml_build_backward_expand` hits:

```c
        // inplace operations are currently not supported
        GGML_ASSERT(!node->view_src || node->op == GGML_OP_CPY || node->op == GGML_OP_VIEW ||
            node->op == GGML_OP_RESHAPE || node->op == GGML_OP_PERMUTE || node->op == GGML_OP_TRANSPOSE);
```

(`vendor/llama.cpp/ggml/src/ggml.c:7093`.) Hard abort, `SIGABRT`, core dumped.

Reproduced on both `ubuntu-latest` and `macos-14`, and locally:

```
$ ./build/vendor-tests/bin/test-backend-ops grad -o SOFT_MAX
  SOFT_MAX(...,inplace=1): OK
ggml.c:7093: GGML_ASSERT(!node->view_src || ...) failed
Aborted (core dumped)
```

**Upstream does not notice** because its CI runs `ctest -L main`
(`vendor/llama.cpp/.github/workflows/*.yml`), which invokes `test-backend-ops` with **no
arguments** — i.e. default `test` mode. `grad` mode is never exercised in upstream CI, so the
abort is latent.

The assert itself is **correct and should stay**: an inplace op overwrites its input, so the
backward pass cannot recover the input it needs. Such an op is genuinely not differentiable in
ggml's autodiff. The bug is that the *test harness* asks for a gradient of one anyway.

This blocks two things we have already written down:

- **ADR-0001, decision 4** — "no submodule-bump PR merges without a full MODE_GRAD run
  attached". Currently impossible: the full run aborts.
- **`ci-cpu`'s nightly full sweep** (S0-07), which is pinned to a working subset with a comment
  pointing here.

It is also the *same assert* that blocks S1-00 (the KV-cache `ggml_set_rows` view). Fixing the
harness does not fix S1-00 — they are independent instances of the same rule — but it does mean
S1-00's work is not confounded by a broken baseline.

## What to do

Two-repo flow (this is a vendored-llama.cpp change; ADR-0001 §5).

1. In the fork, on `learning-llamas-base`: in `tests/test-backend-ops.cpp`, make MODE_GRAD
   **skip** any test case whose graph contains a differentiable node with `view_src != NULL` and
   an op outside the `ggml_build_backward_expand` whitelist — i.e. exactly the cases the assert
   would reject. Skip with a printed reason, in the same style as the existing
   `"skipping large tensors for speed"` message, so the skip is visible rather than silent.
   The natural place is `test_case::eval_grad`, alongside the existing skip conditions
   (`test-backend-ops.cpp:~1700-1710`).
2. Do **not** relax the assert in `ggml.c`. It is correct.
3. Confirm the full sweep now runs: `test-backend-ops grad` with no `-o` filter completes and
   reports `Backend CPU: OK`.
4. Open the fork PR against `learning-llamas-base` titled `[S0-10] …`.
5. In learning-llamas: bump the `vendor/llama.cpp` gitlink, restore the `ci-cpu` nightly to the
   **unfiltered** `test-backend-ops grad`, and widen the per-PR grad subset back to include
   `SOFT_MAX` and `RMS_NORM`.

## Out of scope

- Making ggml's autodiff actually support inplace ops (it should not — see above).
- S1-00's KV-cache bypass. Same assert, different cause, its own ticket.
- Any change to the ops themselves.

## Acceptance criteria

- [ ] `test-backend-ops grad` (no `-o` filter) completes on CPU and reports `Backend CPU: OK`.
- [ ] `test-backend-ops grad -o SOFT_MAX` and `-o RMS_NORM` complete without aborting, and print
      a visible skip line for each inplace case.
- [ ] Non-inplace SOFT_MAX / RMS_NORM grad cases still **run** (they are not skipped along with
      the inplace ones) — verified by the case count.
- [ ] `ggml.c:7093` is unchanged.
- [ ] `ci-cpu` nightly runs the unfiltered `grad` sweep; the per-PR subset includes `SOFT_MAX`
      and `RMS_NORM` again.
- [ ] The fork PR and the submodule-bump PR both carry the ticket ID.

## Testing & verification

`test-backend-ops grad` on the `ci-cpu` lane, both matrix entries. This ticket adds no pytest
tests — it repairs the kernel-acceptance harness that every stage-1-to-4 kernel ticket depends
on.

## PR notes

- Branch: `ticket/S0-10-modegrad-inplace-abort`.
- **Two-repo flow** (ADR-0001 §5): the real PR against the fork's `learning-llamas-base`, then a
  submodule-bump PR here.
- Upstreaming disposition: **upstream-early**. This is a clean bug fix to a test harness, useful
  to anyone who runs MODE_GRAD, with no learning-llamas-specific semantics. Note
  `vendor/llama.cpp/AGENTS.md`: upstream does not accept predominantly AI-generated PRs, so the
  upstream submission needs a human author.
