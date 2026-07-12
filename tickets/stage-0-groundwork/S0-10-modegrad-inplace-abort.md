---
id: S0-10
title: "test-backend-ops MODE_GRAD aborts on inplace ops — skip them so the full grad sweep can run"
stage: 0
track: kernels
size: S
deps: ["S0-02", "S0-07"]
status: pr-open
pr: https://github.com/dillon-blake/llama-farm/pull/10
---

# S0-10 — MODE_GRAD aborts on inplace ops; make the `grad` sweep runnable

**One-line outcome:** `test-backend-ops grad` no longer hard-aborts on inplace ops, so MODE_GRAD
can actually be run and its real coverage measured — which turns out to be far thinner than the
plan assumed, and is now published in `docs/dev/backward-coverage.md`.

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
3. Open the fork PR against `learning-llamas-base` titled `[S0-10] …`.
4. **Now that MODE_GRAD runs, measure what it actually covers** — run it op by op, and audit
   which `test_case` classes call `ggml_set_param` (a case that does not is vacuous). Publish
   `docs/dev/backward-coverage.md`.
5. In learning-llamas: bump the `vendor/llama.cpp` gitlink and set the `ci-cpu` grad subset to
   the ops that are *both* genuinely grad-checked *and* green.

## Out of scope

- Making ggml's autodiff actually support inplace ops (it should not — see above).
- S1-00's KV-cache bypass. Same assert, different cause, its own ticket.
- Any change to the ops themselves.

## What the fix does *not* achieve — and why that is correct

The obvious acceptance criterion — "`test-backend-ops grad` runs unfiltered" — is **not**
achievable in this ticket, and it took running the fixed harness to find out why.

With the inplace cases skipped, the sweep gets much further and then aborts somewhere else
entirely:

```
ggml.c:6897: unsupported glu op for backward pass: REGLU
```

That is not a harness bug. It is `ggml_compute_backward` correctly reporting that **REGLU has no
backward rule** — which is precisely the gap **S1-28** exists to close. Several ops are in the
same position (see the coverage table this ticket produces).

So the honest picture, which the plan did not previously state:

> **`test-backend-ops grad` cannot pass unfiltered at the pinned commit, and will not until the
> stage-1 backward-rule tickets land.** MODE_GRAD is not a working baseline that stage 1 extends
> — it is a baseline that stage 1 *creates*.

Two different failure classes were being conflated:

| Class | Cause | Owner |
|---|---|---|
| **Harness bug** — asks for the gradient of an op that is *inherently* not differentiable | inplace ops | **this ticket** |
| **Missing product** — an op has no backward rule yet | REGLU, GEGLU, … | S1-19, S1-28, and the rest of stage 1 |

This ticket fixes the first and *exposes* the second, which is exactly the right outcome: after
it, every remaining abort is a real, named piece of stage-1 work rather than noise.

**Consequence for ADR-0001.** Its decision 4 ("every vendor bump reruns the full MODE_GRAD
suite") is aspirational today. Until stage 1 completes, the bump gate runs MODE_GRAD over the
ops that *have* backward rules — an allowlist that grows, ticket by ticket, until it is the
whole table and the qualifier can be deleted. This ticket publishes the starting allowlist.

## Acceptance criteria

- [ ] `test-backend-ops grad -o SOFT_MAX` and `-o RMS_NORM` complete **without aborting**, and
      print a visible `not supported [inplace <OP> is not differentiable]` line for each inplace
      case.
- [ ] Non-inplace SOFT_MAX / RMS_NORM grad cases still **run** — they are not skipped along with
      the inplace ones.
- [ ] `ggml.c:7093` is **unchanged**. The assert is correct; only the harness was wrong to
      trip it.
- [ ] The unfiltered `grad` sweep advances past every inplace op, and its next abort is a
      *missing backward rule* (an op owned by a stage-1 ticket), not an inplace case.
- [ ] A **backward-coverage table** is published (`docs/dev/backward-coverage.md`): for every
      ggml op, whether MODE_GRAD currently passes, fails, or aborts for want of a backward rule.
      This is the empirical version of ROADMAP §2's coverage matrix, and it is what the growing
      vendor-bump allowlist is derived from.
- [ ] `ci-cpu`'s grad subset contains only ops that are *both* genuinely grad-checked *and*
      green. (`SOFT_MAX` does **not** qualify: it now runs, and genuinely **fails** — see S1-34.)
- [ ] ADR-0002 records, normatively, that a MODE_GRAD case is vacuous without `ggml_set_param`.
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
