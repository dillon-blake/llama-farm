---
id: S1-29b
title: "SSM backward ops: enums, constructors, and the backward-switch wiring S1-29 never landed"
stage: 1
track: kernels
size: S
deps: ["S1-25"]
status: pr-open
pr: https://github.com/dillon-blake/llama.cpp/pull/23
---

# S1-29b — the SSM backward wiring S1-29 never landed

**One-line outcome:** `SSM_CONV` and `SSM_SCAN` have backward-switch cases that emit
`SSM_CONV_BACK` / `SSM_SCAN_BACK`, so a Mamba graph's backward *builds* instead of aborting — and
S1-30 and S1-31 have a dependency that actually exists.

## Why (context)

**This ticket exists because S1-29 did not do what its own ticket said, and both SSM tickets are
written as if it had.**

S1-29 is marked `pr-open` (PR #27) and its ticket lists six work items. Fork commit `5608bb9aa`
touches exactly **two files** — `ggml/src/ggml.c` (+32) and `tests/test-backend-ops.cpp` (+47) — and
delivers **only the CONCAT VJP**. Items 2, 3, 4 and 6 were never landed:

- no `GGML_OP_SSM_CONV_BACK` / `GGML_OP_SSM_SCAN_BACK` enums
- no `ggml_ssm_conv_back` / `ggml_ssm_scan_back` constructors
- no `SSM_CONV` / `SSM_SCAN` cases in `ggml_compute_backward`
- no mamba backward-build test

Verified: `SSM_CONV_BACK` appears **zero times** in `ggml/include/ggml.h`.

So S1-30 opens with *"S1-29 wired the `SSM_CONV` backward-switch case to emit `SSM_CONV_BACK`, but
the op has no compute kernel"* — and that is false in its first clause. S1-31 makes the same
assumption. Whoever picks either up discovers a missing dependency on day one.

## What to do

Mirror S1-25 exactly — it did this same job for the MoE ops and the traps are identical.

1. **Enums at the TAIL** of `enum ggml_op` (`ggml/include/ggml.h`), after the S1-25 entries.
   Inserting in the middle renumbers every op after it and conflicts across the whole backend
   matrix on every rebase.
2. **The two name/symbol tables** in `ggml.c`, and **both** `static_assert(GGML_OP_COUNT == N)`.
3. **Constructors** `ggml_ssm_conv_back` and `ggml_ssm_scan_back`, with shape asserts mirroring
   `ggml_ssm_conv` / `ggml_ssm_scan`.
4. **Backward cases** in `ggml_compute_backward` emitting them.
5. **CPU `supports_op` must `return false` explicitly** for both. This is the trap S1-25 hit: the
   switch ends in `default: return true`, so a new op with no dispatch case is reported *supported*,
   gets scheduled, and then hits `ggml_compute_forward`'s `default: GGML_ABORT`. It looks
   implemented right up until it kills the process. S1-30/S1-31 flip these on.
6. **A graph-build test**: a small Mamba-shaped subgraph whose backward now *builds* without
   aborting.

## The `SSM_SCAN` packed-grad decision, which must be made here

`ggml_ssm_scan`'s dst is a **packed 1-D tensor**: `y` concatenated with the final states. So
MODE_GRAD's `sum(out)` objective includes the **state region**, which makes
`dL/d(s_final) = 1`, not 0.

S1-31's design says to take the `dy` view (the first `ggml_nelements(x)` elements) and assume the
state-grad region is zero. **Under `test_ssm_scan` that assumption is false**, and the finite
difference will include `d(sum s_final)/dx` while the analytic backward omits it — so the case
fails, and the implementer's instinct will be to "fix" the test rather than the kernel.

**Decide it here:** `SSM_SCAN_BACK` should take the **whole packed grad** and *seed* its working
state-gradient from the state region at `t = n_t`, rather than zeroing it. That is a few lines, it
is correct under both the test's objective and training's (where the region genuinely is zero — the
cache `ggml_cpy` feeds nothing downstream of the loss), and it makes cross-ubatch BPTT nearly free
later. Do **not** silently discard the state-grad region.

## Out of scope

- The CPU kernels themselves — `SSM_CONV_BACK` is S1-30, `SSM_SCAN_BACK` is S1-31.
- GPU ports.
- Cross-ubatch BPTT through the recurrent state.

## Acceptance criteria

- [ ] A Mamba-shaped backward **builds** rather than aborting.
- [ ] Both enums sit at the tail; all existing `test-backend-ops` eval cases still pass.
- [ ] CPU `supports_op` returns **false** for both, and the grad cases report *not-supported*
      rather than aborting on every backend.
- [ ] `test_ssm_conv` and `test_ssm_scan` call `ggml_set_param` — **they call it zero times today**,
      which is why `grad -o SSM_CONV` already prints `OK` with no backward in existence. Their
      default shapes are also over `grad_nmax()` (10000) and would be *silently skipped* even with
      a param, so add tiny shapes. See `docs/dev/backward-coverage.md`.

## PR notes

- Branch: `ticket/S1-29b-ssm-back-ops`.
- Two-repo flow: fork PR + submodule bump.
- Upstreaming disposition: **fork-local** for now; propose upstream with the S1-30/S1-31 kernels
  once the CPU oracle proves the design.
