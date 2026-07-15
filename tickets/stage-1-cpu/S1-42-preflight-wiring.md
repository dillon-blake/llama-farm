---
id: S1-42
title: "Wire the trainability preflight into the training path + a freshness guard for the op table"
stage: 1
track: python
size: M
deps: [S1-11]
status: done
pr: null
---

# S1-42 — Preflight wiring + op-table freshness guard

**One-line outcome:** the S1-11 preflight actually runs before training now — a model with an
un-differentiable op on its gradient path is refused with a clear, op-naming error instead of a raw
`GGML_ABORT` mid-step — and the hand-maintained supported-backward op table can no longer drift from
the fork's real `ggml_compute_backward` switch without a test failing.

## Why (context)

The 2026-07-15 audit confirmed two majors against the training-core (both verified, neither refuted):

1. **The preflight was decorative.** `ll_preflight` (`csrc/farm_preflight.cpp`) and its Python
   wrapper `preflight()` (`src/learning_llamas/preflight.py`) existed, were tested on synthetic
   graphs, and were never *called* by any training entry point. There was no `Model.preflight()` and
   no reference to the preflight anywhere in `src/learning_llamas/train/`. S1-11's item 7 —
   "trainers call it before the first step and raise on `blocked`" — was unmet. So a user could
   start training an unsupported architecture and hit `GGML_ABORT` inside `ggml_compute_backward`,
   naming an op enum and nothing else, on the *first backward pass* — after load and tokenization,
   which is exactly the outcome the preflight was built to prevent.

2. **The op table had no freshness guard.** `op_has_backward()` / `unary_has_backward()` /
   `glu_has_backward()` are a hand-written mirror of the case labels of `ggml_compute_backward`.
   Nothing failed when the fork's actual switch gained or lost an op, so the table could lie silently
   in either direction: claim a backward the switch lacks (says *trainable*, then aborts — the
   dangerous direction) or deny one the switch has (blocks a model that would train fine). It was
   already lying the conservative way: S1-30/S1-31 gave `SSM_CONV`/`SSM_SCAN` backwards and nobody
   flipped the table, so the preflight still reported every Mamba as untrainable.

## What to do

1. **Wire the gate into `Trainer.__init__`** (`src/learning_llamas/train/loop.py`) — the one seam
   all three trainers flow through, since `DPOTrainer` and `GRPOTrainer` both `super().__init__()`.
   Run the walk before the first step; on a blocker raise `PreflightError`, whose message is the
   report's `summary()` plus the escape hatch.
2. **Run the gate on throwaway optimizer state** (`preflight_adapter` in `preflight.py`). The subtle
   part: the walk needs flagged trainable tensors to seed from, but it must **not** borrow the
   training optimizer context. `opt_step_custom` sizes an optimizer context from the first graph it
   sees, and the preflight's forward-only graph is not the training graph — share it and the first
   real backward aborts on `GGML_ASSERT(ggml_is_scalar(b))` in `ggml_build_backward_expand`. So the
   gate stands up its own optimizer state, walks, tears it down (`finally`), and only then does the
   trainer build the real one — which now gets to see the training graph first, as ggml-opt requires.
3. **Escape hatch:** `TrainConfig.preflight: bool = True`. `False` skips the gate — for reaching the
   raw abort deliberately (debugging), or when the op table is believed wrong. Named in the error
   message so it is discoverable.
4. **Python surface:** `Model.preflight()` (S1-11 item 7) returning the structured `Report` with a
   human-readable `summary()`. Standalone — stands up and tears down its own optimizer state — so it
   answers the question before any trainer exists. `PreflightError` is exported from the package.
5. **Reconcile the op table with the fork** (`csrc/farm_preflight.cpp`): add `GGML_OP_SSM_CONV` /
   `GGML_OP_SSM_SCAN` to `op_has_backward` (the drift the guard immediately surfaces), and drop their
   now-false `blocker_detail` "no backward yet" messages.
6. **Freshness guard** (`tests/test_preflight_freshness.py`): parse `ggml_compute_backward`'s case
   labels out of the vendored `ggml.c`, parse the preflight table's out of `farm_preflight.cpp`, and
   fail on any disagreement in either direction — modulo a small documented allowlist of
   `EMITTED_ONLY_OPS` (the backward-only ops `OUT_PROD_ID*` / `GLU_BACK` that the backward pass
   *emits* and never itself differentiates, so they are legitimately in the table but not the
   switch). Same single-registry discipline as `tests/project_ops.py`.

## Out of scope

- Building the SSM backward or a Mamba fixture, or validating SSM gradient numerics (audit open
  finding #6 — a separate decision). This ticket only makes the preflight tell the truth about
  whether the SSM backward *aborts*, which is all the preflight ever claims.
- `docs/support-tiers.md` (a separate missing S1-11 deliverable), the arch mirror generator, and the
  bypass-warning path (already implemented in the shim; untouched here).
- Auto-*generating* the op table from source (S1-11 items 5-6). A hand-maintained table with a test
  that fails on drift is the lighter mechanism the ticket permits, and it is what this delivers.

## Acceptance criteria

- [x] A model the preflight rejects fails to *start* training, raising `PreflightError` whose message
      names the offending op and node and the `preflight=False` escape hatch; with the hatch it
      trains (constructs and steps).
- [x] The gate does not perturb training: the convergence gate, its off-unit variants, and the
      thread bit-identity test all stay green with the gate on by default.
- [x] `Model.preflight()` returns a `trainable` report for the fixture, and errors clearly when no
      adapter is attached.
- [x] The freshness guard cross-checks the table against `ggml_compute_backward` and passes; a
      mutation test proves it flags drift in **both** directions (an op dropped from the table, and a
      bogus op added), and the emitted-only allowlist is a named list, not a blanket pardon.
- [x] The table reconciled: `SSM_CONV`/`SSM_SCAN` moved to the supported side; the guard confirmed it
      would have failed on the pre-fix table, flagging exactly those two.
- [x] Full suite green.

## Testing & verification

`tests/test_preflight_wiring.py` (the seam: refusal, escape hatch, context reuse, real-model
pass-through, `Model.preflight()`) and `tests/test_preflight_freshness.py` (the source cross-parse
plus its mutation test), both per-PR in `ci-cpu / test`. The freshness guard is a source parse, not a
`test-backend-ops` run: the question is purely *which case labels exist* in two switches, which the
text answers exactly and for free — whether each backward is numerically correct remains
`test-backend-ops` MODE_GRAD's job, guarded by `tests/test_backend_ops_grad.py`.

Manual: confirmed the guard flags the exact real drift this ticket fixed — simulating the pre-fix
table (no `SSM_CONV`/`SSM_SCAN`) yields `missing == {SSM_CONV, SSM_SCAN}`.
