---
id: S1-46
title: "GRPO capture wiring: SelfVerified's first customer, and the D6 detach proven nonzero"
stage: 1
track: python
size: S
deps: [S1-13, S1-15, S1-16, S1-40]
status: done
pr: null
---

# S1-46 — GRPO sample-time capture wiring + the D6 reference detach

**One-line outcome:** the GRPO self-verification harness is finally wired to the customer S1-16 §3
named for it — sample-time `logp_old` capture (fast) cross-checked against the S1-13 chunked
recompute (naive) on every real run — and the D6 reference pass's adapter detach is pinned with a
**nonzero** adapter, so "the reference is the base model with the adapter off" is a claim a test can
now fail.

## Why (context)

The 2026-07-15 audit (`docs/dev/audit-2026-07-15.md`, preference-rl section) confirmed two majors,
both adversarially verified:

1. **major/defect** (`grpo.py`): *"Self-verification harness (verify.py) is never wired to its
   stated first customer; runtime logp_old cross-check absent. S1-16 §3 requires wiring the
   SelfVerified harness to its first real customer: sample-time logp_old capture (fast) vs the S1-13
   chunked recompute (naive). S1-15 §4 requires a recompute_logp(engine, rollouts) helper as the
   documented fallback/cross-check. Neither exists."* `verify.SelfVerified` existed and was
   unit-tested with synthetic fast/naive functions, but nothing in `src/` ever constructed one — so
   at runtime a `logp_old` that silently drifted from the policy (a different quantization, a batch
   shape that tripped a different kernel) would train GRPO on the wrong importance ratio with no
   check anywhere. `test_logp_old_matches_an_independent_recompute` (test_rollout.py) covered *one*
   greedy rollout with the last token dropped, by hand — not the helper, not a whole batch, not the
   runtime wiring.

2. **major/vacuous-test** (`test_grpo.py`): *"GRPO reference_logprobs adapter-detach is never
   verified with a nonzero adapter; the only end-to-end KL test cannot fail."* `reference_logprobs`
   (grpo.py) is the BLUEPRINT D6 reference pass: it detaches the adapter with
   `llama_set_adapters_lora(ctx, NULL, 0, NULL)`, scores `logp_ref`, and re-attaches in a `finally`.
   The only test through that path (`test_the_kl_actually_reaches_the_loss`) built its adapter with
   `create_zero_adapter` — `B = 0`, delta exactly zero — so adapter-on and adapter-off are
   numerically identical. Delete the detach and the test still passed.

### Was any of this already closed by S1-40?

Checked, because the S1-40 GRPO trajectory oracle (`tests/test_grpo_trajectory.py`) landed after the
audit ran. It closes **neither**:

- S1-40 builds **fixed, hand-authored** rollouts whose `logp_old` is *seeded noise* (`_rollouts()`
  in test_grpo_trajectory.py) precisely so the ratios start scattered across both clip branches. It
  never samples, so there is no sample-time capture to cross-check, and it never calls
  `recompute_logp` (which did not exist) — finding 1 untouched.
- S1-40 calls `GRPOTrainer.grpo_step` directly with a fixed `_logp_ref` array; it never exercises
  `reference_logprobs`' detach/re-attach, and its adapter is still `create_zero_adapter` (`B = 0`) —
  finding 2 untouched.

So both were implemented here, not de-duplicated.

## What was done

### Finding 1 — the cross-check, wired

- `src/learning_llamas/train/rollout.py`: `recompute_logp(engine, rollouts, lm_head)` — the S1-15 §4
  helper. Scores every rollout's completion tokens through the S1-13 chunked pass
  (`sequence_logprobs`) on the engine's own inference context (adapter attached, so it is the same
  policy), grading position `n_prompt-1+t` for completion token `t`, and returns the per-token
  logprobs concatenated in rollout order. Its twin `captured_logp(rollouts)` returns the sample-time
  numbers in the same layout, so the two are directly comparable. (Unlike the old by-hand check, it
  grades *every* completion token — the last included — and agreement holds.)
- `src/learning_llamas/train/grpo.py`: `train_grpo` now constructs
  `SelfVerified(captured_logp, recompute_logp(...), tolerance, name)` (`_logp_old_verifier`) and
  calls it on each iteration. On the first batch both paths run and are compared; agree, and the free
  capture is used for the rest of the run; disagree, and the run falls back — loudly, via the log and
  the report — to the recompute forever. Whichever the harness returns is scattered back onto the
  rollouts (`_apply_logp_old`) before `collate`, so **a divergent capture never reaches the loss**.
  A new `GRPOConfig.logp_old_tol` (default `5e-3`; `0.0` disables) sets the band; the check needs an
  `lm_head` (the same one the KL already requires) and is silently skipped without one.
  `GRPOResult.logp_verification` surfaces the outcome.

**Oracle / observed agreement.** The naive recompute *is* the independent ground truth (a full
forward + `ce_sparse` log-softmax, sharing no code with the incremental-decode capture; numerically
close, not bitwise, per ADR-0002). Measured capture-vs-recompute over a whole batch:

| base | temperature | max &#124;capture − recompute&#124; |
|---|---|---|
| Q4_K | 0.0 | 9.5e-07 |
| Q4_K | 1.0 | 9.5e-07 |
| F32  | 0.0 | 1.4e-06 |
| F32  | 1.0 | 9.5e-07 |

Band asserted in the tests: `1e-4` (helper) — ~2 orders of margin. Runtime default tolerance `5e-3`.

Tests (`tests/test_rollout.py`, `tests/test_grpo.py`):
- `test_recompute_logp_agrees_with_the_capture_over_a_whole_batch[0.0/1.0]` — the numeric oracle on
  a full multi-rollout batch, greedy and sampled; asserts the arrays are real logprobs (`< 0`,
  finite), not zeros that trivially agree.
- `test_train_grpo_cross_checks_logp_old_on_the_first_batch` — proves the harness is wired and *ran*
  on a real `train_grpo` call (`kl_coef = 0`, `lm_head` given): report present, `verified`, `agreed`,
  `0 < max_deviation < 5e-3`.
- `test_train_grpo_falls_back_when_the_capture_cannot_meet_the_tolerance` — `logp_old_tol = 1e-12`
  can never be met by the ~1e-6 gap, so the end-to-end fallback fires and the run still completes on
  the recompute (all losses finite).
- `test_a_corrupted_logp_old_capture_is_caught_and_the_recompute_is_used` — the mutation-proof: a
  capture poisoned by `-7.0` is caught (`using_fallback`), the values returned equal the recompute
  (not the poison), and `_apply_logp_old` writes them where `collate` reads.

### Finding 2 — the D6 detach, proven nonzero

- `tests/test_grpo.py::test_reference_logprobs_actually_detaches_the_adapter` — fills the adapter's
  `B` tensors with noise via `ll_adapter_set` (`_randomize_b`) so the delta is real (~0.55 on the
  test tokens), then pins three things the zero adapter could not:
  1. the reference differs from the adapter-on policy by that whole margin — the detach happened
     (asserted `> 1e-1`; **observed 5.5e-01**);
  2. the reference equals a context that **never had an adapter** — "off" means the base weights, not
     a scaled-down adapter (**observed ref-vs-base = 0.0 exactly**, asserted within `1e-5`);
  3. scoring the policy again reproduces it (**observed re-attach delta = 0.0 exactly**,
     `np.array_equal`) — the `finally` re-attached; a missing re-attach would collapse onto the base.

## Acceptance criteria

- [x] `recompute_logp` (S1-15 §4) and `captured_logp` exist and are slot-for-slot aligned.
- [x] `SelfVerified` is constructed and invoked in `train_grpo` (S1-16 §3) with capture as the
      default/fast path and the S1-13 recompute as the verifier/naive path; the verified values are
      what the update trains on.
- [x] Capture and recompute agree on real rollouts within a measured band (observed ≤ 1.4e-06,
      band 1e-4), greedy and sampled, on Q4_K and F32.
- [x] The guard is mutation-proofed: a corrupted capture is caught and the recompute is used; an
      impossibly-tight tolerance forces the end-to-end fallback and the run still completes.
- [x] The D6 reference detach is verified with a **nonzero** adapter: detach margin, off == base
      exactly, and re-attach exactly.
- [x] S1-40's coverage checked and found to close neither finding (documented above).
- [x] Full suite green, ruff clean.

## Testing & verification

`pytest tests/ -q`: **362 passed** (356 baseline + 6 new). Ruff clean over `src/` and `tests/`.
No `csrc/`/`vendor/` changes — pure Python over existing FFI, so no rebuild and no new
`test-backend-ops` cases. Observed numbers measured on this host (4-core CPU, F32/Q4_K tiny-llama
fixtures).
