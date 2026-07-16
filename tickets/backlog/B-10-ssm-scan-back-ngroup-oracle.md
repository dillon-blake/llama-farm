---
id: B-10
title: "SSM_SCAN backward n_group>1 gradient oracle + Mamba-2 e2e"
stage: backlog
track: kernels
size: M
deps: [S1-31, S1-47]
status: done
pr: null
---

# B-10 — SSM_SCAN backward n_group>1 oracle, and a Mamba-2 e2e

**One-line outcome:** give the `SSM_SCAN` backward's `n_group > 1` group-index routing a
finite-difference oracle and a real end-to-end training fixture, then lift the loud refusal S1-47
put on the training path.

**Activation trigger:** Mamba-2 / Falcon-H1 / any `n_group > 1` recurrent arch enters training
scope. Mamba-1 (`n_group == 1`) is fully verified and is the only SSM arch trained today (S1-47).

## Why (context)

The `ggml_compute_forward_ssm_scan_back_f32` kernel routes heads to per-group `dB`/`dC` slabs via
`g = h / (nh/ng)` and reduces over the `nh/ng` heads in each group. That routing has **zero**
gradient verification:

- Every grad-enabled `test_ssm_scan` case in `test-backend-ops.cpp` is `n_group == 1` (the Mamba-1
  per-state-`A` case and the Mamba-2 scalar-`A` case both use one group). The `n_group > 1` shapes
  (lines ~9441-9443) exceed `grad_nmax()` (10000 elements) and are silently skipped by MODE_GRAD —
  they run forward-only and print OK having checked nothing (the audit's `moe-ssm` third major).
- S1-47's e2e float64 oracle is Mamba-1 only, so it does not exercise the group reduction either.

Because the routing is unproven, S1-47 made the `SSM_SCAN` backward switch assert `ssm_B->ne[1] ==
1` (`ggml/src/ggml.c`), so a `n_group > 1` training attempt aborts loudly rather than training on a
possibly-silently-wrong gradient. This ticket earns the right to remove that assert.

## What to do

- **MODE_GRAD:** add grad-enabled `test_ssm_scan` cases with `n_group == 2` and `n_group == 4` at
  dims small enough to stay under `grad_nmax()` (e.g. `d_state=8, head_dim=2, n_head=4, n_group=2,
  n_seq_tokens=4, n_seqs=1`), covering both the scalar-`A` (Mamba-2) and the group reduction. Note
  the `random_device` seeding fragility (backward-coverage.md): these exp-heavy recurrences are
  finite-difference-touchy, so keep dims and `A` magnitude modest and justify any per-case eps.
- **Reference + e2e:** a `gen_tiny_mamba2.py` fixture (arch `mamba2`, `ssm.group_count > 1`) and a
  `reference_mamba2.py` (scalar-`A`, grouped `B`/`C`), following the S1-47 pattern — self-audited
  float64 forward/backward, one-step all-gradients, trajectory, e2e loss-falls.
- **Lift the refusal:** once the oracle is green, remove the `ssm_B->ne[1] == 1` assert in the
  `SSM_SCAN` backward switch and let the preflight report `n_group > 1` archs trainable.
- **Determinism (gate G-B):** the group reduction accumulates several heads into one `dB`/`dC`
  slab; the kernel threads by sequence so a single thread owns a whole group, but add the
  thread-count bitwise-identity check S1-31 named and never shipped.

## Acceptance criteria

- [x] `test-backend-ops grad -o SSM_SCAN` passes on `n_group ∈ {2, 4}` cases within ADR-0002
      tolerance, reproducibly. (6 gradients compared now, was 2 — the 4 new `n_group>1` cases cover
      both A branches; `tests/test_backend_ops_grad.py::...[SSM_SCAN]` asserts `compared == 6`
      indirectly via `compared > 0` and the run is green.)
- [x] A `mamba2` e2e: preflight trainable, one-step all-gradients < 1e-3 rel vs float64, trajectory
      < 1e-4, loss falls. (Observed: one-step worst 9.2e-7 over 8 LoRA tensors, loss 1.3e-8;
      24-step trajectory worst 4.7e-7; loss 6.27 -> 6.05.)
- [x] The `ssm_B->ne[1] == 1` refusal is removed and no longer reachable on a supported config.
- [x] Thread-count bitwise-identical `dB`/`dC` on an `n_group > 1` graph. (n_threads 1 vs 4, all 8
      gradient tensors byte-identical.)

## Resolution (2026-07-16)

**The kernel already routed groups.** `ggml_compute_forward_ssm_scan_back_f32` was written (S1-31)
with the group fold in place: `g = h/(nh/ng)` and `+=` into group-sized `dB`/`dC` slabs, threaded by
sequence so a whole group is owned by one thread. B-10 did **not** need to change that arithmetic --
it was correct but unverified and blocked by the `ne[1]==1` assert S1-47 added in `ggml.c`. This
ticket earned the assert's removal by proving the routing two independent ways:

1. **MODE_GRAD (FD of ggml's own forward).** Four tiny `n_group>1` `test_ssm_scan` cases added, small
   enough (`ngrads` a few hundred, `grad_nmax()` is 10000) to be *compared*, not skipped: `n_group`
   in {2,4} x {per-state-A (`head_dim==1`), scalar-A (`head_dim>1`)}, each with `n_head/n_group==2`
   so a group's `dB`/`dC` genuinely folds two heads. `grad -o SSM_SCAN` now compares 6 (was 2).
2. **A float64 Mamba-2 oracle** (`tests/reference_mamba2.py`, self-audited by FD of its own forward
   at 2.1e-7) matches ggml's real `n_group=2` training gradient at ~1e-6, through a `gen_tiny_mamba2`
   fixture (arch `mamba2`, `ssm.group_count=2`) trained on the real stack. `ssm_in`'s gradient flows
   back through the grouped scan's `dB`/`dC` fold, so this is the independent check MODE_GRAD (which
   differences the kernel against itself) structurally cannot be.

**Routing semantics (as derived from the forward kernel):** heads are partitioned into `n_group`
contiguous blocks of `n_head/n_group` heads (`np.repeat(arange(ng), nh/ng)`); head `h` reads B/C row
`g = h // (nh/ng)`; the backward folds every head of a group back into that one row (a GQA-style
sum). Identical routing in the scalar-A and per-state-A branches.

**Coverage boundary after this ticket:**
- Verified (MODE_GRAD + float64 e2e): `SSM_SCAN` backward at `n_group` in {1,2,4}, both A branches;
  Mamba-2 trains end-to-end (`ssm_in`/`ssm_out` LoRA).
- Still eval-only (no gradient): the `xbc_overlap` `test_ssm_scan` case (x/B/C are views of one
  tensor; `ggml_set_param` refuses a view). Real layout, aliasing backward unproven -- unchanged by
  B-10.
- Still refused (by design): `A`/`D`/conv-weight/`dt_bias` gradients (frozen; ROADMAP S5 / B-09).
