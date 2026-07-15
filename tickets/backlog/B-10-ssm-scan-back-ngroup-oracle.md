---
id: B-10
title: "SSM_SCAN backward n_group>1 gradient oracle + Mamba-2 e2e"
stage: backlog
track: kernels
size: M
deps: [S1-31, S1-47]
status: open
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

- [ ] `test-backend-ops grad -o SSM_SCAN` passes on `n_group ∈ {2, 4}` cases within ADR-0002
      tolerance, reproducibly.
- [ ] A `mamba2` e2e: preflight trainable, one-step all-gradients < 1e-3 rel vs float64, trajectory
      < 1e-4, loss falls.
- [ ] The `ssm_B->ne[1] == 1` refusal is removed and no longer reachable on a supported config.
- [ ] Thread-count bitwise-identical `dB`/`dC` on an `n_group > 1` graph.
