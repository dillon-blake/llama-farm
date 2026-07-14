---
id: S1-40
title: "GRPO trajectory oracle: eight updates against float64, every branch live"
stage: 1
track: python
size: S
deps: [S1-15, S1-16, S1-38, S1-39]
status: done
pr: null
---

# S1-40 — GRPO trajectory oracle

**One-line outcome:** a multi-step GRPO run — ratio drift, a binding clip, the k3 anchor, AdamW —
is compared per step against a float64 implementation that shares none of the graph's structure,
on a run that provably exercises every branch of the objective.

## Why (context)

The Stage 0+1 audit confirmed (major): `test_grpo.py` proves the composite at a point — exact
ratio control, one step, the gradient-differentiating clip test — but no multi-step GRPO
trajectory was ever compared against anything independent. The clip only starts *mattering* as the
policy drifts off the behaviour policy, which a one-step test never sees.

## What to do

- `tests/reference_grpo.py`: add `grpo_dlogp` — the objective's gradient w.r.t. `logp_new`,
  branch selection by `np.where` (the `u <= v` selection is exactly where the blueprint's
  `b - relu(b - a)` gradient bug lived; a value-only reference provably cannot see it).
- `tests/test_grpo_trajectory.py`: hand-built FIXED rollouts (no sampling), `logp_old` seeded so
  ratios start scattered across both clip branches, advantages of both signs per group; 8 updates
  of `GRPOTrainer.grpo_step` with `kl_coef > 0` vs the float64 trajectory. Assert branch
  coverage — a trajectory whose clip never binds cannot catch a wrong clip — and assert the
  oracle can fail (moving `clip_eps` / `kl_coef` moves it).

## Acceptance criteria

- [x] 8-step trajectory matches per step: observed 1.45e-06, band 1e-4.
- [x] Branch coverage asserted: clip bound on 161/192 live token-steps, 31 unclipped, advantages
      +/− on 160 each.
- [x] The oracle can fail: `clip_eps` 0.2→0.05 moves it 8.69e-02; `kl_coef` ×5 moves it 1.36e+01 —
      both asserted ≥ 20x the band.

## Testing & verification

`pytest tests/test_grpo_trajectory.py tests/test_grpo.py` green; whole suite green.
