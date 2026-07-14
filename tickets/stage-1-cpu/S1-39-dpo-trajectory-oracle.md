---
id: S1-39
title: "DPO trajectory oracle: twenty steps against float64"
stage: 1
track: python
size: S
deps: [S1-12, S1-14, S1-38]
status: done
pr: null
---

# S1-39 — DPO trajectory oracle

**One-line outcome:** a multi-step DPO run is compared per step against an independent float64
implementation, so a mis-plumbed β, a late-manifesting sign error, or an ignored reference can no
longer hide behind a falling loss.

## Why (context)

The Stage 0+1 audit confirmed (major): the convergence gate is SFT-only. `test_dpo.py`'s
log-2-at-init identity is sharp but only speaks at step zero, where the policy IS the reference;
"the loss falls" and "the model learns to prefer chosen" prove direction, not correctness. No
multi-step DPO trajectory had ever been compared against anything independent.

## What to do

- `tests/reference_dpo.py`: the DPO loss and its dlogits in float64, written from the math —
  `np.logaddexp`, not the graph's softplus-of-negation construction (SIGMOID has no ggml backward,
  which is why the graph is built that way; the oracle must not share that shape).
- `tests/reference_llama.py`: teach the float64 forward packed batches (`seq_ids` → block-causal
  mask), since DPO packs chosen+rejected into one batch.
- `tests/test_dpo_trajectory.py`: two-stage comparison — (1) the frozen reference log-ratios
  (`ll_logp_delta`, B zeroed) vs the float64 base model; (2) 20 optimizer steps of `train_dpo` vs
  the float64 trajectory, using ggml's own (stage-1-vouched) reference deltas so a failure names
  the right stage. Plus a can-fail test: β and Δ_ref must both move the oracle's trajectory.

## Acceptance criteria

- [x] Reference log-ratios match float64 per pair (observed 2.21e-06, band 1e-4).
- [x] 20-step trajectory matches per step (observed 1.11e-05, band 2e-4 — the band is wider than
      SFT's because z = Δ_policy − Δ_ref cancels two ~35-magnitude f32 sums; the ulp-cancellation
      floor is a property of the objective and is documented in the test).
- [x] The oracle can fail: β×3 moves the trajectory 2.86e-01; dropping Δ_ref moves it 3.82e-02 —
      both ≥ 20x the band, asserted.

## Testing & verification

`pytest tests/test_dpo_trajectory.py tests/test_dpo.py tests/test_convergence.py` → 18 passed.
