---
id: S1-38
title: "Convergence gate off-unit variants: scale, weight decay, rank, thread determinism"
stage: 1
track: python
size: S
deps: [S1-12]
status: done
pr: null
---

# S1-38 — Convergence gate off-unit variants

**One-line outcome:** the gate can catch bugs the recorded config is numerically blind to — a
dropped/doubled LoRA scale factor, an ignored weight decay, a rank-dependent shape error, and a
thread-count-dependent reduction.

## Why (context)

The Stage 0+1 audit (2026-07-15) rated this **critical**: every numeric oracle in the suite ran the
recorded config, whose `alpha == rank` makes the effective LoRA scale exactly 1.0, whose
`weight_decay == 0` multiplies by exactly 1, and whose `rank == 4` gives the rank axis a single
data point. A stack that dropped the `alpha/rank` factor (`llama-adapter.h:52-57`), applied it
twice, ignored `user_scale`, or never plumbed `wd` would have passed the entire gate. The
finite-difference checks cannot see this class either: they prove forward/backward *consistency*,
not that either side applies the scale the GGUF asked for.

Separately, `tests/convergence/README.md` claimed the curve is bit-identical across 1/2/4 threads;
nothing asserted it.

## What to do

- `tests/convergence/config.py`: a frozen `RunSpec` (defaults = the recorded run) so the harness
  can run controlled deviations; the dataset stays fixed across variants.
- `tests/convergence/harness.py`: extract the run/reference helpers from `test_convergence.py`,
  parameterized by spec, including the one-step all-gradients comparison.
- `tests/test_convergence_variants.py`: per variant — one-step all-28-gradients vs float64
  (the sharp check), the 40-step curve within a **measured** band, and an asserted separation
  floor proving the variant can fail.
- `tests/test_determinism.py`: 16 steps at 1/2/4 threads, exactly equal.

## Out of scope

DPO/GRPO trajectory oracles, MoE gradient oracle, full-finetune gradients (follow-up tickets from
the same audit).

## Acceptance criteria

- [x] A variant with effective scale ≠ 1.0 (alpha ≠ rank) passes the one-step all-gradients check
      and its curve band; ditto `user_scale ≠ 1`, `weight_decay > 0` (at ggml's max, 1.0),
      `rank ≠ 4`, and `grad_accum = 2` (against a reference that SUMS the window — the
      mean-vs-sum bug class).
- [x] Each variant's reference curve sits ≥ 20x its band from the recorded config's reference
      curve (asserted in the test, not assumed).
- [x] Thread-count bit-identity is asserted, not just documented.
- [x] Every band quotes the observed measurement beside it.

## Testing & verification

Measured on this host: alpha=10 drift 5.7e-04 (conditioning at scale 2.5; gradients exact at
6.2e-06), user_scale=0.5 drift 5.8e-07, wd=1.0 drift 8.5e-07 (dropping decay lands 4.0e-02 away),
rank-8 drift 1.1e-06. ggml asserts `wd <= 1.0` (`ggml-opt.cpp:984`), so 1.0 is the hardest the
decay can be pushed.
