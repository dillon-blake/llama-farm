---
id: S1-37
title: "MODE_GRAD's metric divides by (gn + ga) — and a zero gradient is a NaN free pass"
stage: 1
track: kernels
size: M
deps: [S1-28]
status: pr-open
pr: https://github.com/dillon-blake/llama.cpp/pull/22
---

# S1-37 — `mean_abs_asymm` divides by the signed sum, and NaN passes

**One-line outcome:** MODE_GRAD's error metric is a symmetric relative error, an exactly-zero
gradient is not an automatic pass, and TANH / SIGMOID / CROSS_ENTROPY_LOSS are checked for real.

## Why (context)

`tests/test-backend-ops.cpp`:

```c
const float asymm = (a[i] - b[i]) / (a[i] + b[i]);
sum += fabsf(asymm);
```

Two separate defects, and they compound.

**1. The denominator is the SIGNED sum.** So the ratio goes to infinity whenever the two gradients
nearly cancel — and in particular whenever the *true* gradient is near **zero** and both are
rounding noise. That is not an edge case. Any op whose output is a **product** has near-zero
gradient elements for ordinary inputs (every GLU: `dx = dy · g · act'(x)`), and any op that sums
over a *selected subset* can manufacture them (MUL_MAT_ID's expert routing: with 5 tokens and 3
experts, `d_as` is a sum of one or two random products).

Measured on `GLU_BACK`, with the kernel verified exact (~1e-7) against a float64 reference the
entire time:

| | MAA |
|---|---|
| FD noise, 12 runs | **up to 0.80** |
| a genuinely broken kernel | **0.18 – 0.60** |

**The noise overlaps the defects.** No tolerance can separate them. S1-28 therefore had to ship its
MODE_GRAD cases as a *wiring* check with a 0.9 bound, and put the numerics in a separate float64
oracle (`tests/test-glu-back.cpp`).

**2. An exactly-zero gradient pair is an unconditional PASS.** `gn == ga == 0` gives `0/0 = NaN`,
`sum` becomes NaN, `MAA` becomes NaN — and `NaN > max_maa_err()` is **false**. The case passes.

This is not hypothetical either. `tanh(150)` and `sigmoid(150)` are `1.0` to float precision, so
their derivatives are **exactly zero**, and `test_unary` initializes in `[-150, 150]`. **TANH and
SIGMOID have been on the vendor-bump allowlist since S1-19 on the strength of a NaN.** Likewise
`CROSS_ENTROPY_LOSS` at `[-100, 100]`, where the softmax is one-hot to float precision.

## What to do

1. **Fix the metric:** `denom = |a| + |b|`; `asymm = denom > 0 ? (a-b)/denom : 0`. This is what a
   symmetric relative error is, it is bounded in `[-1, 1]`, and since `|a|+|b| >= |a+b|` it can only
   ever make MAA **smaller** — so no test can start failing *because of the division change*.
   Measured on GLU: noise floor drops from 0.80 to **0.022**, and every injected defect is caught
   with **3.6–12x** margin.
2. **Then fix what the NaN was hiding.** With `0/0 -> 0` instead of NaN, TANH, SIGMOID and
   CROSS_ENTROPY_LOSS are checked for the first time and **fail** — at MAA 0.15, 0.35 and 0.59
   respectively, on rounding noise in their saturated tails, with the kernels fine.

   They need conditioning, and it is not a one-liner: `test_unary` uses `grad_eps = 15.0` against a
   `[-150, 150]` range, so simply narrowing the range makes the FD step larger than the region being
   measured. Expect to co-design the range and the step per op, and to need a float64 oracle for the
   ones where the FD cannot be made to behave — the S1-28 precedent.
3. **Re-tighten `test_glu`'s bound** from the honest-but-useless 0.9 to something that means
   something (~5e-2 measured), and delete the comment saying it is only a wiring check.
4. **Re-audit the allowlist.** Any op whose gradient is exactly zero anywhere has been passing for
   free. Enumerate them before trusting the gate again.

## Out of scope

- New VJPs. This is entirely about the harness and what it was failing to check.

## Acceptance criteria

- [ ] `mean_abs_asymm` divides by `|a| + |b|`, and a zero-zero pair contributes 0, not NaN.
- [ ] TANH, SIGMOID and CROSS_ENTROPY_LOSS pass a MODE_GRAD check that can actually fail —
      demonstrated by breaking each kernel and watching it go red.
- [ ] `test_glu`'s `max_maa_err` is back to a measured value with mutation evidence on both sides.
- [ ] `docs/dev/backward-coverage.md` records which allowlist entries were passing on NaN.

## PR notes

- Branch: `ticket/S1-37-mean-abs-asymm`.
- Upstreaming disposition: **upstream-early**. The signed denominator and the NaN pass are plain
  bugs in upstream's own test harness, and they make its gradient gate weaker than it looks. Note
  llama.cpp does not accept predominantly AI-generated PRs (`AGENTS.md`) — private forks are exempt,
  but an upstream PR is a human commitment. See `docs/dev/fork-changes.md`.
