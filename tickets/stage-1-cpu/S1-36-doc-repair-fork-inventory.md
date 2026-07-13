---
id: S1-36
title: "Repair the docs that were actively causing bugs; inventory the carried fork diff"
stage: 1
track: docs
size: S
deps: [S1-16, S1-34, S1-35]
status: pr-open
pr: https://github.com/dillon-blake/llama-farm/pull/36
---

# S1-36 — Repair the docs that were actively causing bugs

**One-line outcome:** no document in this repo prescribes a defect or excuses one, the 18 carried
llama.cpp commits are inventoried, and an ABI landmine in the `_ffi` mirror is pinned shut.

## Why (context)

A documentation audit found that two documents were not merely stale. They were **causing bugs**.

**BLUEPRINT §6.3 prescribed the vanishing-gradient `min`.** It specified
`min(a,b) = b − relu(b−a)`. That is the same *number* as `a − relu(a−b)` and a different
*gradient*, because `ggml_step(0) == 0`: at a tie — exactly *at* the PPO clip bound — the relu
contributes no derivative, so only the argument outside it carries gradient. `b − relu(b−a)` puts
the **clipped** objective there, whose derivative in `r` is zero. S1-16 had to reject the form its
own specification mandated, and the loss is *identical* either way, so nothing reports it.

**`docs/dev/backward-coverage.md` was wrong in both directions.** It called `SOFT_MAX`'s gradient
"a real gap" that would train sink models wrongly — S1-34 established the gradient was never wrong
and the *test* had never run. And it dismissed `EXPM1` as "not on the LoRA training path" — `EXPM1`
*is* the k3 KL, and ggml's `op_expm1` was `expf(x) - 1.0f`, which makes the KL go negative near zero
and *pay* for divergence.

Separately, ADR-0001 mandates a **monthly** rebase of the llama.cpp fork, and that rebase had no
map: 18 merged fork PRs, inventoried nowhere.

## What to do

1. Correct the `min` identity in the BLUEPRINT, the S1-16 ticket, `docs/dev/grpo.md`,
   `tests/test_grpo.py`, `tests/reference_grpo.py`, and `csrc/farm_train.cpp`'s comment block —
   which still described the old form while the code did the right thing.
2. Rewrite the two wrong rows of `backward-coverage.md`, stating what was *checked* rather than
   what was assumed.
3. `docs/dev/fork-changes.md` — every carried commit: what, why, upstreamability, rebase risk.
4. `README.md` — it said "Status: stage 0" and told the reader the backward aborts (S1-00 fixed
   that long ago).
5. Correct S1-34's ticket, whose *title* still asserted what its own PR disproved.

## Out of scope

- Actually upstreaming the nine upstreamable fork commits. That is its own work; this ticket only
  establishes *which* they are.
- Resolving S1-00's disputed disposition, or the `in-fork-first` / `upstream-later` vocabulary
  split. Both are recorded in `fork-changes.md` as decisions the project owes itself.

## Acceptance criteria

- [x] No document prescribes `min(a,b) = b − relu(b−a)`.
- [x] `backward-coverage.md` states S1-34's actual finding, and lists `EXPM1` as on the training
      path.
- [x] `docs/dev/fork-changes.md` exists, and its facts are verified against the fork: merge-base is
      exactly the pin, 18 substantive commits, 18 files, +2342/−52, zero new files.
- [x] `README.md` reports the real status and links the quickstart.
- [x] The `ggml_opt_params` ctypes mirror declares `grad_clip`, and its layout is pinned from both
      C (`static_assert`/`offsetof`) and Python.

## Testing & verification

`test_the_opt_params_mirror_matches_the_C_layout` in `tests/test_ffi.py`.

The ABI find is the one with teeth. S1-10 inserted `grad_clip` into the **middle** of the public
`ggml_opt_params`, between `opt_period` and `get_opt_pars`, and the ctypes mirror never declared it.
It was harmless **purely by accident**: a float at offset 44 lands in the padding already sitting
between an int32 at 40 and an 8-aligned pointer at 48, so every later field still aligned and
`sizeof` still came to 72. The next field added anywhere before `get_opt_pars` would have shifted a
**function pointer** by four bytes — a call to a wrong address, not a wrong number, which is exactly
the *"silent memory corruption, not a link error"* that ADR-0001 says the mirror discipline exists
to prevent.

## PR notes

- Branch: `ticket/S1-36-doc-repair`.
- Upstreaming disposition: `n/a` — no vendored llama.cpp changes.
