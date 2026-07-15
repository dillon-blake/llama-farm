---
id: S1-49
title: "Doc refresh: regenerate the fork inventory, resolve S1-32, triage the audit's minor tail"
stage: 1
track: docs
size: S
deps: [S1-36, S1-41, S1-47, S1-48]
status: done
pr: null
---

# S1-49 — Doc refresh and minor-findings triage

**One-line outcome:** the carried-fork inventory matches the fork again, S1-32 has an honest status
and a backlog carrier, and every minor finding from the 2026-07-15 audit is either fixed or parked —
so the documentation tail the audit named is closed.

## Why (context)

The 2026-07-15 audit (`docs/dev/audit-2026-07-15.md`) closed everything major as tickets S1-38 … S1-48
(PRs #45–#55). What remained was the documentation/minor tail: `docs/dev/fork-changes.md` had gone
stale as the kernel work landed, S1-32's throughput deliverables were largely missing against its own
acceptance criteria, and 23 `minor/`-tagged findings across the raw per-subsystem sections were
un-dispositioned.

## What was done

1. **Regenerated `docs/dev/fork-changes.md`.** Ground truth from
   `git -C vendor/llama.cpp log --no-merges 4f37f51..HEAD`: merge-base is still exactly the pin
   `4f37f51`; **31** substantive commits (was 18), **22 files, +4900/−97** (was 18/+2342/−52),
   **2 new files** — both test files (was "zero"), **7** new op enums at the tail. Added the 13
   kernel-era commits (S1-24 … S1-47, incl. `41141dd4f` SOFT_MAX_BACK in-place and `555c44643` SSM
   state-cache) with per-commit disposition and rebase risk; updated the "where a rebase hurts"
   footprints. **Fixed the verification method:** the old regenerate recipe used a bare `git log`,
   which counts the 18 early-PR merge commits (49 total) and so disagreed with the doc's own
   "substantive" claim; the recipe now uses `--no-merges` and each command reproduces one cited
   number.
2. **Resolved S1-32.** Wrote an honest `## Status` into the ticket (open → what exists / what
   doesn't / why deferred), filed `tickets/backlog/B-11-cpu-throughput-audit.md` carrying the
   remaining acceptance criteria, and — since this box is the target hardware class — ran
   `benches/cpu_train_step.py` and recorded the numbers in `docs/dev/cpu-throughput-audit.md`.
3. **Triaged the audit's 23 minor findings** — 8 fixed, 15 parked, dispositioned one line each in a
   new "Minor-findings triage (2026-07-16)" section at the end of the audit report.
4. **Added a closure-status table** at the top of the audit report: every confirmed finding → its
   closing PR/ticket, plus the three live bugs the oracles caught.

## Acceptance criteria

- [x] `fork-changes.md` facts match `git ... --no-merges 4f37f51..HEAD` and the regenerate recipe
      reproduces every cited number.
- [x] S1-32 has an honest open status; B-11 carries the remainder; a bench pass is recorded.
- [x] Every one of the 23 minor findings is fixed (named) or parked (reasoned); none dropped.
- [x] The audit report opens with a closure-status table.
- [x] `docs/support-tiers.md` and `docs/dev/p0-proof-of-gradient.md` exist; `backward-coverage.md`
      is present-tense-correct; the clip-bound and device-report tests are non-vacuous.
- [x] Full `pytest tests/ -q` green, ruff clean.

## PR notes

- Branch: `ticket/S1-49-doc-refresh`.
- Docs + small test fixes only; no vendored llama.cpp or csrc changes.
- Upstreaming disposition: **n/a** (project docs and tests).
